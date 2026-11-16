//! k-means nearest neighbour (kMkNN) index. Exact nearest neighbour search
//! that leverages k-means clustering and the triangle inequality to prune
//! distance computations.

use faer::{Accum, MatMut, MatRef, Par, RowRef};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use thousands::Separable;

use crate::prelude::*;
use crate::utils::dist::*;
use crate::utils::gemm::gemm;
use crate::utils::k_means_utils::*;
use crate::utils::verbosity::{BuildLog, QueryProgress};
use crate::utils::*;
use std::time::Instant;

/// Nearest cells every query scans in full before the batched pass, to bound
/// its k-th distance. Fewer leave a looser bound and wider annuli; more scan
/// whole cells the annulus would have cut.
const SEED_CELLS: usize = 4;

/// Bounds on the queries per block of the batched search, see
/// [`KmknnIndex::search_batched`]. Not tuned.
const MIN_QUERY_BLOCK: usize = 256;
/// See [`MIN_QUERY_BLOCK`]. Bounds the per-block range list and heaps.
const MAX_QUERY_BLOCK: usize = 1024;

/// Fewest queries sharing a row range before it is scored with a GEMM. Below
/// it the gather and the GEMM set-up cost more than the four-row SIMD scan
/// saves.
const GEMM_MIN_GROUP: usize = 8;

/// Most queries in one GEMM group. Not tuned.
const MAX_GEMM_GROUP: usize = 64;

/// How far a GEMM group's shared row range may exceed the sum of its queries'
/// own annulus ranges. The GEMM scores every query against the shared range,
/// so this caps the wasted distances. Not tuned.
const MAX_RANGE_WIDENING: usize = 2;

/// Extra candidates kept per query in the batched search, rescored exactly
/// with the rest and then dropped. Covers neighbours that
/// `|q|^2 + |x|^2 - 2 q.x` misranks across the k-th distance in `f32`.
const RERANK_EXTRA: usize = 8;

/// Rows per GEMM tile, bounding the distance tile the heaps scan.
const CELL_ROW_TILE: usize = 512;

/// Fewest queries for which [`KmknnIndex::query_batch`] takes the batched
/// path. Below it the per-call `|x|^2` pass over the whole index is not
/// amortised. Not tuned.
const BATCH_MIN_QUERIES: usize = 1024;

////////////////
// KmknnIndex //
////////////////

/// kMkNN (k-means nearest neighbour) index
///
/// An exact nearest neighbour index that partitions the data via k-means and
/// uses the triangle inequality for pruning. For Cosine distance, vectors are
/// L2-normalised at build time and the index operates in Euclidean space
/// internally (squared Euclidean on the unit sphere is monotone in Cosine
/// distance).
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct KmknnIndex<T> {
    /// Vector data in cluster-sorted order. For Cosine, L2-normalised.
    pub vectors_flat: Vec<T>,
    /// Embedding dimensions
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// The user's original distance metric (used for output conversion)
    metric: Dist,
    /// Cluster centres (nlist * dim elements)
    centroids: Vec<T>,
    /// CSR offsets into `vectors_flat` per cluster (length nlist + 1)
    offsets: Vec<u32>,
    /// Number of clusters
    nlist: usize,
    /// Mapping from internal position to original sample index
    original_ids: Vec<u32>,
    /// Squared distance from each point to its assigned centroid
    point_to_centroid_sq: Vec<T>,
    /// Per-cluster squared maximum point-to-centroid distance
    cluster_radii_sq: Vec<T>,
    /// Global max of `cluster_radii_sq` for early termination
    max_cluster_radius_sq: T,
}

////////////////////
// VectorDistance //
////////////////////

impl<T> VectorDistance<T> for KmknnIndex<T>
where
    T: AnnSearchFloat,
{
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    fn dim(&self) -> usize {
        self.dim
    }

    /// Not populated: for Cosine the vectors are pre-normalised, and for
    /// Euclidean norms are unused. Callers relying on the cosine_distance
    /// default of `VectorDistance` should use `ExhaustiveIndex` instead.
    fn norms(&self) -> &[T] {
        &[]
    }
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for KmknnIndex<T>
where
    T: AnnSearchFloat,
{
    fn centroids(&self) -> &[T] {
        &self.centroids
    }

    fn dim(&self) -> usize {
        self.dim
    }

    /// Always Euclidean: Cosine is handled via normalisation at build time.
    fn metric(&self) -> Dist {
        Dist::SquaredEuclidean
    }

    fn nlist(&self) -> usize {
        self.nlist
    }

    fn centroids_norm(&self) -> &[T] {
        &[]
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for KmknnIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////////
// Index build //
/////////////////

impl<T> KmknnIndex<T>
where
    T: AnnSearchFloat,
{
    /// Build a kMkNN index.
    ///
    /// For Cosine, vectors are L2-normalised and the index operates on
    /// Euclidean distances internally. All triangle-inequality bounds use
    /// true Euclidean distances.
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `nlist` - Optional number of clusters. Defaults to `sqrt(n)`.
    /// * `max_iters` - Optional k-means trainings parameters, see
    ///   [KMeansTrainingParams]. If not provided, will default to sensible
    ///   defaults.
    /// * `seed` - Random seed for reproducibility
    /// * `verbosity` - `Normal` prints start and end, `Detailed` adds timed
    ///   stages
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    pub fn build(
        data: impl AnnMatrix<T>,
        metric: Dist,
        nlist: Option<usize>,
        k_means_params: Option<KMeansTrainingParams>,
        seed: usize,
        verbosity: Verbosity,
    ) -> Result<Self, AnnSearchErrors> {
        let log = BuildLog::new("kMkNN index", verbosity);
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (mut vectors_flat, n, dim) = data.into_row_major();

        // Cosine: normalise in place and proceed in Euclidean space
        if metric == Dist::Cosine {
            for i in 0..n {
                let start = i * dim;
                normalise_vector(&mut vectors_flat[start..start + dim]);
            }
        }

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        let start = Instant::now();
        let centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &Dist::SquaredEuclidean,
            k_means_params,
            seed,
            Verbosity::Quiet,
        )?;
        if verbosity.detailed_verbosity() {
            println!(
                " Trained k-means ({} centroids, {} samples) in {:.2?}",
                nlist.separate_with_underscores(),
                n_train.separate_with_underscores(),
                start.elapsed()
            );
        }

        let dummy_data_norms = vec![T::one(); n];
        let dummy_centroid_norms = vec![T::one(); nlist];
        let start = Instant::now();
        let assignments = assign_all_parallel(
            &vectors_flat,
            &dummy_data_norms,
            dim,
            n,
            &centroids,
            &dummy_centroid_norms,
            nlist,
            &Dist::SquaredEuclidean,
        );
        if verbosity.detailed_verbosity() {
            println!(
                " Assigned {} vectors in {:.2?}",
                n.separate_with_underscores(),
                start.elapsed()
            );
        }

        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);

        // Rows within a cell sorted by distance to its centroid, so the
        // triangle-inequality annulus |d(q,c) - d(p,c)| < r_k is a contiguous
        // row range found by binary search
        let cells: Vec<Vec<(T, u32)>> = (0..nlist)
            .into_par_iter()
            .map(|c| {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let mut rows: Vec<(T, u32)> = all_indices
                    [offsets[c] as usize..offsets[c + 1] as usize]
                    .iter()
                    .map(|&old| {
                        let v = &vectors_flat[old as usize * dim..(old as usize + 1) * dim];
                        (euclidean_distance_static(v, cent), old)
                    })
                    .collect();
                rows.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
                rows
            })
            .collect();

        let mut new_to_old = Vec::with_capacity(n);
        let mut point_to_centroid_sq = Vec::with_capacity(n);
        let mut cluster_radii_sq = vec![T::zero(); nlist];
        let mut max_cluster_radius_sq = T::zero();
        for (c, rows) in cells.iter().enumerate() {
            for &(d_sq, old) in rows {
                point_to_centroid_sq.push(d_sq);
                new_to_old.push(old);
            }
            let max_d_sq = rows.last().map_or(T::zero(), |r| r.0);
            cluster_radii_sq[c] = max_d_sq;
            if max_d_sq > max_cluster_radius_sq {
                max_cluster_radius_sq = max_d_sq;
            }
        }
        let mut new_vectors_flat = Vec::with_capacity(n * dim);
        for &old_id in &new_to_old {
            let start = old_id as usize * dim;
            new_vectors_flat.extend_from_slice(&vectors_flat[start..start + dim]);
        }

        log.finish();

        Ok(Self {
            vectors_flat: new_vectors_flat,
            dim,
            n,
            metric,
            centroids,
            offsets,
            nlist,
            original_ids: new_to_old,
            point_to_centroid_sq,
            cluster_radii_sq,
            max_cluster_radius_sq,
        })
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Index size `in n bytes`
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.vectors_flat.capacity() * std::mem::size_of::<T>()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.offsets.capacity() * std::mem::size_of::<u32>()
            + self.original_ids.capacity() * std::mem::size_of::<u32>()
            + self.point_to_centroid_sq.capacity() * std::mem::size_of::<T>()
            + self.cluster_radii_sq.capacity() * std::mem::size_of::<T>()
    }
}

///////////
// Query //
///////////

impl<T> KmknnIndex<T>
where
    T: AnnSearchFloat,
{
    /// Core search loop. Assumes `q` is already in the correct space (i.e.
    /// L2-normalised if the index is in Cosine mode). Output conversion
    /// (squared Euclidean → Cosine via `d²/2`) still applies.
    ///
    /// ### Params
    ///
    /// * `q` - The query vector/slice
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    #[inline]
    fn query_internal(&self, q: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        let k = k.min(self.n);

        let mut cluster_dists: Vec<(T, usize)> = (0..self.nlist)
            .map(|c| {
                let cent = &self.centroids[c * self.dim..(c + 1) * self.dim];
                (euclidean_distance_static(q, cent), c)
            })
            .collect();
        cluster_dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);
        let mut r_k_sq = T::infinity();
        let mut r_k = T::infinity();

        let max_radius = self.max_cluster_radius_sq.sqrt();

        for &(d_qc_sq, c) in &cluster_dists {
            let d_qc = d_qc_sq.sqrt();

            if d_qc - max_radius >= r_k {
                break;
            }

            let r_c = self.cluster_radii_sq[c].sqrt();
            if d_qc - r_c >= r_k {
                continue;
            }

            let (start, end) = self.annulus(c, d_qc, r_k);

            for pos in start..end {
                let d_pc = self.point_to_centroid_sq[pos].sqrt();
                // rows ascend in d_pc, and r_k only shrinks after `annulus`
                if d_pc - d_qc >= r_k {
                    break;
                }
                if d_qc - d_pc >= r_k {
                    continue;
                }

                let v = &self.vectors_flat[pos * self.dim..(pos + 1) * self.dim];
                let d_qp_sq = euclidean_distance_static(v, q);

                if heap.len() < k {
                    heap.push((OrderedFloat(d_qp_sq), pos));
                    if heap.len() == k {
                        r_k_sq = heap.peek().unwrap().0 .0;
                        r_k = r_k_sq.sqrt();
                    }
                } else if d_qp_sq < r_k_sq {
                    heap.pop();
                    heap.push((OrderedFloat(d_qp_sq), pos));
                    r_k_sq = heap.peek().unwrap().0 .0;
                    r_k = r_k_sq.sqrt();
                }
            }
        }

        let mut results: Vec<_> = heap.into_iter().collect();
        results.sort_unstable_by_key(|&(d, _)| d);

        let half = T::one() / (T::one() + T::one());
        let (distances, indices): (Vec<_>, Vec<_>) = results
            .into_iter()
            .map(|(OrderedFloat(d_sq), pos)| {
                let out = match self.metric {
                    Dist::SquaredEuclidean => d_sq,
                    Dist::Cosine => d_sq * half,
                    Dist::Manhattan => unreachable!(),
                };
                (out, self.original_ids[pos] as usize)
            })
            .unzip();

        (indices, distances)
    }

    /// Rows of a cell that the triangle inequality cannot exclude
    ///
    /// Rows are stored in ascending distance to their centroid, so the points
    /// with `|d(q,c) - d(p,c)| < r_k` form one contiguous range. The bounds are
    /// inclusive on the squared distances, so the range is a superset of the
    /// rows the strict test keeps.
    ///
    /// ### Params
    ///
    /// * `c` - Cell
    /// * `d_qc` - Euclidean (not squared) distance from the query to centroid
    ///   `c`
    /// * `r_k` - Euclidean (not squared) upper bound on the k-th neighbour
    ///   distance; infinity keeps the whole cell
    ///
    /// ### Returns
    ///
    /// `(start, end)` absolute row positions
    #[inline]
    fn annulus(&self, c: usize, d_qc: T, r_k: T) -> (usize, usize) {
        let start = self.offsets[c] as usize;
        let end = self.offsets[c + 1] as usize;
        if r_k.is_infinite() {
            return (start, end);
        }
        let rows = &self.point_to_centroid_sq[start..end];
        let lo = d_qc - r_k;
        let lo = if lo > T::zero() {
            let lo_sq = lo * lo;
            rows.partition_point(|&d| d < lo_sq)
        } else {
            0
        };
        let hi = d_qc + r_k;
        let hi_sq = hi * hi;
        let hi = lo + rows[lo..].partition_point(|&d| d <= hi_sq);
        (start + lo, start + hi)
    }

    /// Batched exact search for many queries at once
    ///
    /// Two phases per block of queries, ordered by nearest centroid so a block
    /// shares cells. First every query scans its [`SEED_CELLS`] nearest cells
    /// in full, which after an exact rerank gives an upper bound on its k-th
    /// distance. With that bound fixed, each query keeps every further cell
    /// the radius test cannot exclude, restricted to the [`Self::annulus`]
    /// row range. Since the bound only overestimates the walk's final
    /// distance, the kept rows are a superset of what [`Self::query`] reads.
    /// Queries sharing a cell are sorted by range start and split into groups
    /// whose shared range stays within [`MAX_RANGE_WIDENING`] of their own
    /// ranges; groups of at least [`GEMM_MIN_GROUP`] are scored by one GEMM
    /// over the shared range, the rest by the four-row SIMD scan. GEMM scores
    /// use `|q|^2 + |x|^2 - 2 q.x`; [`RERANK_EXTRA`] candidates past `k` are
    /// kept and every survivor is rescored exactly before cutting to `k`.
    ///
    /// ### Params
    ///
    /// * `queries` - Row-major queries, `nq * dim`, already normalised for
    ///   Cosine
    /// * `nq` - Number of queries
    /// * `k` - Number of neighbours
    /// * `progress` - Ticked once per query
    ///
    /// ### Returns
    ///
    /// Per query, in input order, `(original ids, distances)` sorted nearest
    /// first, in the index's metric
    fn search_batched(
        &self,
        queries: &[T],
        nq: usize,
        k: usize,
        progress: &QueryProgress,
    ) -> Vec<(Vec<usize>, Vec<T>)> {
        let dim = self.dim;
        let nlist = self.nlist;
        let k = k.min(self.n);
        let k_heap = (k + RERANK_EXTRA).min(self.n);
        let n_seed = SEED_CELLS.min(nlist);
        let two = T::one() + T::one();
        let half = T::one() / two;
        let max_radius = self.max_cluster_radius_sq.sqrt();
        let radii: Vec<T> = self.cluster_radii_sq.iter().map(|r| r.sqrt()).collect();

        let q_sq: Vec<T> = queries[..nq * dim]
            .par_chunks_exact(dim)
            .map(|q| T::dot_simd(q, q))
            .collect();
        let x_sq: Vec<T> = self
            .vectors_flat
            .par_chunks_exact(dim)
            .map(|x| T::dot_simd(x, x))
            .collect();

        let nearest: Vec<u32> = queries[..nq * dim]
            .par_chunks_exact(dim)
            .map(|q| {
                let mut best = (T::infinity(), 0u32);
                for c in 0..nlist {
                    let d = euclidean_distance_static(q, &self.centroids[c * dim..(c + 1) * dim]);
                    if d < best.0 {
                        best = (d, c as u32);
                    }
                }
                best.1
            })
            .collect();
        let mut order: Vec<u32> = (0..nq as u32).collect();
        order.par_sort_by_key(|&i| nearest[i as usize]);
        let block_size =
            (nq / rayon::current_num_threads().max(1)).clamp(MIN_QUERY_BLOCK, MAX_QUERY_BLOCK);

        let block_results: Vec<(u32, Vec<usize>, Vec<T>)> = order
            .par_chunks(block_size)
            .map_init(
                || (Vec::new(), Vec::new()),
                |(q_tile, d_tile): &mut (Vec<T>, Vec<T>), block| {
                    let bq = |bi: usize| {
                        let qi = block[bi] as usize;
                        &queries[qi * dim..(qi + 1) * dim]
                    };
                    let mut heaps: Vec<BoundedMaxHeap<T>> = (0..block.len())
                        .map(|_| BoundedMaxHeap::new(k_heap))
                        .collect();

                    // score rows `lo..hi` for the block queries in `members`
                    let mut score =
                        |lo: usize, hi: usize, members: &[u32], heaps: &mut [BoundedMaxHeap<T>]| {
                            if members.len() < GEMM_MIN_GROUP {
                                for &bi in members {
                                    let q = bq(bi as usize);
                                    let heap = &mut heaps[bi as usize];
                                    let tail = lo + (hi - lo) / 4 * 4;
                                    for base in (lo..tail).step_by(4) {
                                        let d = self.euclidean_distance_to_query_4(base, q);
                                        for o in 0..4 {
                                            heap.push(d[o], base + o);
                                        }
                                    }
                                    for pos in tail..hi {
                                        heap.push(self.euclidean_distance_to_query(pos, q), pos);
                                    }
                                }
                                return;
                            }
                            let g = members.len();
                            q_tile.clear();
                            for &bi in members {
                                q_tile.extend_from_slice(bq(bi as usize));
                            }
                            let q_mat = MatRef::from_row_major_slice(q_tile, g, dim);
                            for r0 in (lo..hi).step_by(CELL_ROW_TILE) {
                                let r1 = (r0 + CELL_ROW_TILE).min(hi);
                                let rows = r1 - r0;
                                d_tile.resize(g * rows, T::zero());
                                gemm(
                                    MatMut::from_row_major_slice_mut(
                                        &mut d_tile[..g * rows],
                                        g,
                                        rows,
                                    ),
                                    Accum::Replace,
                                    q_mat,
                                    MatRef::from_row_major_slice(
                                        &self.vectors_flat[r0 * dim..r1 * dim],
                                        rows,
                                        dim,
                                    )
                                    .transpose(),
                                    T::one(),
                                    Par::Seq,
                                );
                                for (gi, &bi) in members.iter().enumerate() {
                                    let qn = q_sq[block[bi as usize] as usize];
                                    let heap = &mut heaps[bi as usize];
                                    for (j, &d) in
                                        d_tile[gi * rows..(gi + 1) * rows].iter().enumerate()
                                    {
                                        let dist = qn + x_sq[r0 + j] - two * d;
                                        if dist < heap.threshold() {
                                            heap.push(dist, r0 + j);
                                        }
                                    }
                                }
                            }
                        };

                    // per query: centroids by ascending distance
                    let cds: Vec<Vec<(T, u32)>> = (0..block.len())
                        .map(|bi| {
                            let q = bq(bi);
                            let mut cd: Vec<(T, u32)> = (0..nlist)
                                .map(|c| {
                                    let cent = &self.centroids[c * dim..(c + 1) * dim];
                                    (euclidean_distance_static(q, cent).sqrt(), c as u32)
                                })
                                .collect();
                            cd.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                            cd
                        })
                        .collect();

                    // phase 1: seed cells in full
                    let mut pairs: Vec<u64> = Vec::with_capacity(block.len() * n_seed);
                    for (bi, cd) in cds.iter().enumerate() {
                        for &(_, c) in &cd[..n_seed] {
                            pairs.push(((c as u64) << 32) | bi as u64);
                        }
                    }
                    pairs.sort_unstable();
                    let mut members: Vec<u32> = Vec::new();
                    let mut g0 = 0;
                    while g0 < pairs.len() {
                        let c = (pairs[g0] >> 32) as usize;
                        let mut g1 = g0;
                        members.clear();
                        while g1 < pairs.len() && (pairs[g1] >> 32) as usize == c {
                            members.push((pairs[g1] & 0xFFFF_FFFF) as u32);
                            g1 += 1;
                        }
                        let (lo, hi) = (self.offsets[c] as usize, self.offsets[c + 1] as usize);
                        score(lo, hi, &members, &mut heaps);
                        g0 = g1;
                    }

                    // exact rerank, then the k-th distance bounds phase 2
                    let mut r_k = vec![T::infinity(); block.len()];
                    for (bi, heap) in heaps.iter_mut().enumerate() {
                        let q = bq(bi);
                        let mut e: Vec<(T, usize)> = heap
                            .ids()
                            .iter()
                            .map(|&id| (self.euclidean_distance_to_query(id, q), id))
                            .collect();
                        e.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                        if e.len() >= k {
                            r_k[bi] = e[k - 1].0.sqrt();
                        }
                        heap.clear();
                        for (d, id) in e {
                            heap.push(d, id);
                        }
                    }

                    // phase 2: annulus ranges in every cell the bound keeps
                    // (cell, lo, hi, block query)
                    let mut ranges: Vec<(u32, u32, u32, u32)> = Vec::new();
                    for (bi, cd) in cds.iter().enumerate() {
                        let rk = r_k[bi];
                        for &(d_qc, c) in &cd[n_seed..] {
                            if d_qc - max_radius >= rk {
                                break;
                            }
                            if d_qc - radii[c as usize] >= rk {
                                continue;
                            }
                            let (lo, hi) = self.annulus(c as usize, d_qc, rk);
                            if lo < hi {
                                ranges.push((c, lo as u32, hi as u32, bi as u32));
                            }
                        }
                    }
                    ranges.sort_unstable();

                    let mut g0 = 0;
                    while g0 < ranges.len() {
                        let c = ranges[g0].0;
                        let mut g1 = g0;
                        while g1 < ranges.len() && ranges[g1].0 == c {
                            g1 += 1;
                        }
                        // greedy groups along range start
                        let mut s = g0;
                        while s < g1 {
                            let (mut lo, mut hi) = (ranges[s].1 as usize, ranges[s].2 as usize);
                            let mut own = hi - lo;
                            let mut e = s + 1;
                            while e < g1 && e - s < MAX_GEMM_GROUP {
                                let (l, h) = (ranges[e].1 as usize, ranges[e].2 as usize);
                                let nlo = lo.min(l);
                                let nhi = hi.max(h);
                                let nown = own + (h - l);
                                if (nhi - nlo) * (e - s + 1) > MAX_RANGE_WIDENING * nown {
                                    break;
                                }
                                (lo, hi, own) = (nlo, nhi, nown);
                                e += 1;
                            }
                            members.clear();
                            members.extend(ranges[s..e].iter().map(|r| r.3));
                            if members.len() >= GEMM_MIN_GROUP {
                                score(lo, hi, &members, &mut heaps);
                            } else {
                                for r in &ranges[s..e] {
                                    score(r.1 as usize, r.2 as usize, &[r.3], &mut heaps);
                                }
                            }
                            s = e;
                        }
                        g0 = g1;
                    }

                    block
                        .iter()
                        .zip(heaps)
                        .enumerate()
                        .map(|(bi, (&qi, heap))| {
                            progress.tick();
                            let q = bq(bi);
                            let mut e: Vec<(T, usize)> = heap
                                .ids()
                                .iter()
                                .map(|&id| (self.euclidean_distance_to_query(id, q), id))
                                .collect();
                            e.sort_unstable_by(|a, b| {
                                a.0.partial_cmp(&b.0)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                                    .then(a.1.cmp(&b.1))
                            });
                            e.truncate(k);
                            let ids = e
                                .iter()
                                .map(|&(_, r)| self.original_ids[r] as usize)
                                .collect();
                            let dists = e
                                .iter()
                                .map(|&(d, _)| match self.metric {
                                    Dist::Cosine => d * half,
                                    _ => d,
                                })
                                .collect();
                            (qi, ids, dists)
                        })
                        .collect::<Vec<_>>()
                },
            )
            .flatten()
            .collect();

        let mut results: Vec<(Vec<usize>, Vec<T>)> = vec![(Vec::new(), Vec::new()); nq];
        for (qi, ids, dists) in block_results {
            results[qi as usize] = (ids, dists);
        }
        results
    }

    /// Exact k-nearest neighbour query.
    ///
    /// Walks clusters in ascending query-to-centroid distance order, uses the
    /// cluster radius to skip entire clusters, and uses per-point centroid
    /// distances to skip individual candidates via the triangle inequality. For
    /// Cosine, normalises the query and converts the final squared Euclidean
    /// distances back to Cosine distance (`d² / 2`).
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)` sorted nearest-first. Euclidean mode
    /// returns squared distances; Cosine mode returns Cosine distance.
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        if self.metric == Dist::Cosine {
            let mut q = query_vec.to_vec();
            normalise_vector(&mut q);
            Ok(self.query_internal(&q, k))
        } else {
            Ok(self.query_internal(query_vec, k))
        }
    }

    /// Exact k-nearest neighbour query from a row reference.
    ///
    /// Fast path for contiguous rows (stride == 1); otherwise clones.
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k)
    }

    /// Query the index with a batch of vectors
    ///
    /// From [`BATCH_MIN_QUERIES`] queries up, runs the batched search,
    /// otherwise one [`KmknnIndex::query`] per query in parallel.
    ///
    /// ### Params
    ///
    /// * `queries` - Row-major queries, `nq * query_dim`
    /// * `query_dim` - Query dimensionality, must match the index
    /// * `k` - Number of neighbours to return
    /// * `return_dist` - Whether to return distances
    /// * `verbosity` - Prints start, progress and end at `Normal`
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)`, one row per query
    pub fn query_batch(
        &self,
        queries: &[T],
        query_dim: usize,
        k: usize,
        return_dist: bool,
        verbosity: Verbosity,
    ) -> KnnOptionResult<T> {
        self.check_dim(query_dim)?;
        let nq = queries.len() / query_dim;
        let progress = QueryProgress::new("kMkNN index", nq, false, verbosity);
        let results = if nq >= BATCH_MIN_QUERIES {
            if self.metric == Dist::Cosine {
                let mut q = queries[..nq * query_dim].to_vec();
                q.par_chunks_exact_mut(query_dim).for_each(normalise_vector);
                self.search_batched(&q, nq, k, &progress)
            } else {
                self.search_batched(queries, nq, k, &progress)
            }
        } else {
            queries
                .par_chunks_exact(query_dim)
                .map(|q| {
                    progress.tick();
                    self.query(q, k)
                })
                .collect::<Result<Vec<_>, AnnSearchErrors>>()?
        };
        progress.finish();
        Ok(pack_knn_results(results, return_dist))
    }

    /// Generate exact kNN graph from vectors stored in the index.
    ///
    /// Runs the batched search with the stored vectors as queries; in Cosine
    /// mode they are already normalised.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `return_dist` - Whether to return distances
    /// * `verbosity` - Prints start, progress and end at `Normal`
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` indexed by original id
    pub fn generate_knn(
        &self,
        k: usize,
        return_dist: bool,
        verbosity: Verbosity,
    ) -> KnnOptionResult<T> {
        let progress = QueryProgress::new("kMkNN index", self.n, true, verbosity);
        let results = self.search_batched(&self.vectors_flat, self.n, k, &progress);
        progress.finish();

        let mut final_indices = vec![Vec::new(); self.n];
        let mut final_dists = if return_dist {
            Some(vec![Vec::new(); self.n])
        } else {
            None
        };

        for (i, (indices, dists)) in results.into_iter().enumerate() {
            let orig_id = self.original_ids[i] as usize;
            final_indices[orig_id] = indices;
            if let Some(ref mut fd) = final_dists {
                fd[orig_id] = dists;
            }
        }

        Ok((final_indices, final_dists))
    }
}

///////////////////
// KnnValidation //
///////////////////

impl<T> KnnValidation<T> for KmknnIndex<T>
where
    T: AnnSearchFloat,
{
    fn query_for_validation(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.query(query_vec, k)
    }

    fn n(&self) -> usize {
        self.n
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn original_ids(&self) -> &[u32] {
        &self.original_ids
    }
}

///////////
// Tests //
///////////

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for KmknnIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "kmknn";

    /// Sort each cell's rows by distance to its centroid if they are not.
    ///
    /// Indices saved before the rows were sorted decode under the same
    /// format, and [`KmknnIndex::annulus`] would silently cut neighbours from
    /// them. Sorting on load keeps those files usable; a file already sorted
    /// only pays the check.
    ///
    /// ### Params
    ///
    /// * `dir` - Unused
    ///
    /// ### Returns
    ///
    /// `Ok(())`
    fn load_aux(&mut self, dir: &std::path::Path) -> Result<(), AnnSearchErrors> {
        let _ = dir;
        let dim = self.dim;
        for c in 0..self.nlist {
            let (start, end) = (self.offsets[c] as usize, self.offsets[c + 1] as usize);
            let d = &self.point_to_centroid_sq[start..end];
            if d.windows(2).all(|w| w[0] <= w[1]) {
                continue;
            }
            let mut perm: Vec<usize> = (start..end).collect();
            perm.sort_unstable_by(|&a, &b| {
                self.point_to_centroid_sq[a]
                    .partial_cmp(&self.point_to_centroid_sq[b])
                    .unwrap()
                    .then(self.original_ids[a].cmp(&self.original_ids[b]))
            });
            let vecs: Vec<T> = perm
                .iter()
                .flat_map(|&p| self.vectors_flat[p * dim..(p + 1) * dim].iter().copied())
                .collect();
            let ids: Vec<u32> = perm.iter().map(|&p| self.original_ids[p]).collect();
            let dists: Vec<T> = perm.iter().map(|&p| self.point_to_centroid_sq[p]).collect();
            self.vectors_flat[start * dim..end * dim].copy_from_slice(&vecs);
            self.original_ids[start..end].copy_from_slice(&ids);
            self.point_to_centroid_sq[start..end].copy_from_slice(&dists);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::exhaustive::ExhaustiveIndex;
    use approx::assert_relative_eq;
    use faer::Mat;

    fn create_simple_matrix() -> Mat<f32> {
        let data = [
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0,
        ];
        Mat::from_fn(5, 3, |i, j| data[i * 3 + j])
    }

    #[test]
    fn test_kmknn_index_creation() {
        let data = create_simple_matrix();
        let _ = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        );
    }

    #[test]
    fn test_kmknn_query_finds_self_euclidean() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kmknn_query_finds_self_cosine() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(
            data.as_ref(),
            Dist::Cosine,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 1).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kmknn_query_sorted() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (_, distances) = index.query(&query, 5).unwrap();

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_kmknn_query_k_larger_than_dataset() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _) = index.query(&query, 10).unwrap();

        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn test_kmknn_orthogonal_cosine() {
        let data = Mat::from_fn(3, 3, |i, j| if i == j { 1.0 } else { 0.0 });
        let index = KmknnIndex::build(
            data.as_ref(),
            Dist::Cosine,
            Some(3),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances) = index.query(&query, 3).unwrap();

        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(distances[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(distances[2], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kmknn_reproducibility() {
        let data = create_simple_matrix();

        let idx1 = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let idx2 = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query = vec![0.5, 0.5, 0.0];
        let (i1, _) = idx1.query(&query, 3).unwrap();
        let (i2, _) = idx2.query(&query, 3).unwrap();
        assert_eq!(i1, i2);
    }

    #[test]
    fn test_kmknn_query_row() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let (indices, distances) = index.query_row(data.row(0), 1).unwrap();
        assert_eq!(indices[0], 0);
        assert_relative_eq!(distances[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_kmknn_larger_dataset() {
        let n = 100;
        let dim = 10;
        let data = Mat::from_fn(n, dim, |i, j| (i * j) as f32 / 10.0);

        let index = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(10),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let query: Vec<f32> = (0..dim).map(|_| 0.0).collect();
        let (indices, _) = index.query(&query, 5).unwrap();

        assert_eq!(indices.len(), 5);
        assert_eq!(indices[0], 0);
    }

    #[test]
    fn test_kmknn_generate_knn() {
        let data = create_simple_matrix();
        let index = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(2),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let (indices, dists) = index.generate_knn(3, true, Verbosity::Quiet).unwrap();

        assert_eq!(indices.len(), 5);
        assert!(dists.is_some());
        // Each row's nearest neighbour is itself at distance 0
        for (i, row) in indices.iter().enumerate() {
            assert_eq!(row[0], i);
        }
        for row in dists.unwrap() {
            assert_relative_eq!(row[0], 0.0, epsilon = 1e-5);
        }
    }

    // Exactness check: kMkNN must return the same neighbours as the
    // brute-force index. This is the core correctness guarantee.
    #[test]
    fn test_kmknn_matches_exhaustive_euclidean() {
        let n = 200;
        let dim = 16;
        let data = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 97) as f32 / 10.0);

        let kmknn = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            Some(14),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let exhaustive =
            ExhaustiveIndex::new(data.as_ref(), Dist::SquaredEuclidean, Verbosity::Quiet);

        let query: Vec<f32> = (0..dim).map(|j| (j * 3 % 17) as f32 / 5.0).collect();
        let k = 10;

        let (_, kmknn_dist) = kmknn.query(&query, k).unwrap();
        let (_, exh_dist) = exhaustive.query(&query, k).unwrap();

        // Distances must match exactly (up to float noise); indices may
        // differ only on ties
        for i in 0..k {
            assert_relative_eq!(kmknn_dist[i], exh_dist[i], epsilon = 1e-4);
        }
    }

    #[test]
    fn test_kmknn_matches_exhaustive_cosine() {
        let n = 200;
        let dim = 16;
        let data = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 97) as f32 / 10.0 + 0.1);

        let kmknn = KmknnIndex::build(
            data.as_ref(),
            Dist::Cosine,
            Some(14),
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let exhaustive = ExhaustiveIndex::new(data.as_ref(), Dist::Cosine, Verbosity::Quiet);

        let query: Vec<f32> = (0..dim).map(|j| (j * 3 % 17) as f32 / 5.0 + 0.1).collect();
        let k = 10;

        let (_, kmknn_dist) = kmknn.query(&query, k).unwrap();
        let (_, exh_dist) = exhaustive.query(&query, k).unwrap();

        for i in 0..k {
            assert_relative_eq!(kmknn_dist[i], exh_dist[i], epsilon = 1e-4);
        }
    }

    /// Batched self-kNN and the batched `query_batch` path both give the
    /// exhaustive distances
    fn assert_batched_matches_exhaustive(metric: Dist) {
        use crate::synthetic::generate_clustered_data;
        let (data, _) = generate_clustered_data::<f32>(3000, 16, 10, 42);
        let (queries, _) = generate_clustered_data::<f32>(BATCH_MIN_QUERIES + 200, 16, 10, 7);
        let k = 10;
        let kmknn =
            KmknnIndex::build(data.as_ref(), metric, None, None, 42, Verbosity::Quiet).unwrap();
        let exhaustive = ExhaustiveIndex::new(data.as_ref(), metric, Verbosity::Quiet);

        let (_, self_dist) = kmknn.generate_knn(k, true, Verbosity::Quiet).unwrap();
        let self_dist = self_dist.unwrap();
        for i in 0..data.nrows() {
            let row: Vec<f32> = data.row(i).iter().copied().collect();
            let (_, exh) = exhaustive.query(&row, k).unwrap();
            for j in 0..k {
                assert_relative_eq!(self_dist[i][j], exh[j], epsilon = 1e-3, max_relative = 1e-4);
            }
        }

        let nq = queries.nrows();
        let flat: Vec<f32> = (0..nq)
            .flat_map(|i| queries.row(i).iter().copied().collect::<Vec<_>>())
            .collect();
        let (_, q_dist) = kmknn
            .query_batch(&flat, 16, k, true, Verbosity::Quiet)
            .unwrap();
        let q_dist = q_dist.unwrap();
        for i in 0..nq {
            let (_, exh) = exhaustive.query(&flat[i * 16..(i + 1) * 16], k).unwrap();
            for j in 0..k {
                assert_relative_eq!(q_dist[i][j], exh[j], epsilon = 1e-3, max_relative = 1e-4);
            }
        }
    }

    #[test]
    fn test_kmknn_batched_matches_exhaustive_euclidean() {
        assert_batched_matches_exhaustive(Dist::SquaredEuclidean);
    }

    #[test]
    fn test_kmknn_batched_matches_exhaustive_cosine() {
        assert_batched_matches_exhaustive(Dist::Cosine);
    }

    /// An index saved with unsorted cells, as before the rows were sorted,
    /// is re-sorted on load and still matches exhaustive search
    #[cfg(feature = "serialise")]
    #[test]
    fn test_kmknn_load_sorts_unsorted_cells() {
        use crate::synthetic::generate_clustered_data;
        let (data, _) = generate_clustered_data::<f32>(2000, 16, 10, 42);
        let k = 10;
        let mut kmknn = KmknnIndex::build(
            data.as_ref(),
            Dist::SquaredEuclidean,
            None,
            None,
            42,
            Verbosity::Quiet,
        )
        .unwrap();
        let dim = kmknn.dim;
        for c in 0..kmknn.nlist {
            let (s, e) = (kmknn.offsets[c] as usize, kmknn.offsets[c + 1] as usize);
            for i in 0..(e - s) / 2 {
                let (a, b) = (s + i, e - 1 - i);
                kmknn.original_ids.swap(a, b);
                kmknn.point_to_centroid_sq.swap(a, b);
                for j in 0..dim {
                    kmknn.vectors_flat.swap(a * dim + j, b * dim + j);
                }
            }
        }

        let dir = tempfile::TempDir::new().unwrap();
        kmknn.save_index(dir.path()).unwrap();
        let loaded = KmknnIndex::<f32>::load_index(dir.path()).unwrap();
        let exhaustive =
            ExhaustiveIndex::new(data.as_ref(), Dist::SquaredEuclidean, Verbosity::Quiet);
        for i in 0..data.nrows() {
            let row: Vec<f32> = data.row(i).iter().copied().collect();
            let (_, got) = loaded.query(&row, k).unwrap();
            let (_, exh) = exhaustive.query(&row, k).unwrap();
            for j in 0..k {
                assert_relative_eq!(got[j], exh[j], epsilon = 1e-4);
            }
        }
    }
}
