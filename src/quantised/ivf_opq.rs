//! Inverted file OPQ index: quantises the original data via optimised product
//! quantisation and uses Voronoi cells to identify the most interesting
//! candidates.

use faer::RowRef;
use rayon::prelude::*;
use std::ops::AddAssign;
use thousands::*;

use crate::prelude::*;
use crate::quantised::pq_search::*;
use crate::quantised::quantisers::*;
use crate::utils::k_means_utils::*;
use crate::utils::pack_knn_results;
use crate::utils::permute_rows_in_place;
use crate::utils::verbosity::{BuildLog, QueryProgress};
use std::time::Instant;

////////////////
// Main index //
////////////////

/// IVF index with optimised product quantisation
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct IvfOpqIndex<T> {
    ///Encoded vectors (M u8 codes per vector)
    quantised_codes: Vec<u8>,
    /// Original dimensions
    dim: usize,
    /// Number of samples in the index
    n: usize,
    /// Distance metric
    metric: Dist,
    /// K-means cluster centroids
    centroids: Vec<T>,
    /// Centroids pre-rotated into the OPQ codebook space, `nlist * dim`.
    ///
    /// The ADC table has to be built from `R * (q - c)`. Rotating that residual
    /// per probed cell would cost `O(nprobe * dim^2)` scalar work per query,
    /// several times the table build it feeds at dim 512. `R` is linear, so
    /// `R(q - c) == R q - R c`: rotate the query once and subtract these.
    centroids_rotated: Vec<T>,
    /// Norms of the centroids - not relevant for this index.
    centroids_norm: Vec<T>,
    /// Vector indices for each cluster (CSR format)
    all_indices: Vec<u32>,
    /// Offsets for each inverted list
    offsets: Vec<u32>,
    /// Product quantiser with M codebooks
    codebook: OptimisedProductQuantiser<T>,
    /// Number of k-means clusters
    nlist: usize,
    /// /// Original indices
    original_ids: Vec<u32>,
    /// Precomputed per-cell residual-table terms, see [`cell_terms`]. Empty
    /// when over [`PRECOMP_TABLE_MAX_BYTES`]. Derived, so not serialised and
    /// rebuilt on load.
    #[cfg_attr(feature = "serialise", serde(skip, default = "Vec::new"))]
    cell_terms: Vec<T>,
}

//////////////////////
// CentroidDistance //
//////////////////////

impl<T> CentroidDistance<T> for IvfOpqIndex<T>
where
    T: AnnSearchFloat,
{
    fn centroids(&self) -> &[T] {
        &self.centroids
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn nlist(&self) -> usize {
        self.nlist
    }

    fn centroids_norm(&self) -> &[T] {
        &self.centroids_norm
    }
}

///////////////////////
// VectorDistanceAdc //
///////////////////////

impl<T> VectorDistanceAdc<T> for IvfOpqIndex<T>
where
    T: AnnSearchFloat + AddAssign,
{
    fn codebook_m(&self) -> usize {
        self.codebook.m()
    }

    fn codebook_n_centroids(&self) -> usize {
        self.codebook.n_centroids()
    }

    fn codebook_subvec_dim(&self) -> usize {
        self.codebook.subvec_dim()
    }

    fn centroids(&self) -> &[T] {
        &self.centroids
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn quantised_codes(&self) -> &[u8] {
        &self.quantised_codes
    }

    fn metric(&self) -> Dist {
        self.metric
    }

    fn codebooks(&self) -> &[Vec<T>] {
        self.codebook.codebooks()
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for IvfOpqIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

///////////
// Shape //
///////////

impl<T> IvfOpqIndex<T> {
    /// Number of indexed samples.
    ///
    /// ### Returns
    ///
    /// The row count the index was built from. Pairs with
    /// [`DimensionValidation::dim`] so every index exposes its shape the same
    /// way, which is what the Python bindings' handle macro reads.
    pub fn n(&self) -> usize {
        self.n
    }
}

////////////////
// Main index //
////////////////

impl<T> IvfOpqIndex<T>
where
    T: AnnSearchFloat + AddAssign,
{
    /// Build an IVF index with optimised product quantisation
    ///
    /// ### Params
    ///
    /// * `data` - Matrix reference with vectors as rows (n × dim)
    /// * `nlist` - Optional number of clusters. Defaults to `sqrt(n)`.
    /// * `m` - Number of subspaces for PQ (dim must be divisible by m)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `k_means_params` - Optional k-means trainings parameters, see
    ///   [KMeansTrainingParams]. If not provided, will default to sensible
    ///   defaults.
    /// * `opq_iter` - Optional number of iterations to get the rotation matrix.
    ///   Defaults to `3`.
    /// * `n_opq_centroids` - Number of centroids to use for the product
    ///   quantisation. If not provided, it uses the default `256`.
    /// * `seed` - Random seed
    /// * `verbosity` - Prints start, progress and end at `Normal`
    ///
    /// ### Returns
    ///
    /// Index ready for querying
    #[allow(clippy::too_many_arguments)]
    pub fn build(
        data: impl AnnMatrix<T>,
        nlist: Option<usize>,
        m: usize,
        metric: Dist,
        k_means_params: Option<KMeansTrainingParams>,
        opq_iter: Option<usize>,
        n_opq_centroids: Option<usize>,
        seed: usize,
        verbosity: Verbosity,
    ) -> Result<Self, AnnSearchErrors> {
        let log = BuildLog::new("IVF-OPQ index", verbosity);
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }

        let (mut vectors_flat, n, dim) = data.into_row_major();

        let nlist = nlist.unwrap_or((n as f32).sqrt() as usize).max(1);

        // normalise for cosine distance
        if metric == Dist::Cosine {
            vectors_flat
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        // 1. Subsample training data
        let n_train = (256 * nlist).min(250_000).min(n).max(1);
        let (training_data, _) = sample_vectors(&vectors_flat, dim, n, n_train, seed);

        // 2. train IVF centroids
        let start = Instant::now();
        let mut centroids = train_centroids(
            &training_data,
            dim,
            n_train,
            nlist,
            &metric,
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

        // normalise centroids for cosine
        if metric == Dist::Cosine {
            centroids
                .par_chunks_mut(dim)
                .for_each(|chunk| normalise_vector(chunk));
        }

        // 3. compute residuals for training data
        let training_norms = vec![T::one(); n_train];
        let centroid_norms = vec![T::one(); nlist];
        let training_assignments = assign_all_parallel(
            &training_data,
            &training_norms,
            dim,
            n_train,
            &centroids,
            &centroid_norms,
            nlist,
            &metric,
        );

        // For normalised vectors (cosine), ||q-v||² = 2(1 - <q,v>) so Euclidean distance
        // on residuals correctly approximates cosine distance
        // residuals in place: the sample is not needed once they exist
        let mut training_residuals = training_data;
        training_residuals
            .par_chunks_mut(dim)
            .zip(training_assignments.par_iter())
            .for_each(|(row, &cluster_id)| {
                let centroid =
                    &centroids[cluster_id as usize * dim..(cluster_id as usize + 1) * dim];
                row.iter_mut().zip(centroid).for_each(|(x, &c)| *x = *x - c);
            });

        // 4. train OPQ on residuals
        let start = Instant::now();
        let codebook = OptimisedProductQuantiser::train(
            &training_residuals,
            dim,
            m,
            n_opq_centroids,
            opq_iter,
            30,
            seed + 1000,
            verbosity,
        )?;
        drop(training_residuals);
        if verbosity.detailed_verbosity() {
            println!(
                " Trained optimised product quantiser (m = {}) in {:.2?}",
                m,
                start.elapsed()
            );
        }

        // 5. assign all vectors to IVF clusters
        let data_norms = vec![T::one(); n];
        let assignments = assign_all_parallel(
            &vectors_flat,
            &data_norms,
            dim,
            n,
            &centroids,
            &centroid_norms,
            nlist,
            &metric,
        );

        // 6. encode all vectors with product quantiser
        if verbosity.detailed_verbosity() {
            println!(" Encoding residuals ...");
        }
        let mut quantised_codes = vec![0u8; n * m];

        quantised_codes
            .par_chunks_mut(ENCODE_ROW_TILE * m)
            .enumerate()
            .for_each_init(
                || Vec::with_capacity(ENCODE_ROW_TILE * dim),
                |tile, (t, codes)| {
                    tile.clear();
                    for vec_idx in t * ENCODE_ROW_TILE..t * ENCODE_ROW_TILE + codes.len() / m {
                        let vec = &vectors_flat[vec_idx * dim..(vec_idx + 1) * dim];
                        let c = assignments[vec_idx] as usize;
                        let centroid = &centroids[c * dim..(c + 1) * dim];
                        tile.extend(vec.iter().zip(centroid).map(|(&a, &b)| a - b));
                    }
                    codebook.encode_rows(tile, codes);
                },
            );

        // Pre-rotate once at build so the query path never pays `dim^2` per
        // probed cell.
        let centroids_rotated: Vec<T> = centroids
            .chunks_exact(dim)
            .flat_map(|c| codebook.rotate(c))
            .collect();

        let (all_indices, offsets) = build_csr_layout(assignments, n, nlist);
        drop(vectors_flat);

        let mut idx = Self {
            quantised_codes,
            centroids,
            centroids_rotated,
            all_indices,
            offsets,
            codebook,
            dim,
            n,
            nlist,
            metric,
            centroids_norm: Vec::new(),
            original_ids: Vec::new(),
            cell_terms: Vec::new(),
        };

        let new_to_old = idx.optimise_memory_layout();
        idx.original_ids = new_to_old;
        idx.cell_terms = cell_terms(
            &idx.centroids_rotated,
            idx.codebook.pq(),
            PRECOMP_TABLE_MAX_BYTES,
        );

        log.finish();

        Ok(idx)
    }

    /// Residual ADC view over this index
    ///
    /// ### Returns
    ///
    /// The shared IVF scan state, see [`IvfAdc`]
    fn adc(&self) -> IvfAdc<'_, T> {
        IvfAdc {
            centroids: &self.centroids_rotated,
            books_t: self.codebook.pq().codebooks_t(),
            cell_terms: &self.cell_terms,
            codes: &self.quantised_codes,
            offsets: &self.offsets,
            dim: self.dim,
            m: self.codebook.m(),
            sd: self.codebook.subvec_dim(),
            k: self.codebook.n_centroids(),
            scale: lut_scale(self.metric),
        }
    }

    /// Scratch for one thread
    ///
    /// ### Returns
    ///
    /// Buffers sized for this index
    fn scratch(&self) -> AdcScratch<T> {
        AdcScratch::new(self.dim, self.codebook.m(), self.codebook.n_centroids())
    }

    /// Resolve the probe count
    ///
    /// ### Params
    ///
    /// * `nprobe` - Requested cells, `sqrt(nlist)` when `None`
    ///
    /// ### Returns
    ///
    /// Cells to probe, at most `nlist`
    fn resolve_nprobe(&self, nprobe: Option<usize>) -> usize {
        nprobe
            .unwrap_or_else(|| ((self.nlist as f64).sqrt() as usize).max(1))
            .min(self.nlist)
    }

    /// Map a query into codebook space
    ///
    /// Codes are OPQ-encoded residuals, so the codebooks live in the rotated
    /// space and the table has to be built there too. `R` is orthogonal and
    /// linear, so `R (q - c) == R q - R c` against the pre-rotated centroids,
    /// distances are preserved, and normalising after the rotation is the same
    /// as before it.
    ///
    /// ### Params
    ///
    /// * `q` - Query, length `dim`
    /// * `out` - Output, `R q`, length `dim`
    #[inline]
    fn prepare(&self, q: &[T], out: &mut [T]) {
        self.codebook.rotate_to(q, out);
        if self.metric == Dist::Cosine {
            normalise_vector(out);
        }
    }

    /// Probe list of a prepared query
    ///
    /// ### Params
    ///
    /// * `q` - Query in codebook space
    /// * `nprobe` - Requested cells
    /// * `k` - Neighbours the probe list has to reach
    /// * `buf` - Scratch for the centroid scores
    ///
    /// ### Returns
    ///
    /// Cell ids, nearest first
    #[inline]
    fn probe(&self, q: &[T], nprobe: usize, k: usize, buf: &mut Vec<(T, usize)>) -> Vec<usize> {
        probe_cells(
            q,
            &self.centroids_rotated,
            self.dim,
            &self.offsets,
            nprobe,
            k,
            buf,
        )
    }

    /// Score the query held in `s.q` against its probed cells
    ///
    /// ### Params
    ///
    /// * `probes` - Cells to scan
    /// * `k` - Neighbours to return
    /// * `s` - Scratch holding the prepared query
    /// * `qlut` - Query-only table from the batched path, see [`IvfAdc::search`]
    ///
    /// ### Returns
    ///
    /// `(original ids, distances)`, nearest first
    fn search_prepared(
        &self,
        probes: &[usize],
        k: usize,
        s: &mut AdcScratch<T>,
        qlut: Option<&[T]>,
    ) -> (Vec<usize>, Vec<T>) {
        s.heap.reset(k);
        self.adc().search(probes, s, qlut, None, |_| false);
        finish_heap(&mut s.heap, Some(&self.original_ids))
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Uses asymmetric distance computation (ADC). With the precomputed cell
    /// terms, one query-only table is built and each probed cell adds its
    /// precomputed term to it; otherwise the residual table is built per cell.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to `sqrt(nlist)`)
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances) sorted by distance
    #[inline]
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;
        let nprobe = self.resolve_nprobe(nprobe);
        let k = k.min(self.n);

        let mut s = self.scratch();
        self.prepare(query_vec, &mut s.q);
        let probes = self.probe(&s.q, nprobe, k, &mut s.probe_buf);
        Ok(self.search_prepared(&probes, k, &mut s, None))
    }

    /// Query many vectors at once
    ///
    /// From `BATCH_MIN_QUERIES` queries up, every query is probed first and
    /// the queries are then scored in nearest-cell order (see
    /// `run_ordered`); below it, one query per task. Both give identical
    /// results.
    ///
    /// ### Params
    ///
    /// * `queries` - Row-major queries, `nq * query_dim`
    /// * `query_dim` - Query dimensionality, must match the index
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search (defaults to `sqrt(nlist)`)
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
        nprobe: Option<usize>,
        return_dist: bool,
        verbosity: Verbosity,
    ) -> KnnOptionResult<T> {
        self.check_dim(query_dim)?;
        let dim = self.dim;
        let nq = queries.len() / dim;
        let nprobe = self.resolve_nprobe(nprobe);
        let k = k.min(self.n);
        let progress = QueryProgress::new("IVF-OPQ index", nq, false, verbosity);

        let results = if nq >= BATCH_MIN_QUERIES {
            run_ordered(
                queries,
                &BatchPlan {
                    adc: self.adc(),
                    probe_offsets: &self.offsets,
                    rotation: Some(self.codebook.rotation_matrix()),
                    cosine: self.metric == Dist::Cosine,
                    nprobe,
                    k,
                },
                || self.scratch(),
                |probes, s, qlut| self.search_prepared(probes, k, s, qlut),
                &progress,
            )
        } else {
            (0..nq)
                .into_par_iter()
                .map_init(
                    || self.scratch(),
                    |s, i| {
                        self.prepare(&queries[i * dim..(i + 1) * dim], &mut s.q);
                        let probes = self.probe(&s.q, nprobe, k, &mut s.probe_buf);
                        progress.tick();
                        self.search_prepared(&probes, k, s, None)
                    },
                )
                .collect()
        };
        progress.finish();
        Ok(pack_knn_results(results, return_dist))
    }

    /// Query using a matrix row reference
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference
    /// * `k` - Number of neighbours to return
    /// * `nprobe` - Number of clusters to search
    ///
    /// ### Returns
    ///
    /// Tuple of (indices, distances) sorted by distance
    #[inline]
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        nprobe: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, nprobe);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, nprobe)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// Reconstructs each point in codebook space (centroid plus decoded
    /// residual, both already rotated, so no rotation is needed) from its code row and queries with it. Positions
    /// are walked in storage order, which is cell order, so consecutive
    /// queries on a thread share their cells.
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `nprobe` - Number of clusters to search (defaults to sqrt(nlist) if None)
    /// * `return_dist` - Whether to return distances
    /// * `verbosity` - Prints start, progress and end at `Normal`
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` where each row corresponds
    /// to a vector in the index
    pub fn generate_knn(
        &self,
        k: usize,
        nprobe: Option<usize>,
        return_dist: bool,
        verbosity: Verbosity,
    ) -> KnnOptionResult<T> {
        let (m, dim) = (self.codebook.m(), self.dim);
        let nprobe = self.resolve_nprobe(nprobe);
        let k = k.min(self.n);
        let progress = QueryProgress::new("IVF-OPQ index", self.n, true, verbosity);

        let n_pos = self.quantised_codes.len() / m;
        let mut pos_cell = vec![0u32; n_pos];
        for cell in 0..self.nlist {
            for p in self.offsets[cell] as usize..self.offsets[cell + 1] as usize {
                pos_cell[p] = cell as u32;
            }
        }

        let rows: Vec<u32> = (0..n_pos as u32).collect();
        let results = run_self(
            &rows,
            &BatchPlan {
                adc: self.adc(),
                probe_offsets: &self.offsets,
                rotation: None,
                cosine: self.metric == Dist::Cosine,
                nprobe,
                k,
            },
            || self.scratch(),
            |i, out| {
                let c = pos_cell[i] as usize;
                self.codebook
                    .pq()
                    .decode_into(&self.quantised_codes[i * m..(i + 1) * m], out);
                for (v, &cv) in out
                    .iter_mut()
                    .zip(&self.centroids_rotated[c * dim..(c + 1) * dim])
                {
                    *v += cv;
                }
            },
            |probes, s, qlut| self.search_prepared(probes, k, s, qlut),
            &progress,
        );
        progress.finish();
        let results = rows
            .iter()
            .zip(results)
            .map(|(&p, (ids, dists))| (self.original_ids[p as usize] as usize, ids, dists))
            .collect();

        Ok(scatter_self(results, self.n, return_dist))
    }

    /// Set the memory budget for the precomputed residual-table terms
    ///
    /// The table holds `nlist * m * n_pq_centroids` values and lets a probed
    /// cell cost one add per entry instead of a full residual-table build.
    /// It is rebuilt or dropped straight away. The budget is not saved with
    /// the index: a loaded index starts from the default of 256 MiB.
    ///
    /// ### Params
    ///
    /// * `max_bytes` - Largest table to keep; `0` turns the table off and
    ///   queries build the residual table per probed cell instead
    pub fn set_cell_terms_budget(&mut self, max_bytes: usize) {
        self.cell_terms = cell_terms(&self.centroids_rotated, self.codebook.pq(), max_bytes);
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.quantised_codes.capacity() * std::mem::size_of::<u8>()
            + self.centroids.capacity() * std::mem::size_of::<T>()
            + self.centroids_rotated.capacity() * std::mem::size_of::<T>()
            + self.centroids_norm.capacity() * std::mem::size_of::<T>()
            + self.all_indices.capacity() * std::mem::size_of::<u32>()
            + self.offsets.capacity() * std::mem::size_of::<u32>()
            + self.cell_terms.capacity() * std::mem::size_of::<T>()
            + self.codebook.memory_usage_bytes()
    }

    /// Function will optimise memory layout and put vectors sorted by cluster
    /// id
    ///
    /// ### Returns
    ///
    /// New to old mapping
    fn optimise_memory_layout(&mut self) -> Vec<u32> {
        let m = self.codebook.m();
        let mut new_to_old = Vec::with_capacity(self.n);

        for cluster in 0..self.nlist {
            let start = self.offsets[cluster] as usize;
            let end = self.offsets[cluster + 1] as usize;
            new_to_old.extend_from_slice(&self.all_indices[start..end]);
        }

        permute_rows_in_place(&mut self.quantised_codes, m, &new_to_old);
        self.all_indices.clear();
        self.all_indices.shrink_to_fit();

        new_to_old
    }
}

///////////
// Tests //
///////////

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for IvfOpqIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "ivf_opq";

    fn load_aux(&mut self, _dir: &std::path::Path) -> Result<(), AnnSearchErrors> {
        self.codebook.pq_mut().rebuild_codebooks_t();
        self.cell_terms = cell_terms(
            &self.centroids_rotated,
            self.codebook.pq(),
            PRECOMP_TABLE_MAX_BYTES,
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn create_simple_dataset() -> Mat<f32> {
        let mut data = Vec::new();
        // Create 6 vectors of 32 dimensions
        // First 3 near origin
        for i in 0..3 {
            for j in 0..32 {
                data.push(i as f32 * 0.1 + j as f32 * 0.01);
            }
        }
        // Next 3 far from origin
        for i in 0..3 {
            for j in 0..32 {
                data.push(10.0 + i as f32 * 0.1 + j as f32 * 0.01);
            }
        }
        Mat::from_fn(6, 32, |i, j| data[i * 32 + j])
    }

    fn get_default_k_means() -> Option<KMeansTrainingParams> {
        Some(KMeansTrainingParams::new(10, None, None))
    }

    #[test]
    fn test_build_euclidean() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        assert_eq!(index.dim, 32);
        assert_eq!(index.n, 6);
        assert_eq!(index.nlist, 2);
        assert_eq!(index.metric, Dist::SquaredEuclidean);
        assert_eq!(index.quantised_codes.len(), 48);
        assert_eq!(index.centroids.len(), 64);
        assert_eq!(index.offsets.len(), 3);
    }

    #[test]
    fn test_build_cosine() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Cosine,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        assert_eq!(index.metric, Dist::Cosine);
    }

    #[test]
    fn test_query_returns_k_results() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, distances) = index.query(&query, 3, None).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_query_k_exceeds_n() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, _) = index.query(&query, 100, None).unwrap();

        assert!(indices.len() <= 6);
    }

    #[test]
    fn test_query_distances_sorted() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (_, distances) = index.query(&query, 3, Some(2)).unwrap();

        for i in 1..distances.len() {
            assert!(distances[i] >= distances[i - 1]);
        }
    }

    #[test]
    fn test_query_cosine() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::Cosine,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| if x < 16 { 1.0 } else { 0.0 }).collect();
        let (indices, distances) = index.query(&query, 3, None).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_query_different_nprobe() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| 5.0 + x as f32 * 0.01).collect();

        let (indices1, _) = index.query(&query, 3, Some(1)).unwrap();
        let (indices2, _) = index.query(&query, 3, Some(2)).unwrap();

        assert_eq!(indices1.len(), 3);
        assert_eq!(indices2.len(), 3);
    }

    #[test]
    fn test_query_expands_nprobe_when_probed_cells_underfill_k() {
        // 6 vectors split into 3 cells (~2 per cell). nprobe=1 alone cannot
        // reach k=4 - fix must walk further cells.
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(3),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        let (indices, distances) = index.query(&query, 4, Some(1)).unwrap();

        assert_eq!(indices.len(), 4);
        assert_eq!(distances.len(), 4);
    }

    #[test]
    fn test_query_deterministic() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| 0.5 + x as f32 * 0.01).collect();

        let (indices1, distances1) = index.query(&query, 3, Some(2)).unwrap();
        let (indices2, distances2) = index.query(&query, 3, Some(2)).unwrap();

        assert_eq!(indices1, indices2);
        assert_eq!(distances1, distances2);
    }

    #[test]
    fn test_query_row() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query_mat = Mat::<f32>::from_fn(1, 32, |_, j| 0.5 + j as f32 * 0.01);
        let row = query_mat.row(0);

        let (indices, distances) = index.query_row(row, 3, None).unwrap();

        assert_eq!(indices.len(), 3);
        assert_eq!(distances.len(), 3);
    }

    #[test]
    fn test_build_different_m() {
        let data = Mat::from_fn(20, 32, |i, j| (i + j) as f32);

        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        assert_eq!(index.codebook.m(), 8);
        assert_eq!(index.codebook.subvec_dim(), 4);
        assert_eq!(index.quantised_codes.len(), 160);
    }

    #[test]
    fn test_build_lookup_tables() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        // Rotated residual, as the query path does: the codebooks live in
        // R-space, so a table built from the unrotated residual scores in the
        // wrong basis.
        let residual = f32::subtract_simd(&query, &index.centroids[0..32]);
        let table = index.build_lookup_tables_direct(&index.codebook.rotate(&residual));

        assert_eq!(table.len(), 32);
    }

    #[test]
    fn test_compute_distance_adc() {
        let data = create_simple_dataset();
        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(2),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32 * 0.01).collect();
        // Rotated residual, as the query path does: the codebooks live in
        // R-space, so a table built from the unrotated residual scores in the
        // wrong basis.
        let residual = f32::subtract_simd(&query, &index.centroids[0..32]);
        let table = index.build_lookup_tables_direct(&index.codebook.rotate(&residual));

        let dist = index.compute_distance_adc(0, &table);

        assert!(dist >= 0.0);
    }

    #[test]
    fn test_opq_iterations() {
        let data = Mat::from_fn(50, 32, |i, j| (i + j) as f32);

        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(5),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(2),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        assert_eq!(index.codebook.m(), 8);
    }

    #[test]
    fn test_residual_encoding() {
        let data = Mat::from_fn(50, 32, |i, j| (i + j) as f32);

        let index = IvfOpqIndex::build(
            data.as_ref(),
            Some(5),
            8,
            Dist::SquaredEuclidean,
            get_default_k_means(),
            Some(1),
            Some(4),
            42,
            Verbosity::Quiet,
        )
        .unwrap();

        let query: Vec<f32> = (0..32).map(|x| x as f32).collect();
        let (indices, _) = index.query(&query, 1, Some(5)).unwrap();

        assert_eq!(indices[0], 0);
    }
}
