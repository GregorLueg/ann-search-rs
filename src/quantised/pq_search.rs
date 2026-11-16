//! Shared query kernels for the PQ and OPQ indices.
//!
//! Lookup tables are built against dimension-major codebooks, so each subspace
//! row vectorises across its centroids instead of calling a SIMD distance on
//! `subvec_dim`-long slices. The IVF variants can skip the per-cell residual
//! table entirely through the decomposition
//!
//! `||q - c - r||^2 = ||q - c||^2 + (||r||^2 + 2 <c, r>) - 2 <q, r>`
//!
//! taken per subspace: the middle term depends only on the cell and is
//! precomputed, the last only on the query and is built once per query, so a
//! probed cell costs `m * n_centroids` adds rather than `n_centroids * dim`
//! multiply-adds.
//!
//! ### References
//!
//! Jégou, Douze & Schmid, IEEE TPAMI, 2011 (IVFADC)

use num_traits::Float;
use rayon::prelude::*;

use crate::prelude::*;
use crate::quantised::k_means::{dots_transposed, sq_dists_transposed, transpose_centroids};
use crate::quantised::quantisers::ProductQuantiser;
use crate::utils::k_means_utils::select_probed_clusters;

///////////////
// Constants //
///////////////

/// Largest precomputed cell-term table, in bytes. Above it the IVF indices
/// fall back to building the residual table per probed cell. The table is
/// `nlist * m * n_centroids` elements, so it outgrows the codes themselves for
/// small datasets with many subspaces. Not tuned.
pub(crate) const PRECOMP_TABLE_MAX_BYTES: usize = 256 << 20;

/// Fewest queries for which `query_batch` takes the batched path. Below it the
/// query pre-pass and the reordering are not amortised. Not tuned.
pub(crate) const BATCH_MIN_QUERIES: usize = 1024;

/// Queries scored together against one tile of codes on the exhaustive path.
/// Each query's lookup table has to stay cache resident across the tile. Not
/// tuned.
pub(crate) const EXH_QUERY_BLOCK: usize = 16;

/// Code rows per tile on the exhaustive path, so a tile is re-read from L1
/// by every query of the block. Not tuned.
pub(crate) const EXH_CODE_TILE: usize = 256;

////////////////////////
// Tables and kernels //
////////////////////////

/// Lookup-table scale for a metric.
///
/// The ADC sum is a squared Euclidean distance. Under cosine both sides are
/// unit length, so it equals `2 (1 - cos)`: halving every table entry makes
/// the sum the cosine distance exactly, at `m * k` multiplies per table rather
/// than one per candidate.
///
/// ### Params
///
/// * `metric` - Index metric
///
/// ### Returns
///
/// 0.5 under cosine, else 1
#[inline]
pub(crate) fn lut_scale<T: Float>(metric: Dist) -> T {
    match metric {
        Dist::Cosine => T::one() / (T::one() + T::one()),
        _ => T::one(),
    }
}

/// Dimension-major copy of every codebook.
///
/// ### Params
///
/// * `books` - One row-major `k * sd` codebook per subspace
/// * `sd` - Subvector dimension
/// * `k` - Centroids per codebook
///
/// ### Returns
///
/// `m * sd * k` elements; subspace `s` occupies `[s * sd * k, (s + 1) * sd * k)`
/// laid out as [`transpose_centroids`] does.
pub(crate) fn transpose_codebooks<T: Float>(books: &[Vec<T>], sd: usize, k: usize) -> Vec<T> {
    let mut out = vec![T::zero(); books.len() * sd * k];
    for (book, dst) in books.iter().zip(out.chunks_exact_mut(sd * k)) {
        transpose_centroids(book, sd, k, dst);
    }
    out
}

/// Squared Euclidean lookup table of a query against every codebook.
///
/// ### Params
///
/// * `q` - Query in codebook space, length `m * sd`
/// * `books_t` - Codebooks from [`transpose_codebooks`]
/// * `m` - Subspaces
/// * `sd` - Subvector dimension
/// * `k` - Centroids per codebook
/// * `scale` - Factor applied to every entry
/// * `lut` - Output, `m * k`, row `s` is subspace `s`
#[inline]
pub(crate) fn fill_lut_sq<T: Float>(
    q: &[T],
    books_t: &[T],
    m: usize,
    sd: usize,
    k: usize,
    scale: T,
    lut: &mut [T],
) {
    for s in 0..m {
        let row = &mut lut[s * k..(s + 1) * k];
        sq_dists_transposed(
            &q[s * sd..(s + 1) * sd],
            &books_t[s * sd * k..(s + 1) * sd * k],
            k,
            row,
        );
        if scale != T::one() {
            row.iter_mut().for_each(|v| *v = *v * scale);
        }
    }
}

/// Scaled dot-product lookup table of a vector against every codebook.
///
/// ### Params
///
/// * `q` - Vector in codebook space, length `m * sd`
/// * `books_t` - Codebooks from [`transpose_codebooks`]
/// * `m` - Subspaces
/// * `sd` - Subvector dimension
/// * `k` - Centroids per codebook
/// * `factor` - Factor applied to every dot product
/// * `lut` - Output, `m * k`
#[inline]
pub(crate) fn fill_lut_dot<T: Float>(
    q: &[T],
    books_t: &[T],
    m: usize,
    sd: usize,
    k: usize,
    factor: T,
    lut: &mut [T],
) {
    for s in 0..m {
        dots_transposed(
            &q[s * sd..(s + 1) * sd],
            &books_t[s * sd * k..(s + 1) * sd * k],
            k,
            factor,
            &mut lut[s * k..(s + 1) * k],
        );
    }
}

/// ADC distance of one code row.
///
/// Four independent accumulators, so the sum is a chain of `m / 4` adds
/// rather than `m`.
///
/// ### Params
///
/// * `codes` - The row's `m` codes, each `< k`
/// * `lut` - Lookup table, `m * k`
/// * `k` - Centroids per codebook
///
/// ### Returns
///
/// Sum over subspaces of `lut[s * k + codes[s]]`
#[inline(always)]
pub(crate) fn adc_dist<T: Float>(codes: &[u8], lut: &[T], k: usize) -> T {
    debug_assert!(lut.len() >= codes.len() * k);
    let mut acc = [T::zero(); 4];
    let (chunks, rem) = codes.as_chunks::<4>();
    for (ci, ch) in chunks.iter().enumerate() {
        let base = ci * 4 * k;
        for l in 0..4 {
            // SAFETY: codes are produced by the encoder and are `< k`, and the
            // table holds `m * k` entries, so `(4 ci + l) * k + code < m * k`
            acc[l] = acc[l] + unsafe { *lut.get_unchecked(base + l * k + ch[l] as usize) };
        }
    }
    let rem_start = codes.len() - rem.len();
    for (j, &c) in rem.iter().enumerate() {
        acc[j] = acc[j] + lut[(rem_start + j) * k + c as usize];
    }
    (acc[0] + acc[1]) + (acc[2] + acc[3])
}

/// Score a contiguous range of code rows into a heap.
///
/// ### Params
///
/// * `codes` - All code rows, `m` per position
/// * `m` - Subspaces
/// * `k` - Centroids per codebook
/// * `lut` - Lookup table, `m * k`
/// * `bias` - Added to every distance (the `||q - c||^2` term, or zero)
/// * `start` - First position
/// * `end` - One past the last position
/// * `heap` - Receives `(distance, position)`
/// * `skip` - Positions for which this returns `true` are not scored
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn scan_range<T: Float, F: Fn(usize) -> bool>(
    codes: &[u8],
    m: usize,
    k: usize,
    lut: &[T],
    bias: T,
    start: usize,
    end: usize,
    heap: &mut BoundedMaxHeap<T>,
    skip: F,
) {
    for (i, row) in codes[start * m..end * m].chunks_exact(m).enumerate() {
        let pos = start + i;
        if skip(pos) {
            continue;
        }
        heap.push(bias + adc_dist(row, lut, k), pos);
    }
}

/// Sort a heap and map its positions to caller-facing ids.
///
/// ### Params
///
/// * `heap` - Filled heap; left sorted
/// * `ids` - Position to id map; identity when `None`
///
/// ### Returns
///
/// `(ids, distances)`, nearest first
pub(crate) fn finish_heap<T: Float>(
    heap: &mut BoundedMaxHeap<T>,
    ids: Option<&[u32]>,
) -> (Vec<usize>, Vec<T>) {
    heap.sort();
    let out_ids = match ids {
        Some(map) => heap.ids().iter().map(|&p| map[p] as usize).collect(),
        None => heap.ids().to_vec(),
    };
    (out_ids, heap.dists().to_vec())
}

/// Precompute the per-cell term `||r||^2 + 2 <c_s, r>` of the residual table.
///
/// ### Params
///
/// * `centroids` - IVF centroids in codebook space, `nlist * dim`
/// * `pq` - The product quantiser the codes were encoded with
///
/// ### Returns
///
/// `nlist * m * k` elements, cell-major, or an empty vector when the table
/// would exceed [`PRECOMP_TABLE_MAX_BYTES`].
pub(crate) fn cell_terms<T: AnnSearchFloat>(centroids: &[T], pq: &ProductQuantiser<T>) -> Vec<T> {
    let (m, sd, k) = (pq.m(), pq.subvec_dim(), pq.n_centroids());
    let dim = m * sd;
    let nlist = centroids.len() / dim;
    if nlist * m * k * std::mem::size_of::<T>() > PRECOMP_TABLE_MAX_BYTES {
        return Vec::new();
    }

    let norms: Vec<T> = pq
        .codebooks()
        .iter()
        .flat_map(|book| book.chunks_exact(sd).map(|c| T::dot_simd(c, c)))
        .collect();
    let books_t = pq.codebooks_t();
    let two = T::one() + T::one();

    let mut out = vec![T::zero(); nlist * m * k];
    out.par_chunks_mut(m * k)
        .zip(centroids.par_chunks_exact(dim))
        .for_each(|(dst, c)| {
            fill_lut_dot(c, books_t, m, sd, k, two, dst);
            for (v, &nr) in dst.iter_mut().zip(&norms) {
                *v = *v + nr;
            }
        });
    out
}

/// Probe list of a query in codebook space.
///
/// Scores every centroid by squared Euclidean distance. Under cosine both the
/// query and the centroids are unit length, so this orders the cells exactly
/// as `1 - dot` does.
///
/// ### Params
///
/// * `q` - Query in the same space as `centroids`
/// * `centroids` - `nlist * dim`
/// * `dim` - Dimension
/// * `offsets` - Per-cell counts for the underfill expansion, `nlist + 1`
/// * `nprobe` - Requested cells
/// * `k` - Neighbours the probe list has to be able to reach
/// * `buf` - Scratch for `(distance, cell)` pairs
///
/// ### Returns
///
/// Cell ids nearest first, see [`select_probed_clusters`]
pub(crate) fn probe_cells<T: AnnSearchFloat>(
    q: &[T],
    centroids: &[T],
    dim: usize,
    offsets: &[u32],
    nprobe: usize,
    k: usize,
    buf: &mut Vec<(T, usize)>,
) -> Vec<usize> {
    let nlist = centroids.len() / dim;
    let tail = nlist / 4 * 4;
    buf.clear();
    for base in (0..tail).step_by(4) {
        let s = base * dim;
        let d = T::euclidean_simd_batch_4(
            q,
            [
                &centroids[s..s + dim],
                &centroids[s + dim..s + 2 * dim],
                &centroids[s + 2 * dim..s + 3 * dim],
                &centroids[s + 3 * dim..s + 4 * dim],
            ],
        );
        for (j, &dj) in d.iter().enumerate() {
            buf.push((dj, base + j));
        }
    }
    for c in tail..nlist {
        buf.push((T::euclidean_simd(q, &centroids[c * dim..(c + 1) * dim]), c));
    }
    select_probed_clusters(buf, offsets, nprobe, k)
}

/////////////
// Scratch //
/////////////

/// Per-thread buffers for one query at a time.
pub(crate) struct AdcScratch<T> {
    /// Query in codebook space
    pub q: Vec<T>,
    /// Lookup table of the cell being scanned, `m * k`
    pub lut: Vec<T>,
    /// Query-only term `-2 <q_s, r>`, `m * k`; unused without cell terms
    pub query_lut: Vec<T>,
    /// Query residual against a cell; unused with cell terms
    pub resid: Vec<T>,
    /// Centroid scores for probing
    pub probe_buf: Vec<(T, usize)>,
    /// Result heap
    pub heap: BoundedMaxHeap<T>,
}

impl<T: Float> AdcScratch<T> {
    /// Allocate scratch for one index shape.
    ///
    /// ### Params
    ///
    /// * `dim` - Dimension
    /// * `m` - Subspaces
    /// * `k` - Centroids per codebook
    ///
    /// ### Returns
    ///
    /// Zeroed buffers and an empty heap
    pub fn new(dim: usize, m: usize, k: usize) -> Self {
        Self {
            q: vec![T::zero(); dim],
            lut: vec![T::zero(); m * k],
            query_lut: vec![T::zero(); m * k],
            resid: vec![T::zero(); dim],
            probe_buf: Vec::new(),
            heap: BoundedMaxHeap::new(0),
        }
    }
}

//////////////////
// IVF residual //
//////////////////

/// Residual ADC over IVF cells, shared by the IVF and SOAR PQ/OPQ indices.
pub(crate) struct IvfAdc<'a, T> {
    /// IVF centroids in codebook space, `nlist * dim`
    pub centroids: &'a [T],
    /// Codebooks from [`transpose_codebooks`]
    pub books_t: &'a [T],
    /// Output of [`cell_terms`]; empty means the per-cell table is built
    /// from the residual directly
    pub cell_terms: &'a [T],
    /// All code rows, `m` per position
    pub codes: &'a [u8],
    /// Position offsets per cell, `nlist + 1`
    pub offsets: &'a [u32],
    /// Dimension
    pub dim: usize,
    /// Subspaces
    pub m: usize,
    /// Subvector dimension
    pub sd: usize,
    /// Centroids per codebook
    pub k: usize,
    /// 0.5 under cosine (the ADC sum is `2 (1 - cos)` on unit vectors), else 1
    pub scale: T,
}

impl<T: AnnSearchFloat> IvfAdc<'_, T> {
    /// Score the probed cells of the query held in `s.q` into `s.heap`.
    ///
    /// ### Params
    ///
    /// * `probes` - Cells to scan
    /// * `s` - Scratch; `s.q` holds the query, `s.heap` is reset by the caller
    /// * `skip` - Positions for which this returns `true` are not scored
    pub fn search<F: Fn(usize) -> bool>(&self, probes: &[usize], s: &mut AdcScratch<T>, skip: F) {
        let (m, sd, k, dim) = (self.m, self.sd, self.k, self.dim);
        let precomputed = !self.cell_terms.is_empty();
        let two = T::one() + T::one();
        if precomputed {
            fill_lut_dot(&s.q, self.books_t, m, sd, k, -two, &mut s.query_lut);
        }
        for &cell in probes {
            let c = &self.centroids[cell * dim..(cell + 1) * dim];
            let bias = if precomputed {
                let t1 = &self.cell_terms[cell * m * k..(cell + 1) * m * k];
                for ((o, &a), &b) in s.lut.iter_mut().zip(t1).zip(&s.query_lut) {
                    *o = self.scale * (a + b);
                }
                self.scale * T::euclidean_simd(&s.q, c)
            } else {
                for ((r, &a), &b) in s.resid.iter_mut().zip(&s.q).zip(c) {
                    *r = a - b;
                }
                fill_lut_sq(&s.resid, self.books_t, m, sd, k, self.scale, &mut s.lut);
                T::zero()
            };
            scan_range(
                self.codes,
                m,
                k,
                &s.lut,
                bias,
                self.offsets[cell] as usize,
                self.offsets[cell + 1] as usize,
                &mut s.heap,
                &skip,
            );
        }
    }
}

/// Run prepared queries in parallel, nearest-centroid order first.
///
/// Every query is mapped to codebook space and probed up front, then the
/// queries are scored ordered by their nearest cell, so consecutive queries
/// on a thread share cells and their codes and tables stay cache resident.
/// Results come back in input order.
///
/// ### Params
///
/// * `nq` - Number of queries
/// * `dim` - Dimension
/// * `scratch` - Builds one thread's scratch
/// * `prepare` - Writes query `i` in codebook space into the slice, returns
///   its probe list
/// * `search` - Scores the query held in the scratch against its probes and
///   returns `(ids, distances)`
/// * `progress` - Ticked once per query
///
/// ### Returns
///
/// Per query `(ids, distances)`, nearest first
pub(crate) fn run_ordered<T, S, P, Q>(
    nq: usize,
    dim: usize,
    scratch: S,
    prepare: P,
    search: Q,
    progress: &crate::utils::verbosity::QueryProgress,
) -> Vec<(Vec<usize>, Vec<T>)>
where
    T: AnnSearchFloat,
    S: Fn() -> AdcScratch<T> + Sync + Send,
    P: Fn(usize, &mut [T], &mut AdcScratch<T>) -> Vec<usize> + Sync + Send,
    Q: Fn(&[usize], &mut AdcScratch<T>) -> (Vec<usize>, Vec<T>) + Sync + Send,
{
    let mut prepared = vec![T::zero(); nq * dim];
    let probes: Vec<Vec<usize>> = prepared
        .par_chunks_mut(dim)
        .enumerate()
        .map_init(&scratch, |s, (i, q)| prepare(i, q, s))
        .collect();

    let mut order: Vec<u32> = (0..nq as u32).collect();
    order.par_sort_by_key(|&i| probes[i as usize].first().copied().unwrap_or(0));

    let chunk = nq.div_ceil(rayon::current_num_threads().max(1) * 8).max(1);
    let scored: Vec<(u32, (Vec<usize>, Vec<T>))> = order
        .par_chunks(chunk)
        .map_init(&scratch, |s, block| {
            block
                .iter()
                .map(|&i| {
                    let i_us = i as usize;
                    s.q.copy_from_slice(&prepared[i_us * dim..(i_us + 1) * dim]);
                    progress.tick();
                    (i, search(&probes[i_us], s))
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect();

    let mut out = vec![(Vec::new(), Vec::new()); nq];
    for (i, r) in scored {
        out[i as usize] = r;
    }
    out
}

/// Scatter per-query results into original-id order and pack them.
///
/// ### Params
///
/// * `results` - `(original id, ids, distances)` per indexed point
/// * `n` - Number of points
/// * `return_dist` - Whether distances are kept
///
/// ### Returns
///
/// `(knn_indices, optional distances)` indexed by original id
pub(crate) fn scatter_self<T: Float>(
    results: Vec<(usize, Vec<usize>, Vec<T>)>,
    n: usize,
    return_dist: bool,
) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
    let mut idx = vec![Vec::new(); n];
    let mut dists = if return_dist {
        Some(vec![Vec::new(); n])
    } else {
        None
    };
    for (orig, i, d) in results {
        idx[orig] = i;
        if let Some(ref mut fd) = dists {
            fd[orig] = d;
        }
    }
    fix_neg_dist(&mut dists);
    (idx, dists)
}

////////////////
// Exhaustive //
////////////////

/// Score a block of queries against every code row.
///
/// Rows are visited in tiles of [`EXH_CODE_TILE`] and every query of the block
/// scans a tile before the next is loaded, so the codes stream from memory
/// once per block rather than once per query.
///
/// ### Params
///
/// * `queries` - Queries in codebook space, `nb * dim`
/// * `codes` - All code rows, `m` per row
/// * `books_t` - Codebooks from [`transpose_codebooks`]
/// * `m` - Subspaces
/// * `sd` - Subvector dimension
/// * `k_cents` - Centroids per codebook
/// * `scale` - Table scale, see [`IvfAdc::scale`]
/// * `k` - Neighbours per query
/// * `luts` - Scratch, resized to `nb * m * k_cents`
/// * `heaps` - Scratch heaps, one per query after the call
///
/// ### Returns
///
/// Per query `(row ids, distances)`, nearest first
#[allow(clippy::too_many_arguments)]
pub(crate) fn exhaustive_block<T: Float>(
    queries: &[T],
    codes: &[u8],
    books_t: &[T],
    m: usize,
    sd: usize,
    k_cents: usize,
    scale: T,
    k: usize,
    luts: &mut Vec<T>,
    heaps: &mut Vec<BoundedMaxHeap<T>>,
) -> Vec<(Vec<usize>, Vec<T>)> {
    let dim = m * sd;
    let nb = queries.len() / dim;
    let n = codes.len() / m;
    let lut_len = m * k_cents;
    luts.resize(nb * lut_len, T::zero());
    heaps.resize_with(nb, || BoundedMaxHeap::new(k));
    for (q, (lut, heap)) in queries
        .chunks_exact(dim)
        .zip(luts.chunks_exact_mut(lut_len).zip(heaps.iter_mut()))
    {
        fill_lut_sq(q, books_t, m, sd, k_cents, scale, lut);
        heap.reset(k);
    }

    for t0 in (0..n).step_by(EXH_CODE_TILE) {
        let t1 = (t0 + EXH_CODE_TILE).min(n);
        for (lut, heap) in luts.chunks_exact(lut_len).zip(heaps.iter_mut()) {
            scan_range(codes, m, k_cents, lut, T::zero(), t0, t1, heap, |_| false);
        }
    }

    heaps[..nb]
        .iter_mut()
        .map(|h| finish_heap(h, None))
        .collect()
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Deterministic values in roughly [-1, 1)
    fn noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
            })
            .collect()
    }

    #[test]
    fn test_adc_dist_matches_naive_sum_for_any_m() {
        let k = 16;
        for m in [1usize, 3, 4, 8, 12, 13, 64] {
            let lut = noise(m * k, m as u64);
            let codes: Vec<u8> = (0..m).map(|s| ((s * 7 + 3) % k) as u8).collect();
            let want: f64 = (0..m).map(|s| lut[s * k + codes[s] as usize]).sum();
            assert_relative_eq!(adc_dist(&codes, &lut, k), want, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_fill_lut_sq_matches_row_major() {
        let (m, sd, k) = (4, 3, 10);
        let books: Vec<Vec<f64>> = (0..m).map(|s| noise(k * sd, 100 + s as u64)).collect();
        let books_t = transpose_codebooks(&books, sd, k);
        let q = noise(m * sd, 7);
        let mut lut = vec![0.0; m * k];
        fill_lut_sq(&q, &books_t, m, sd, k, 0.5, &mut lut);
        for s in 0..m {
            for c in 0..k {
                let want: f64 = (0..sd)
                    .map(|d| (q[s * sd + d] - books[s][c * sd + d]).powi(2))
                    .sum::<f64>()
                    * 0.5;
                assert_relative_eq!(lut[s * k + c], want, epsilon = 1e-12);
            }
        }
    }

    /// `||q - c||^2 + T1[c] + T2` has to reproduce the residual table entry by
    /// entry, summed over subspaces.
    #[test]
    fn test_decomposed_table_matches_residual_table() {
        let (m, sd, k, nlist) = (4, 8, 16, 3);
        let dim = m * sd;
        let train = noise(200 * dim, 3);
        let pq = ProductQuantiser::<f64>::train(
            &train,
            dim,
            m,
            Some(k),
            &Dist::SquaredEuclidean,
            5,
            1,
            Verbosity::Quiet,
        )
        .unwrap();
        let centroids = noise(nlist * dim, 9);
        let terms = cell_terms(&centroids, &pq);
        assert_eq!(terms.len(), nlist * m * k);

        let q = noise(dim, 11);
        let books_t = pq.codebooks_t();
        let mut t2 = vec![0.0; m * k];
        fill_lut_dot(&q, books_t, m, sd, k, -2.0, &mut t2);

        let codes: Vec<u8> = (0..m).map(|s| ((s * 5 + 1) % k) as u8).collect();
        for cell in 0..nlist {
            let c = &centroids[cell * dim..(cell + 1) * dim];
            let resid: Vec<f64> = q.iter().zip(c).map(|(a, b)| a - b).collect();
            let mut direct = vec![0.0; m * k];
            fill_lut_sq(&resid, books_t, m, sd, k, 1.0, &mut direct);

            let mut comp = vec![0.0; m * k];
            for i in 0..m * k {
                comp[i] = terms[cell * m * k + i] + t2[i];
            }
            let bias = f64::euclidean_simd(&q, c);
            assert_relative_eq!(
                bias + adc_dist(&codes, &comp, k),
                adc_dist(&codes, &direct, k),
                epsilon = 1e-9,
                max_relative = 1e-9
            );
        }
    }
}
