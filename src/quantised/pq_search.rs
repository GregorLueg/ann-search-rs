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

use faer::{Accum, MatMut, MatRef, Par};
use num_traits::Float;
use rayon::prelude::*;

use crate::prelude::*;
use crate::quantised::k_means::{dots_transposed, sq_dists_transposed, transpose_centroids};
use crate::quantised::quantisers::ProductQuantiser;
use crate::utils::dist::normalise_vector;
use crate::utils::gemm::gemm;
use crate::utils::k_means_utils::select_probed_clusters;

///////////////
// Constants //
///////////////

/// Default budget for the precomputed cell-term table, in bytes. Above it the
/// IVF indices fall back to building the residual table per probed cell. The
/// table is `nlist * m * n_centroids` elements, so it outgrows the codes
/// themselves for small datasets with many subspaces. Callers can change it
/// per index with `set_cell_terms_budget`. Not tuned.
pub(crate) const PRECOMP_TABLE_MAX_BYTES: usize = 256 << 20;

/// Fewest queries for which `query_batch` takes the batched path. Below it the
/// query pre-pass and the reordering are not amortised. Not tuned.
pub(crate) const BATCH_MIN_QUERIES: usize = 1024;

/// Queries per GEMM when the batched path rotates and probes. Not tuned.
const PROBE_BLOCK: usize = 64;

/// Queries per GEMM when the batched path builds query-only tables. Each
/// query's table is `m * n_centroids` elements, so this bounds the scratch.
/// Not tuned.
const LUT_BLOCK: usize = 32;

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
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn scan_range<T: Float>(
    codes: &[u8],
    m: usize,
    k: usize,
    lut: &[T],
    bias: T,
    start: usize,
    end: usize,
    heap: &mut BoundedMaxHeap<T>,
) {
    for (i, row) in codes[start * m..end * m].chunks_exact(m).enumerate() {
        heap.push(bias + adc_dist(row, lut, k), start + i);
    }
}

////////////////
// SOAR dedup //
////////////////

/// Runs of equal dedup key within each cell of a SOAR index.
///
/// A SOAR entry carries `u32::MAX` when it is its point's primary copy and
/// otherwise the cell holding the primary. The scan decides per run rather
/// than per entry whether to skip, so a cell laid out by
/// [`sort_spills_by_primary`] costs one check per source cell. Derived from
/// the keys, so not serialised; any layout is correct, a sorted one just has
/// fewer runs.
#[derive(Default)]
pub(crate) struct KeyRuns {
    /// Run ranges per cell, `nlist + 1`
    pub offsets: Vec<u32>,
    /// `(key, one past the last position)` per run
    pub runs: Vec<(u32, u32)>,
}

impl KeyRuns {
    /// Run-length encode the keys of every cell.
    ///
    /// ### Params
    ///
    /// * `dedup_key` - Key per position
    /// * `cell_offsets` - Positions per cell, `nlist + 1`
    ///
    /// ### Returns
    ///
    /// The run table
    pub fn new(dedup_key: &[u32], cell_offsets: &[u32]) -> Self {
        let mut offsets = Vec::with_capacity(cell_offsets.len());
        let mut runs = Vec::new();
        offsets.push(0);
        for w in cell_offsets.windows(2) {
            for pos in w[0]..w[1] {
                let key = dedup_key[pos as usize];
                match runs.last_mut() {
                    Some((k, end)) if *k == key && *end == pos && pos != w[0] => *end = pos + 1,
                    _ => runs.push((key, pos + 1)),
                }
            }
            offsets.push(runs.len() as u32);
        }
        Self { offsets, runs }
    }

    /// Bytes held by the table
    ///
    /// ### Returns
    ///
    /// Capacity of both buffers in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        self.offsets.capacity() * std::mem::size_of::<u32>()
            + self.runs.capacity() * std::mem::size_of::<(u32, u32)>()
    }
}

/// Order every cell as primaries first, then spilled entries grouped by the
/// cell holding their primary, so [`KeyRuns`] finds long runs.
///
/// ### Params
///
/// * `members` - Point id per position, reordered within each cell
/// * `offsets` - Positions per cell, `nlist + 1`
/// * `primary` - Primary cell per point
pub(crate) fn sort_spills_by_primary(members: &mut [u32], offsets: &[u32], primary: &[u32]) {
    for (c, w) in offsets.windows(2).enumerate() {
        members[w[0] as usize..w[1] as usize].sort_by_key(|&p| {
            let pc = primary[p as usize];
            (pc != c as u32, pc, p)
        });
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
/// * `max_bytes` - Budget; a larger table is not built
///
/// ### Returns
///
/// `nlist * m * k` elements, cell-major, or an empty vector when the table
/// would exceed `max_bytes`.
pub(crate) fn cell_terms<T: AnnSearchFloat>(
    centroids: &[T],
    pq: &ProductQuantiser<T>,
    max_bytes: usize,
) -> Vec<T> {
    let (m, sd, k) = (pq.m(), pq.subvec_dim(), pq.n_centroids());
    let dim = m * sd;
    let nlist = centroids.len() / dim;
    if nlist * m * k * std::mem::size_of::<T>() > max_bytes {
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
                *v += nr;
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
    /// * `qlut` - This query's row of [`Self::query_luts`]; built here when
    ///   `None`. Ignored without cell terms.
    /// * `runs` - SOAR dedup runs; `None` scans every entry
    /// * `skip_key` - Runs whose (non-primary) key this returns `true` for are
    ///   not scored
    pub fn search<F: Fn(u32) -> bool>(
        &self,
        probes: &[usize],
        s: &mut AdcScratch<T>,
        qlut: Option<&[T]>,
        runs: Option<&KeyRuns>,
        skip_key: F,
    ) {
        let (m, sd, k, dim) = (self.m, self.sd, self.k, self.dim);
        let precomputed = !self.cell_terms.is_empty();
        let two = T::one() + T::one();
        let qlut = match qlut {
            Some(row) => row,
            None => {
                if precomputed {
                    fill_lut_dot(&s.q, self.books_t, m, sd, k, -two, &mut s.query_lut);
                }
                &s.query_lut
            }
        };
        for &cell in probes {
            let c = &self.centroids[cell * dim..(cell + 1) * dim];
            let bias = if precomputed {
                let t1 = &self.cell_terms[cell * m * k..(cell + 1) * m * k];
                for ((o, &a), &b) in s.lut.iter_mut().zip(t1).zip(qlut) {
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
            let (lo, hi) = (self.offsets[cell] as usize, self.offsets[cell + 1] as usize);
            let Some(r) = runs else {
                scan_range(self.codes, m, k, &s.lut, bias, lo, hi, &mut s.heap);
                continue;
            };
            let mut start = lo;
            for &(key, end) in &r.runs[r.offsets[cell] as usize..r.offsets[cell + 1] as usize] {
                let end = end as usize;
                if key == u32::MAX || !skip_key(key) {
                    scan_range(self.codes, m, k, &s.lut, bias, start, end, &mut s.heap);
                }
                start = end;
            }
        }
    }

    /// Query-only tables `-2 <q_s, c_sj>` of a block of queries, one GEMM per
    /// subspace. Same values as the per-query build up to summation order.
    ///
    /// ### Params
    ///
    /// * `tile` - Row-major queries in codebook space, `b * dim`
    /// * `out` - Output, `b * m * k`, one table per query
    pub fn query_luts(&self, tile: &[T], out: &mut [T]) {
        let (m, sd, k, dim) = (self.m, self.sd, self.k, self.dim);
        let b = tile.len() / dim;
        let two = T::one() + T::one();
        for s in 0..m {
            // Not `from_row_major_slice_with_stride_mut`: faer 0.24.4 lays that
            // out column-major (unit row stride), which writes out of bounds.
            gemm(
                MatMut::from_column_major_slice_with_stride_mut(&mut out[s * k..], k, b, m * k)
                    .transpose_mut(),
                Accum::Replace,
                MatRef::from_row_major_slice_with_stride(&tile[s * sd..], b, sd, dim),
                MatRef::from_row_major_slice(&self.books_t[s * sd * k..(s + 1) * sd * k], sd, k),
                -two,
                Par::Seq,
            );
        }
    }
}

/// What the batched path needs to prepare and probe queries on its own.
pub(crate) struct BatchPlan<'a, T> {
    /// Scan state; its centroids are the ones probed
    pub adc: IvfAdc<'a, T>,
    /// Per-cell counts for the probe underfill, `nlist + 1`
    pub probe_offsets: &'a [u32],
    /// OPQ rotation, `dim * dim` row-major; `None` for plain PQ
    pub rotation: Option<&'a [T]>,
    /// Normalise prepared queries
    pub cosine: bool,
    /// Requested cells
    pub nprobe: usize,
    /// Neighbours the probe list has to reach
    pub k: usize,
}

/// One query's `(ids, distances)`, nearest first.
type QueryResult<T> = (Vec<usize>, Vec<T>);

/// Rotate, normalise and probe a block of queries, see [`probe_rows`].
///
/// ### Params
///
/// * `plan` - Index state
/// * `c_sq` - Squared centroid norms, `nlist`
/// * `queries` - Raw queries, `b * dim`
/// * `out` - Prepared queries, `b * dim`
/// * `probes` - Probe lists, `b`
/// * `dots` - Scratch for the `b * nlist` dot products
/// * `buf` - Scratch for one query's `(distance, cell)` pairs
fn prepare_block<T: AnnSearchFloat>(
    plan: &BatchPlan<'_, T>,
    c_sq: &[T],
    queries: &[T],
    out: &mut [T],
    probes: &mut [Vec<usize>],
    dots: &mut Vec<T>,
    buf: &mut Vec<(T, usize)>,
) {
    let dim = plan.adc.dim;
    let b = out.len() / dim;
    match plan.rotation {
        Some(r) => gemm(
            MatMut::from_row_major_slice_mut(out, b, dim),
            Accum::Replace,
            MatRef::from_row_major_slice(queries, b, dim),
            MatRef::from_row_major_slice(r, dim, dim).transpose(),
            T::one(),
            Par::Seq,
        ),
        None => out.copy_from_slice(queries),
    }
    if plan.cosine {
        out.chunks_exact_mut(dim).for_each(normalise_vector);
    }
    probe_rows(plan, c_sq, out, probes, dots, buf);
}

/// Probe a block of prepared queries with one GEMM against the centroids.
///
/// Centroid distances come from `||q||^2 + ||c||^2 - 2 <q, c>`, so they match
/// the per-query [`probe_cells`] up to rounding and a near tie can order
/// differently.
///
/// ### Params
///
/// * `plan` - Index state
/// * `c_sq` - Squared centroid norms, `nlist`
/// * `prepared` - Queries in codebook space, `b * dim`
/// * `probes` - Probe lists, `b`
/// * `dots` - Scratch for the `b * nlist` dot products
/// * `buf` - Scratch for one query's `(distance, cell)` pairs
fn probe_rows<T: AnnSearchFloat>(
    plan: &BatchPlan<'_, T>,
    c_sq: &[T],
    prepared: &[T],
    probes: &mut [Vec<usize>],
    dots: &mut Vec<T>,
    buf: &mut Vec<(T, usize)>,
) {
    let dim = plan.adc.dim;
    let nlist = c_sq.len();
    let b = prepared.len() / dim;
    let two = T::one() + T::one();
    dots.resize(b * nlist, T::zero());
    gemm(
        MatMut::from_row_major_slice_mut(dots, b, nlist),
        Accum::Replace,
        MatRef::from_row_major_slice(prepared, b, dim),
        MatRef::from_row_major_slice(plan.adc.centroids, nlist, dim).transpose(),
        T::one(),
        Par::Seq,
    );
    for (i, q) in prepared.chunks_exact(dim).enumerate() {
        let q_sq = T::dot_simd(q, q);
        buf.clear();
        buf.extend(
            dots[i * nlist..(i + 1) * nlist]
                .iter()
                .zip(c_sq)
                .enumerate()
                .map(|(c, (&d, &cs))| (q_sq + cs - two * d, c)),
        );
        probes[i] = select_probed_clusters(buf, plan.probe_offsets, plan.nprobe, plan.k);
    }
}

/// Run a batch of queries in parallel, nearest-centroid order first.
///
/// Queries are rotated, normalised and probed up front in blocks
/// ([`prepare_block`]), then scored ordered by their nearest cell, so
/// consecutive queries on a thread share cells and their codes and tables
/// stay cache resident. With cell terms, each run of [`LUT_BLOCK`] queries
/// gets its query-only tables from [`IvfAdc::query_luts`]. Both GEMM steps
/// change summation order, so results can differ from the per-query path in
/// the last bits and at near ties. Results come back in input order.
///
/// ### Params
///
/// * `queries` - Raw queries, row-major
/// * `plan` - Index state for preparing and probing
/// * `scratch` - Builds one thread's scratch
/// * `search` - Scores the query held in the scratch against its probes,
///   given its query-only table when there is one, and returns
///   `(ids, distances)`
/// * `progress` - Ticked once per query
///
/// ### Returns
///
/// Per query `(ids, distances)`, nearest first
pub(crate) fn run_ordered<T, S, Q>(
    queries: &[T],
    plan: &BatchPlan<'_, T>,
    scratch: S,
    search: Q,
    progress: &crate::utils::verbosity::QueryProgress,
) -> Vec<(Vec<usize>, Vec<T>)>
where
    T: AnnSearchFloat,
    S: Fn() -> AdcScratch<T> + Sync + Send,
    Q: Fn(&[usize], &mut AdcScratch<T>, Option<&[T]>) -> (Vec<usize>, Vec<T>) + Sync + Send,
{
    let adc = &plan.adc;
    let dim = adc.dim;
    let nq = queries.len() / dim;
    let c_sq = centroid_sq_norms(adc);

    let mut prepared = vec![T::zero(); nq * dim];
    let mut probes: Vec<Vec<usize>> = vec![Vec::new(); nq];
    prepared
        .par_chunks_mut(PROBE_BLOCK * dim)
        .zip(queries.par_chunks(PROBE_BLOCK * dim))
        .zip(probes.par_chunks_mut(PROBE_BLOCK))
        .for_each_init(
            || (Vec::new(), Vec::new()),
            |(dots, buf), ((out, raw), pr)| prepare_block(plan, &c_sq, raw, out, pr, dots, buf),
        );

    let mut order: Vec<u32> = (0..nq as u32).collect();
    order.par_sort_by_key(|&i| probes[i as usize].first().copied().unwrap_or(0));

    let precomputed = !adc.cell_terms.is_empty();
    let mk = adc.m * adc.k;
    let chunk = nq.div_ceil(rayon::current_num_threads().max(1) * 8).max(1);
    let scored: Vec<(u32, QueryResult<T>)> = order
        .par_chunks(chunk)
        .map_init(
            || (scratch(), Vec::new(), Vec::new()),
            |(s, tile, qluts), block| {
                let mut res = Vec::with_capacity(block.len());
                for sub in block.chunks(LUT_BLOCK) {
                    if precomputed {
                        tile.clear();
                        for &i in sub {
                            let i = i as usize;
                            tile.extend_from_slice(&prepared[i * dim..(i + 1) * dim]);
                        }
                        qluts.resize(sub.len() * mk, T::zero());
                        adc.query_luts(tile, qluts);
                    }
                    for (j, &i) in sub.iter().enumerate() {
                        let i_us = i as usize;
                        s.q.copy_from_slice(&prepared[i_us * dim..(i_us + 1) * dim]);
                        progress.tick();
                        let qlut = precomputed.then(|| &qluts[j * mk..(j + 1) * mk]);
                        res.push((i, search(&probes[i_us], s, qlut)));
                    }
                }
                res
            },
        )
        .flatten()
        .collect();

    let mut out = vec![(Vec::new(), Vec::new()); nq];
    for (i, r) in scored {
        out[i as usize] = r;
    }
    out
}

/// Squared norms of the probed centroids.
///
/// ### Params
///
/// * `adc` - Scan state holding the centroids
///
/// ### Returns
///
/// One norm per cell
fn centroid_sq_norms<T: AnnSearchFloat>(adc: &IvfAdc<'_, T>) -> Vec<T> {
    adc.centroids
        .chunks_exact(adc.dim)
        .map(|c| T::dot_simd(c, c))
        .collect()
}

/// Self-kNN over indexed positions, in blocks of [`PROBE_BLOCK`].
///
/// Each block is reconstructed, normalised under cosine and probed with one
/// GEMM ([`probe_rows`]), then scored in runs of [`LUT_BLOCK`] with their
/// query-only tables from [`IvfAdc::query_luts`] when there are cell terms.
/// The positions arrive in cell order, so consecutive points already share
/// cells and no reordering is needed. Reconstructed points are already in
/// codebook space, so `plan.rotation` is ignored.
///
/// ### Params
///
/// * `rows` - Positions to query
/// * `plan` - Index state for probing
/// * `scratch` - Builds one thread's scratch
/// * `reconstruct` - Writes the point at a position, in codebook space and
///   not normalised
/// * `search` - Scores the point held in the scratch, see [`run_ordered`]
/// * `progress` - Ticked once per point
///
/// ### Returns
///
/// Per row `(ids, distances)`, in the order of `rows`
pub(crate) fn run_self<T, S, R, Q>(
    rows: &[u32],
    plan: &BatchPlan<'_, T>,
    scratch: S,
    reconstruct: R,
    search: Q,
    progress: &crate::utils::verbosity::QueryProgress,
) -> Vec<QueryResult<T>>
where
    T: AnnSearchFloat,
    S: Fn() -> AdcScratch<T> + Sync + Send,
    R: Fn(usize, &mut [T]) + Sync + Send,
    Q: Fn(&[usize], &mut AdcScratch<T>, Option<&[T]>) -> QueryResult<T> + Sync + Send,
{
    let adc = &plan.adc;
    let dim = adc.dim;
    let c_sq = centroid_sq_norms(adc);
    let precomputed = !adc.cell_terms.is_empty();
    let mk = adc.m * adc.k;

    rows.par_chunks(PROBE_BLOCK)
        .map_init(
            || {
                let probes: Vec<Vec<usize>> = vec![Vec::new(); PROBE_BLOCK];
                (
                    scratch(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    probes,
                )
            },
            |(s, tile, qluts, dots, buf, probes), block| {
                tile.resize(block.len() * dim, T::zero());
                for (&pos, row) in block.iter().zip(tile.chunks_exact_mut(dim)) {
                    reconstruct(pos as usize, row);
                    if plan.cosine {
                        normalise_vector(row);
                    }
                }
                probe_rows(plan, &c_sq, tile, &mut probes[..block.len()], dots, buf);

                let mut res = Vec::with_capacity(block.len());
                for j0 in (0..block.len()).step_by(LUT_BLOCK) {
                    let j1 = (j0 + LUT_BLOCK).min(block.len());
                    if precomputed {
                        qluts.resize((j1 - j0) * mk, T::zero());
                        adc.query_luts(&tile[j0 * dim..j1 * dim], qluts);
                    }
                    for j in j0..j1 {
                        s.q.copy_from_slice(&tile[j * dim..(j + 1) * dim]);
                        progress.tick();
                        let qlut = precomputed.then(|| &qluts[(j - j0) * mk..(j - j0 + 1) * mk]);
                        res.push(search(&probes[j], s, qlut));
                    }
                }
                res
            },
        )
        .flatten()
        .collect()
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
            scan_range(codes, m, k_cents, lut, T::zero(), t0, t1, heap);
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
        let terms = cell_terms(&centroids, &pq, PRECOMP_TABLE_MAX_BYTES);
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
