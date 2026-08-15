//! Multi-probe LSH over quantile-bucketed random projections.
//!
//! Each table holds `n_proj` orthogonalised random projections. A projection is
//! turned into a slot by binary-searching `2^slot_bits - 1` **quantile
//! boundaries** measured on a subsample of the data, and the per-table code is
//! the concatenation of those slots. That one construction covers both metrics:
//!
//! * `slot_bits == 1` reduces to SimHash whose threshold is the median of the
//!   projection rather than zero. Classic SimHash puts every hyperplane through
//!   the origin, so on data with a large shared mean offset (foundation-model
//!   cell embeddings being the motivating case) the sign is decided by the
//!   offset and nearly every point lands in the same bucket. A median threshold
//!   splits each bit 50/50 by construction and the degeneracy disappears.
//! * `slot_bits >= 2` quantises the projection into levels, which is the
//!   p-stable construction of Datar et al. (2004) with the fixed width `w`
//!   replaced by data-derived boundaries. Unlike SimHash it is sensitive to
//!   vector magnitude, which matters because squared Euclidean distance is.
//!
//! Trading `w` for quantiles makes the collision probability data-dependent
//! instead of a clean function of `||u - v||`, so the theoretical guarantee of
//! Datar et al. is given up in exchange for bucket balance on skewed data. That
//! is a deliberate choice.
//!
//! Cosine hashes the L2-normalised vector (a threshold is not scale-invariant,
//! so it has to); Euclidean hashes the raw one. Neither stores a second copy of
//! the data: projection is linear, so the cosine path just divides the
//! projection by the norm it already keeps.
//!
//! Buckets live in a directly addressed CSR layout rather than a hash map,
//! which is what bounds `bits_per_hash` to [`MAX_BITS_PER_HASH`].
//!
//! ### References
//!
//! Datar, Immorlica, Indyk & Mirrokni, SoCG, 2004 (p-stable LSH);
//! Lv, Josephson, Wang, Charikar & Li, VLDB, 2007 (query-directed multi-probe)

use faer::{linalg::matmul::matmul, Accum, Mat, MatRef, Par, RowRef};
use fixedbitset::FixedBitSet;
use num_traits::Float;
use rand::{prelude::*, rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use std::cell::RefCell;
use std::num::NonZero;
use thousands::*;

use crate::prelude::*;
use crate::utils::*;

///////////////
// Constants //
///////////////

/// Largest supported `bits_per_hash`.
///
/// The bucket table is directly addressed, so a table costs
/// `(1 << bits_per_hash) + 1` `u32` offsets. 20 bits is ~4 MB per table, which
/// is the point where the offsets array stops fitting any sensible cache
/// budget. It is also far past useful: at 20 bits every bucket is a singleton
/// long before a realistic dataset fills the code space.
pub const MAX_BITS_PER_HASH: usize = 20;

/// Number of vectors sampled when fitting the quantile boundaries.
///
/// Quantiles converge quickly, so 20k rows pin the boundaries well within the
/// noise of the projections themselves while keeping the fitting GEMM's
/// intermediate at a few tens of MB.
const BOUNDARY_SAMPLE: usize = 20_000;

/// Rows per tile in the projection GEMM.
///
/// Bounds the intermediate to `GEMM_ROW_TILE * num_tables * n_proj` elements,
/// a few MB at any legal parameter combination.
const GEMM_ROW_TILE: usize = 4096;

/// Vectors sampled at random when no bucket yielded a single candidate.
const FALLBACK_SAMPLE: usize = 1000;

//////////////////////////
// Thread-local buffers //
//////////////////////////

// Reused across queries on the same thread. `visited` deduplicates candidates
// across tables and probes; `touched` records which bits were set so the reset
// is O(candidates) rather than O(n / 64). The top-k buffer is not stored here
// because it is typed by T and tiny (k ~ 15).
thread_local! {
    static LSH_VISITED: RefCell<FixedBitSet> = const { RefCell::new(FixedBitSet::new()) };
    static LSH_TOUCHED: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

/////////////
// Helpers //
/////////////

/// Resolve `slot_bits` when the caller did not pick one.
///
/// Cosine wants the angular family, which is the sign of the projection, so a
/// single bit. Squared Euclidean is magnitude-sensitive and needs at least two
/// levels per projection to see it; two is the cheapest value that does, and it
/// keeps `n_proj` at half the bit budget so multi-probe still has projections
/// to perturb.
///
/// ### Params
///
/// * `slot_bits` - Caller's choice, if any
/// * `metric` - Distance metric the index was built for
/// * `bits_per_hash` - Total bits in a code, used as the upper clamp
///
/// ### Returns
///
/// Bits per quantised projection, in `1..=bits_per_hash`
fn resolve_slot_bits(slot_bits: Option<usize>, metric: Dist, bits_per_hash: usize) -> usize {
    let auto = match metric {
        Dist::Cosine => 1,
        _ => 2,
    };

    slot_bits.unwrap_or(auto).clamp(1, bits_per_hash)
}

/// Orthogonalise random projections within each table via modified
/// Gram-Schmidt
///
/// Orthogonal projections decorrelate the slots, which improves bucket balance
/// on top of what the quantile boundaries already give per projection.
///
/// ### Params
///
/// * `vecs` - The random projection vectors to orthogonalise (mutated in
///   place), row-major `(num_tables * n_proj) x dim`
/// * `num_tables` - Number of tables for the LSH index
/// * `n_proj` - Projections per table
/// * `dim` - Number of dimensions
fn orthogonalise_table_projections<T>(vecs: &mut [T], num_tables: usize, n_proj: usize, dim: usize)
where
    T: Float,
{
    for table_idx in 0..num_tables {
        let base = table_idx * n_proj * dim;

        for i in 0..n_proj {
            let i_base = base + i * dim;

            // Orthogonalise against previous
            for j in 0..i {
                let j_base = base + j * dim;
                let mut dot = T::zero();
                for d in 0..dim {
                    dot = dot + vecs[i_base + d] * vecs[j_base + d];
                }
                for d in 0..dim {
                    vecs[i_base + d] = vecs[i_base + d] - dot * vecs[j_base + d];
                }
            }

            // Normalise
            let mut norm_sq = T::zero();
            for d in 0..dim {
                norm_sq = norm_sq + vecs[i_base + d] * vecs[i_base + d];
            }
            let norm = norm_sq.sqrt();
            if norm > T::epsilon() {
                for d in 0..dim {
                    vecs[i_base + d] = vecs[i_base + d] / norm;
                }
            }
        }
    }
}

/// Scale factor applied to a vector's projections before bucketing
///
/// Cosine hashes the L2-normalised vector; because projection is linear this is
/// a division of the projection by the norm rather than a second copy of the
/// data. A degenerate (zero) norm collapses every projection to zero, which
/// puts the vector in whichever bucket the boundaries assign to zero.
///
/// ### Params
///
/// * `metric` - Distance metric the index was built for
/// * `norm` - L2 norm of the vector being hashed
///
/// ### Returns
///
/// Multiplier for the raw projections
#[inline]
fn hash_scale<T: Float>(metric: Dist, norm: T) -> T {
    match metric {
        Dist::Cosine if norm > T::epsilon() => T::one() / norm,
        Dist::Cosine => T::zero(),
        _ => T::one(),
    }
}

/// Turn one row of projections into per-table codes
///
/// ### Params
///
/// * `proj_row` - Projections for a single vector, layout
///   `[table_idx * n_proj + proj_idx]`
/// * `boundaries` - Quantile boundaries, layout
///   `[(table_idx * n_proj + proj_idx) * n_bounds + b]`, ascending
/// * `num_tables` - Number of hash tables
/// * `n_proj` - Projections per table
/// * `slot_bits` - Bits per quantised projection
/// * `out` - Destination for the `num_tables` codes
#[inline]
fn encode_row<T: Float>(
    proj_row: &[T],
    boundaries: &[T],
    num_tables: usize,
    n_proj: usize,
    slot_bits: usize,
    out: &mut [u32],
) {
    let n_bounds = (1usize << slot_bits) - 1;

    for table_idx in 0..num_tables {
        let mut code: u32 = 0;
        for j in 0..n_proj {
            let col = table_idx * n_proj + j;
            let bounds = &boundaries[col * n_bounds..(col + 1) * n_bounds];
            let slot = bounds.partition_point(|&b| b <= proj_row[col]);
            code |= (slot as u32) << (j * slot_bits);
        }
        out[table_idx] = code;
    }
}

/// Generate probe codes ordered by how close the query sits to a slot boundary
///
/// Query-directed multi-probe: a projection whose value is a hair away from a
/// boundary is the one most likely to have put the true neighbour in the
/// adjacent slot, so it is perturbed first. Single-projection shifts come
/// first, then pairs scored by the sum of their gaps.
///
/// Every perturbation is bounds-checked when it is generated, so shifting a
/// slot can never carry into a neighbouring projection's bits and a probe code
/// is plain arithmetic on the base code.
///
/// ### Params
///
/// * `base_code` - Code of the query itself
/// * `perturb` - Candidate shifts as `(gap, proj_idx, delta)`, sorted ascending
/// * `max_probes` - Maximum number of probe codes to emit
/// * `out` - Destination, cleared by the caller
///
/// ### References
///
/// Lv, Josephson, Wang, Charikar & Li, VLDB, 2007
fn build_probes<T: Float>(
    base_code: u32,
    perturb: &[(OrderedFloat<T>, usize, i64)],
    max_probes: usize,
    out: &mut Vec<u32>,
) {
    let base = base_code as i64;

    for &(_, _, delta) in perturb {
        if out.len() >= max_probes {
            return;
        }
        out.push((base + delta) as u32);
    }

    for (i, &(_, proj_i, delta_i)) in perturb.iter().enumerate() {
        for &(_, proj_j, delta_j) in &perturb[i + 1..] {
            // Two shifts of the same projection contradict each other
            if proj_i == proj_j {
                continue;
            }
            if out.len() >= max_probes {
                return;
            }
            out.push((base + delta_i + delta_j) as u32);
        }
    }
}

/// Project a block of vectors onto every table's projections
///
/// One GEMM rather than `rows * num_tables * n_proj` strided dot products.
///
/// ### Params
///
/// * `rows_flat` - Vectors, row-major `rows x dim`
/// * `rows` - Number of vectors in the block
/// * `random_vecs` - Projections, row-major `n_cols x dim`
/// * `n_cols` - `num_tables * n_proj`
/// * `dim` - Embedding dimensionality
/// * `par` - Parallelism for the GEMM itself; `Par::Seq` when the caller
///   already fanned out over blocks
/// * `out` - Output tile, resized to `rows x n_cols` if needed
fn project_block<T>(
    rows_flat: &[T],
    rows: usize,
    random_vecs: &[T],
    n_cols: usize,
    dim: usize,
    par: Par,
    out: &mut Mat<T>,
) where
    T: AnnSearchFloat,
{
    let data = MatRef::from_row_major_slice(rows_flat, rows, dim);
    let proj = MatRef::from_row_major_slice(random_vecs, n_cols, dim);

    if out.nrows() != rows || out.ncols() != n_cols {
        *out = Mat::<T>::zeros(rows, n_cols);
    }

    matmul(
        out.as_mut(),
        Accum::Replace,
        data,
        proj.transpose(),
        T::one(),
        par,
    );
}

/// Fit the per-projection quantile boundaries on a subsample
///
/// Boundary `b` of a projection is the `(b + 1) / 2^slot_bits` quantile of that
/// projection over the sample, so every slot holds the same share of the
/// sampled data regardless of how skewed the projection is. This is what makes
/// bucket occupancy independent of any shared mean offset in the data.
///
/// Successive quantiles are taken with `select_nth_unstable_by` over a
/// shrinking suffix rather than a full sort: there are at most 15 boundaries
/// per projection, so a sort would be the more expensive of the two.
///
/// ### Params
///
/// * `vectors_flat` - All vectors, row-major
/// * `norms` - Per-vector L2 norms, empty unless the metric is cosine
/// * `n` - Number of vectors
/// * `dim` - Embedding dimensionality
/// * `random_vecs` - Projections, row-major `n_cols x dim`
/// * `n_cols` - `num_tables * n_proj`
/// * `n_slots` - `1 << slot_bits`
/// * `metric` - Distance metric the index is built for
/// * `seed` - Random seed, so the subsample is reproducible
///
/// ### Returns
///
/// Ascending boundaries, layout `[col * (n_slots - 1) + b]`
#[allow(clippy::too_many_arguments)]
fn fit_boundaries<T>(
    vectors_flat: &[T],
    norms: &[T],
    n: usize,
    dim: usize,
    random_vecs: &[T],
    n_cols: usize,
    n_slots: usize,
    metric: Dist,
    seed: usize,
) -> Vec<T>
where
    T: AnnSearchFloat,
{
    let n_bounds = n_slots - 1;
    let sample_size = BOUNDARY_SAMPLE.min(n);

    let mut rng = StdRng::seed_from_u64(seed as u64 ^ 0x9E37_79B9);
    let sample_ids: Vec<usize> = if sample_size == n {
        (0..n).collect()
    } else {
        rand::seq::index::sample(&mut rng, n, sample_size).into_vec()
    };

    let mut sample_flat = Vec::with_capacity(sample_size * dim);
    for &i in &sample_ids {
        sample_flat.extend_from_slice(&vectors_flat[i * dim..(i + 1) * dim]);
    }

    let mut tile = Mat::<T>::zeros(0, 0);
    let par = Par::Rayon(NonZero::new(rayon::current_num_threads().max(1)).unwrap());
    project_block(
        &sample_flat,
        sample_size,
        random_vecs,
        n_cols,
        dim,
        par,
        &mut tile,
    );

    // Column-major copy so each projection's values are contiguous for the
    // selection below.
    let mut columns = vec![T::zero(); n_cols * sample_size];
    for c in 0..n_cols {
        for (r, &id) in sample_ids.iter().enumerate() {
            let scale = hash_scale(
                metric,
                if norms.is_empty() {
                    T::one()
                } else {
                    norms[id]
                },
            );
            columns[c * sample_size + r] = tile[(r, c)] * scale;
        }
    }

    let mut boundaries = vec![T::zero(); n_cols * n_bounds];
    boundaries
        .par_chunks_mut(n_bounds)
        .zip(columns.par_chunks_mut(sample_size))
        .for_each(|(bounds, col)| {
            let mut lo = 0usize;
            for (b, slot) in bounds.iter_mut().enumerate() {
                let target = ((b + 1) * sample_size) / n_slots;
                if target <= lo {
                    // Fewer samples than slots: collapse onto the previous
                    // boundary, which just leaves the slot in between empty.
                    *slot = col[lo.min(sample_size - 1)];
                    continue;
                }
                col[lo..].select_nth_unstable_by(target - lo, |a, b| {
                    a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                });
                *slot = col[target];
                lo = target + 1;
            }
        });

    boundaries
}

/// Assign every vector its per-table code
///
/// ### Params
///
/// * `vectors_flat` - All vectors, row-major
/// * `norms` - Per-vector L2 norms, empty unless the metric is cosine
/// * `n` - Number of vectors
/// * `dim` - Embedding dimensionality
/// * `random_vecs` - Projections, row-major `n_cols x dim`
/// * `boundaries` - Quantile boundaries from [`fit_boundaries`]
/// * `num_tables` - Number of hash tables
/// * `n_proj` - Projections per table
/// * `slot_bits` - Bits per quantised projection
/// * `metric` - Distance metric the index is built for
///
/// ### Returns
///
/// Codes laid out `[vec_idx * num_tables + table_idx]`; sample-major so the
/// GEMM tiles write contiguously.
#[allow(clippy::too_many_arguments)]
fn assign_codes<T>(
    vectors_flat: &[T],
    norms: &[T],
    n: usize,
    dim: usize,
    random_vecs: &[T],
    boundaries: &[T],
    num_tables: usize,
    n_proj: usize,
    slot_bits: usize,
    metric: Dist,
) -> Vec<u32>
where
    T: AnnSearchFloat,
{
    let n_cols = num_tables * n_proj;
    let mut codes = vec![0u32; n * num_tables];

    vectors_flat
        .par_chunks(GEMM_ROW_TILE * dim)
        .zip(codes.par_chunks_mut(GEMM_ROW_TILE * num_tables))
        .enumerate()
        .for_each_init(
            || (Mat::<T>::zeros(0, 0), vec![T::zero(); n_cols]),
            |(tile, scratch), (chunk_idx, (rows_flat, out))| {
                let rows = rows_flat.len() / dim;
                let row_base = chunk_idx * GEMM_ROW_TILE;

                project_block(rows_flat, rows, random_vecs, n_cols, dim, Par::Seq, tile);

                for r in 0..rows {
                    let scale = hash_scale(
                        metric,
                        if norms.is_empty() {
                            T::one()
                        } else {
                            norms[row_base + r]
                        },
                    );
                    for (c, slot) in scratch.iter_mut().enumerate() {
                        *slot = tile[(r, c)] * scale;
                    }
                    encode_row(
                        scratch,
                        boundaries,
                        num_tables,
                        n_proj,
                        slot_bits,
                        &mut out[r * num_tables..(r + 1) * num_tables],
                    );
                }
            },
        );

    codes
}

/// Counting-sort the codes of every table into a CSR bucket layout
///
/// Mirrors [`crate::utils::build_csr_layout`], but keeps `u32` throughout: the
/// bucket contents are half the size of a `Vec<usize>` and, unlike the map of
/// per-bucket `Vec`s this replaces, they are one contiguous allocation.
///
/// ### Params
///
/// * `codes` - Per-vector codes, layout `[vec_idx * num_tables + table_idx]`
/// * `n` - Number of vectors
/// * `num_tables` - Number of hash tables
/// * `n_buckets` - `1 << (n_proj * slot_bits)`
///
/// ### Returns
///
/// Tuple of `(offsets, ids)` with layouts
/// `[table_idx * (n_buckets + 1) + code]` and `[table_idx * n + pos]`
fn build_bucket_csr(
    codes: &[u32],
    n: usize,
    num_tables: usize,
    n_buckets: usize,
) -> (Vec<u32>, Vec<u32>) {
    let mut offsets = vec![0u32; num_tables * (n_buckets + 1)];
    let mut ids = vec![0u32; num_tables * n];

    offsets
        .par_chunks_mut(n_buckets + 1)
        .zip(ids.par_chunks_mut(n))
        .enumerate()
        .for_each(|(table_idx, (offs, out))| {
            for i in 0..n {
                offs[codes[i * num_tables + table_idx] as usize + 1] += 1;
            }
            for b in 1..=n_buckets {
                offs[b] += offs[b - 1];
            }

            let mut cursor: Vec<u32> = offs[..n_buckets].to_vec();
            for i in 0..n {
                let code = codes[i * num_tables + table_idx] as usize;
                out[cursor[code] as usize] = i as u32;
                cursor[code] += 1;
            }
        });

    (offsets, ids)
}

//////////////
// LSHIndex //
//////////////

/// LSH index for approximate nearest neighbour search
///
/// Multiple hash tables of quantile-bucketed random projections partition the
/// space; vectors sharing a code share a bucket. Multi-probe querying visits
/// neighbouring buckets ordered by how close the query sits to each slot
/// boundary, which buys recall without paying for more tables.
///
/// See the module documentation for why the boundaries are quantiles rather
/// than the origin (SimHash) or a fixed width (p-stable LSH).
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct LSHIndex<T> {
    /// Original data, flattened row-major for cache efficiency
    pub vectors_flat: Vec<T>,
    /// Embedding dimensionality
    pub dim: usize,
    /// Number of vectors
    pub n: usize,
    /// Pre-computed L2 norms for Cosine distance (empty for Euclidean)
    norms: Vec<T>,
    /// Distance metric (Euclidean or Cosine)
    metric: Dist,
    /// Orthogonalised random projections from N(0,1), row-major
    /// `(num_tables * n_proj) x dim`
    random_vecs: Vec<T>,
    /// Ascending quantile boundaries per projection, layout
    /// `[(table_idx * n_proj + proj_idx) * (n_slots - 1) + b]`
    boundaries: Vec<T>,
    /// CSR bucket starts, layout `[table_idx * (n_buckets + 1) + code]`
    bucket_offsets: Vec<u32>,
    /// CSR bucket contents, layout `[table_idx * n + pos]`
    bucket_ids: Vec<u32>,
    /// Number of hash tables
    num_tables: usize,
    /// Bits in each hash code (higher = fewer collisions)
    bits_per_hash: usize,
    /// Projections per table, `bits_per_hash / slot_bits`
    n_proj: usize,
    /// Bits per quantised projection
    slot_bits: usize,
    /// Addressable codes per table, `1 << (n_proj * slot_bits)`
    n_buckets: usize,
    /// Orignal indices
    original_ids: Vec<usize>,
}

////////////////////
// VectorDistance //
////////////////////

/// VectorDistance trait
impl<T> VectorDistance<T> for LSHIndex<T>
where
    T: AnnSearchFloat,
{
    fn vectors_flat(&self) -> &[T] {
        &self.vectors_flat
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn norms(&self) -> &[T] {
        &self.norms
    }
}

/////////////////////////
// DimensionValidation //
/////////////////////////

impl<T> DimensionValidation for LSHIndex<T> {
    fn dim(&self) -> usize {
        self.dim
    }
}

/////////////////
// Index build //
/////////////////

impl<T> LSHIndex<T>
where
    T: AnnSearchFloat,
{
    /// Construct a new LSH index
    ///
    /// Fits the quantile boundaries on a subsample, projects the whole dataset
    /// through a tiled GEMM, and counting-sorts the resulting codes into a CSR
    /// bucket layout per table.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (rows = samples, columns = dimensions)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `num_tables` - Number of hash tables (with multi-probe, 4-8 is
    ///   typically sufficient)
    /// * `bits_per_hash` - Bits per hash code, at most [`MAX_BITS_PER_HASH`]
    ///   (more = fewer collisions, smaller buckets)
    /// * `slot_bits` - Bits per quantised projection; `None` picks 1 for cosine
    ///   and 2 for squared Euclidean
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Constructed index ready for querying
    pub fn new(
        data: MatRef<T>,
        metric: Dist,
        num_tables: usize,
        bits_per_hash: usize,
        slot_bits: Option<usize>,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(metric));
        }
        if bits_per_hash == 0 || bits_per_hash > MAX_BITS_PER_HASH {
            return Err(AnnSearchErrors::InvalidLshBits {
                bits_per_hash,
                max: MAX_BITS_PER_HASH,
            });
        }
        if data.nrows() > u32::MAX as usize {
            return Err(AnnSearchErrors::LshTooManySamples {
                n: data.nrows(),
                max: u32::MAX as usize,
            });
        }

        let (vectors_flat, n, dim) = matrix_to_flat(data);

        let slot_bits = resolve_slot_bits(slot_bits, metric, bits_per_hash);
        let n_proj = bits_per_hash / slot_bits;
        let n_slots = 1usize << slot_bits;
        let n_buckets = 1usize << (n_proj * slot_bits);
        let n_cols = num_tables * n_proj;

        // Cosine needs the norms to hash with as well as to score with.
        let norms: Vec<T> = if metric == Dist::Cosine {
            (0..n)
                .map(|i| T::calculate_l2_norm(&vectors_flat[i * dim..(i + 1) * dim]))
                .collect()
        } else {
            Vec::new()
        };

        // generate random projection vectors from N(0,1)
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut random_vecs: Vec<T> = (0..n_cols * dim)
            .map(|_| {
                let val: f64 = rng.sample(StandardNormal);
                T::from_f64(val).unwrap()
            })
            .collect();

        orthogonalise_table_projections(&mut random_vecs, num_tables, n_proj, dim);

        let boundaries = fit_boundaries(
            &vectors_flat,
            &norms,
            n,
            dim,
            &random_vecs,
            n_cols,
            n_slots,
            metric,
            seed,
        );

        let codes = assign_codes(
            &vectors_flat,
            &norms,
            n,
            dim,
            &random_vecs,
            &boundaries,
            num_tables,
            n_proj,
            slot_bits,
            metric,
        );

        let (bucket_offsets, bucket_ids) = build_bucket_csr(&codes, n, num_tables, n_buckets);

        Ok(Self {
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            random_vecs,
            boundaries,
            bucket_offsets,
            bucket_ids,
            num_tables,
            bits_per_hash,
            n_proj,
            slot_bits,
            n_buckets,
            original_ids: (0..n).collect(),
        })
    }

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Index size `in n bytes`
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);

        total += self.vectors_flat.capacity() * std::mem::size_of::<T>();
        total += self.norms.capacity() * std::mem::size_of::<T>();
        total += self.random_vecs.capacity() * std::mem::size_of::<T>();
        total += self.boundaries.capacity() * std::mem::size_of::<T>();
        total += self.bucket_offsets.capacity() * std::mem::size_of::<u32>();
        total += self.bucket_ids.capacity() * std::mem::size_of::<u32>();
        total += self.original_ids.capacity() * std::mem::size_of::<usize>();

        total
    }

    /// Returns the number of bits used for each hash.
    pub fn num_bits(&self) -> usize {
        self.bits_per_hash
    }

    /// Returns the number of projections per hash table.
    ///
    /// This is the natural unit for `n_probes`: there are `2 * n_projections()`
    /// single-slot perturbations available per table.
    pub fn num_projections(&self) -> usize {
        self.n_proj
    }

    /// Returns the number of bits each quantised projection contributes.
    pub fn slot_bits(&self) -> usize {
        self.slot_bits
    }
}

///////////
// Query //
///////////

impl<T> LSHIndex<T>
where
    T: AnnSearchFloat,
{
    /// Project a single vector onto every table's projections
    ///
    /// ### Params
    ///
    /// * `vec` - Vector to project
    /// * `scale` - Multiplier from [`hash_scale`]
    ///
    /// ### Returns
    ///
    /// Projections laid out `[table_idx * n_proj + proj_idx]`
    fn project_one(&self, vec: &[T], scale: T) -> Vec<T> {
        (0..self.num_tables * self.n_proj)
            .map(|c| T::dot_simd(&self.random_vecs[c * self.dim..(c + 1) * self.dim], vec) * scale)
            .collect()
    }

    /// Scan one bucket, deduplicating and scoring as it goes
    ///
    /// Dedup, distance and top-k are fused into a single pass so no
    /// duplicate-heavy candidate list is ever materialised. The budget is
    /// checked per candidate rather than per bucket, so `max_cand` bounds the
    /// work even when one bucket is enormous.
    ///
    /// ### Params
    ///
    /// * `table_idx` - Table the code belongs to
    /// * `code` - Bucket to scan
    /// * `query_vec` - Query vector
    /// * `query_norm` - Query L2 norm, only read for cosine
    /// * `k` - Number of neighbours to keep
    /// * `budget` - Maximum number of unique candidates to examine
    /// * `visited` - Per-query dedup bitset
    /// * `touched` - Ids whose bit was set, for the O(candidates) reset
    /// * `buffer` - Running top-k
    ///
    /// ### Returns
    ///
    /// `false` once the budget is exhausted, `true` otherwise
    #[allow(clippy::too_many_arguments)]
    fn scan_bucket(
        &self,
        table_idx: usize,
        code: u32,
        query_vec: &[T],
        query_norm: T,
        k: usize,
        budget: usize,
        visited: &mut FixedBitSet,
        touched: &mut Vec<u32>,
        buffer: &mut SortedBuffer<(OrderedFloat<T>, usize)>,
    ) -> bool {
        let offset_base = table_idx * (self.n_buckets + 1) + code as usize;
        let start = self.bucket_offsets[offset_base] as usize;
        let end = self.bucket_offsets[offset_base + 1] as usize;
        let id_base = table_idx * self.n;

        for pos in start..end {
            if touched.len() >= budget {
                return false;
            }

            let idx = self.bucket_ids[id_base + pos] as usize;
            if visited.contains(idx) {
                continue;
            }
            visited.insert(idx);
            touched.push(idx as u32);

            let dist = match self.metric {
                Dist::SquaredEuclidean => self.euclidean_distance_to_query(idx, query_vec),
                Dist::Cosine => self.cosine_distance_to_query(idx, query_vec, query_norm),
                Dist::Manhattan => unreachable!(),
            };
            buffer.insert((OrderedFloat(dist), idx), k);
        }

        true
    }

    /// Shared search over pre-computed projections
    ///
    /// Both the cross-set and the self-query path land here; they differ only
    /// in where the projections come from.
    ///
    /// ### Params
    ///
    /// * `proj` - Projections of the query, layout `[table_idx * n_proj + j]`
    /// * `query_vec` - Query vector, used for the distance computations
    /// * `query_norm` - Query L2 norm, only read for cosine
    /// * `k` - Number of neighbours to return
    /// * `max_cand` - Optional cap on unique candidates examined
    /// * `n_probes` - Additional buckets to visit per table
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances, fallback_triggered)` sorted by distance
    fn search(
        &self,
        proj: &[T],
        query_vec: &[T],
        query_norm: T,
        k: usize,
        max_cand: Option<usize>,
        n_probes: usize,
    ) -> (Vec<usize>, Vec<T>, bool) {
        let n_slots = 1usize << self.slot_bits;
        let n_bounds = n_slots - 1;
        let budget = max_cand.unwrap_or(self.n);

        LSH_VISITED.with(|vis_cell| {
            LSH_TOUCHED.with(|touch_cell| {
                let mut visited = vis_cell.borrow_mut();
                let mut touched = touch_cell.borrow_mut();

                if visited.len() < self.n {
                    visited.grow(self.n);
                }
                touched.clear();

                let mut buffer = SortedBuffer::with_capacity(k);
                let mut perturb: Vec<(OrderedFloat<T>, usize, i64)> =
                    Vec::with_capacity(2 * self.n_proj);
                let mut probes: Vec<u32> = Vec::with_capacity(n_probes);

                'tables: for table_idx in 0..self.num_tables {
                    let mut code: u32 = 0;
                    perturb.clear();

                    for j in 0..self.n_proj {
                        let col = table_idx * self.n_proj + j;
                        let value = proj[col];
                        let bounds = &self.boundaries[col * n_bounds..(col + 1) * n_bounds];
                        let slot = bounds.partition_point(|&b| b <= value);
                        code |= (slot as u32) << (j * self.slot_bits);

                        let step = 1i64 << (j * self.slot_bits);
                        if slot + 1 < n_slots {
                            perturb.push((OrderedFloat(bounds[slot] - value), j, step));
                        }
                        if slot > 0 {
                            perturb.push((OrderedFloat(value - bounds[slot - 1]), j, -step));
                        }
                    }

                    if !self.scan_bucket(
                        table_idx,
                        code,
                        query_vec,
                        query_norm,
                        k,
                        budget,
                        &mut visited,
                        &mut touched,
                        &mut buffer,
                    ) {
                        break 'tables;
                    }

                    if n_probes > 0 {
                        perturb.sort_unstable();
                        probes.clear();
                        build_probes(code, &perturb, n_probes, &mut probes);

                        for &probe_code in probes.iter() {
                            if !self.scan_bucket(
                                table_idx,
                                probe_code,
                                query_vec,
                                query_norm,
                                k,
                                budget,
                                &mut visited,
                                &mut touched,
                                &mut buffer,
                            ) {
                                break 'tables;
                            }
                        }
                    }
                }

                let fallback_triggered = touched.is_empty();
                if fallback_triggered {
                    let mut rng = rng();
                    let sample_size = FALLBACK_SAMPLE.min(self.n);
                    for idx in (0..self.n).choose_multiple(&mut rng, sample_size) {
                        let dist = match self.metric {
                            Dist::SquaredEuclidean => {
                                self.euclidean_distance_to_query(idx, query_vec)
                            }
                            Dist::Cosine => {
                                self.cosine_distance_to_query(idx, query_vec, query_norm)
                            }
                            Dist::Manhattan => unreachable!(),
                        };
                        buffer.insert((OrderedFloat(dist), idx), k);
                    }
                }

                // Only the bits we set, so the reset is O(candidates)
                for &idx in touched.iter() {
                    visited.remove(idx as usize);
                }

                let (indices, dists) = buffer
                    .data()
                    .iter()
                    .map(|&(OrderedFloat(d), idx)| (self.original_ids[idx], d))
                    .unzip();

                (indices, dists, fallback_triggered)
            })
        })
    }

    /// Query the index for approximate nearest neighbours
    ///
    /// Hashes the query vector and retrieves candidates from the matching
    /// bucket in every table. With `n_probes > 0`, also visits neighbouring
    /// buckets ordered by how close the query sits to each slot boundary.
    ///
    /// If no candidates are found across all tables and probes, falls back to
    /// random sampling.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `max_cand` - Optional limit on unique candidates examined
    /// * `n_probes` - Number of additional buckets to probe per table (0 =
    ///   exact bucket only). Good default: `num_projections()`.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances, fallback_triggered)` sorted by distance
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
        max_cand: Option<usize>,
        n_probes: usize,
    ) -> Result<(Vec<usize>, Vec<T>, bool), AnnSearchErrors> {
        self.check_dim(query_vec.len())?;

        let query_norm = if self.metric == Dist::Cosine {
            T::calculate_l2_norm(query_vec)
        } else {
            T::one()
        };
        let proj = self.project_one(query_vec, hash_scale(self.metric, query_norm));

        Ok(self.search(&proj, query_vec, query_norm, k, max_cand, n_probes))
    }

    /// Query using a matrix row reference
    ///
    /// Optimised path for contiguous memory (stride == 1), otherwise copies to
    /// a temporary vector.
    ///
    /// ### Params
    ///
    /// * `query_row` - Row reference to query vector
    /// * `k` - Number of neighbours to return
    /// * `max_cand` - Optional candidate limit
    /// * `n_probes` - Number of multi-probe buckets per table
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances, fallback_triggered)`
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
        max_cand: Option<usize>,
        n_probes: usize,
    ) -> Result<(Vec<usize>, Vec<T>, bool), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k, max_cand, n_probes);
        }

        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k, max_cand, n_probes)
    }

    /// Generate kNN graph from vectors stored in the index
    ///
    /// ### Params
    ///
    /// * `k` - Number of neighbours per vector
    /// * `max_cand` - Optional candidate limit per query
    /// * `n_probes` - Number of additional buckets to probe per table
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Controls verbosity
    ///
    /// ### Returns
    ///
    /// Tuple of `(knn_indices, optional distances)` where each row corresponds
    /// to a vector in the index
    pub fn generate_knn(
        &self,
        k: usize,
        max_cand: Option<usize>,
        n_probes: usize,
        return_dist: bool,
        verbose: bool,
    ) -> (Vec<Vec<usize>>, Option<Vec<Vec<T>>>) {
        use std::sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        };

        let counter = Arc::new(AtomicUsize::new(0));

        let results: Vec<(Vec<usize>, Vec<T>, bool)> = (0..self.n)
            .into_par_iter()
            .map(|vec_idx| {
                if verbose {
                    let count = counter.fetch_add(1, Ordering::Relaxed) + 1;
                    if count.is_multiple_of(100_000) {
                        println!(
                            "  Processed {} / {} samples.",
                            count.separate_with_underscores(),
                            self.n.separate_with_underscores()
                        );
                    }
                }

                self.self_query_at(vec_idx, k, max_cand, n_probes)
            })
            .collect();

        if verbose {
            let missed = results.iter().filter(|(_, _, fallback)| *fallback).count();
            if (missed as f32) / (self.n as f32) >= 0.01 {
                println!("More than 1% of samples were not represented in the buckets.");
                println!("Please verify underlying data");
            }
        }

        if return_dist {
            let mut indices = Vec::with_capacity(results.len());
            let mut distances = Vec::with_capacity(results.len());

            for (idx, dist, _) in results {
                indices.push(idx);
                distances.push(dist);
            }
            (indices, Some(distances))
        } else {
            let indices: Vec<Vec<usize>> = results.into_iter().map(|(idx, _, _)| idx).collect();
            (indices, None)
        }
    }

    /// Self-query for a vector already stored in the index
    ///
    /// Re-projects the stored vector rather than reading a cached code. That
    /// costs `num_tables * n_proj` dot products against the ~`budget * dim`
    /// flops the candidate distances cost, and it buys the same
    /// boundary-ranked probe order the cross-set path gets. Caching codes
    /// instead would force probes to be generated blind.
    ///
    /// ### Params
    ///
    /// * `vec_idx` - Index of the vector to query.
    /// * `k` - Number of neighbours to return.
    /// * `max_cand` - Optional maximum number of candidates.
    /// * `n_probes` - Number of additional buckets to probe per table
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances, random_sampling triggered)`
    fn self_query_at(
        &self,
        vec_idx: usize,
        k: usize,
        max_cand: Option<usize>,
        n_probes: usize,
    ) -> (Vec<usize>, Vec<T>, bool) {
        let query_vec = &self.vectors_flat[vec_idx * self.dim..(vec_idx + 1) * self.dim];
        let query_norm = if self.metric == Dist::Cosine {
            self.norms[vec_idx]
        } else {
            T::one()
        };
        let proj = self.project_one(query_vec, hash_scale(self.metric, query_norm));

        self.search(&proj, query_vec, query_norm, k, max_cand, n_probes)
    }
}

//////////////////////
// Validation trait //
//////////////////////

impl<T> KnnValidation<T> for LSHIndex<T>
where
    T: AnnSearchFloat,
{
    fn query_for_validation(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let (indices, dist, _) = self.query(query_vec, k, None, self.n_proj)?;
        Ok((indices, dist))
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

    fn original_ids(&self) -> &[usize] {
        &self.original_ids
    }
}

/////////////
// IndexIo //
/////////////

#[cfg(feature = "serialise")]
impl<T> IndexIo for LSHIndex<T>
where
    T: AnnSearchFloat,
{
    type Elem = T;

    const KIND: &'static str = "lsh";
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn simple_test_data() -> Mat<f32> {
        Mat::from_fn(5, 3, |i, j| match i {
            0 => [1.0, 0.0, 0.0][j],
            1 => [0.0, 1.0, 0.0][j],
            2 => [0.0, 0.0, 1.0][j],
            3 => [1.0, 1.0, 0.0][j],
            4 => [0.5, 0.5, 0.7][j],
            _ => 0.0,
        })
    }

    /// Reproducible Gaussian-ish spread, centred well away from the origin so
    /// the median thresholds have something to correct.
    fn offset_data(n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| {
            let a = ((i * 2654435761 + j * 40503) % 10007) as f32 / 10007.0;
            50.0 + a * 4.0 - 2.0
        })
    }

    #[test]
    fn test_index_creation_euclidean() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.num_tables, 4);
        assert_eq!(index.bits_per_hash, 8);
        // Euclidean auto-resolves slot_bits to 2
        assert_eq!(index.slot_bits, 2);
        assert_eq!(index.n_proj, 4);
        assert_eq!(index.n_buckets, 256);
        assert_eq!(index.vectors_flat.len(), 15);
        assert_eq!(index.bucket_ids.len(), 4 * 5);
        assert_eq!(index.bucket_offsets.len(), 4 * 257);
    }

    #[test]
    fn test_index_creation_cosine() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Cosine, 4, 8, None, 42).unwrap();

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.norms.len(), 5);
        // Cosine auto-resolves slot_bits to 1, so the code is a sign pattern
        assert_eq!(index.slot_bits, 1);
        assert_eq!(index.n_proj, 8);
        assert_eq!(index.n_buckets, 256);
    }

    #[test]
    fn test_bits_per_hash_above_limit_errors() {
        let mat = simple_test_data();

        let too_many = LSHIndex::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            4,
            MAX_BITS_PER_HASH + 1,
            None,
            42,
        );
        assert!(matches!(
            too_many,
            Err(AnnSearchErrors::InvalidLshBits { .. })
        ));

        let zero = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 0, None, 42);
        assert!(matches!(zero, Err(AnnSearchErrors::InvalidLshBits { .. })));

        // The limit itself must still build
        assert!(LSHIndex::new(
            mat.as_ref(),
            Dist::SquaredEuclidean,
            1,
            MAX_BITS_PER_HASH,
            None,
            42
        )
        .is_ok());
    }

    #[test]
    fn test_manhattan_not_supported() {
        let mat = simple_test_data();
        let result = LSHIndex::new(mat.as_ref(), Dist::Manhattan, 4, 8, None, 42);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::DistanceNotSupported(_))
        ));
    }

    #[test]
    fn test_boundaries_split_evenly() {
        // With slot_bits = 1 the single boundary is the median, so every
        // projection must split the data close to 50/50 even though the data
        // sits ~50 units from the origin. This is the property plain SimHash
        // does not have.
        let mat = offset_data(2000, 16);
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 2, 8, Some(1), 7).unwrap();

        assert_eq!(index.slot_bits, 1);

        // slot_bits == 1 means exactly one boundary per projection, so
        // `boundaries[col]` is that projection's median.
        let n_cols = index.num_tables * index.n_proj;
        let mut above = vec![0usize; n_cols];

        for i in 0..index.n {
            let vec = &index.vectors_flat[i * index.dim..(i + 1) * index.dim];
            let proj = index.project_one(vec, 1.0);
            for col in 0..n_cols {
                above[col] += usize::from(proj[col] > index.boundaries[col]);
            }
        }

        for (col, &count) in above.iter().enumerate() {
            let share = count as f32 / index.n as f32;
            assert!(
                (share - 0.5).abs() < 0.05,
                "projection {col} put {share} of the data above its median, expected ~0.5"
            );
        }
    }

    #[test]
    fn test_csr_layout_covers_every_vector() {
        let mat = offset_data(500, 8);
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 3, 8, None, 11).unwrap();

        for table_idx in 0..index.num_tables {
            let offs = &index.bucket_offsets
                [table_idx * (index.n_buckets + 1)..(table_idx + 1) * (index.n_buckets + 1)];

            assert_eq!(offs[0], 0);
            assert_eq!(offs[index.n_buckets] as usize, index.n);
            for b in 1..=index.n_buckets {
                assert!(offs[b] >= offs[b - 1], "offsets must be monotone");
            }

            let mut ids = index.bucket_ids[table_idx * index.n..(table_idx + 1) * index.n].to_vec();
            ids.sort_unstable();
            assert_eq!(ids, (0..index.n as u32).collect::<Vec<_>>());
        }
    }

    #[test]
    fn test_probe_order_follows_boundary_gap() {
        // Projection 1 sits closest to its boundary, so it must be perturbed
        // first; projection 0 is furthest and comes last.
        let perturb = vec![
            (OrderedFloat(0.9f32), 0usize, 1i64),
            (OrderedFloat(0.1f32), 1usize, 2i64),
            (OrderedFloat(0.4f32), 2usize, 4i64),
        ];
        let mut sorted = perturb.clone();
        sorted.sort_unstable();

        let mut probes = Vec::new();
        build_probes(0u32, &sorted, 8, &mut probes);

        assert_eq!(probes[0], 2, "smallest gap first");
        assert_eq!(probes[1], 4);
        assert_eq!(probes[2], 1);
        // then the pairs, cheapest sum first
        assert_eq!(probes[3], 2 + 4);
    }

    #[test]
    fn test_probes_never_pair_the_same_projection() {
        // Two shifts of the same projection contradict each other and must not
        // be combined.
        let mut perturb = vec![
            (OrderedFloat(0.1f32), 0usize, 1i64),
            (OrderedFloat(0.2f32), 0usize, -1i64),
        ];
        perturb.sort_unstable();

        let mut probes = Vec::new();
        build_probes(4u32, &perturb, 16, &mut probes);

        assert_eq!(probes, vec![5, 3], "only the two single shifts");
    }

    #[test]
    fn test_probe_respects_max() {
        let perturb: Vec<(OrderedFloat<f32>, usize, i64)> = (0..8)
            .map(|j| (OrderedFloat(j as f32), j, 1i64 << j))
            .collect();

        let mut probes = Vec::new();
        build_probes(0u32, &perturb, 5, &mut probes);
        assert_eq!(probes.len(), 5);
    }

    #[test]
    fn test_slot_bits_gt_one_separates_by_magnitude() {
        // Same direction, very different norms. A sign-only hash cannot tell
        // them apart; a quantised one must, because squared Euclidean does.
        let n = 400;
        let dim = 8;
        let mat = Mat::from_fn(n, dim, |i, j| {
            let scale = 1.0 + (i as f32) * 0.5;
            scale * (1.0 + j as f32 * 0.1)
        });

        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 1, 8, Some(4), 3).unwrap();

        let short = &index.vectors_flat[0..dim];
        let long = &index.vectors_flat[(n - 1) * dim..n * dim];

        let proj_short = index.project_one(short, 1.0);
        let proj_long = index.project_one(long, 1.0);

        let mut code_short = [0u32; 1];
        let mut code_long = [0u32; 1];
        encode_row(
            &proj_short,
            &index.boundaries,
            1,
            index.n_proj,
            index.slot_bits,
            &mut code_short,
        );
        encode_row(
            &proj_long,
            &index.boundaries,
            1,
            index.n_proj,
            index.slot_bits,
            &mut code_long,
        );

        assert_ne!(
            code_short[0], code_long[0],
            "magnitude must change the bucket when slot_bits > 1"
        );
    }

    #[test]
    fn test_basic_query_no_probes() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 3, None, 0).unwrap();

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
        assert!(indices.contains(&0));

        for i in 1..distances.len() {
            assert!(distances[i - 1] <= distances[i]);
        }
    }

    #[test]
    fn test_basic_query_with_probes() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 3, None, 8).unwrap();

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
        assert!(indices.contains(&0));

        for i in 1..distances.len() {
            assert!(distances[i - 1] <= distances[i]);
        }
    }

    #[test]
    fn test_multi_probe_finds_more_candidates() {
        let n = 100;
        let dim = 50;
        let mat = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 100) as f32 / 100.0);

        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 2, 12, None, 42).unwrap();

        let query = vec![0.5; dim];
        let (idx_no_probe, _, _) = index.query(&query, 10, None, 0).unwrap();
        let (idx_probed, _, _) = index.query(&query, 10, None, 12).unwrap();

        assert!(idx_probed.len() >= idx_no_probe.len());
    }

    #[test]
    fn test_query_cosine() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Cosine, 4, 8, None, 42).unwrap();

        let query = vec![2.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 2, None, 0).unwrap();

        assert!(!indices.is_empty());
        assert!(indices.len() <= 2);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_query_row() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let query_mat = Mat::from_fn(1, 3, |_, j| [1.0, 0.0, 0.0][j]);
        let (indices, distances, _) = index.query_row(query_mat.row(0), 3, None, 0).unwrap();

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_max_cand_bounds_candidates_examined() {
        // Every vector is identical, so a single bucket holds all of them. The
        // old between-buckets budget check let a bucket like this blow straight
        // past max_cand.
        let n = 5000;
        let dim = 8;
        let mat = Mat::from_fn(n, dim, |_, j| j as f32);

        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let query = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let budget = 64;
        let (indices, _, _) = index.query(&query, 10, Some(budget), 8).unwrap();

        assert!(indices.len() <= 10);

        LSH_TOUCHED.with(|cell| {
            assert!(
                cell.borrow().len() <= budget,
                "examined {} candidates against a budget of {budget}",
                cell.borrow().len()
            );
        });
    }

    #[test]
    fn test_k_larger_than_n() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 100, None, 0).unwrap();

        assert!(indices.len() <= 5);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();
        let query = vec![1.0, 0.0];
        let result = index.query(&query, 3, None, 0);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_query_row_dimension_mismatch() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();
        let query_mat = Mat::from_fn(1, 2, |_, j| [1.0, 0.0][j]);
        let result = index.query_row(query_mat.row(0), 3, None, 0);

        assert!(matches!(
            result,
            Err(AnnSearchErrors::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn test_fallback_mechanism() {
        let mat = Mat::from_fn(10, 100, |i, j| if j == i * 10 { 1.0 } else { 0.0 });

        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 2, 16, None, 42).unwrap();

        let query = vec![1.0; 100];
        let (indices, distances, _) = index.query(&query, 3, None, 0).unwrap();

        assert!(!indices.is_empty());
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_deterministic_with_seed() {
        let mat = simple_test_data();

        let index1 = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();
        let index2 = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices1, _, _) = index1.query(&query, 3, None, 4).unwrap();
        let (indices2, _, _) = index2.query(&query, 3, None, 4).unwrap();

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_f64_query() {
        let mat = Mat::from_fn(3, 3, |i, j| if i == j { 1.0f64 } else { 0.0f64 });
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 2, None, 0).unwrap();

        assert!(!indices.is_empty());
        assert!(indices.len() <= 2);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_distances_sorted() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (_, distances, _) = index.query(&query, 5, None, 8).unwrap();

        for i in 1..distances.len() {
            assert!(distances[i - 1] <= distances[i]);
        }
    }

    #[test]
    fn test_query_returns_k_or_fewer() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 8, 6, None, 42).unwrap();

        let query = vec![1.0, 0.0, 0.0];

        for k in 1..=5 {
            let (indices, distances, _) = index.query(&query, k, None, 0).unwrap();
            assert!(indices.len() <= k);
            assert_eq!(indices.len(), distances.len());
        }
    }

    #[test]
    fn test_no_duplicate_results() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 8, 6, None, 42).unwrap();

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _, _) = index.query(&query, 5, None, 6).unwrap();

        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();

        assert_eq!(indices.len(), sorted.len(), "Results contain duplicates");
    }

    #[test]
    fn test_no_duplicate_results_with_probes() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 8, 6, None, 42).unwrap();

        let query = vec![0.5, 0.5, 0.5];
        let (indices, _, _) = index.query(&query, 5, None, 10).unwrap();

        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();

        assert_eq!(
            indices.len(),
            sorted.len(),
            "Results contain duplicates with multi-probe"
        );
    }

    #[test]
    fn test_self_query_matches_query_for_stored_vector() {
        // Self-query re-projects rather than reading a cached code, so it must
        // land on exactly the same buckets and probes as the cross-set path.
        let mat = offset_data(300, 12);
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 3, 10, None, 5).unwrap();

        for vec_idx in [0usize, 7, 42, 299] {
            let stored = &index.vectors_flat[vec_idx * index.dim..(vec_idx + 1) * index.dim];
            let (via_query, _, _) = index.query(stored, 10, None, index.n_proj).unwrap();
            let (via_self, _, _) = index.self_query_at(vec_idx, 10, None, index.n_proj);

            assert_eq!(via_query, via_self, "paths diverged for vector {vec_idx}");
        }
    }

    #[test]
    fn test_self_query_cosine_matches_query() {
        let mat = offset_data(300, 12);
        let index = LSHIndex::new(mat.as_ref(), Dist::Cosine, 3, 10, None, 5).unwrap();

        for vec_idx in [0usize, 13, 250] {
            let stored = &index.vectors_flat[vec_idx * index.dim..(vec_idx + 1) * index.dim];
            let (via_query, _, _) = index.query(stored, 10, None, index.n_proj).unwrap();
            let (via_self, _, _) = index.self_query_at(vec_idx, 10, None, index.n_proj);

            assert_eq!(via_query, via_self, "paths diverged for vector {vec_idx}");
        }
    }

    #[test]
    fn test_self_query_finds_self() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let (indices, distances, _) = index.self_query_at(0, 3, None, 0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
        assert!(indices.contains(&0));
    }

    #[test]
    fn test_generate_knn() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let (knn_indices, knn_dists) = index.generate_knn(2, None, 4, true, false);

        assert_eq!(knn_indices.len(), 5);
        assert!(knn_dists.is_some());
        let dists = knn_dists.unwrap();
        assert_eq!(dists.len(), 5);

        for i in 0..5 {
            assert!(!knn_indices[i].is_empty());
            assert!(knn_indices[i].len() <= 2);
            assert_eq!(knn_indices[i].len(), dists[i].len());
        }
    }

    #[test]
    fn test_generate_knn_no_distances() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let (knn_indices, knn_dists) = index.generate_knn(2, None, 0, false, false);

        assert_eq!(knn_indices.len(), 5);
        assert!(knn_dists.is_none());
    }

    #[test]
    fn test_larger_dataset() {
        let n = 1000;
        let dim = 50;
        let mat = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 100) as f32 / 100.0);

        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 6, 10, None, 42).unwrap();

        let query = vec![0.5; dim];
        let (indices, distances, _) = index.query(&query, 10, None, 10).unwrap();

        assert!(!indices.is_empty());
        assert!(indices.len() <= 10);
        assert_eq!(indices.len(), distances.len());

        for &idx in &indices {
            assert!(idx < n);
        }
    }

    #[test]
    fn test_recall_on_offset_data() {
        // The regression guard for the whole rewrite: clustered data sitting
        // far from the origin, which is where plain SimHash collapses into one
        // bucket.
        let n = 2000;
        let dim = 16;
        let mat = Mat::from_fn(n, dim, |i, j| {
            let cluster = (i % 8) as f32;
            let jitter = ((i * 2654435761 + j * 40503) % 1009) as f32 / 1009.0;
            40.0 + cluster * 3.0 + jitter
        });

        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 12, None, 42).unwrap();
        let recall = index.validate_index(10, 42, Some(200)).unwrap();

        assert!(recall > 0.8, "recall on offset data was {recall}");
    }

    #[test]
    fn test_memory_usage_accounts_for_buckets() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 4, 8, None, 42).unwrap();

        let mem = index.memory_usage_bytes();
        assert!(mem >= index.bucket_ids.len() * std::mem::size_of::<u32>());
    }

    #[test]
    fn test_slot_bits_clamped_to_bits_per_hash() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::SquaredEuclidean, 2, 4, Some(9), 42).unwrap();

        assert_eq!(index.slot_bits, 4);
        assert_eq!(index.n_proj, 1);
    }
}
