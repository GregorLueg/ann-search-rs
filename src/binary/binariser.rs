//! Contains the binarisers for pure binary indices, i.e., RandomProjections,
//! ITQ-PCA hashing and sign-based binarisation.

use faer::linalg::matmul::matmul;
use faer::{Accum, Col, ColRef, Mat, MatRef, Par, Side};
use num_traits::{Float, FromPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use rayon::prelude::*;

use crate::prelude::*;

///////////////
// Binariser //
///////////////

const MAX_SAMPLES_PCA: usize = 100_000;

/// Rows used to fit the ITQ rotation.
///
/// The rotation has `k^2` free parameters and is fitted by plain Procrustes,
/// so it settles on far fewer rows than the PCA itself wants. Each iteration
/// costs two `n * k * k` products, so this bound is what keeps the ITQ stage
/// from dominating index construction.
const MAX_SAMPLES_ITQ: usize = 10_000;

/// ITQ alternating-minimisation iterations.
///
/// Gong and Lazebnik run 50. The objective is essentially flat well before
/// that, so 30 is where further iterations stop paying for their two GEMMs.
const ITQ_ITERATIONS: usize = 30;

/// Share of total variance the retained components must explain.
///
/// Judging each component against `sigma_0` alone cannot separate genuine rank
/// one from a long informative tail sitting under one dominant axis, which is
/// the normal shape for an embedding carrying a depth or library-size
/// direction. A cumulative rule reads the whole spectrum instead.
const PCA_VARIANCE_EXPLAINED: f64 = 0.9;

/// Ceiling on retained components, as a share of the bit budget.
///
/// On a flat, noisy spectrum the cumulative rule wants nearly every component,
/// because no small prefix explains 90% of anything. That is the case this cap
/// exists for: past the genuinely structured directions a PCA bit is worse than
/// a random one, since a random hyperplane preserves angular distance by
/// construction and a low-variance loading does not. Set to 10%.
const PCA_MAX_COMPONENT_SHARE: f64 = 0.1;

/// Target footprint of the projection GEMM's output tile.
///
/// The tile holds `rows * n_bits` scores and the sign-packing pass that follows
/// walks it one bit-column at a time, so it wants to stay resident in L2. The
/// row count is derived from this rather than fixed, because `n_bits` reaches
/// 512 and a flat 4096-row tile would be 8 MiB of `f32`.
const ENCODE_TILE_BYTES: usize = 2 << 20;

/// Floor on the projection tile's row count.
///
/// Below this the GEMM has too little work per rayon task to amortise the
/// dispatch, whatever `n_bits` says.
const ENCODE_TILE_MIN_ROWS: usize = 256;

/// Ceiling on the projection tile's row count.
///
/// Matches the tile `cpu::lsh` uses for the same shape of work.
const ENCODE_TILE_MAX_ROWS: usize = 4096;

/// Rows per tile of the parallel reductions over the training sample.
///
/// Shared by the mean and the Gram accumulation. Only has to be large enough
/// that the per-tile work dominates the rayon dispatch; fixing it also fixes
/// the reduction order, which is what keeps a seed reproducible.
const REDUCTION_ROW_TILE: usize = 4096;

/// Initialisation of the binariser
#[derive(Default, Eq, PartialEq)]
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub enum BinarisationInit {
    /// Random projection with orthogonalisation
    #[default]
    RandomProjections,
    /// PCA-based hashing
    PcaHashing,
    /// Sign-based binarisation
    SignBased,
}

/// Helper function to parse the Binarisation initialisation
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Returns
///
/// `Option<BinarisationInit>`
pub fn parse_binarisation_init(s: &str) -> Option<BinarisationInit> {
    match s.to_lowercase().as_str() {
        "pca" | "pca_hashing" => Some(BinarisationInit::PcaHashing),
        "random" | "random_projections" => Some(BinarisationInit::RandomProjections),
        "sign" | "signed" | "sign_based" => Some(BinarisationInit::SignBased),
        _ => None,
    }
}

/////////////
// Helpers //
/////////////

/// Enum representing different binarisation methods
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub enum BinarisationMethod<T> {
    /// SimHash with random orthogonalised projections
    SimHash {
        /// The random, orthogonal projection
        projections: Vec<T>,
        /// The mean value across each feature. Random hyperplanes pass through
        /// the origin, so without centring every bit is biased on data whose
        /// mean sits far from it and the codes collapse.
        mean: Vec<T>,
    },
    /// PCA hashing with learned projections and mean centring
    PcaHashing {
        /// Projections based on PCA loadings
        projections: Vec<T>,
        /// The mean value across each feature
        mean: Vec<T>,
    },
    /// Sign-based binarisation (no projections needed)
    SignBased,
}

/// Generate random projections and orthogonalise them
///
/// Creates orthonormal random hyperplanes for better hash quality.
/// Orthogonalisation via Gram-Schmidt ensures projections are independent.
///
/// ### Params
///
/// * `dim` - Input vector dimensionality
/// * `n_bits` - Number of bits in output
/// * `seed` - Random seed for reproducible projection generation
///
/// ### Returns
///
/// Flattened projection matrix (n_bits × dim)
fn prepare_simhash_projections<T>(dim: usize, n_bits: usize, seed: usize) -> Vec<T>
where
    T: Float + FromPrimitive + Copy,
{
    let n_orthogonal = n_bits.min(dim);

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut random_projections: Vec<T> = (0..n_bits * dim)
        .map(|_| {
            let val: f64 = rng.sample(StandardNormal);
            T::from_f64(val).unwrap()
        })
        .collect();

    // Orthogonalise the projections via Gram-Schmidt
    for i in 0..n_orthogonal {
        let i_base = i * dim;

        // Subtract projection onto all previous vectors
        for j in 0..i {
            let j_base = j * dim;
            let mut dot = T::zero();
            for d in 0..dim {
                dot = dot + random_projections[i_base + d] * random_projections[j_base + d];
            }
            for d in 0..dim {
                random_projections[i_base + d] =
                    random_projections[i_base + d] - dot * random_projections[j_base + d];
            }
        }

        // Normalise to unit length
        let mut norm_sq = T::zero();
        for d in 0..dim {
            norm_sq = norm_sq + random_projections[i_base + d] * random_projections[i_base + d];
        }
        let norm = norm_sq.sqrt();
        if norm > T::epsilon() {
            for d in 0..dim {
                random_projections[i_base + d] = random_projections[i_base + d] / norm;
            }
        }
    }

    for i in n_orthogonal..n_bits {
        let i_base = i * dim;
        let mut norm_sq = T::zero();
        for d in 0..dim {
            norm_sq = norm_sq + random_projections[i_base + d] * random_projections[i_base + d];
        }
        let norm = norm_sq.sqrt();
        if norm > T::epsilon() {
            for d in 0..dim {
                random_projections[i_base + d] = random_projections[i_base + d] / norm;
            }
        }
    }

    random_projections
}

/// Row indices to train on, capped at [`MAX_SAMPLES_PCA`]
///
/// Shared by the two trained binarisers so they agree on the subsample for a
/// given seed.
///
/// ### Params
///
/// * `n` - Number of rows in the training matrix
/// * `seed` - Random seed for reproducible subsampling
///
/// ### Returns
///
/// All row indices when `n <= MAX_SAMPLES_PCA`, otherwise a random subset of
/// that size.
fn training_sample_indices(n: usize, seed: usize) -> Vec<usize> {
    if n <= MAX_SAMPLES_PCA {
        return (0..n).collect();
    }

    let mut rng = StdRng::seed_from_u64(seed as u64);
    let mut idx: Vec<usize> = (0..n).collect();
    idx.shuffle(&mut rng);
    idx.truncate(MAX_SAMPLES_PCA);

    idx
}

/// Per-feature mean over the sampled rows
///
/// ### Params
///
/// * `data` - Training data, row-major, `n_samples * dim`
/// * `dim` - Feature dimensionality
/// * `sample_indices` - Rows to average over, must be non-empty
///
/// ### Returns
///
/// The mean of each feature, length `dim`.
fn feature_mean<T>(data: &[T], dim: usize, sample_indices: &[usize]) -> Vec<T>
where
    T: AnnSearchFloat,
{
    // Fixed chunks summed in index order, for the reason on `gram_matrix`
    let partials: Vec<Vec<T>> = sample_indices
        .par_chunks(REDUCTION_ROW_TILE)
        .map(|chunk| {
            let mut acc = vec![T::zero(); dim];
            for &idx in chunk {
                T::add_assign_simd(&mut acc, &data[idx * dim..(idx + 1) * dim]);
            }
            acc
        })
        .collect();

    let mut mean = vec![T::zero(); dim];
    for partial in &partials {
        T::add_assign_simd(&mut mean, partial);
    }

    let n_t = T::from_usize(sample_indices.len().max(1)).unwrap();
    for m in mean.iter_mut() {
        *m = *m / n_t;
    }

    mean
}

/// Centred Gram matrix `X_c^T X_c` of a row-major block
///
/// Rows are centred as they are copied into the tile, so the GEMM only ever
/// sees mean-zero data. Accumulating `X^T X` and subtracting `n * mean mean^T`
/// afterwards would be one pass cheaper, but it computes the covariance by
/// cancellation: the accumulated magnitude is `n * mean^2` while the signal is
/// `n * Var`, so at `T = f32` on data sitting far from the origin the leading
/// digits cancel and the loadings come back as noise. The tile copy is
/// `rows * dim` and cache resident, against a `rows * dim * dim` GEMM.
///
/// Tiles are reduced in index order rather than by `fold`/`reduce`, whose
/// split points follow work stealing: float addition is not associative, and
/// the crate promises a given seed reproduces a given index.
///
/// ### Params
///
/// * `data` - Rows, row-major, a multiple of `dim` long
/// * `mean` - Per-feature mean to centre against, length `dim`
/// * `dim` - Feature dimensionality
///
/// ### Returns
///
/// The `dim x dim` symmetric product of the centred rows.
fn gram_matrix<T>(data: &[T], mean: &[T], dim: usize) -> Mat<f64>
where
    T: AnnSearchFloat,
{
    let partials: Vec<Mat<f64>> = data
        .par_chunks(REDUCTION_ROW_TILE * dim)
        .map(|rows_flat| {
            let rows = rows_flat.len() / dim;

            let mut centred = Vec::with_capacity(rows * dim);
            for row in rows_flat.chunks_exact(dim) {
                centred.extend(row.iter().zip(mean).map(|(&v, &m)| v - m));
            }

            let tile = MatRef::from_row_major_slice(&centred, rows, dim);
            let mut local = Mat::<T>::zeros(dim, dim);
            matmul(
                local.as_mut(),
                Accum::Replace,
                tile.transpose(),
                tile,
                T::one(),
                Par::Seq,
            );

            Mat::<f64>::from_fn(dim, dim, |i, j| local[(i, j)].to_f64().unwrap())
        })
        .collect();

    let mut gram = Mat::<f64>::zeros(dim, dim);
    for partial in &partials {
        for j in 0..dim {
            for i in 0..dim {
                gram[(i, j)] += partial[(i, j)];
            }
        }
    }

    gram
}

/// Number of principal components worth spending bits on
///
/// Keeps the leading components until they explain [`PCA_VARIANCE_EXPLAINED`]
/// of the total variance, then clamps to `cap`.
///
/// The cumulative rule is what makes this robust on real spectra: judging each
/// component against `sigma_0` alone cannot tell genuine rank one apart from a
/// long tail sitting under one dominant axis, which is the normal shape for an
/// embedding with a depth or library-size direction in it.
///
/// The cap carries the opposite case. On a flat, noisy spectrum the cumulative
/// rule wants nearly every component, which is precisely the failure the whole
/// variance cut exists to prevent: past the genuinely structured directions a
/// PCA bit is worse than a random one, because a random hyperplane preserves
/// angular distance by construction and a low-variance loading does not.
///
/// ### Params
///
/// * `singular_values` - Singular values of the centred training data,
///   descending.
/// * `max_k` - Upper bound from the bit budget and the matrix rank
/// * `cap` - Hard ceiling, see [`PCA_MAX_COMPONENT_SHARE`]
///
/// ### Returns
///
/// Number of components to retain, at least one unless `max_k` is zero
fn retained_components<T>(singular_values: ColRef<T>, max_k: usize, cap: usize) -> usize
where
    T: Float + FromPrimitive,
{
    if max_k == 0 {
        return 0;
    }

    let mut total = T::zero();
    for j in 0..singular_values.nrows() {
        let s = singular_values[j];
        total = total + s * s;
    }

    if total <= T::zero() {
        return 1;
    }

    let target = total * T::from_f64(PCA_VARIANCE_EXPLAINED).unwrap();

    let mut acc = T::zero();
    let mut k = 0;
    while k < max_k && acc < target {
        let s = singular_values[k];
        acc = acc + s * s;
        k += 1;
    }

    k.clamp(1, cap.max(1)).min(max_k)
}

/// Learn an ITQ rotation over PCA loadings and fold it into them
///
/// PCA orders components by variance, so the sign bit of a low-variance
/// component is close to noise while still weighing exactly as much in a
/// Hamming distance as the leading component's. Taking `k = dim` bits is the
/// worst case: the trailing components have almost no variance left and their
/// bits are decided by rounding.
///
/// ITQ (Gong and Lazebnik, CVPR 2011) rotates the projected space to minimise
/// the quantisation loss `||B - V R||_F` with `B = sign(VR)`, which spreads
/// variance across the bits. Alternating minimisation: fixing `R` gives `B` by
/// taking signs, and fixing `B` is an orthogonal Procrustes problem solved by
/// the SVD of `V^T B`.
///
/// The rotation is linear, so it folds straight back into the projections and
/// query-time encoding cost is unchanged.
///
/// ### Params
///
/// * `data` - Training data, row-major, uncentred
/// * `sample_indices` - Rows of `data` the PCA was fitted on
/// * `mean` - Per-feature mean, length `dim`
/// * `projections` - The `k` PCA loadings, row-major `k * dim`, rotated in place
/// * `k` - Number of PCA components
/// * `dim` - Feature dimensionality
/// * `seed` - Random seed for the initial rotation
///
/// ### References
///
/// Gong and Lazebnik, "Iterative Quantization: A Procrustean Approach to
/// Learning Binary Codes", CVPR 2011
fn itq_rotate_projections<T>(
    data: &[T],
    sample_indices: &[usize],
    mean: &[T],
    projections: &mut [T],
    k: usize,
    dim: usize,
    seed: usize,
) where
    T: AnnSearchFloat,
{
    if k == 0 {
        return;
    }

    let n_rows = sample_indices.len();
    let n_itq = n_rows.min(MAX_SAMPLES_ITQ);

    let itq_rows: Vec<usize> = if n_itq == n_rows {
        (0..n_rows).collect()
    } else {
        let mut rng = StdRng::seed_from_u64(seed as u64 ^ 0x9E37_79B9);
        let mut idx: Vec<usize> = (0..n_rows).collect();
        idx.shuffle(&mut rng);
        idx.truncate(n_itq);
        idx
    };

    let mut sample_flat = Vec::with_capacity(n_itq * dim);
    for &i in &itq_rows {
        let row = &data[sample_indices[i] * dim..(sample_indices[i] + 1) * dim];
        sample_flat.extend(row.iter().zip(mean).map(|(&v, &m)| v - m));
    }
    let sample = MatRef::from_row_major_slice(&sample_flat, n_itq, dim);

    // PCA scores of the ITQ subsample: V = centred * loadings, n_itq x k.
    let loadings = Mat::<T>::from_fn(dim, k, |d, j| projections[j * dim + d]);
    let mut scores = Mat::<T>::zeros(n_itq, k);
    matmul(
        scores.as_mut(),
        Accum::Replace,
        sample,
        loadings.as_ref(),
        T::one(),
        Par::Seq,
    );

    let r_flat = prepare_simhash_projections::<T>(k, k, seed);
    let mut rotation = Mat::<T>::from_fn(k, k, |i, j| r_flat[i * k + j]);
    let mut rotated = Mat::<T>::zeros(n_itq, k);
    let mut b = Mat::<T>::zeros(n_itq, k);
    let mut m = Mat::<T>::zeros(k, k);

    for _ in 0..ITQ_ITERATIONS {
        // B = sign(V R), with zero mapped to +1 so no entry is dropped
        matmul(
            rotated.as_mut(),
            Accum::Replace,
            scores.as_ref(),
            rotation.as_ref(),
            T::one(),
            Par::Seq,
        );
        for j in 0..k {
            for i in 0..n_itq {
                b[(i, j)] = if rotated[(i, j)] >= T::zero() {
                    T::one()
                } else {
                    -T::one()
                };
            }
        }

        // Orthogonal Procrustes: argmin ||B - V R|| is U W^T for V^T B = U S W^T
        matmul(
            m.as_mut(),
            Accum::Replace,
            scores.as_ref().transpose(),
            b.as_ref(),
            T::one(),
            Par::Seq,
        );
        let svd = match m.as_ref().thin_svd() {
            Ok(svd) => svd,
            // A degenerate cross-product leaves the current rotation in place;
            // the codes stay valid, they just miss the balancing.
            Err(_) => return,
        };
        matmul(
            rotation.as_mut(),
            Accum::Replace,
            svd.U(),
            svd.V().transpose(),
            T::one(),
            Par::Seq,
        );
    }

    // Fold the rotation back into the loadings: bit j reads the direction
    // `sum_l R[l][j] * loading_l`.
    let original = projections[..k * dim].to_vec();
    for j in 0..k {
        for d in 0..dim {
            let mut acc = T::zero();
            for l in 0..k {
                acc = acc + rotation[(l, j)] * original[l * dim + d];
            }
            projections[j * dim + d] = acc;
        }
    }
}

/// Initialise binariser using PCA hashing
///
/// Learns a projection from the top principal components of the training
/// data. Each bit corresponds to the sign of a data point's score on one
/// principal component. The top PCs capture the directions of greatest
/// variance, so the resulting bits preserve the most informative structure
/// in the data.
///
/// ### Algorithm
///
/// 1. Sample the training data and take its per-feature mean
/// 2. Accumulate the centred Gram matrix `X^T X - n * mean mean^T` and take its
///    eigendecomposition; the leading eigenvectors are the loadings. Going
///    through the `dim x dim` Gram matrix rather than an SVD of a centred copy
///    of the data avoids materialising that copy at all.
/// 3. Rotate the loadings with ITQ so variance is spread evenly across bits
///
/// Eigenvector signs are arbitrary, so the loadings are not reproducible
/// against an implementation that takes the SVD instead. On the raw loadings a
/// sign flip would be a relabelling, flipping bit `j` for data and query alike
/// and leaving Hamming distances untouched, but ITQ runs afterwards from a
/// fixed initial rotation that does not flip with them, so the codes really do
/// differ. Recall parity is therefore a measurement, not a proof.
///
/// Step 3 is not optional in practice. Raw PCA loadings put nearly all the
/// variance in the leading components, so the trailing sign bits are decided
/// by rounding noise yet count for just as much in a Hamming distance. See
/// [`itq_rotate_projections`] for the fix and the reference.
///
/// ### Limitations
///
/// PCA hashing can only produce `min(n_bits, dim)` meaningful bits. If
/// `n_bits > dim`, the excess bits are filled with random orthogonal
/// projections, as there are no additional variance directions to capture.
/// Those padding bits are left out of the ITQ rotation, which only spans the
/// PCA block.
///
/// ### Params
///
/// * `data` - Training data, row-major, `n * dim`. Automatically downsampled
///   if `n` exceeds MAX_SAMPLES_PCA
/// * `n` - Number of samples
/// * `dim` - Feature dimensionality
/// * `n_bits` - Number of bits in output
/// * `seed` - Random seed for reproducibility (used for subsampling and
///   fallback random projections)
///
/// ### Returns
///
/// Tuple of (projections, mean) where projections is flattened
/// (n_bits x dim) and mean is the per-feature mean (length dim)
fn prepare_pca_projections<T>(
    data: &[T],
    n: usize,
    dim: usize,
    n_bits: usize,
    seed: usize,
) -> (Vec<T>, Vec<T>)
where
    T: AnnSearchFloat,
{
    let effective_bits = n_bits.min(dim);

    let sample_indices = training_sample_indices(n, seed);
    let n_samples = sample_indices.len();
    let mean = feature_mean(data, dim, &sample_indices);

    let gathered: Option<Vec<T>> = if n_samples == n {
        None
    } else {
        let mut buf = Vec::with_capacity(n_samples * dim);
        for &idx in &sample_indices {
            buf.extend_from_slice(&data[idx * dim..(idx + 1) * dim]);
        }
        Some(buf)
    };
    let sample: &[T] = gathered.as_deref().unwrap_or(&data[..n_samples * dim]);
    let gram = gram_matrix(sample, &mean, dim);

    let eigen = match gram.as_ref().self_adjoint_eigen(Side::Lower) {
        Ok(eigen) => eigen,
        Err(_) => return (prepare_simhash_projections(dim, n_bits, seed), mean),
    };

    let eigenvalues = eigen.S().column_vector();
    let n_eig = eigenvalues.nrows();
    let singular = Col::<T>::from_fn(n_eig, |j| {
        let lambda = eigenvalues[n_eig - 1 - j].max(0.0);
        T::from_f64(lambda.sqrt()).unwrap()
    });

    let cap = ((n_bits as f64 * PCA_MAX_COMPONENT_SHARE) as usize).max(1);
    let k = retained_components(singular.as_ref(), effective_bits.min(n_eig), cap);

    let u = eigen.U();
    let mut projections = Vec::with_capacity(n_bits * dim);
    for j in 0..k {
        let col = n_eig - 1 - j;
        for i in 0..dim {
            projections.push(T::from_f64(u[(i, col)]).unwrap());
        }
    }

    itq_rotate_projections(data, &sample_indices, &mean, &mut projections, k, dim, seed);

    if n_bits > k {
        let extra = prepare_simhash_projections::<T>(dim, n_bits - k, seed + 1);
        projections.extend(extra);
    }

    (projections, mean)
}

/// Encode a vector to binary using projection-based methods
///
/// Projects the input vector onto learned or random hyperplanes and
/// quantises the result to a binary code.
///
/// ### Params
///
/// * `vec` - Input vector (length must equal dim)
/// * `projections` - Hyperplane projections (length must equal n_bits × dim)
/// * `mean` - Mean vector for centring (empty slice for SimHash, populated for
///   ITQ)
/// * `n_bits` - Number of bits to encode
/// * `dim` - Dimension of the input vector
///
/// ### Returns
///
/// Binary code as Vec<u8> (length = n_bits / 8)
fn encode_with_projections<T>(
    vec: &[T],
    projections: &[T],
    mean: &[T],
    n_bits: usize,
    dim: usize,
) -> Result<Vec<u8>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    if vec.len() != dim {
        return Err(AnnSearchErrors::DimensionMismatch {
            index_dim: dim,
            query_dim: vec.len(),
        });
    }

    let n_bytes = n_bits.div_ceil(8);
    let mut binary = vec![0u8; n_bytes];

    // Centre once, not once per bit: the subtraction used to sit in the
    // innermost loop and ran `n_bits` times for every dimension.
    let centred: Vec<T> = if mean.is_empty() {
        Vec::new()
    } else {
        vec.iter().zip(mean).map(|(&v, &m)| v - m).collect()
    };
    let row = if mean.is_empty() { vec } else { &centred };

    for bit_idx in 0..n_bits {
        let dot = T::dot_simd(&projections[bit_idx * dim..(bit_idx + 1) * dim], row);

        if dot >= T::zero() {
            let byte_idx = bit_idx / 8;
            let bit_pos = bit_idx % 8;
            binary[byte_idx] |= 1u8 << bit_pos;
        }
    }

    Ok(binary)
}

/// Rows per tile of the projection GEMM
///
/// Derived from [`ENCODE_TILE_BYTES`] so the intermediate stays cache resident
/// whatever `n_bits` is, then clamped to [`ENCODE_TILE_MIN_ROWS`] and
/// [`ENCODE_TILE_MAX_ROWS`].
///
/// ### Params
///
/// * `n_bits` - Number of bits the binariser emits
///
/// ### Returns
///
/// Number of rows per tile.
fn encode_tile_rows<T>(n_bits: usize) -> usize {
    let per_row = n_bits.max(1) * std::mem::size_of::<T>();
    (ENCODE_TILE_BYTES / per_row.max(1)).clamp(ENCODE_TILE_MIN_ROWS, ENCODE_TILE_MAX_ROWS)
}

/// Encode every row against a set of hyperplanes, tiled and in parallel
///
/// The scalar counterpart, [`encode_with_projections`], is an `n_bits * dim`
/// dot-product loop per vector; over the whole dataset that is a dense
/// `(n x dim) * (dim x n_bits)` product, so it goes through faer instead. Outer
/// parallelism is rayon over row tiles, hence `Par::Seq` on the GEMM itself.
///
/// Rows are centred into a scratch buffer rather than folding the mean into a
/// per-bit threshold. The threshold form saves the copy and is algebraically
/// identical, but `dot(v, p) >= dot(mean, p)` and `dot(v - mean, p) >= 0` are
/// not identical in floating point. Queries go through the scalar encoder, so
/// the two must agree bit for bit or a vector in the index stops matching
/// itself at Hamming distance zero.
///
/// ### Params
///
/// * `data` - Rows to encode, row-major, `n * dim`
/// * `projections` - Hyperplanes, row-major `n_bits * dim`
/// * `mean` - Per-feature mean, empty when the method does not centre
/// * `n_bits` - Number of bits
/// * `dim` - Feature dimensionality
/// * `out` - Destination codes, `n * n_bits.div_ceil(8)` bytes, overwritten
fn encode_all_with_projections<T>(
    data: &[T],
    projections: &[T],
    mean: &[T],
    n_bits: usize,
    dim: usize,
    out: &mut [u8],
) where
    T: AnnSearchFloat,
{
    let n_bytes = n_bits.div_ceil(8);
    let tile_rows = encode_tile_rows::<T>(n_bits);
    let proj = MatRef::from_row_major_slice(projections, n_bits, dim);

    data.par_chunks(tile_rows * dim)
        .zip(out.par_chunks_mut(tile_rows * n_bytes))
        .for_each_init(
            || (Mat::<T>::zeros(0, 0), Vec::<T>::new()),
            |(tile, centred), (rows_flat, codes)| {
                let rows = rows_flat.len() / dim;

                let rows_ref = if mean.is_empty() {
                    MatRef::from_row_major_slice(rows_flat, rows, dim)
                } else {
                    centred.clear();
                    centred.reserve(rows * dim);
                    for row in rows_flat.chunks_exact(dim) {
                        centred.extend(row.iter().zip(mean).map(|(&v, &m)| v - m));
                    }
                    MatRef::from_row_major_slice(centred, rows, dim)
                };

                if tile.nrows() != rows || tile.ncols() != n_bits {
                    *tile = Mat::<T>::zeros(rows, n_bits);
                }

                matmul(
                    tile.as_mut(),
                    Accum::Replace,
                    rows_ref,
                    proj.transpose(),
                    T::one(),
                    Par::Seq,
                );

                // Load-bearing: the packing loop below only ORs
                codes.fill(0);

                for j in 0..n_bits {
                    let byte = j / 8;
                    let mask = 1u8 << (j % 8);

                    for r in 0..rows {
                        if tile[(r, j)] >= T::zero() {
                            codes[r * n_bytes + byte] |= mask;
                        }
                    }
                }
            },
        );
}

/// Encode a vector to binary using sign-based binarisation
///
/// Simply takes the sign of each component. Positive values (including zero)
/// map to 1, negative values map to 0.
///
/// ### Params
///
/// * `vec` - Input vector (length must equal dim)
/// * `dim` - Dimension of the input vector
///
/// ### Returns
///
/// Binary code as Vec<u8> (length = (dim + 7) / 8)
fn encode_sign_based<T: Float>(vec: &[T], dim: usize) -> Vec<u8> {
    let mut binary = vec![0u8; dim.div_ceil(8)];
    encode_sign_based_into(vec, &mut binary);

    binary
}

/// Sign-encode a vector into a caller-owned buffer
///
/// The allocation-free half of [`encode_sign_based`], so the batch encoder can
/// write straight into the flat code array.
///
/// ### Params
///
/// * `vec` - Input vector
/// * `out` - Destination code, `vec.len().div_ceil(8)` bytes. Zeroed on entry.
fn encode_sign_based_into<T: Float>(vec: &[T], out: &mut [u8]) {
    // Load-bearing: the bit loop only ORs, so a reused buffer keeps stale bits
    out.fill(0);

    for (bit_idx, &val) in vec.iter().enumerate() {
        if val >= T::zero() {
            out[bit_idx / 8] |= 1u8 << (bit_idx % 8);
        }
    }
}

/// Sign-encode the residual of `vec` against `centroid`, in place
///
/// Bit `d` is set when `scale.0 * vec[d] - scale.1 * centroid[d] >= 0`.
///
/// The scale pair exists so Cosine can compare unit-length vectors without
/// dividing. Since `‖vec‖ * ‖centroid‖ > 0`,
///
/// ```text
/// [vec_d/‖vec‖ - c_d/‖c‖ >= 0]  ==  [‖c‖ * vec_d - ‖vec‖ * c_d >= 0]
/// ```
///
/// so passing `(‖centroid‖, ‖vec‖)` scales the comparison by a positive
/// constant and leaves every sign untouched. Squared Euclidean passes `(1, 1)`
/// and gets the plain residual.
///
/// ### Params
///
/// * `vec` - Input vector, length `dim`
/// * `centroid` - Centroid of the vector's assigned cell, length `dim`
/// * `scale` - `(vector scale, centroid scale)`, both strictly positive
/// * `dim` - Dimensionality
/// * `out` - Destination code, length `dim.div_ceil(8)`. Zeroed on entry.
pub(crate) fn encode_sign_residual_into<T: Float>(
    vec: &[T],
    centroid: &[T],
    scale: (T, T),
    dim: usize,
    out: &mut [u8],
) {
    // Load-bearing: the bit loop only ORs, so a reused buffer keeps stale bits
    out.fill(0);

    for bit_idx in 0..dim {
        if scale.0 * vec[bit_idx] - scale.1 * centroid[bit_idx] >= T::zero() {
            out[bit_idx / 8] |= 1u8 << (bit_idx % 8);
        }
    }
}

/// Binariser for converting float vectors to binary codes
///
/// Supports three binarisation methods:
///
/// - **SimHash**: Random orthogonalised projections
/// - **PcaHashing**: Signs of the principal components that clear
///   [`PCA_VARIANCE_EXPLAINED`], rotated by ITQ so variance is spread evenly across
///   the bits, with every remaining bit filled by a random orthogonal
///   direction. The padding is the common case, not an edge case: on real data
///   only a few dozen components typically clear the floor.
/// - **SignBased**: Simple sign binarisation (no training required)
#[cfg_attr(feature = "serialise", derive(serde::Serialize, serde::Deserialize))]
pub struct Binariser<T> {
    /// The binarisation method and its parameters
    pub method: BinarisationMethod<T>,
    /// Number of bits in output codes
    pub n_bits: usize,
    /// Input vector dimensionality
    pub dim: usize,
}

impl<T> Binariser<T>
where
    T: AnnSearchFloat,
{
    /// Create a new binariser using random projections (SimHash)
    ///
    /// Generates random orthogonalised projections for binary encoding. The
    /// hyperplanes pass through the origin, so the data is centred on its
    /// per-feature mean first: without that, data sitting far from the origin
    /// puts almost every point on the same side of almost every hyperplane and
    /// the codes carry next to no information.
    ///
    /// ### Params
    ///
    /// * `data` - Training data, row-major, `n * dim`, used only to fit the
    ///   centring mean. Automatically downsampled if `n` exceeds
    ///   MAX_SAMPLES_PCA
    /// * `n` - Number of samples
    /// * `dim` - Input vector dimensionality
    /// * `n_bits` - Number of bits in output (must be multiple of 8)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised binariser
    pub fn new_simhash(
        data: &[T],
        n: usize,
        dim: usize,
        n_bits: usize,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if !n_bits.is_multiple_of(8) {
            return Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits });
        }

        let projections = prepare_simhash_projections(dim, n_bits, seed);
        let mean = feature_mean(data, dim, &training_sample_indices(n, seed));

        Ok(Self {
            method: BinarisationMethod::SimHash { projections, mean },
            n_bits,
            dim,
        })
    }

    /// Initialise binariser using PCA hashing
    ///
    /// Uses Principal Component Analysis to find the directions of maximum
    /// variance in the training data, then binarises by taking the sign of
    /// each data point's projection onto these directions.
    ///
    /// ### Params
    ///
    /// * `data` - Training data, row-major, `n * dim`
    /// * `n` - Number of samples
    /// * `dim` - Input vector dimensionality
    /// * `n_bits` - Number of bits in output (must be multiple of 8)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised binariser with PCA projections
    pub fn new_pca_hashing(
        data: &[T],
        n: usize,
        dim: usize,
        n_bits: usize,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if !n_bits.is_multiple_of(8) {
            return Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits });
        }

        let (projections, mean) = prepare_pca_projections(data, n, dim, n_bits, seed);

        Ok(Self {
            method: BinarisationMethod::PcaHashing { projections, mean },
            n_bits,
            dim,
        })
    }

    /// Create a new binariser using sign-based binarisation
    ///
    /// No training required. Simply binarises based on the sign of each
    /// component. Output has exactly `dim` bits (one per dimension).
    ///
    /// ### Params
    ///
    /// * `dim` - Input vector dimensionality
    ///
    /// ### Returns
    ///
    /// Initialised binariser
    pub fn new_sign_based(dim: usize) -> Self {
        Self {
            method: BinarisationMethod::SignBased,
            n_bits: dim, // sign-based always produces dim bits
            dim,
        }
    }

    /// Length of the codes this binariser emits
    ///
    /// This is the only correct stride for a flattened code array. It is *not*
    /// derivable from the `n_bits` a caller passed to a `build_*` function:
    /// sign-based binarisation ignores that argument and always produces `dim`
    /// bits, so taking the stride from the argument corrupts the layout
    /// whenever `n_bits != dim`.
    ///
    /// ### Returns
    ///
    /// Bytes per encoded vector.
    pub fn n_bytes(&self) -> usize {
        self.n_bits.div_ceil(8)
    }

    /// Encode a vector to binary
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length must equal dim)
    ///
    /// ### Returns
    ///
    /// Binary code as `Vec<u8>`
    pub fn encode(&self, vec: &[T]) -> Result<Vec<u8>, AnnSearchErrors> {
        if vec.len() != self.dim {
            return Err(AnnSearchErrors::DimensionMismatch {
                index_dim: self.dim,
                query_dim: vec.len(),
            });
        }

        match &self.method {
            BinarisationMethod::SimHash { projections, mean } => {
                encode_with_projections(vec, projections, mean, self.n_bits, self.dim)
            }
            BinarisationMethod::PcaHashing { projections, mean } => {
                encode_with_projections(vec, projections, mean, self.n_bits, self.dim)
            }
            BinarisationMethod::SignBased => Ok(encode_sign_based(vec, self.dim)),
        }
    }

    /// Encode a whole dataset in one go
    ///
    /// The projection methods go through a tiled, rayon-parallel GEMM rather
    /// than `n` calls to [`Self::encode`]: encoding the dataset *is* a dense
    /// `(n x dim) * (dim x n_bits)` product, and doing it one row at a time
    /// costs both the arithmetic and a heap allocation per vector. Sign-based
    /// codes have no projection to batch, so they simply fan out over rows.
    ///
    /// ### Params
    ///
    /// * `data` - Rows to encode, row-major, `n * dim`
    /// * `n` - Number of rows
    /// * `out` - Destination, `n * self.n_bytes()` bytes. Overwritten in full.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or [`AnnSearchErrors::BufferLengthMismatch`] when `data` is
    /// not `n * dim` long or `out` is not `n * self.n_bytes()` long.
    pub fn encode_all(&self, data: &[T], n: usize, out: &mut [u8]) -> Result<(), AnnSearchErrors> {
        if data.len() != n * self.dim {
            return Err(AnnSearchErrors::BufferLengthMismatch {
                expected: n * self.dim,
                actual: data.len(),
            });
        }

        let n_bytes = self.n_bytes();
        if out.len() != n * n_bytes {
            return Err(AnnSearchErrors::BufferLengthMismatch {
                expected: n * n_bytes,
                actual: out.len(),
            });
        }

        match &self.method {
            BinarisationMethod::SimHash { projections, mean }
            | BinarisationMethod::PcaHashing { projections, mean } => {
                encode_all_with_projections(data, projections, mean, self.n_bits, self.dim, out);
            }
            BinarisationMethod::SignBased => {
                out.par_chunks_mut(n_bytes)
                    .zip(data.par_chunks(self.dim))
                    .for_each(|(code, row)| encode_sign_based_into(row, code));
            }
        }

        Ok(())
    }

    /// Encode a vector as the sign of its residual against a centroid
    ///
    /// Sign bits taken in the global frame encode which cluster a point sits
    /// in, not where it sits inside that cluster, because a cluster far from
    /// the origin puts every one of its members on the same side of every
    /// coordinate plane. Taking the residual against the cluster's own centroid
    /// moves the frame to the cluster, so the bits carry the within-cluster
    /// structure the search funnel actually needs.
    ///
    /// Codes produced this way are only comparable against other codes taken
    /// against the *same* centroid.
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length must equal `dim`)
    /// * `centroid` - Centroid to take the residual against (length must equal
    ///   `dim`)
    /// * `scale` - `(vector scale, centroid scale)`, see
    ///   [`encode_sign_residual_into`]
    ///
    /// ### Returns
    ///
    /// Binary code as `Vec<u8>`, or
    /// [`AnnSearchErrors::ResidualEncodingUnsupported`] for the projection-based
    /// methods, which carry a frame of their own.
    pub fn encode_residual(
        &self,
        vec: &[T],
        centroid: &[T],
        scale: (T, T),
    ) -> Result<Vec<u8>, AnnSearchErrors> {
        if vec.len() != self.dim {
            return Err(AnnSearchErrors::DimensionMismatch {
                index_dim: self.dim,
                query_dim: vec.len(),
            });
        }
        if !matches!(self.method, BinarisationMethod::SignBased) {
            return Err(AnnSearchErrors::ResidualEncodingUnsupported);
        }

        let mut out = vec![0u8; self.n_bytes()];
        encode_sign_residual_into(vec, centroid, scale, self.dim, &mut out);

        Ok(out)
    }

    /// Returns memory usage in bytes
    ///
    /// ### Returns
    ///
    /// Total bytes used by the binariser
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);
        match &self.method {
            BinarisationMethod::SimHash { projections, mean } => {
                total += projections.capacity() * std::mem::size_of::<T>();
                total += mean.capacity() * std::mem::size_of::<T>();
            }
            BinarisationMethod::PcaHashing { projections, mean } => {
                total += projections.capacity() * std::mem::size_of::<T>();
                total += mean.capacity() * std::mem::size_of::<T>();
            }
            BinarisationMethod::SignBased => {}
        }
        total
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use crate::binary::dist_binary::hamming_distance;
    use faer::Mat;

    #[test]
    fn test_simhash_basic() {
        let dim = 128;
        let n_bits = 256;
        let zero_mean = Mat::<f64>::zeros(8, dim);
        let binariser = Binariser::<f64>::new_simhash(
            &matrix_to_flat(zero_mean.as_ref()).0,
            zero_mean.nrows(),
            dim,
            n_bits,
            42,
        )
        .unwrap();

        let vec1: Vec<f64> = (0..dim).map(|i| (i as f64) / (dim as f64)).collect();
        let binary = binariser.encode(&vec1).unwrap();

        assert_eq!(binary.len(), n_bits / 8);
    }

    #[test]
    fn test_simhash_preserves_similarity() {
        let dim = 64;
        let n_bits = 128;
        let zero_mean = Mat::<f64>::zeros(8, dim);
        let binariser = Binariser::<f64>::new_simhash(
            &matrix_to_flat(zero_mean.as_ref()).0,
            zero_mean.nrows(),
            dim,
            n_bits,
            42,
        )
        .unwrap();

        let vec1: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let vec2: Vec<f64> = (0..dim).map(|i| i as f64 + 0.1).collect();
        let vec3: Vec<f64> = (0..dim).map(|i| -(i as f64)).collect();

        let bin1 = binariser.encode(&vec1).unwrap();
        let bin2 = binariser.encode(&vec2).unwrap();
        let bin3 = binariser.encode(&vec3).unwrap();

        let dist_12 = hamming_distance(&bin1, &bin2);
        let dist_13 = hamming_distance(&bin1, &bin3);

        assert!(
            dist_12 < dist_13,
            "Similar vectors should have smaller Hamming distance"
        );
    }

    #[test]
    fn test_pca_hashing_basic() {
        let n_samples = 1000;
        let dim = 64;
        let n_bits = 128;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = ((i + j) as f64).sin();
            }
        }

        let binariser = Binariser::<f64>::new_pca_hashing(
            &matrix_to_flat(data.as_ref()).0,
            data.nrows(),
            dim,
            n_bits,
            42,
        )
        .unwrap();

        let vec1: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
        let binary = binariser.encode(&vec1).unwrap();

        assert_eq!(binary.len(), n_bits / 8);
    }

    /// The property ITQ exists to deliver: after the rotation, no bit should
    /// be reading a direction with orders of magnitude less variance than
    /// another. The fixture's coordinate `j` has standard deviation `2^-j`, so
    /// the raw PCA loadings would give per-bit variances spanning the full
    /// `4^-(dim-1)` range, roughly nine orders of magnitude at dim 16. A bit
    /// sitting at the bottom of that range is decided by rounding noise but
    /// still counts for one unit of Hamming distance.
    #[test]
    fn test_itq_balances_variance_across_bits() {
        let n_samples = 2000;
        let dim = 16;
        let n_bits = 16; // every bit is a PCA bit, the worst case for balance

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                let mut x = (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                x ^= x >> 33;
                x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                x ^= x >> 33;
                let unit = (x as f64 / u64::MAX as f64) * 2.0 - 1.0;
                data[(i, j)] = unit * 2.0_f64.powi(-(j as i32));
            }
        }

        let flat = matrix_to_flat(data.as_ref()).0;
        let binariser =
            Binariser::<f64>::new_pca_hashing(&flat, n_samples, dim, n_bits, 42).unwrap();

        let BinarisationMethod::PcaHashing { projections, mean } = &binariser.method else {
            panic!("Expected PcaHashing method");
        };

        // Per-bit variance of the projected scores
        let mut variances = vec![0.0f64; n_bits];
        for (b, var) in variances.iter_mut().enumerate() {
            let proj = &projections[b * dim..(b + 1) * dim];
            let mut sum = 0.0;
            let mut sum_sq = 0.0;
            for i in 0..n_samples {
                let score: f64 = (0..dim).map(|d| proj[d] * (data[(i, d)] - mean[d])).sum();
                sum += score;
                sum_sq += score * score;
            }
            let n = n_samples as f64;
            *var = sum_sq / n - (sum / n).powi(2);
        }

        let max = variances.iter().cloned().fold(f64::MIN, f64::max);
        let min = variances.iter().cloned().fold(f64::MAX, f64::min);

        // Raw PCA loadings on this fixture spread over ~1e9; ITQ must pull that
        // in by orders of magnitude. The bound is deliberately loose: ITQ
        // minimises quantisation loss, it does not equalise variance exactly.
        assert!(
            max / min < 1e3,
            "ITQ left the bit variances spread over {:e} (max {max:e}, min {min:e})",
            max / min
        );
    }

    /// `retained_components` is the whole point of the variance cut, so pin its
    /// behaviour directly rather than inferring it from downstream recall.
    #[test]
    fn test_retained_components_cumulative_rule() {
        use faer::Col;

        // Variances 100, 10, 1, 0.1 sum to 111.1; 90% is 99.99, which the
        // leading component alone clears.
        let decaying = Col::<f64>::from_fn(4, |i| [10.0, 3.1623, 1.0, 0.3162][i]);
        assert_eq!(retained_components(decaying.as_ref(), 4, 4), 1);

        // Flat spectrum: 90% of six equal components needs six of them, so the
        // cap is the only thing standing between this and spending every bit on
        // a noise direction.
        let flat = Col::<f64>::from_fn(6, |_| 1.0);
        assert_eq!(retained_components(flat.as_ref(), 6, 6), 6);
        assert_eq!(retained_components(flat.as_ref(), 6, 2), 2);

        // One dominant axis over an informative tail. The old ratio-to-sigma_0
        // rule returned 1 here; the cumulative rule keeps the tail.
        let spiked = Col::<f64>::from_fn(21, |i| if i == 0 { 10.0 } else { 1.0 });
        assert!(
            retained_components(spiked.as_ref(), 21, 21) > 1,
            "a long tail under one dominant axis must not collapse to rank one"
        );

        // The bit budget still caps it, and degenerate inputs stay sane.
        assert_eq!(retained_components(flat.as_ref(), 3, 6), 3);
        assert_eq!(retained_components(flat.as_ref(), 0, 6), 0);
        let zeros = Col::<f64>::from_fn(3, |_| 0.0);
        assert_eq!(retained_components(zeros.as_ref(), 3, 3), 1);
    }

    /// Data on a 4-dimensional subspace of a 32-dimensional space. Only four
    /// components clear the variance floor, so the other 60 bits must be random
    /// hyperplanes rather than sign bits of a noise direction. Every row still
    /// has to be a usable unit-norm hyperplane.
    #[test]
    fn test_rank_deficient_data_falls_back_to_random_bits() {
        let n_samples = 800;
        let dim = 32;
        let intrinsic = 4;
        let n_bits = 64;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                let mut v = 0.0;
                for c in 0..intrinsic {
                    let mut x = (i as u64)
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add((c as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
                    x ^= x >> 33;
                    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                    x ^= x >> 33;
                    let coord = (x as f64 / u64::MAX as f64) * 2.0 - 1.0;
                    v += coord * ((c * dim + j) as f64 * 0.7).cos();
                }
                data[(i, j)] = v;
            }
        }

        let binariser = Binariser::<f64>::new_pca_hashing(
            &matrix_to_flat(data.as_ref()).0,
            n_samples,
            dim,
            n_bits,
            42,
        )
        .unwrap();

        let BinarisationMethod::PcaHashing { projections, .. } = &binariser.method else {
            panic!("Expected PcaHashing method");
        };
        assert_eq!(projections.len(), n_bits * dim);

        for i in 0..n_bits {
            let base = i * dim;
            let norm: f64 = projections[base..base + dim]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "projection {i} is not a unit hyperplane: {norm}"
            );
        }

        // The codes must still separate points: a vector and its negation sit on
        // opposite sides of every hyperplane through the mean.
        let row: Vec<f64> = (0..dim).map(|j| data[(0, j)]).collect();
        let code = binariser.encode(&row).unwrap();
        assert_eq!(code.len(), n_bits / 8);
    }

    #[test]
    fn test_pca_hashing_orthogonality() {
        let n_samples = 500;
        let dim = 32;
        let n_bits = 64;
        let rank = 4;

        // Exactly `rank` equal-variance directions plus a negligible floor. The
        // variances are equal, so 90% of the total needs all four and the
        // cumulative rule retains exactly `rank` loadings; rows past that are
        // random padding, which is deliberately not orthogonal to the block.
        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                let mut x = (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                x ^= x >> 33;
                x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                x ^= x >> 33;
                let unit = (x as f64 / u64::MAX as f64) * 2.0 - 1.0;
                data[(i, j)] = if j < rank { unit } else { unit * 1e-6 };
            }
        }

        let binariser = Binariser::<f64>::new_pca_hashing(
            &matrix_to_flat(data.as_ref()).0,
            data.nrows(),
            dim,
            n_bits,
            42,
        )
        .unwrap();

        let BinarisationMethod::PcaHashing { projections, .. } = &binariser.method else {
            panic!("Expected PcaHashing method");
        };

        // Every hyperplane, loading or padding, must be usable.
        for i in 0..n_bits {
            let base = i * dim;
            let norm: f64 = projections[base..base + dim]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "projection {i} not normalised: {norm}"
            );
        }

        // The retained loadings stay mutually orthogonal through the ITQ fold.
        for i in 0..rank {
            for j in (i + 1)..rank {
                let dot: f64 = (0..dim)
                    .map(|d| projections[i * dim + d] * projections[j * dim + d])
                    .sum();
                assert!(
                    dot.abs() < 1e-6,
                    "loadings {i} and {j} not orthogonal: {dot}"
                );
            }
        }
    }

    #[test]
    fn test_pca_hashing_centring() {
        let n_samples = 100;
        let dim = 16;
        let n_bits = 32;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = (i as f64) + 10.0;
            }
        }

        let binariser = Binariser::<f64>::new_pca_hashing(
            &matrix_to_flat(data.as_ref()).0,
            data.nrows(),
            dim,
            n_bits,
            42,
        )
        .unwrap();

        if let BinarisationMethod::PcaHashing { mean, .. } = &binariser.method {
            for d in 0..dim {
                let expected_mean = (n_samples as f64 - 1.0) / 2.0 + 10.0;
                assert!((mean[d] - expected_mean).abs() < 1e-6);
            }
        } else {
            panic!("Expected PcaHashing method");
        }
    }

    #[test]
    fn test_sign_based_basic() {
        let dim = 128;
        let binariser = Binariser::<f64>::new_sign_based(dim);

        assert_eq!(binariser.n_bits, dim);

        let vec: Vec<f64> = (0..dim)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let binary = binariser.encode(&vec).unwrap();

        assert_eq!(binary.len(), dim.div_ceil(8));

        // Check first few bits match expected pattern
        for i in 0..8 {
            let byte_idx = i / 8;
            let bit_pos = i % 8;
            let bit_set = (binary[byte_idx] & (1u8 << bit_pos)) != 0;
            assert_eq!(bit_set, i % 2 == 0, "Bit {} should be {}", i, i % 2 == 0);
        }
    }

    #[test]
    fn test_sign_based_preserves_similarity() {
        let dim = 64;
        let binariser = Binariser::<f64>::new_sign_based(dim);

        let vec1: Vec<f64> = (0..dim).map(|i| (i + 1) as f64).collect();
        let vec2: Vec<f64> = (0..dim).map(|i| (i + 1) as f64 + 0.1).collect();
        let vec3: Vec<f64> = (0..dim).map(|i| -((i + 1) as f64)).collect();

        let bin1 = binariser.encode(&vec1).unwrap();
        let bin2 = binariser.encode(&vec2).unwrap();
        let bin3 = binariser.encode(&vec3).unwrap();

        let dist_12 = hamming_distance(&bin1, &bin2);
        let dist_13 = hamming_distance(&bin1, &bin3);

        assert_eq!(
            dist_12, 0,
            "Vectors with same signs should have zero Hamming distance"
        );
        assert_eq!(
            dist_13, dim as u32,
            "Vectors with opposite signs should have maximum Hamming distance"
        );
    }

    #[test]
    fn test_deterministic() {
        let dim = 32;
        let n_bits = 64;

        let zero_mean = Mat::<f64>::zeros(8, dim);
        let binariser1 = Binariser::<f64>::new_simhash(
            &matrix_to_flat(zero_mean.as_ref()).0,
            zero_mean.nrows(),
            dim,
            n_bits,
            42,
        )
        .unwrap();
        let binariser2 = Binariser::<f64>::new_simhash(
            &matrix_to_flat(zero_mean.as_ref()).0,
            zero_mean.nrows(),
            dim,
            n_bits,
            42,
        )
        .unwrap();

        let vec: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let bin1 = binariser1.encode(&vec).unwrap();
        let bin2 = binariser2.encode(&vec).unwrap();

        assert_eq!(bin1, bin2);
    }

    #[test]
    fn test_parse_binarisation_init() {
        assert!(matches!(
            parse_binarisation_init("pca"),
            Some(BinarisationInit::PcaHashing)
        ));
        assert!(matches!(
            parse_binarisation_init("pca_hashing"),
            Some(BinarisationInit::PcaHashing)
        ));
        assert!(matches!(
            parse_binarisation_init("random"),
            Some(BinarisationInit::RandomProjections)
        ));
        assert!(matches!(
            parse_binarisation_init("random_projections"),
            Some(BinarisationInit::RandomProjections)
        ));
        assert!(matches!(
            parse_binarisation_init("sign"),
            Some(BinarisationInit::SignBased)
        ));
        assert!(matches!(
            parse_binarisation_init("sign_based"),
            Some(BinarisationInit::SignBased)
        ));
        assert!(parse_binarisation_init("invalid").is_none());
    }

    #[test]
    fn test_invalid_n_bits_simhash() {
        let zero_mean = Mat::<f64>::zeros(8, 64);
        let result = Binariser::<f64>::new_simhash(
            &matrix_to_flat(zero_mean.as_ref()).0,
            zero_mean.nrows(),
            64,
            123,
            42,
        );
        assert!(matches!(
            result,
            Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits: 123 })
        ));
    }

    #[test]
    fn test_invalid_n_bits_pca_hashing() {
        let data = Mat::<f64>::zeros(100, 64);
        let result = Binariser::<f64>::new_pca_hashing(
            &matrix_to_flat(data.as_ref()).0,
            data.nrows(),
            64,
            123,
            42,
        );
        assert!(matches!(
            result,
            Err(AnnSearchErrors::NBitsMustBe8Multiple { n_bits: 123 })
        ));
    }

    #[test]
    fn test_dimension_mismatch() {
        let zero_mean = Mat::<f64>::zeros(8, 64);
        let binariser = Binariser::<f64>::new_simhash(
            &matrix_to_flat(zero_mean.as_ref()).0,
            zero_mean.nrows(),
            64,
            128,
            42,
        )
        .unwrap();
        let result = binariser.encode(&vec![0.0; 32]);
        assert!(matches!(
            result,
            Err(AnnSearchErrors::DimensionMismatch {
                index_dim: 64,
                query_dim: 32
            })
        ));
    }

    /// SimHash hyperplanes pass through the origin, so an uncentred fit on data
    /// sitting far from it puts nearly every point on the same side of nearly
    /// every plane. Codes then collapse towards a single value and Hamming
    /// distance stops discriminating. Centring on the training mean restores a
    /// roughly balanced bit per plane.
    #[test]
    fn test_simhash_centres_off_origin_data() {
        let (n, dim, n_bits) = (256, 32, 64);

        // Tight cloud a long way from the origin
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(19);
        let data = Mat::<f64>::from_fn(n, dim, |_, _| 50.0 + rng.random::<f64>() * 2.0 - 1.0);

        let binariser = Binariser::<f64>::new_simhash(
            &matrix_to_flat(data.as_ref()).0,
            data.nrows(),
            dim,
            n_bits,
            42,
        )
        .unwrap();

        let codes: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let row: Vec<f64> = data.row(i).iter().cloned().collect();
                binariser.encode(&row).unwrap()
            })
            .collect();

        // Every bit position should be set by a decent share of the points
        for bit in 0..n_bits {
            let set = codes
                .iter()
                .filter(|c| (c[bit / 8] >> (bit % 8)) & 1 == 1)
                .count();
            let balance = set as f64 / n as f64;

            assert!(
                (0.05..=0.95).contains(&balance),
                "bit {bit} is near-constant across the data (balance {balance:.3}), \
                 which is what an uncentred fit produces"
            );
        }

        // and distinct points must not collapse onto one code
        let mut unique = codes.clone();
        unique.sort_unstable();
        unique.dedup();

        assert!(
            unique.len() > n / 2,
            "only {} distinct codes for {n} points",
            unique.len()
        );
    }

    /// The `(1, 1)` scale pair is the plain residual.
    #[test]
    fn test_encode_residual_squared_euclidean_matches_plain_sign() {
        let dim = 24;
        let binariser = Binariser::<f64>::new_sign_based(dim);

        let vec: Vec<f64> = (0..dim)
            .map(|i| (i as f64 * 0.7).sin() * 3.0 + 1.0)
            .collect();
        let centroid: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.3).cos()).collect();

        let residual: Vec<f64> = vec.iter().zip(&centroid).map(|(v, c)| v - c).collect();

        assert_eq!(
            binariser
                .encode_residual(&vec, &centroid, (1.0, 1.0))
                .unwrap(),
            encode_sign_based(&residual, dim)
        );
    }

    /// The `(‖c‖, ‖v‖)` pair is the residual between unit-length vectors, which
    /// is what Cosine wants, expressed without a division.
    #[test]
    fn test_encode_residual_cosine_matches_normalised_sign() {
        let dim = 24;
        let binariser = Binariser::<f64>::new_sign_based(dim);

        let vec: Vec<f64> = (0..dim)
            .map(|i| (i as f64 * 0.7).sin() * 3.0 + 1.0)
            .collect();
        let centroid: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.3).cos()).collect();

        let vn = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        let cn = centroid.iter().map(|x| x * x).sum::<f64>().sqrt();

        let residual: Vec<f64> = vec
            .iter()
            .zip(&centroid)
            .map(|(v, c)| v / vn - c / cn)
            .collect();

        assert_eq!(
            binariser
                .encode_residual(&vec, &centroid, (cn, vn))
                .unwrap(),
            encode_sign_based(&residual, dim)
        );
    }

    /// The projection methods carry a learned or random frame of their own, so
    /// a per-cell threshold shift is a different method, not a variant.
    #[test]
    fn test_encode_residual_rejects_projection_methods() {
        let dim = 32;
        let zero_mean = Mat::<f64>::zeros(8, dim);

        let simhash = Binariser::<f64>::new_simhash(
            &matrix_to_flat(zero_mean.as_ref()).0,
            zero_mean.nrows(),
            dim,
            64,
            42,
        )
        .unwrap();
        let pca = Binariser::<f64>::new_pca_hashing(
            &matrix_to_flat(zero_mean.as_ref()).0,
            zero_mean.nrows(),
            dim,
            64,
            42,
        )
        .unwrap();

        let vec = vec![1.0; dim];
        let centroid = vec![0.5; dim];

        for binariser in [simhash, pca] {
            assert!(matches!(
                binariser.encode_residual(&vec, &centroid, (1.0, 1.0)),
                Err(AnnSearchErrors::ResidualEncodingUnsupported)
            ));
        }
    }

    /// A `dim` that is not a multiple of 8 leaves padding bits, which must stay
    /// zero so they XOR away in the Hamming kernels.
    #[test]
    fn test_encode_residual_zeroes_padding_bits() {
        for dim in [30, 31, 32] {
            let binariser = Binariser::<f64>::new_sign_based(dim);

            // All residuals positive, so every real bit is set
            let vec = vec![1.0; dim];
            let centroid = vec![0.0; dim];
            let code = binariser
                .encode_residual(&vec, &centroid, (1.0, 1.0))
                .unwrap();

            assert_eq!(code.len(), dim.div_ceil(8));

            let set: u32 = code.iter().map(|b| b.count_ones()).sum();
            assert_eq!(set, dim as u32, "padding bits leaked at dim = {dim}");
        }
    }

    /// `encode_all` is the only encoder the build path uses now, so it has to
    /// agree with the per-vector one exactly. `dim` is deliberately not a
    /// multiple of the SIMD width and `n` not a multiple of the tile, so the
    /// remainder handling in both is exercised.
    #[test]
    fn test_encode_all_matches_per_vector_encode() {
        let n = 9_001;
        let dim = 37;
        let n_bits = 128;

        let mut data = vec![0.0f64; n * dim];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i as f64) * 0.7).sin() * 3.0 + ((i % 17) as f64) - 8.0;
        }

        let simhash = Binariser::<f64>::new_simhash(&data, n, dim, n_bits, 42).unwrap();
        let pca = Binariser::<f64>::new_pca_hashing(&data, n, dim, n_bits, 42).unwrap();
        let sign = Binariser::<f64>::new_sign_based(dim);

        for binariser in [simhash, pca, sign] {
            let n_bytes = binariser.n_bytes();
            let mut batched = vec![0u8; n * n_bytes];
            binariser.encode_all(&data, n, &mut batched).unwrap();

            for i in 0..n {
                let one = binariser.encode(&data[i * dim..(i + 1) * dim]).unwrap();
                assert_eq!(
                    &batched[i * n_bytes..(i + 1) * n_bytes],
                    &one[..],
                    "row {i} disagrees at n_bits {}",
                    binariser.n_bits
                );
            }
        }
    }

    /// Length mistakes are the easy way to corrupt a flat code array, so they
    /// have to be errors rather than a panic or a silent partial write.
    #[test]
    fn test_encode_all_rejects_bad_lengths() {
        let (n, dim, n_bits) = (16, 8, 32);
        let data = vec![0.5f64; n * dim];
        let binariser = Binariser::<f64>::new_simhash(&data, n, dim, n_bits, 42).unwrap();

        let mut out = vec![0u8; n * binariser.n_bytes()];
        assert!(matches!(
            binariser.encode_all(&data[..(n - 1) * dim], n, &mut out),
            Err(AnnSearchErrors::BufferLengthMismatch { .. })
        ));

        let mut short = vec![0u8; n * binariser.n_bytes() - 1];
        assert!(matches!(
            binariser.encode_all(&data, n, &mut short),
            Err(AnnSearchErrors::BufferLengthMismatch { .. })
        ));
    }

    /// The Gram path replaced a `thin_svd` of the centred data. It has to
    /// recover the same principal subspace and the same spectrum; only the sign
    /// of each eigenvector is free, and a sign flip flips one bit for every
    /// vector alike, so it leaves Hamming distances untouched.
    #[test]
    fn test_gram_matches_thin_svd_loadings() {
        let (n, dim) = (600, 12);

        let mut data = vec![0.0f64; n * dim];
        for i in 0..n {
            for j in 0..dim {
                let mut x = (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                x ^= x >> 33;
                x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                x ^= x >> 33;
                let unit = (x as f64 / u64::MAX as f64) * 2.0 - 1.0;
                data[i * dim + j] = unit * (1.0 + j as f64) + 4.0;
            }
        }

        let indices: Vec<usize> = (0..n).collect();
        let mean = feature_mean(&data, dim, &indices);

        // Reference: SVD of the explicitly centred matrix
        let centred = Mat::<f64>::from_fn(n, dim, |i, d| data[i * dim + d] - mean[d]);
        let svd = centred.as_ref().thin_svd().unwrap();

        // Under test: eigendecomposition of the centred Gram matrix
        let gram = gram_matrix(&data, &mean, dim);
        let eigen = gram.as_ref().self_adjoint_eigen(Side::Lower).unwrap();
        let eigenvalues = eigen.S().column_vector();

        for j in 0..dim {
            let from_gram = eigenvalues[dim - 1 - j].max(0.0).sqrt();
            approx::assert_relative_eq!(from_gram, svd.S()[j], epsilon = 1e-6, max_relative = 1e-6);

            // Same direction up to sign
            let dot: f64 = (0..dim)
                .map(|i| eigen.U()[(i, dim - 1 - j)] * svd.V()[(i, j)])
                .sum();
            approx::assert_relative_eq!(dot.abs(), 1.0, epsilon = 1e-6);
        }
    }

    /// The tiled accumulation must agree with the obvious double loop, and it
    /// has to handle a row count that is not a multiple of the tile.
    #[test]
    fn test_gram_matrix_matches_direct_accumulation() {
        let (n, dim) = (REDUCTION_ROW_TILE + 137, 5);

        let mut data = vec![0.0f32; n * dim];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i % 23) as f32) * 0.25 - 2.0;
        }

        let mean = feature_mean(&data, dim, &(0..n).collect::<Vec<_>>());
        let gram = gram_matrix(&data, &mean, dim);

        for a in 0..dim {
            for b in 0..dim {
                let direct: f64 = (0..n)
                    .map(|i| {
                        (data[i * dim + a] - mean[a]) as f64 * (data[i * dim + b] - mean[b]) as f64
                    })
                    .sum();
                approx::assert_relative_eq!(gram[(a, b)], direct, max_relative = 1e-4);
            }
        }
    }

    /// The reason `gram_matrix` centres on the way in rather than accumulating
    /// `X^T X` and subtracting `n * mean mean^T` afterwards. The correction form
    /// computes the covariance by cancellation: accumulated magnitude is
    /// `n * mean^2` while the signal is `n * Var`, so at `f32` on data far from
    /// the origin the leading digits cancel and the loadings come back as noise.
    /// This is the crate's off-origin shape pushed out to 500.
    #[test]
    fn test_gram_matrix_survives_large_mean_at_f32() {
        let (n, dim) = (60_000, 8);

        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                let mut x = (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
                x ^= x >> 33;
                x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                x ^= x >> 33;
                let unit = (x as f32 / u64::MAX as f32) * 2.0 - 1.0;
                data[i * dim + j] = 500.0 + unit;
            }
        }

        let mean = feature_mean(&data, dim, &(0..n).collect::<Vec<_>>());
        let gram = gram_matrix(&data, &mean, dim);

        // Reference in f64, centred on the same f32 mean, so only the
        // accumulation precision is under test.
        for a in 0..dim {
            for b in 0..dim {
                let direct: f64 = (0..n)
                    .map(|i| {
                        (data[i * dim + a] - mean[a]) as f64 * (data[i * dim + b] - mean[b]) as f64
                    })
                    .sum();
                let scale = (n as f64) / 3.0;
                assert!(
                    (gram[(a, b)] - direct).abs() < 0.02 * scale,
                    "entry ({a}, {b}) is {} against {direct}; cancellation is back",
                    gram[(a, b)]
                );
            }
        }
    }

    /// PCA hashing end to end on off-origin `f32` data.
    ///
    /// Asserts the recovered loadings against an `f64` reference eigen-
    /// decomposition of the explicitly centred data, because that is what
    /// actually degrades. Bit balance is not a usable proxy here: ITQ rotates
    /// the retained block to spread variance evenly, so the bits come out
    /// balanced whether or not the loadings are directions in the data.
    #[test]
    fn test_pca_hashing_off_origin_f32() {
        let (n, dim, n_bits) = (40_000, 32, 256);

        let mut data = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                let mut x = (i as u64)
                    .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add((j as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93));
                x ^= x >> 33;
                x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
                x ^= x >> 33;
                let unit = (x as f32 / u64::MAX as f32) * 2.0 - 1.0;
                // Structure worth finding, sitting 500 away from the origin
                data[i * dim + j] = 500.0 + unit * (1.0 + (j % 4) as f32);
            }
        }

        let indices: Vec<usize> = (0..n).collect();
        let mean = feature_mean(&data, dim, &indices);

        // Reference: centre in f64 first, so no cancellation anywhere
        let centred = Mat::<f64>::from_fn(n, dim, |i, d| data[i * dim + d] as f64 - mean[d] as f64);
        let mut reference = Mat::<f64>::zeros(dim, dim);
        matmul(
            reference.as_mut(),
            Accum::Replace,
            centred.as_ref().transpose(),
            centred.as_ref(),
            1.0,
            Par::Seq,
        );
        let want = reference.as_ref().self_adjoint_eigen(Side::Lower).unwrap();

        let binariser = Binariser::<f32>::new_pca_hashing(&data, n, dim, n_bits, 42).unwrap();
        let BinarisationMethod::PcaHashing { projections, .. } = &binariser.method else {
            panic!("Expected PcaHashing method");
        };
        assert_eq!(projections.len(), n_bits * dim);

        // `k` retained loadings, then random padding. ITQ mixes within the
        // retained block, so compare the subspace the block spans rather than
        // each direction: every retained loading must lie in the span of the
        // reference's leading `k` eigenvectors.
        //
        // `k` is what the cumulative rule keeps, not the cap: the cap is only
        // an upper bound and on this fixture the rule stops well short of it,
        // so reading `k` off the cap grades random padding bits against the PCA
        // subspace. Driven by the reference's `f64` spectrum, so the `f32` path
        // is not choosing its own comparison.
        let reference_singular = Col::<f64>::from_fn(dim, |j| {
            want.S().column_vector()[dim - 1 - j].max(0.0).sqrt()
        });
        let k = retained_components(
            reference_singular.as_ref(),
            n_bits.min(dim),
            ((n_bits as f64 * PCA_MAX_COMPONENT_SHARE) as usize).max(1),
        );
        assert!(
            k < dim,
            "with k == dim the reference subspace is the whole space and this \
             assertion is vacuous"
        );

        for b in 0..k {
            let loading = &projections[b * dim..(b + 1) * dim];

            let norm: f32 = loading.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "loading {b} is not a unit hyperplane: {norm}"
            );

            let captured: f64 = (0..k)
                .map(|j| {
                    let col = dim - 1 - j;
                    let dot: f64 = (0..dim)
                        .map(|i| loading[i] as f64 * want.U()[(i, col)])
                        .sum();
                    dot * dot
                })
                .sum();

            assert!(
                captured > 0.99,
                "loading {b} lies mostly outside the true leading subspace \
                 (captured {captured:.4}); the covariance is noise"
            );
        }
    }

    /// `encode_all` and `encode` must agree at `f32`, the precision that
    /// actually separates centring the rows from folding the mean into a
    /// per-bit threshold.
    #[test]
    fn test_encode_all_matches_per_vector_encode_f32() {
        let n = 5_003;
        let dim = 24;
        let n_bits = 64;

        let mut data = vec![0.0f32; n * dim];
        for (i, v) in data.iter_mut().enumerate() {
            *v = 200.0 + ((i as f32) * 0.13).sin() * 2.0;
        }

        for binariser in [
            Binariser::<f32>::new_simhash(&data, n, dim, n_bits, 7).unwrap(),
            Binariser::<f32>::new_pca_hashing(&data, n, dim, n_bits, 7).unwrap(),
        ] {
            let n_bytes = binariser.n_bytes();
            let mut batched = vec![0u8; n * n_bytes];
            binariser.encode_all(&data, n, &mut batched).unwrap();

            for i in 0..n {
                let one = binariser.encode(&data[i * dim..(i + 1) * dim]).unwrap();
                assert_eq!(
                    &batched[i * n_bytes..(i + 1) * n_bytes],
                    &one[..],
                    "row {i} disagrees"
                );
            }
        }
    }

    /// A given seed has to reproduce a given index. The parallel reductions in
    /// `feature_mean` and `gram_matrix` are the risk: `fold`/`reduce` splits on
    /// work stealing and float addition is not associative. Unlike
    /// `test_deterministic` this fixture is not all zeros, so the partial sums
    /// have something to disagree about.
    #[test]
    fn test_pca_fit_is_reproducible_across_runs() {
        let (n, dim, n_bits) = (30_000, 12, 64);

        let mut data = vec![0.0f32; n * dim];
        for (i, v) in data.iter_mut().enumerate() {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            *v = 30.0 + (x as f32 / u64::MAX as f32) * 4.0;
        }

        let first = Binariser::<f32>::new_pca_hashing(&data, n, dim, n_bits, 11).unwrap();
        let second = Binariser::<f32>::new_pca_hashing(&data, n, dim, n_bits, 11).unwrap();

        let (
            BinarisationMethod::PcaHashing {
                projections: p1,
                mean: m1,
            },
            BinarisationMethod::PcaHashing {
                projections: p2,
                mean: m2,
            },
        ) = (&first.method, &second.method)
        else {
            panic!("Expected PcaHashing method");
        };

        assert_eq!(m1, m2, "the fitted mean is not reproducible");
        assert_eq!(p1, p2, "the fitted projections are not reproducible");
    }

    /// A fit with fewer rows than features has a null space, and a null-space
    /// eigenvector is not a direction in the data. Every emitted bit still has
    /// to be a usable unit hyperplane.
    #[test]
    fn test_fewer_samples_than_features() {
        let (n, dim, n_bits) = (20, 64, 128);

        let mut data = vec![0.0f64; n * dim];
        for (i, v) in data.iter_mut().enumerate() {
            *v = ((i as f64) * 0.37).cos();
        }

        let binariser = Binariser::<f64>::new_pca_hashing(&data, n, dim, n_bits, 3).unwrap();
        let BinarisationMethod::PcaHashing { projections, .. } = &binariser.method else {
            panic!("Expected PcaHashing method");
        };
        assert_eq!(projections.len(), n_bits * dim);

        for i in 0..n_bits {
            let norm: f64 = projections[i * dim..(i + 1) * dim]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "projection {i} is not a unit hyperplane: {norm}"
            );
        }
    }

    #[test]
    fn test_memory_usage() {
        let dim = 32;
        let n_bits = 64;

        let zero_mean = Mat::<f64>::zeros(8, dim);
        let simhash = Binariser::<f64>::new_simhash(
            &matrix_to_flat(zero_mean.as_ref()).0,
            zero_mean.nrows(),
            dim,
            n_bits,
            42,
        )
        .unwrap();
        let simhash_mem = simhash.memory_usage_bytes();
        assert!(simhash_mem > 0);

        let sign_based = Binariser::<f64>::new_sign_based(dim);
        let sign_mem = sign_based.memory_usage_bytes();
        assert!(sign_mem > 0);
        assert!(simhash_mem > sign_mem, "SimHash should use more memory");
    }
}
