//! Synthetic datasets with realistic structure.
//!
//! These are the generators behind the numbers in `docs/benchmarks_*.md`. They
//! used to live in `examples/commons/`, where nothing outside the repository
//! could reach them, so a published benchmark table was not reproducible by
//! anyone else. They are here so the gridsearch examples, downstream Rust
//! crates and the Python bindings all draw from one implementation and one
//! seed.
//!
//! Uniform Gaussian noise is a poor benchmark for nearest-neighbour search:
//! past a few dozen dimensions every point sits at roughly the same distance
//! from every other, so recall stops discriminating between indices. Each
//! generator here puts back a structure real single-cell data has, and each
//! one stresses a different part of an index.
//!
//! * [`generate_clustered_data`] - Separated Gaussian clusters joined by
//!   inter-cluster bridges. The baseline.
//! * [`generate_clustered_data_high_dim`] - Adds per-cluster anisotropy plus a
//!   globally shared off-axis subspace, which is the structure OPQ's rotation
//!   is built to exploit and PQ's axis-aligned split is not.
//! * [`generate_low_rank_rotated_data`] - Data on a low-dimensional manifold
//!   inside a high-dimensional ambient space, with differentiation
//!   trajectories between cell types.
//! * [`generate_cell_embeddings`] - Foundation-model embeddings in the style of
//!   Geneformer or scGPT: heavy-tailed spectrum, a handful of high-variance
//!   rogue dimensions, and a shared mean offset that puts everything in an
//!   anisotropy cone.
//!
//! Every generator returns `(data, cluster_labels)`, so ground-truth labels are
//! free for recall and for scoring a downstream clustering.
//!
//! ### Note
//!
//! The tuning constants are fixed rather than exposed as parameters. They are
//! what the published benchmark tables were produced with, and changing one
//! silently invalidates the comparison. If you need to sweep them, that wants a
//! parameter struct and a deliberate decision about which tables get regenerated.

use faer::traits::ComplexField;
use faer::Mat;
use num_traits::{Float, FromPrimitive};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::StandardNormal;

////////////
// Consts //
////////////

/// Default intrinsic dimensionality of the `LowRank` manifold.
pub const DEFAULT_INTRINSIC_DIM: usize = 16;

/// Share of structured variance routed to the global off-axis subspace, from
/// 0.0 to 1.0.
pub const DEFAULT_COR_STRENGTH: f64 = 0.5;

/// Fraction of `GaussianNoise` samples placed on inter-cluster bridges.
pub const DEFAULT_BRIDGE_FRACTION: f64 = 0.2;

/// Fraction of `LowRank` samples placed along differentiation trajectories.
pub const DEFAULT_TRAJECTORY_FRACTION: f64 = 0.15;

/// Strong directions per cluster (local anisotropy) in the `Correlated`
/// modality.
pub const DEFAULT_LOCAL_RANK: usize = 16;

/// Globally shared correlated directions (off-axis, what OPQ can exploit).
pub const DEFAULT_CORR_RANK: usize = 32;

/// Power-law decay exponent for the structured variance spectra.
pub const DEFAULT_ANISO_DECAY: f64 = 1.0;

/// Number of high-variance rogue dimensions in cell embeddings.
pub const DEFAULT_N_ROGUE: usize = 4;

/// Magnitude of the shared mean offset (anisotropy cone).
pub const DEFAULT_CONE_SHIFT: f64 = 8.0;

/// Local intrinsic rank of each cell type's variation.
pub const DEFAULT_CELL_LOCAL_RANK: usize = 10;

/// Fraction of samples placed on differentiation trajectories.
pub const DEFAULT_CELL_TRAJ_FRACTION: f64 = 0.25;

/// Gentle power-law decay: heavy-tailed but high participation ratio.
pub const DEFAULT_CELL_DECAY: f64 = 0.6;

////////////////
// Generators //
////////////////

/// Enum defining the synthetic data
#[derive(Default)]
pub enum SyntheticData {
    /// Default Gaussian noise cluster
    #[default]
    GaussianNoise,
    /// Correlated structure
    Correlated,
    /// LowRank type of data
    LowRank,
    /// Foundation-model cell embeddings (Geneformer/scGPT-like)
    CellEmbedding,
}

/// Helper function to parse the data type
///
/// ### Params
///
/// * `s` - The string to parse
///
/// ### Returns
///
/// `Option<SyntheticData>`
pub fn parse_data(s: &str) -> Option<SyntheticData> {
    match s.to_lowercase().as_str() {
        "gaussian" => Some(SyntheticData::GaussianNoise),
        "correlated" => Some(SyntheticData::Correlated),
        "lowrank" => Some(SyntheticData::LowRank),
        "cell" | "embedding" => Some(SyntheticData::CellEmbedding),
        _ => None,
    }
}

/// Generate synthetic single-cell-like data with cluster structure
///
/// Creates data with multiple Gaussian clusters to simulate clusters, cell
/// types in the data
///
/// ### Params
///
/// * `n_samples` - Number of samples (samples)
/// * `dim` - Embedding dimensionality
/// * `n_clusters` - Number of distinct clusters
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (n_samples, dim)
pub fn generate_clustered_data<T>(
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> (Mat<T>, Vec<usize>)
where
    T: Float + FromPrimitive + ComplexField,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Mat::<T>::zeros(n_samples, dim);

    let mut centres: Vec<Vec<f64>> = Vec::with_capacity(n_clusters);
    let mut cluster_stds = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        centres.push((0..dim).map(|_| rng.random_range(-7.5..7.5)).collect());
        cluster_stds.push(rng.random_range(0.5..2.5));
    }

    // bridge edges: connect each cluster to its nearest neighbour, dedup
    let mut edges: Vec<(usize, usize)> = Vec::new();
    if n_clusters >= 2 {
        for a in 0..n_clusters {
            let mut best = usize::MAX;
            let mut best_d = f64::INFINITY;
            for b in 0..n_clusters {
                if a == b {
                    continue;
                }
                let d: f64 = centres[a]
                    .iter()
                    .zip(&centres[b])
                    .map(|(x, y)| (x - y).powi(2))
                    .sum();
                if d < best_d {
                    best_d = d;
                    best = b;
                }
            }
            let edge = (a.min(best), a.max(best));
            if !edges.contains(&edge) {
                edges.push(edge);
            }
        }
    }

    let n_bridge = if edges.is_empty() {
        0
    } else {
        (n_samples as f64 * DEFAULT_BRIDGE_FRACTION) as usize
    };
    let n_blob = n_samples - n_bridge;

    // variable cluster sizes over the blob budget
    let mut assignments = Vec::with_capacity(n_blob);
    for cluster_idx in 0..n_clusters {
        let weight = rng.random_range(0.5..2.5);
        let n_in_cluster = ((n_blob as f64 * weight) / (n_clusters as f64 * 1.25)) as usize;
        assignments.extend(vec![cluster_idx; n_in_cluster]);
    }
    while assignments.len() < n_blob {
        assignments.push(rng.random_range(0..n_clusters));
    }
    assignments.shuffle(&mut rng);
    assignments.truncate(n_blob);

    let mut labels = Vec::with_capacity(n_samples);
    let mut row = 0;

    for &cluster_idx in &assignments {
        let centre = &centres[cluster_idx];
        let std = cluster_stds[cluster_idx];
        for j in 0..dim {
            let noise: f64 = rng.sample(StandardNormal);
            data[(row, j)] = T::from_f64(centre[j] + noise * std).unwrap();
        }
        labels.push(cluster_idx);
        row += 1;
    }

    // bridge points: thin Gaussian tube interpolating between connected centres
    for _ in 0..n_bridge {
        let (a, b) = edges[rng.random_range(0..edges.len())];
        let t: f64 = rng.random();
        let tube = (cluster_stds[a] + cluster_stds[b]) * 0.5 * 0.3;
        for j in 0..dim {
            let mid = (1.0 - t) * centres[a][j] + t * centres[b][j];
            let noise: f64 = rng.sample(StandardNormal);
            data[(row, j)] = T::from_f64(mid + noise * tube).unwrap();
        }
        labels.push(if t < 0.5 { a } else { b });
        row += 1;
    }

    (data, labels)
}

/// Random matrix with `rank` orthonormal columns in `dim`-space (Gram-Schmidt).
///
/// ### Params
///
/// * `dim` - Dimensionality of the data
/// * `rank` - Rank of the data
/// * `rng` - Random number generator
///
/// ### Returns
///
/// The orthonormalised matrix.
fn random_orthonormal_basis<T>(dim: usize, rank: usize, rng: &mut StdRng) -> Mat<T>
where
    T: Float + FromPrimitive + ComplexField,
{
    let r = rank.min(dim);
    let mut b = Mat::<T>::zeros(dim, r);
    for i in 0..dim {
        for j in 0..r {
            b[(i, j)] = T::from_f64(rng.sample(StandardNormal)).unwrap();
        }
    }
    for col in 0..r {
        for prev in 0..col {
            let mut dot = T::zero();
            for row in 0..dim {
                dot = dot + b[(row, col)] * b[(row, prev)];
            }
            for row in 0..dim {
                b[(row, col)] = b[(row, col)] - dot * b[(row, prev)];
            }
        }
        let mut norm_sq = T::zero();
        for row in 0..dim {
            norm_sq = norm_sq + b[(row, col)] * b[(row, col)];
        }
        let norm = norm_sq.sqrt();
        if norm > T::epsilon() {
            for row in 0..dim {
                b[(row, col)] = b[(row, col)] / norm;
            }
        }
    }
    b
}

/// Generate synthetic single-cell-like data with cluster structure and
/// off-axis correlated dimensions.
///
/// Well-separated clusters, each an arbitrarily-oriented ellipsoid (low-rank
/// power-law covariance), plus a globally-shared off-axis subspace carrying
/// inter-dimension correlation. The structured variance does not align with the
/// coordinate axes, so a learned rotation (OPQ) can recover it while axis-aligned
/// PQ cannot. `correlation_strength` splits structured variance between the
/// shared global subspace (1.0) and the cluster-local one (0.0).
///
/// ### Params
///
/// * `n_samples` - Number of samples
/// * `dim` - Embedding dimensionality
/// * `n_clusters` - Number of distinct clusters
/// * `correlation_strength` - Share of structured variance in the global
///   off-axis subspace (0.0-1.0)
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (n_samples, dim) and cluster assignments
pub fn generate_clustered_data_high_dim<T>(
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    correlation_strength: f64,
    seed: u64,
) -> (Mat<T>, Vec<usize>)
where
    T: Float + FromPrimitive + ComplexField,
{
    let mut rng = StdRng::seed_from_u64(seed);
    let mut data = Mat::<T>::zeros(n_samples, dim);

    let scale = (dim as f64).sqrt() * 2.0;
    let min_separation = scale * 0.8;

    // well-separated centres
    let mut centres: Vec<Vec<f64>> = Vec::with_capacity(n_clusters);
    for _ in 0..n_clusters {
        let centre = loop {
            let candidate: Vec<f64> = (0..dim).map(|_| rng.random_range(-scale..scale)).collect();
            let too_close = centres.iter().any(|existing: &Vec<f64>| {
                let d: f64 = candidate
                    .iter()
                    .zip(existing)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                d < min_separation.powi(2)
            });
            if !too_close {
                break candidate;
            }
        };
        centres.push(centre);
    }

    // globally-shared off-axis correlated subspace
    let corr_rank = DEFAULT_CORR_RANK.min(dim);
    let global_basis = random_orthonormal_basis::<T>(dim, corr_rank, &mut rng);
    let global_spec: Vec<f64> = (0..corr_rank)
        .map(|i| (scale / 10.0) / ((i + 1) as f64).powf(DEFAULT_ANISO_DECAY))
        .collect();

    // per-cluster oriented covariance: random basis + power-law spectrum
    let rank = DEFAULT_LOCAL_RANK.min(dim);
    let bases: Vec<Mat<T>> = (0..n_clusters)
        .map(|_| random_orthonormal_basis::<T>(dim, rank, &mut rng))
        .collect();
    let spectra: Vec<Vec<f64>> = (0..n_clusters)
        .map(|_| {
            let s = rng.random_range(0.3..1.0) * scale / 10.0;
            (0..rank)
                .map(|i| s / ((i + 1) as f64).powf(DEFAULT_ANISO_DECAY))
                .collect()
        })
        .collect();
    let floor = scale / 100.0;

    let sg = correlation_strength.clamp(0.0, 1.0).sqrt();
    let sl = (1.0 - correlation_strength.clamp(0.0, 1.0)).sqrt();

    // variable cluster sizes
    let mut assignments = Vec::new();
    for cluster_idx in 0..n_clusters {
        let weight = rng.random_range(0.5..2.5);
        let n_in_cluster = ((n_samples as f64 * weight) / (n_clusters as f64 * 1.25)) as usize;
        assignments.extend(vec![cluster_idx; n_in_cluster]);
    }
    while assignments.len() < n_samples {
        assignments.push(rng.random_range(0..n_clusters));
    }
    assignments.shuffle(&mut rng);
    assignments.truncate(n_samples);

    // sample = centre + isotropic floor + shared global component + cluster-local component
    for (i, &cluster_idx) in assignments.iter().enumerate() {
        let centre = &centres[cluster_idx];
        for j in 0..dim {
            let g: f64 = rng.sample(StandardNormal);
            data[(i, j)] = T::from_f64(centre[j] + g * floor).unwrap();
        }

        for k in 0..corr_rank {
            let z: f64 = rng.sample(StandardNormal);
            let amp = T::from_f64(z * global_spec[k] * sg).unwrap();
            for j in 0..dim {
                data[(i, j)] = data[(i, j)] + amp * global_basis[(j, k)];
            }
        }

        let basis = &bases[cluster_idx];
        let spec = &spectra[cluster_idx];
        for k in 0..rank {
            let z: f64 = rng.sample(StandardNormal);
            let amp = T::from_f64(z * spec[k] * sl).unwrap();
            for j in 0..dim {
                data[(i, j)] = data[(i, j)] + amp * basis[(j, k)];
            }
        }
    }

    (data, assignments)
}

/// Generate manifold-based data
///
/// Creates high-dimensional data that actually lives in a low-dimensional
/// subspace with rotated cluster structure.
///
/// ### Params
///
/// * `n_samples` - Number of samples
/// * `embedding_dim` - Full dimensionality (e.g., 128, 256)
/// * `intrinsic_dim` - True dimensionality of data (e.g., 16, 32)
/// * `n_clusters` - Number of clusters
/// * `seed` - Random seed
///
/// ### Returns
///
/// Matrix of shape (n_samples, embedding_dim)
pub fn generate_low_rank_rotated_data<T>(
    n_samples: usize,
    embedding_dim: usize,
    intrinsic_dim: usize,
    n_clusters: usize,
    seed: u64,
) -> (Mat<T>, Vec<usize>)
where
    T: Float + FromPrimitive + ComplexField,
{
    assert!(
        intrinsic_dim <= embedding_dim,
        "Intrinsic dim must be <= embedding dim"
    );

    let mut rng = StdRng::seed_from_u64(seed);

    // hierarchy: roots far apart, leaves clustered tightly around their root
    let n_roots = (n_clusters as f64).sqrt().ceil().max(1.0) as usize;
    let root_sep = (intrinsic_dim as f64).sqrt() * 3.0;
    let leaf_offset = root_sep * 0.25;

    let roots: Vec<Vec<f64>> = (0..n_roots)
        .map(|_| {
            (0..intrinsic_dim)
                .map(|_| rng.random_range(-root_sep..root_sep))
                .collect()
        })
        .collect();

    let leaf_root: Vec<usize> = (0..n_clusters).map(|l| l % n_roots).collect();

    let centres: Vec<Vec<f64>> = (0..n_clusters)
        .map(|leaf| {
            let root = &roots[leaf_root[leaf]];
            (0..intrinsic_dim)
                .map(|d| root[d] + rng.sample::<f64, _>(StandardNormal) * leaf_offset)
                .collect()
        })
        .collect();

    // trajectories: chain leaves within each lineage (state transitions)
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for r in 0..n_roots {
        let members: Vec<usize> = (0..n_clusters).filter(|&l| leaf_root[l] == r).collect();
        for w in members.windows(2) {
            edges.push((w[0], w[1]));
        }
    }

    let n_traj = if edges.is_empty() {
        0
    } else {
        (n_samples as f64 * DEFAULT_TRAJECTORY_FRACTION) as usize
    };
    let n_blob = n_samples - n_traj;

    let mut assignments = Vec::with_capacity(n_blob);
    for leaf in 0..n_clusters {
        assignments.extend(vec![leaf; n_blob / n_clusters]);
    }
    while assignments.len() < n_blob {
        assignments.push(rng.random_range(0..n_clusters));
    }
    assignments.shuffle(&mut rng);
    assignments.truncate(n_blob);

    let mut low = Mat::<T>::zeros(n_samples, intrinsic_dim);
    let mut labels = Vec::with_capacity(n_samples);
    let cluster_std = 0.3;
    let mut row = 0;

    for &leaf in &assignments {
        let centre = &centres[leaf];
        for j in 0..intrinsic_dim {
            let noise: f64 = rng.sample(StandardNormal);
            low[(row, j)] = T::from_f64(centre[j] + noise * cluster_std).unwrap();
        }
        labels.push(leaf);
        row += 1;
    }

    // trajectory points: quadratic Bezier with a random control-point bend
    for _ in 0..n_traj {
        let (a, b) = edges[rng.random_range(0..edges.len())];
        let ctrl: Vec<f64> = (0..intrinsic_dim)
            .map(|d| {
                0.5 * (centres[a][d] + centres[b][d])
                    + rng.sample::<f64, _>(StandardNormal) * leaf_offset * 0.5
            })
            .collect();
        let t: f64 = rng.random();
        let (u, v, w) = ((1.0 - t) * (1.0 - t), 2.0 * (1.0 - t) * t, t * t);
        for j in 0..intrinsic_dim {
            let pos = u * centres[a][j] + v * ctrl[j] + w * centres[b][j];
            let noise: f64 = rng.sample(StandardNormal);
            low[(row, j)] = T::from_f64(pos + noise * cluster_std).unwrap();
        }
        labels.push(if t < 0.5 { a } else { b });
        row += 1;
    }

    // isometric embedding: intrinsic_dim orthonormal ROWS in embedding space
    let mut rotation = Mat::<T>::zeros(intrinsic_dim, embedding_dim);
    for i in 0..intrinsic_dim {
        for j in 0..embedding_dim {
            let val: f64 = rng.sample(StandardNormal);
            rotation[(i, j)] = T::from_f64(val).unwrap();
        }
    }
    for r in 0..intrinsic_dim {
        for prev in 0..r {
            let mut dot = T::zero();
            for c in 0..embedding_dim {
                dot = dot + rotation[(r, c)] * rotation[(prev, c)];
            }
            for c in 0..embedding_dim {
                rotation[(r, c)] = rotation[(r, c)] - dot * rotation[(prev, c)];
            }
        }
        let mut norm_sq = T::zero();
        for c in 0..embedding_dim {
            norm_sq = norm_sq + rotation[(r, c)] * rotation[(r, c)];
        }
        let norm = norm_sq.sqrt();
        if norm > T::epsilon() {
            for c in 0..embedding_dim {
                rotation[(r, c)] = rotation[(r, c)] / norm;
            }
        }
    }

    let mut high = Mat::<T>::zeros(n_samples, embedding_dim);
    let noise_std = 0.01;
    for i in 0..n_samples {
        for j in 0..embedding_dim {
            let mut sum = T::zero();
            for k in 0..intrinsic_dim {
                sum = sum + low[(i, k)] * rotation[(k, j)];
            }
            let noise: f64 = rng.sample(StandardNormal);
            high[(i, j)] = sum + T::from_f64(noise * noise_std).unwrap();
        }
    }

    (high, labels)
}

/// Generate synthetic foundation-model cell embeddings
///
/// Reproduces the geometry of Geneformer/scGPT-style embeddings: a strong
/// anisotropy cone (large shared mean offset), a few axis-aligned rogue
/// dimensions that dominate dot products, per-cell-type low-rank subspaces with
/// independent orientations (full-rank globally), differentiation trajectories
/// between related types, and per-cell norm variation standing in for library
/// size. Each property targets a distinct quantisation failure mode.
///
/// ### Params
///
/// * `n_samples` - Number of cells
/// * `dim` - Embedding dimensionality (256-768 typical)
/// * `n_clusters` - Number of cell types
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (n_samples, dim) and cell-type assignments
pub fn generate_cell_embeddings<T>(
    n_samples: usize,
    dim: usize,
    n_clusters: usize,
    seed: u64,
) -> (Mat<T>, Vec<usize>)
where
    T: Float + FromPrimitive + ComplexField,
{
    let mut rng = StdRng::seed_from_u64(seed);

    // shared cone: every cell carries this offset, pushing the mean far from the
    // origin and raising pairwise cosine.
    let cone: Vec<f64> = (0..dim)
        .map(|_| rng.sample::<f64, _>(StandardNormal))
        .collect();
    let cn: f64 = cone.iter().map(|x| x * x).sum::<f64>().sqrt();
    let cone: Vec<f64> = cone
        .into_iter()
        .map(|x| x / cn * DEFAULT_CONE_SHIFT * (dim as f64).sqrt())
        .collect();

    // lineage hierarchy: roots are lineages, leaves are types within a lineage.
    let n_roots = (n_clusters as f64).sqrt().ceil().max(1.0) as usize;
    let spread = (dim as f64).sqrt();
    let roots: Vec<Vec<f64>> = (0..n_roots)
        .map(|_| {
            (0..dim)
                .map(|_| rng.random_range(-spread..spread))
                .collect()
        })
        .collect();
    let leaf_root: Vec<usize> = (0..n_clusters).map(|l| l % n_roots).collect();
    let centres: Vec<Vec<f64>> = (0..n_clusters)
        .map(|leaf| {
            let root = &roots[leaf_root[leaf]];
            (0..dim)
                .map(|d| root[d] + rng.sample::<f64, _>(StandardNormal) * spread * 0.3)
                .collect()
        })
        .collect();

    // differentiation trajectories chain leaves within a lineage.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for r in 0..n_roots {
        let members: Vec<usize> = (0..n_clusters).filter(|&l| leaf_root[l] == r).collect();
        for w in members.windows(2) {
            edges.push((w[0], w[1]));
        }
    }

    // per-type oriented low-rank covariance; the union over types is full-rank.
    let rank = DEFAULT_CELL_LOCAL_RANK.min(dim);
    let bases: Vec<Mat<T>> = (0..n_clusters)
        .map(|_| random_orthonormal_basis::<T>(dim, rank, &mut rng))
        .collect();
    let spectra: Vec<Vec<f64>> = (0..n_clusters)
        .map(|_| {
            let s = rng.random_range(0.5..1.5);
            (0..rank)
                .map(|i| s / ((i + 1) as f64).powf(DEFAULT_CELL_DECAY))
                .collect()
        })
        .collect();
    let floor = 0.05;

    // rogue dimensions: a few coordinate axes with huge variance, lineage-biased.
    let n_rogue = DEFAULT_N_ROGUE.min(dim);
    let rogue_dims: Vec<usize> = {
        let mut all: Vec<usize> = (0..dim).collect();
        all.shuffle(&mut rng);
        all.truncate(n_rogue);
        all
    };
    let rogue_scale = DEFAULT_CONE_SHIFT * (dim as f64).sqrt() * 0.5;

    let n_traj = if edges.is_empty() {
        0
    } else {
        (n_samples as f64 * DEFAULT_CELL_TRAJ_FRACTION) as usize
    };
    let n_blob = n_samples - n_traj;

    let mut assignments = Vec::with_capacity(n_blob);
    for leaf in 0..n_clusters {
        let weight = rng.random_range(0.5..2.5);
        let n = ((n_blob as f64 * weight) / (n_clusters as f64 * 1.25)) as usize;
        assignments.extend(vec![leaf; n]);
    }
    while assignments.len() < n_blob {
        assignments.push(rng.random_range(0..n_clusters));
    }
    assignments.shuffle(&mut rng);
    assignments.truncate(n_blob);

    let mut data = Mat::<T>::zeros(n_samples, dim);
    let mut labels = Vec::with_capacity(n_samples);
    let mut row = 0;

    for &leaf in &assignments {
        let centre = &centres[leaf];
        for j in 0..dim {
            let g: f64 = rng.sample(StandardNormal);
            data[(row, j)] = T::from_f64(cone[j] + centre[j] + g * floor).unwrap();
        }
        let basis = &bases[leaf];
        let spec = &spectra[leaf];
        for k in 0..rank {
            let z: f64 = rng.sample(StandardNormal);
            let amp = T::from_f64(z * spec[k]).unwrap();
            for j in 0..dim {
                data[(row, j)] = data[(row, j)] + amp * basis[(j, k)];
            }
        }
        labels.push(leaf);
        row += 1;
    }

    // trajectory cells: quadratic Bezier between connected types, local variation
    // drawn from the nearer endpoint's subspace.
    for _ in 0..n_traj {
        let (a, b) = edges[rng.random_range(0..edges.len())];
        let ctrl: Vec<f64> = (0..dim)
            .map(|d| {
                0.5 * (centres[a][d] + centres[b][d])
                    + rng.sample::<f64, _>(StandardNormal) * spread * 0.15
            })
            .collect();
        let t: f64 = rng.random();
        let (u, v, w) = ((1.0 - t) * (1.0 - t), 2.0 * (1.0 - t) * t, t * t);
        let leaf = if t < 0.5 { a } else { b };
        for j in 0..dim {
            let pos = u * centres[a][j] + v * ctrl[j] + w * centres[b][j];
            let g: f64 = rng.sample(StandardNormal);
            data[(row, j)] = T::from_f64(cone[j] + pos + g * floor).unwrap();
        }
        let basis = &bases[leaf];
        let spec = &spectra[leaf];
        for k in 0..rank {
            let z: f64 = rng.sample(StandardNormal);
            let amp = T::from_f64(z * spec[k]).unwrap();
            for j in 0..dim {
                data[(row, j)] = data[(row, j)] + amp * basis[(j, k)];
            }
        }
        labels.push(leaf);
        row += 1;
    }

    // inject rogue dims after the base embedding so they are axis-aligned.
    for i in 0..n_samples {
        let lineage = leaf_root[labels[i]] as f64;
        for (ri, &d) in rogue_dims.iter().enumerate() {
            let bias = (lineage + ri as f64) * 0.7;
            let g: f64 = rng.sample(StandardNormal);
            let val = data[(i, d)].to_f64().unwrap() + (bias + g) * rogue_scale;
            data[(i, d)] = T::from_f64(val).unwrap();
        }
    }

    // per-cell norm variation (library size / depth): lognormal whole-vector scale.
    for i in 0..n_samples {
        let g: f64 = rng.sample(StandardNormal);
        let scale = (g * 0.3).exp();
        for j in 0..dim {
            let val = data[(i, j)].to_f64().unwrap() * scale;
            data[(i, j)] = T::from_f64(val).unwrap();
        }
    }

    (data, labels)
}

///////////////
// Utilities //
///////////////

/// Randomly subsample a matrix and add fixed Gaussian noise
///
/// ### Params
///
/// * `data` - The input matrix to subsample
/// * `n_samples` - Number of samples to draw
/// * `seed` - Random seed for reproducibility
///
/// ### Returns
///
/// Matrix of shape (min(n_samples, n_rows), dim) with noise added
pub fn subsample_with_noise<T>(data: &Mat<T>, n_samples: usize, seed: u64) -> Mat<T>
where
    T: Float + FromPrimitive + ComplexField,
{
    let mut rng = StdRng::seed_from_u64(seed + 1000);
    let (n_rows, n_cols) = data.shape();

    let mut indices: Vec<usize> = (0..n_rows).collect();
    indices.shuffle(&mut rng);
    indices.truncate(n_samples.min(n_rows));

    let mut result = Mat::<T>::zeros(n_samples.min(n_rows), n_cols);

    for (i, &row_idx) in indices.iter().enumerate() {
        for j in 0..n_cols {
            let u1: f64 = rng.random();
            let u2: f64 = rng.random();
            let noise = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let noised_value = data[(row_idx, j)].to_f64().unwrap() + noise * 0.05;
            result[(i, j)] = T::from_f64(noised_value).unwrap();
        }
    }

    result
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Cheap order-sensitive digest of a matrix, for cross-language checks.
    ///
    /// The Python bindings assert the same values, which is what proves both
    /// languages see identical points for a given seed.
    fn checksum(m: &Mat<f32>) -> f64 {
        let (n, dim) = m.shape();
        let mut acc = 0.0f64;
        for i in 0..n {
            for j in 0..dim {
                acc += m[(i, j)] as f64 * ((i * dim + j) % 97 + 1) as f64;
            }
        }
        acc / (n * dim) as f64
    }

    #[test]
    fn test_generators_are_reproducible() {
        let (a, la) = generate_clustered_data::<f32>(500, 16, 4, 42);
        let (b, lb) = generate_clustered_data::<f32>(500, 16, 4, 42);
        assert_eq!(checksum(&a), checksum(&b));
        assert_eq!(la, lb);
    }

    #[test]
    fn test_seed_changes_the_draw() {
        let (a, _) = generate_clustered_data::<f32>(500, 16, 4, 42);
        let (b, _) = generate_clustered_data::<f32>(500, 16, 4, 7);
        assert_ne!(checksum(&a), checksum(&b));
    }

    #[test]
    fn test_shapes_and_labels() {
        for (data, labels) in [
            generate_clustered_data::<f32>(300, 12, 5, 1),
            generate_clustered_data_high_dim::<f32>(300, 12, 5, DEFAULT_COR_STRENGTH, 1),
            generate_low_rank_rotated_data::<f32>(300, 12, 6, 5, 1),
            generate_cell_embeddings::<f32>(300, 12, 5, 1),
        ] {
            assert_eq!(data.shape(), (300, 12));
            assert_eq!(labels.len(), 300);
            assert!(labels.iter().all(|&l| l < 5));
        }
    }

    #[test]
    fn test_all_values_are_finite() {
        for (data, _) in [
            generate_clustered_data::<f32>(200, 8, 3, 5),
            generate_clustered_data_high_dim::<f32>(200, 8, 3, DEFAULT_COR_STRENGTH, 5),
            generate_low_rank_rotated_data::<f32>(200, 8, 4, 3, 5),
            generate_cell_embeddings::<f32>(200, 8, 3, 5),
        ] {
            let (n, dim) = data.shape();
            for i in 0..n {
                for j in 0..dim {
                    assert!(data[(i, j)].is_finite());
                }
            }
        }
    }

    #[test]
    fn test_cell_embeddings_are_the_most_anisotropic() {
        // The whole point of the four modalities is that they stress different
        // things. Cell embeddings carry rogue high-variance dimensions, so
        // their variance spectrum must be far more concentrated than plain
        // clusters. If this ever flips, a generator has been broken.
        let participation = |m: &Mat<f32>| {
            let (n, dim) = m.shape();
            let vars: Vec<f64> = (0..dim)
                .map(|j| {
                    let mean = (0..n).map(|i| m[(i, j)] as f64).sum::<f64>() / n as f64;
                    (0..n)
                        .map(|i| (m[(i, j)] as f64 - mean).powi(2))
                        .sum::<f64>()
                        / n as f64
                })
                .collect();
            let sum: f64 = vars.iter().sum();
            sum * sum / vars.iter().map(|v| v * v).sum::<f64>()
        };
        let (plain, _) = generate_clustered_data::<f32>(2000, 32, 8, 42);
        let (cells, _) = generate_cell_embeddings::<f32>(2000, 32, 8, 42);
        assert!(participation(&cells) < participation(&plain));
    }

    #[test]
    fn test_subsample_is_capped_and_perturbed() {
        let (data, _) = generate_clustered_data::<f32>(100, 8, 2, 3);
        let sub = subsample_with_noise(&data, 250, 3);
        assert_eq!(sub.shape(), (100, 8));
        assert_ne!(checksum(&sub), checksum(&data));
    }

    #[test]
    fn test_print_cross_language_checksums() {
        // Printed, not asserted: the values are pinned on the Python side, and
        // this is how they get regenerated if a generator changes on purpose.
        let (a, _) = generate_clustered_data::<f32>(500, 16, 4, 42);
        let (b, _) = generate_clustered_data_high_dim::<f32>(500, 16, 4, DEFAULT_COR_STRENGTH, 42);
        let (c, _) = generate_low_rank_rotated_data::<f32>(500, 16, 8, 4, 42);
        let (d, _) = generate_cell_embeddings::<f32>(500, 16, 4, 42);
        println!("CHECKSUM clustered   {:.10}", checksum(&a));
        println!("CHECKSUM correlated  {:.10}", checksum(&b));
        println!("CHECKSUM low_rank    {:.10}", checksum(&c));
        println!("CHECKSUM cell        {:.10}", checksum(&d));
    }
}
