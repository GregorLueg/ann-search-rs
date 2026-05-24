#![allow(dead_code)]

use clap::Parser;
use faer::traits::ComplexField;
use faer::Mat;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use rand_distr::StandardNormal;
use rustc_hash::FxHashSet;
use thousands::*;

////////////
// Consts //
////////////

/// Default number of samples
pub const DEFAULT_N_SAMPLES: usize = 150_000;

/// Default number for the querying
pub const DEFAULT_N_QUERY: usize = DEFAULT_N_SAMPLES / 10;

/// Default dimensionality -> typical for single cell
pub const DEFAULT_DIM: usize = 32;

/// Number of default clusters
pub const DEFAULT_N_CLUSTERS: usize = 25;

/// Default number of neighbours
pub const DEFAULT_K: usize = 15;

/// Default random seed
pub const DEFAULT_SEED: u64 = 42;

/// Default distance metric
pub const DEFAULT_DISTANCE: &str = "euclidean";

/// Share of structured variance routed to the global off-axis subspace (0.0 to
/// 1.0).
pub const DEFAULT_COR_STRENGTH: f64 = 0.5;

/// Default data type
pub const DEFAULT_DATA: &str = "gaussian";

/// Default intrinsic dimensions
pub const DEFAULT_INTRINSIC_DIM: usize = 16;

/// Fraction of GaussianNoise samples placed on inter-cluster bridges.
pub const DEFAULT_BRIDGE_FRACTION: f64 = 0.2;

/// Fraction of LowRank samples placed along differentiation trajectories.
pub const DEFAULT_TRAJECTORY_FRACTION: f64 = 0.15;

/// Strong directions per cluster (local anisotropy) in the Correlated modality.
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

////////////
// Parser //
////////////

/// Parsing structure
///
/// ### Fields
///
/// * `n_samples` - Number of samples/samples
/// * `dim` - Number of dimensions to use
/// * `n_clusters` - Number of clusters in the data
/// * `k` - Number of neighbours to search
/// * `seed` - Random seed for reproducibility
/// * `distance` - The distance to use. One of `"euclidean"` or `"cosine"`.
/// * `data` - The data to use. One of `"gaussian"`, `"correlated"`, `"lowrank"`
///   or `"cell"`.
/// * `intrinsic_dim` - True dimensionality for the `"lowrank"` manifold data.
/// * `spectral_decay` - Currently unused (was the quantisation-stress decay exponent).
#[derive(Parser, Clone)]
pub struct Cli {
    #[arg(long, default_value_t = DEFAULT_N_SAMPLES)]
    pub n_samples: usize,

    #[arg(long, default_value_t = DEFAULT_DIM)]
    pub dim: usize,

    #[arg(long, default_value_t = DEFAULT_N_CLUSTERS)]
    pub n_clusters: usize,

    #[arg(long, default_value_t = DEFAULT_K)]
    pub k: usize,

    #[arg(long, default_value_t = DEFAULT_SEED)]
    pub seed: u64,

    #[arg(long, default_value = DEFAULT_DISTANCE)]
    pub distance: String,

    #[arg(long, default_value = DEFAULT_DATA)]
    pub data: String,

    #[arg(long, default_value_t = DEFAULT_INTRINSIC_DIM)]
    pub intrinsic_dim: usize,
}

//////////
// Data //
//////////

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

////////////////////////////
// Main dispatch function //
////////////////////////////

/// Wrapper function to generate the synthetic data to use
///
/// ### Params
///
/// * `cli` - The Cli structure with the data
///
/// ### Returns
///
/// `(syn data, cluster assignments)`
pub fn generate_data(cli: &Cli) -> (Mat<f32>, Vec<usize>) {
    let data_type = parse_data(&cli.data).unwrap_or_default();
    let res: (Mat<f32>, Vec<usize>) = match data_type {
        SyntheticData::GaussianNoise => {
            println!(">>> Using simple Gaussian cluster data. <<<");
            generate_clustered_data(cli.n_samples, cli.dim, cli.n_clusters, cli.seed)
        }
        SyntheticData::Correlated => {
            println!(">>> Using data with subspace structure and correlated features. <<<");
            generate_clustered_data_high_dim(
                cli.n_samples,
                cli.dim,
                cli.n_clusters,
                DEFAULT_COR_STRENGTH,
                cli.seed,
            )
        }
        SyntheticData::LowRank => {
            println!(">>> Using data that simulating manifold hypothesis. <<<");
            generate_low_rank_rotated_data(
                cli.n_samples,
                cli.dim,
                cli.intrinsic_dim,
                cli.n_clusters,
                cli.seed,
            )
        }
        SyntheticData::CellEmbedding => {
            println!(">>> Using foundation-model cell embedding data. <<<");
            generate_cell_embeddings(cli.n_samples, cli.dim, cli.n_clusters, cli.seed)
        }
    };

    res
}

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

////////////////
// Benchmarks //
////////////////

/// BenchmarkResult
pub struct BenchmarkResultSize {
    /// Name of the method
    pub method: String,
    /// The build time of the index in ms
    pub build_time_ms: f64,
    /// The query time of the index in ms
    pub query_time_ms: f64,
    ///  Total time the index build & query takes in ms
    pub total_time_ms: f64,
    /// Recall@k neighbours against ground truth. Overlap in top k neighbours
    /// for given k
    pub recall_at_k: f64,
    /// Mean distance ratio
    pub mean_dist_rat: f64,
    /// Size of the index
    pub index_size_mb: f64,
}

/////////////
// Helpers //
/////////////

/// Calculate Recall@k
///
/// ### Params
///
/// * `true_neighbors` - Slice of true neighbours
/// * `approx_neighbors` - Slice of the approximate neighbours
/// * `k` - Number of selected k
///
/// ### Returns
///
/// The Recall@k
pub fn calculate_recall(
    true_neighbors: &[Vec<usize>],
    approx_neighbors: &[Vec<usize>],
    k: usize,
) -> f64 {
    let mut total_recall = 0.0;

    for (true_nn, approx_nn) in true_neighbors.iter().zip(approx_neighbors.iter()) {
        let true_set: FxHashSet<_> = true_nn.iter().take(k).collect();
        let approx_set: FxHashSet<_> = approx_nn.iter().take(k).collect();

        let matches = approx_set.intersection(&true_set).count();

        total_recall += matches as f64 / k as f64;
    }

    total_recall / true_neighbors.len() as f64
}

/// Calculate mean distance ratio across queries
///
/// For each query, computes the ratio of the sum of approximate distances
/// to the sum of true distances across the top-k neighbours. A ratio of
/// 1.0 indicates perfect results; values above 1.0 indicate how much
/// worse the approximate distances are on average (e.g. 1.05 means 5%
/// worse than optimal).
///
/// This metric is stable across distance metrics and dimensionalities
/// because summing over k neighbours avoids the division-by-near-zero
/// instability that plagues per-pair relative error, particularly with
/// cosine distance. Queries where the true distance sum is negligible
/// (< 1e-12) are excluded.
///
/// ### Params
///
/// * `true_dist` - Slice of true distances to the neighbours (one vec per
///   query)
/// * `approx_dist` - Slice of approximate distances to the neighbours (one vec
///   per query)
/// * `k` - Number of neighbours to consider per query
///
/// ### Returns
///
/// The mean distance ratio (1.0 = perfect, >1.0 = proportionally worse).
/// Returns `NaN` if no queries have a non-negligible true distance sum.
pub fn calculate_mean_distance_ratio<T>(
    true_dist: &[Vec<T>],
    approx_dist: &[Vec<T>],
    k: usize,
) -> f64
where
    T: Float + ToPrimitive,
{
    let mut total_ratio = 0.0;
    let mut count = 0usize;
    for (td, ad) in true_dist.iter().zip(approx_dist.iter()) {
        let n = k.min(td.len()).min(ad.len());
        let sum_true: f64 = td[..n].iter().map(|v| v.to_f64().unwrap()).sum();
        let sum_approx: f64 = ad[..n].iter().map(|v| v.to_f64().unwrap()).sum();
        if sum_true > 1e-12 {
            total_ratio += sum_approx / sum_true;
            count += 1;
        }
    }
    total_ratio / count as f64
}

////////////
// Prints //
////////////

fn format_with_underscores(value: f64) -> String {
    let formatted = format!("{:.2}", value);
    let parts: Vec<&str> = formatted.split('.').collect();
    let int_part = parts[0].parse::<i64>().unwrap().separate_with_underscores();
    format!("{}.{}", int_part, parts[1])
}

/// Helper to print results to console
///
/// This version also returns the size of the index
///
/// ### Params
///
/// * `config` - Benchmark configuration
/// * `results` - Benchmark results to print
pub fn print_results_size(config: &str, results: &[BenchmarkResultSize]) {
    println!("\n{:=>131}", "");
    println!("Benchmark: {}", config);
    println!("{:=>131}", "");
    println!(
        "{:<50} {:>12} {:>12} {:>12} {:>12} {:>15} {:>12}",
        "Method",
        "Build (ms)",
        "Query (ms)",
        "Total (ms)",
        "Recall@k",
        "Mean dist ratio",
        "Size (MB)"
    );
    println!("{:->131}", "");
    for result in results {
        println!(
            "{:<50} {:>12} {:>12} {:>12} {:>12.4} {:>15.4} {:>12.2}",
            result.method,
            format_with_underscores(result.build_time_ms),
            format_with_underscores(result.query_time_ms),
            format_with_underscores(result.total_time_ms),
            result.recall_at_k,
            result.mean_dist_rat,
            result.index_size_mb
        );
    }
    println!("{:->131}\n", "");
}
