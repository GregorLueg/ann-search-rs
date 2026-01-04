use faer::{Mat, MatRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use std::iter::Sum;

use crate::utils::dist::*;
use crate::utils::ivf_utils::*;
use crate::utils::*;

///////////////
// Binariser //
///////////////

const MAX_SAMPLES_PCA: usize = 100_000;
const ITQ_ITERATIONS: usize = 10;
const RABITQ_K_MEANS_ITER: usize = 30;

/// Initialisation of the binariser
#[derive(Default)]
pub enum BinarisationInit {
    /// Random projection with orthogonalisation
    #[default]
    RandomProjections,
    /// Iterative Quantisation
    ITQ,
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
        "itq" => Some(BinarisationInit::ITQ),
        "random" | "random_projections" => Some(BinarisationInit::RandomProjections),
        _ => None,
    }
}

/// Binariser using random hyperplane projections
///
/// Converts float vectors to binary codes using locality-sensitive hashing.
/// Supports SimHash (for Cosine similarity) and E2LSH (for Euclidean distance).
///
/// ### Fields
///
/// * `random_projections` - Random vectors from N(0,1), flattened (n_bits *
///   dim)
/// * `random_offsets` - Random offsets for E2LSH (None for SimHash)
/// * `bucket_width` - Bucket width for E2LSH (None for SimHash)
/// * `n_bits` - Number of bits in binary code (e.g., 256, 512)
/// * `dim` - Input vector dimensionality
pub struct Binariser<T> {
    pub projections: Vec<T>,
    pub n_bits: usize,
    pub mean: Vec<T>,
    pub dim: usize,
}

impl<T> Binariser<T>
where
    T: Float + FromPrimitive + ToPrimitive + ComplexField,
{
    /// Create a new binariser
    ///
    /// Generates random projections and initialises hash function parameters.
    /// SimHash orthogonalises projections for better quality.
    ///
    /// ### Params
    ///
    /// * `dim` - Input vector dimensionality
    /// * `n_bits` - Number of bits in output (must be multiple of 8)
    /// * `bucket_width` - Bucket width for E2LSH (ignored for SimHash, defaults to 4.0 if None)
    /// * `hash_func` - Hash function type (SimHash or E2LSH)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised binariser
    pub fn new(dim: usize, n_bits: usize, seed: usize) -> Self {
        assert!(n_bits % 8 == 0, "n_bits must be multiple of 8");

        let mut binariser = Binariser {
            projections: Vec::new(),
            n_bits,
            dim,
            mean: vec![T::zero(); dim],
        };

        binariser.prepare_simhash(seed);

        binariser
    }

    /// Initialise binariser using PCA followed by ITQ rotation
    ///
    /// Uses Principal Component Analysis to find the directions of maximum variance,
    /// then applies Iterative Quantisation (ITQ) to rotate these components for
    /// optimal binary quantisation. This typically produces better quality codes
    /// than random projections.
    ///
    /// ### Params
    ///
    /// * `data` - Training data matrix (n_samples × dim)
    /// * `dim` - Input vector dimensionality
    /// * `n_bits` - Number of bits in output (must be multiple of 8)
    /// * `seed` - Random seed for reproducibility
    ///
    /// ### Returns
    ///
    /// Initialised binariser with PCA+ITQ projections
    pub fn initialise_with_pca(data: MatRef<T>, dim: usize, n_bits: usize, seed: usize) -> Self {
        assert!(n_bits % 8 == 0, "n_bits must be multiple of 8");

        let mut binariser = Binariser {
            projections: Vec::new(),
            n_bits,
            dim,
            mean: vec![T::zero(); dim],
        };

        binariser.prepare_pca_with_itq(data, seed, ITQ_ITERATIONS);

        binariser
    }

    /// Encode a vector to binary
    ///
    /// Computes hash code by projecting onto random vectors and quantising.
    /// Uses different schemes based on initialised hash function.
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length must equal dim)
    ///
    /// ### Returns
    ///
    /// Binary code as Vec<u8> (length = n_bits / 8)
    pub fn encode(&self, vec: &[T]) -> Vec<u8> {
        assert_eq!(vec.len(), self.dim, "Vector dimension mismatch");

        let n_bytes = self.n_bits / 8;
        let mut binary = vec![0u8; n_bytes];

        for bit_idx in 0..self.n_bits {
            let proj_base = bit_idx * self.dim;
            let mut dot = T::zero();
            for d in 0..self.dim {
                let centered = vec[d] - self.mean[d];
                dot = dot + centered * self.projections[proj_base + d];
            }

            if dot >= T::zero() {
                let byte_idx = bit_idx / 8;
                let bit_pos = bit_idx % 8;
                binary[byte_idx] |= 1u8 << bit_pos;
            }
        }

        binary
    }

    /// Returns memory usage in bytes
    ///
    /// ### Returns
    ///
    /// Total bytes used by the binariser
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);
        total += self.projections.capacity() * std::mem::size_of::<T>();
        total
    }

    /// Generate random projections and orthogonalise them for SimHash
    ///
    /// Creates orthonormal random hyperplanes for better hash quality.
    /// Orthogonalisation via Gram-Schmidt ensures projections are independent.
    ///
    /// ### Params
    ///
    /// * `seed` - Random seed for reproducible projection generation
    fn prepare_simhash(&mut self, seed: usize) {
        let n_orthogonal = self.n_bits.min(self.dim);

        let mut rng = StdRng::seed_from_u64(seed as u64);
        let mut random_projections: Vec<T> = (0..self.n_bits * self.dim)
            .map(|_| {
                let val: f64 = rng.sample(StandardNormal);
                T::from_f64(val).unwrap()
            })
            .collect();

        // Orthogonalise the projections via Gram-Schmidt
        for i in 0..n_orthogonal {
            let i_base = i * self.dim;

            // Subtract projection onto all previous vectors
            for j in 0..i {
                let j_base = j * self.dim;
                let mut dot = T::zero();
                for d in 0..self.dim {
                    dot = dot + random_projections[i_base + d] * random_projections[j_base + d];
                }
                for d in 0..self.dim {
                    random_projections[i_base + d] =
                        random_projections[i_base + d] - dot * random_projections[j_base + d];
                }
            }

            // Normalise to unit length
            let mut norm_sq = T::zero();
            for d in 0..self.dim {
                norm_sq = norm_sq + random_projections[i_base + d] * random_projections[i_base + d];
            }
            let norm = norm_sq.sqrt();
            if norm > T::epsilon() {
                for d in 0..self.dim {
                    random_projections[i_base + d] = random_projections[i_base + d] / norm;
                }
            }
        }

        for i in n_orthogonal..self.n_bits {
            let i_base = i * self.dim;
            let mut norm_sq = T::zero();
            for d in 0..self.dim {
                norm_sq = norm_sq + random_projections[i_base + d] * random_projections[i_base + d];
            }
            let norm = norm_sq.sqrt();
            if norm > T::epsilon() {
                for d in 0..self.dim {
                    random_projections[i_base + d] = random_projections[i_base + d] / norm;
                }
            }
        }

        self.projections = random_projections;
    }

    /// Prepare PCA projections with ITQ rotation for binary quantisation
    ///
    /// Steps:
    ///
    /// 1. Sample up to MAX_SAMPLES_PCA points if dataset is large
    /// 2. Centre the data by subtracting the mean
    /// 3. Compute PCA to find principal components
    /// 4. Apply ITQ to find optimal rotation for binary quantisation
    /// 5. If n_bits > dim, add orthogonalised random projections for remaining
    ///    bits
    ///
    /// ### Params
    ///
    /// * `data` - Training data matrix (n_samples × dim)
    /// * `seed` - Random seed for sampling and random projections
    /// * `itq_iterations` - Number of ITQ iterations (defaults to 10)
    fn prepare_pca_with_itq(&mut self, data: MatRef<T>, seed: usize, itq_iterations: usize) {
        let n = data.nrows();
        let dim = data.ncols();
        let mut rng = StdRng::seed_from_u64(seed as u64);

        let sample_indices: Vec<usize> = if n > MAX_SAMPLES_PCA {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.shuffle(&mut rng);
            idx.into_iter().take(MAX_SAMPLES_PCA).collect()
        } else {
            (0..n).collect()
        };
        let n_samples = sample_indices.len();

        let mut sampled_data = Mat::<T>::zeros(n_samples, dim);
        let mut mean = vec![T::zero(); dim];

        for &old_idx in &sample_indices {
            for d in 0..dim {
                mean[d] = mean[d] + data[(old_idx, d)];
            }
        }
        let n_samples_t = T::from_usize(n_samples).unwrap();
        for d in 0..dim {
            mean[d] = mean[d] / n_samples_t;
        }

        for (i, &old_idx) in sample_indices.iter().enumerate() {
            for d in 0..dim {
                sampled_data[(i, d)] = data[(old_idx, d)] - mean[d];
            }
        }
        self.mean = mean;

        let svd = sampled_data.as_ref().svd().unwrap();
        let full_v = svd.V();

        let n_pca_bits = self.n_bits.min(dim);
        let mut v_pc = Mat::<T>::zeros(dim, n_pca_bits);
        for j in 0..n_pca_bits {
            for i in 0..dim {
                v_pc[(i, j)] = full_v[(i, j)];
            }
        }

        let projected_data = &sampled_data * &v_pc;

        let mut r_mat = Mat::<T>::zeros(n_pca_bits, n_pca_bits);
        for i in 0..n_pca_bits {
            for j in 0..n_pca_bits {
                let val: f64 = rng.sample(StandardNormal);
                r_mat[(i, j)] = T::from_f64(val).unwrap();
            }
        }
        // QR decomposition to get orthogonal matrix
        let qr = r_mat.as_ref().qr();
        r_mat = qr.compute_Q();

        for _ in 0..itq_iterations {
            let rotated = &projected_data * &r_mat;
            let mut b_mat = Mat::<T>::zeros(n_samples, n_pca_bits);
            for i in 0..n_samples {
                for j in 0..n_pca_bits {
                    b_mat[(i, j)] = if rotated[(i, j)] >= T::zero() {
                        T::one()
                    } else {
                        -T::one()
                    };
                }
            }

            let c_mat = projected_data.transpose() * &b_mat;
            let svd_itq = c_mat.as_ref().thin_svd().unwrap();

            r_mat = svd_itq.U() * svd_itq.V().transpose();
        }

        let final_projections_mat = v_pc * r_mat;
        let mut projections = Vec::with_capacity(self.n_bits * dim);

        for j in 0..n_pca_bits {
            for i in 0..dim {
                projections.push(final_projections_mat[(i, j)]);
            }
        }

        if self.n_bits > dim {
            for _ in n_pca_bits..self.n_bits {
                let mut proj = Vec::with_capacity(dim);
                for _ in 0..dim {
                    let val: f64 = rng.sample(StandardNormal);
                    proj.push(T::from_f64(val).unwrap());
                }

                // Just normalise
                let mut norm_sq = T::zero();
                for d in 0..dim {
                    norm_sq = norm_sq + proj[d] * proj[d];
                }
                let norm = norm_sq.sqrt();
                if norm > T::epsilon() {
                    for d in 0..dim {
                        proj[d] = proj[d] / norm;
                    }
                }

                projections.extend(proj);
            }
        }

        self.projections = projections;
    }
}

/////////////////////
// RaBitQQuantiser //
/////////////////////

/// Encoded query for RaBitQ distance estimation
///
/// ### Fields
///
/// * `quantised` - Int4 quantised values (one per dimension, stored as u8)
/// * `dist_to_centroid` - Distance from query to centroid
/// * `lower` - Lower bound used in quantisation
/// * `width` - Bucket width used in quantisation
/// * `sum_quantised` - Sum of all quantised values
#[repr(C)]
pub struct RaBitQQuery<T> {
    pub quantised: Vec<u8>,
    pub dist_to_centroid: T,
    pub lower: T,
    pub width: T,
    pub sum_quantised: u32,
}

/// Per-cluster RaBitQ data
///
/// ### Fields
///
/// * `centroid` - The centroid of the this cluster
/// * `vector_indices` - The indices of this cluster
/// * `binary_codes` - The binarised codes of this cluster
/// * `dist_to_centroid` - The distance to the centroid
/// * `dot_corrections` - The dot corrections of this cluster
pub struct RaBitQCluster<T> {
    pub centroid: Vec<T>,
    pub vector_indices: Vec<usize>,
    pub binary_codes: Vec<u8>,
    pub dist_to_centroid: Vec<T>,
    pub dot_corrections: Vec<T>,
}

impl<T: Float> RaBitQCluster<T> {
    /// Generate a new cluster
    ///
    /// ### Params
    ///
    /// * `centroid` - The centroid of this cluster
    ///
    /// ### Returns
    ///
    /// Initialised self
    fn new(centroid: Vec<T>) -> Self {
        Self {
            centroid,
            vector_indices: Vec::new(),
            binary_codes: Vec::new(),
            dist_to_centroid: Vec::new(),
            dot_corrections: Vec::new(),
        }
    }

    /// Returns the number of vectors stored
    ///
    /// ### Returns
    ///
    /// Number of vectors in this cluster
    fn n_vectors(&self) -> usize {
        self.vector_indices.len()
    }
}

/// RaBitQ quantiser with multi-centroid support
///
/// ### Fields
///
/// * `clusters` - The stored RaBitQ clusters per rotation
/// * `rotations` - The rotation matrix
/// * `dim` - Dimensionality of the data set
/// * `n_bytes` - Number of used bytes
/// * `metric` - Used distance metric
pub struct RaBitQQuantiser<T> {
    pub clusters: Vec<RaBitQCluster<T>>,
    pub rotation: Vec<T>,
    pub dim: usize,
    pub n_bytes: usize,
    pub metric: Dist,
}

impl<T> RaBitQQuantiser<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField,
{
    /// Create a new multi-centroid RaBitQ quantiser
    ///
    /// ### Params
    ///
    /// * `data` - Training data matrix (n_samples × dim)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `n_clusters` - Number of clusters. If None, uses 0.5 * sqrt(n)
    /// * `seed` - Random seed
    ///
    /// ### Returns
    ///
    /// Initialised quantiser with encoded vectors
    pub fn new(data: MatRef<T>, metric: &Dist, n_clusters: Option<usize>, seed: usize) -> Self {
        let (mut data_flat, n, dim) = matrix_to_flat(data);

        let n_bytes = dim.div_ceil(8);

        // determine number of clusters
        let k = n_clusters
            .unwrap_or_else(|| ((n as f64).sqrt() * 0.5).ceil() as usize)
            .max(1)
            .min(n);

        let mut data_norms = Vec::with_capacity(n);

        for i in 0..n {
            let row = data.row(i);
            let vec: Vec<T> = row.iter().cloned().collect();

            let norm = compute_norm(&vec);
            data_norms.push(norm);

            // for cosine, store normalised vectors for clustering
            match metric {
                Dist::Cosine => {
                    if norm > T::epsilon() {
                        data_flat.extend(vec.iter().map(|&x| x / norm));
                    } else {
                        data_flat.extend(vec);
                    }
                }
                Dist::Euclidean => {
                    data_flat.extend(vec);
                }
            }
        }

        // recompute norms after normalisation for cosine (they're all ~1.0 now)
        let cluster_norms = if matches!(metric, Dist::Cosine) {
            vec![T::one(); n]
        } else {
            data_norms.clone()
        };

        // train centroids
        let centroids_flat = train_centroids(
            &data_flat,
            dim,
            n,
            k,
            metric,
            RABITQ_K_MEANS_ITER,
            seed,
            false,
        );

        // compute centroid norms for assignment
        let centroid_norms: Vec<T> = (0..k)
            .map(|c| {
                let cent = &centroids_flat[c * dim..(c + 1) * dim];
                cent.iter()
                    .map(|&x| x * x)
                    .fold(T::zero(), |a, b| a + b)
                    .sqrt()
            })
            .collect();

        // assign vectors to clusters
        let assignments = assign_all_parallel(
            &data_flat,
            &cluster_norms,
            dim,
            n,
            &centroids_flat,
            &centroid_norms,
            k,
            metric,
        );

        // generate shared rotation matrix
        let rotation = Self::generate_random_orthogonal(dim, seed as u64);

        // build clusters
        let mut clusters: Vec<RaBitQCluster<T>> = (0..k)
            .map(|c| {
                let centroid = centroids_flat[c * dim..(c + 1) * dim].to_vec();
                RaBitQCluster::new(centroid)
            })
            .collect();

        // Encode each vector into its assigned cluster
        for (vec_idx, &cluster_idx) in assignments.iter().enumerate() {
            let vec = &data_flat[vec_idx * dim..(vec_idx + 1) * dim];
            let cluster = &mut clusters[cluster_idx];

            let (binary, dist, dot_corr) =
                Self::encode_vec_internal(vec, &cluster.centroid, &rotation, dim);

            cluster.vector_indices.push(vec_idx);
            cluster.binary_codes.extend(binary);
            cluster.dist_to_centroid.push(dist);
            cluster.dot_corrections.push(dot_corr);
        }

        Self {
            clusters,
            rotation,
            dim,
            n_bytes,
            metric: *metric,
        }
    }

    /// Encode a vector relative to a specific centroid
    ///
    /// Function to encode the index-internal data.
    ///
    /// ### Params
    ///
    /// * `vec` - The vector to encode.
    /// * `centroid` - The centroid of the cluster.
    /// * `rotation` - The rotations to apply.
    /// * `dim` - The dimensions of the data.
    ///
    /// ### Returns
    ///
    /// The `(binarised code, dist to centroid, correction for the dot product)`
    #[inline]
    fn encode_vec_internal(
        vec: &[T],
        centroid: &[T],
        rotation: &[T],
        dim: usize,
    ) -> (Vec<u8>, T, T) {
        // compute residual
        let res: Vec<T> = vec
            .iter()
            .zip(centroid.iter())
            .map(|(&v, &c)| v - c)
            .collect();

        let dist_to_centroid = compute_norm(&res);

        // normalise residual to unit vector
        let v_c: Vec<T> = if dist_to_centroid > T::epsilon() {
            res.iter().map(|&r| r / dist_to_centroid).collect()
        } else {
            vec![T::zero(); dim]
        };

        // apply rotation
        let v_c_rotated = Self::apply_rotation(&v_c, rotation, dim);

        // binary encode
        let n_bytes = dim.div_ceil(8);
        let mut binary = vec![0u8; n_bytes];
        for d in 0..dim {
            if v_c_rotated[d] >= T::zero() {
                binary[d / 8] |= 1u8 << (d % 8);
            }
        }

        // dot correction: L1 norm of rotated unit residual
        let dot_correction: T = v_c_rotated
            .iter()
            .map(|&x| x.abs())
            .fold(T::zero(), |a, b| a + b);

        (binary, dist_to_centroid, dot_correction)
    }

    /// Find nearest clusters to a query vector
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    /// * `n_probe` - Number of clusters to return
    ///
    /// ### Returns
    ///
    /// Indices of nearest clusters, sorted by distance
    #[inline]
    pub fn find_nearest_clusters(&self, query: &[T], n_probe: usize) -> Vec<usize> {
        let query_norm: Vec<T> = match self.metric {
            Dist::Cosine => {
                let norm = compute_norm(query);
                if norm > T::epsilon() {
                    query.iter().map(|&x| x / norm).collect()
                } else {
                    query.to_vec()
                }
            }
            Dist::Euclidean => query.to_vec(),
        };

        let mut dists: Vec<(T, usize)> = self
            .clusters
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let d: T = query_norm
                    .iter()
                    .zip(c.centroid.iter())
                    .map(|(&q, &c)| (q - c) * (q - c))
                    .fold(T::zero(), |a, b| a + b);
                (d, i)
            })
            .collect();

        let n_probe = n_probe.min(self.clusters.len());

        if n_probe < self.clusters.len() {
            dists.select_nth_unstable_by(n_probe - 1, |a, b| a.0.partial_cmp(&b.0).unwrap());
            dists.truncate(n_probe);
        }

        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        dists.into_iter().map(|(_, i)| i).collect()
    }

    /// Encode a query vector relative to a specific cluster
    ///
    /// ### Params
    ///
    /// * `query` - Query vector
    /// * `cluster_idx` - Which cluster's centroid to use
    ///
    /// ### Returns
    ///
    /// Encoded query for distance estimation
    #[inline]
    pub fn encode_query(&self, query: &[T], cluster_idx: usize) -> RaBitQQuery<T> {
        let centroid = &self.clusters[cluster_idx].centroid;

        let query_norm: Vec<T> = match self.metric {
            Dist::Cosine => {
                let norm = compute_norm(query);
                if norm > T::epsilon() {
                    query.iter().map(|&x| x / norm).collect()
                } else {
                    query.to_vec()
                }
            }
            Dist::Euclidean => query.to_vec(),
        };

        // residual relative to cluster centroid
        let res: Vec<T> = query_norm
            .iter()
            .zip(centroid.iter())
            .map(|(&q, &c)| q - c)
            .collect();

        let dist_to_centroid = compute_norm(&res);

        // normalise residual
        let q_c: Vec<T> = if dist_to_centroid > T::epsilon() {
            res.iter().map(|&r| r / dist_to_centroid).collect()
        } else {
            vec![T::zero(); self.dim]
        };

        // apply rotation
        let q_c_rotated = Self::apply_rotation(&q_c, &self.rotation, self.dim);

        // scalar quantise to int4
        let mut lower = q_c_rotated[0];
        let mut upper = q_c_rotated[0];
        for d in 1..self.dim {
            if q_c_rotated[d] < lower {
                lower = q_c_rotated[d];
            }
            if q_c_rotated[d] > upper {
                upper = q_c_rotated[d];
            }
        }

        let range = upper - lower;
        let width = if range > T::epsilon() {
            range / T::from_f32(15.0).unwrap()
        } else {
            T::one()
        };

        let mut quantised = vec![0u8; self.dim];
        let mut sum_quantised: u32 = 0;

        for d in 0..self.dim {
            let val = ((q_c_rotated[d] - lower) / width)
                .round()
                .to_u8()
                .unwrap_or(0)
                .min(15);
            quantised[d] = val;
            sum_quantised += val as u32;
        }

        RaBitQQuery {
            quantised,
            dist_to_centroid,
            lower,
            width,
            sum_quantised,
        }
    }

    /// Number of clusters
    ///
    /// ### Returns
    ///
    /// The number of clusters
    pub fn n_clusters(&self) -> usize {
        self.clusters.len()
    }

    /// Total number of vectors across all clusters
    ///
    /// ### Returns
    ///
    /// Total number of vectors stored in the quantiser
    pub fn n_vectors(&self) -> usize {
        self.clusters.iter().map(|c| c.n_vectors()).sum()
    }

    /// Memory usage in bytes
    ///
    /// ### Returns
    ///
    /// The memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        let base = std::mem::size_of_val(self);
        let rotation = self.rotation.capacity() * std::mem::size_of::<T>();
        let clusters: usize = self
            .clusters
            .iter()
            .map(|c| {
                c.centroid.capacity() * std::mem::size_of::<T>()
                    + c.vector_indices.capacity() * std::mem::size_of::<usize>()
                    + c.binary_codes.capacity()
                    + c.dist_to_centroid.capacity() * std::mem::size_of::<T>()
                    + c.dot_corrections.capacity() * std::mem::size_of::<T>()
            })
            .sum();
        base + rotation + clusters
    }

    /// Generate a random orthogonal matrix
    ///
    /// ### Params
    ///
    /// * `dim` - The dimensions of the rotation matrix
    /// * `seed` - Seed for reproducibility
    ///
    /// ### Returns
    ///
    /// A flattened orthogonal rotation matrix
    fn generate_random_orthogonal(dim: usize, seed: u64) -> Vec<T> {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut mat = Mat::<T>::zeros(dim, dim);
        for i in 0..dim {
            for j in 0..dim {
                let val: f64 = rng.sample(StandardNormal);
                mat[(i, j)] = T::from_f64(val).unwrap();
            }
        }

        let qr = mat.as_ref().qr();
        let q = qr.compute_Q();

        let mut rotation = Vec::with_capacity(dim * dim);
        for i in 0..dim {
            for j in 0..dim {
                rotation.push(q[(i, j)]);
            }
        }
        rotation
    }

    /// Apply rotation to a vector
    ///
    /// ### Params
    ///
    /// * `vec` - The vector to which to apply the rotation.
    /// * `rotation` - The rotation matrix to apply to the vector
    /// * `dim` - Dimensions of the data
    ///
    /// ### Returns
    ///
    /// The vector with rotation applied
    #[inline]
    fn apply_rotation(vec: &[T], rotation: &[T], dim: usize) -> Vec<T> {
        let mut rotated = vec![T::zero(); dim];
        for i in 0..dim {
            let base = i * dim;
            let mut sum = T::zero();
            for j in 0..dim {
                sum = sum + rotation[base + j] * vec[j];
            }
            rotated[i] = sum;
        }
        rotated
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
        let binariser = Binariser::<f64>::new(dim, n_bits, 42);

        let vec1: Vec<f64> = (0..dim).map(|i| (i as f64) / (dim as f64)).collect();
        let binary = binariser.encode(&vec1);

        assert_eq!(binary.len(), n_bits / 8);
    }

    #[test]
    fn test_simhash_preserves_similarity() {
        let dim = 64;
        let n_bits = 128;
        let binariser = Binariser::<f64>::new(dim, n_bits, 42);

        let vec1: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let vec2: Vec<f64> = (0..dim).map(|i| i as f64 + 0.1).collect();
        let vec3: Vec<f64> = (0..dim).map(|i| -(i as f64)).collect();

        let bin1 = binariser.encode(&vec1);
        let bin2 = binariser.encode(&vec2);
        let bin3 = binariser.encode(&vec3);

        let dist_12 = hamming_distance(&bin1, &bin2);
        let dist_13 = hamming_distance(&bin1, &bin3);

        assert!(
            dist_12 < dist_13,
            "Similar vectors should have smaller Hamming distance"
        );
    }

    #[test]
    fn test_pca_itq_basic() {
        let n_samples = 1000;
        let dim = 64;
        let n_bits = 128;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = ((i + j) as f64).sin();
            }
        }

        let binariser = Binariser::<f64>::initialise_with_pca(data.as_ref(), dim, n_bits, 42);

        let vec1: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
        let binary = binariser.encode(&vec1);

        assert_eq!(binary.len(), n_bits / 8);
    }

    #[test]
    fn test_pca_itq_orthogonality() {
        let n_samples = 500;
        let dim = 32;
        let n_bits = 128;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = ((i * j) as f64).sin();
            }
        }

        let binariser = Binariser::<f64>::initialise_with_pca(data.as_ref(), dim, n_bits, 42);

        for i in 0..n_bits.min(dim) {
            let i_base = i * dim;
            let mut norm_sq = 0.0;
            for d in 0..dim {
                norm_sq += binariser.projections[i_base + d] * binariser.projections[i_base + d];
            }
            let norm = norm_sq.sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-6,
                "Projection {} not normalised: {}",
                i,
                norm
            );

            for j in (i + 1)..n_bits.min(dim) {
                let j_base = j * dim;
                let mut dot = 0.0;
                for d in 0..dim {
                    dot += binariser.projections[i_base + d] * binariser.projections[j_base + d];
                }
                assert!(
                    dot.abs() < 1e-6,
                    "Projections {} and {} not orthogonal: {}",
                    i,
                    j,
                    dot
                );
            }
        }
    }

    #[test]
    fn test_centering() {
        let n_samples = 100;
        let dim = 16;
        let n_bits = 32;

        let mut data = Mat::<f64>::zeros(n_samples, dim);
        for i in 0..n_samples {
            for j in 0..dim {
                data[(i, j)] = (i as f64) + 10.0;
            }
        }

        let binariser = Binariser::<f64>::initialise_with_pca(data.as_ref(), dim, n_bits, 42);

        for d in 0..dim {
            let expected_mean = (n_samples as f64 - 1.0) / 2.0 + 10.0;
            assert!((binariser.mean[d] - expected_mean).abs() < 1e-6);
        }
    }

    #[test]
    fn test_deterministic() {
        let dim = 32;
        let n_bits = 64;

        let binariser1 = Binariser::<f64>::new(dim, n_bits, 42);
        let binariser2 = Binariser::<f64>::new(dim, n_bits, 42);

        let vec: Vec<f64> = (0..dim).map(|i| i as f64).collect();
        let bin1 = binariser1.encode(&vec);
        let bin2 = binariser2.encode(&vec);

        assert_eq!(bin1, bin2);
    }

    #[test]
    fn test_parse_binarisation_init() {
        assert!(matches!(
            parse_binarisation_init("itq"),
            Some(BinarisationInit::ITQ)
        ));
        assert!(matches!(
            parse_binarisation_init("ITQ"),
            Some(BinarisationInit::ITQ)
        ));
        assert!(matches!(
            parse_binarisation_init("random"),
            Some(BinarisationInit::RandomProjections)
        ));
        assert!(matches!(
            parse_binarisation_init("random_projections"),
            Some(BinarisationInit::RandomProjections)
        ));
        assert!(parse_binarisation_init("invalid").is_none());
    }

    #[test]
    #[should_panic(expected = "n_bits must be multiple of 8")]
    fn test_invalid_n_bits() {
        let _binariser = Binariser::<f64>::new(64, 123, 42);
    }

    #[test]
    #[should_panic(expected = "Vector dimension mismatch")]
    fn test_dimension_mismatch() {
        let binariser = Binariser::<f64>::new(64, 128, 42);
        let wrong_vec: Vec<f64> = vec![0.0; 32];
        let _binary = binariser.encode(&wrong_vec);
    }
}
