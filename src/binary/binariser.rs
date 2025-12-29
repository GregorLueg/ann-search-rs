use num_traits::{Float, FromPrimitive, ToPrimitive};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rand_distr::StandardNormal;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

///////////////
// Binariser //
///////////////

/// Binariser using random hyperplane projections
///
/// Converts float vectors to binary codes using locality-sensitive hashing.
/// Supports SimHash (for Cosine similarity) and E2LSH (for Euclidean distance).
///
/// ### Fields
///
/// * `random_projections` - Random vectors from N(0,1), flattened (n_bits * dim)
/// * `random_offsets` - Random offsets for E2LSH (None for SimHash)
/// * `bucket_width` - Bucket width for E2LSH (None for SimHash)
/// * `n_bits` - Number of bits in binary code (e.g., 256, 512)
/// * `dim` - Input vector dimensionality
pub struct Binariser<T> {
    pub random_projections: Vec<T>,
    pub n_bits: usize,
    pub dim: usize,
}

impl<T> Binariser<T>
where
    T: Float + FromPrimitive + ToPrimitive,
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
            random_projections: Vec::new(),
            n_bits,
            dim,
        };

        binariser.prepare_simhash(seed);

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

        // SimHash: sign of dot product
        for bit_idx in 0..self.n_bits {
            let proj_base = bit_idx * self.dim;

            let mut dot = T::zero();
            for d in 0..self.dim {
                dot = dot + vec[d] * self.random_projections[proj_base + d];
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
        total += self.random_projections.capacity() * std::mem::size_of::<T>();
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

        self.random_projections = random_projections;
    }
}

/////////////
// Helpers //
/////////////

/// Distance metric for 4-bit quantisation
#[derive(Clone, Debug, Copy, PartialEq, Default)]
pub enum Dist4Bit {
    /// Euclidean distance (L2)
    #[default]
    Euclidean,
    /// Cosine distance (1 - cosine_similarity)
    Cosine,
}

/// Parse distance metric from string
pub fn parse_dist_4bit(s: &str) -> Option<Dist4Bit> {
    match s.to_lowercase().as_str() {
        "euclidean" | "l2" => Some(Dist4Bit::Euclidean),
        "cosine" => Some(Dist4Bit::Cosine),
        _ => None,
    }
}

/////////////////
// Quantiser4Bit //
/////////////////

/// 4-bit asymmetric quantiser using random projections
///
/// Encodes vectors as sequences of 4-bit bucket indices (16 levels per projection).
/// Supports both Euclidean and Cosine distance metrics.
///
/// ### Fields
///
/// * `random_projections` - Unit-length random vectors, flattened (n_projections * dim)
/// * `projection_mins` - Per-projection minimum values for quantisation
/// * `projection_scales` - Per-projection scale factors: 16 / (max - min)
/// * `bucket_centres` - Centre value of each bucket per projection (n_projections * 16)
/// * `n_projections` - Number of random projections
/// * `dim` - Input vector dimensionality
/// * `metric` - Distance metric (Euclidean or Cosine)
pub struct Quantiser4Bit<T> {
    pub random_projections: Vec<T>,
    pub projection_mins: Vec<T>,
    pub projection_scales: Vec<T>,
    pub bucket_centres: Vec<T>, // Flattened: [proj_0_bucket_0..15, proj_1_bucket_0..15, ...]
    pub n_projections: usize,
    pub dim: usize,
    pub metric: Dist4Bit,
    fitted: bool,
}

impl<T> Quantiser4Bit<T>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    /// Create a new 4-bit quantiser
    ///
    /// Generates random projections. Call `fit()` before encoding.
    ///
    /// ### Params
    ///
    /// * `dim` - Input vector dimensionality
    /// * `n_projections` - Number of random projections (output = n_projections * 4 bits)
    /// * `metric` - Distance metric
    /// * `seed` - Random seed for reproducibility
    pub fn new(dim: usize, n_projections: usize, metric: Dist4Bit, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let mut random_projections: Vec<T> = Vec::with_capacity(n_projections * dim);

        for _ in 0..n_projections {
            let mut proj: Vec<T> = (0..dim)
                .map(|_| {
                    let val: f64 = rng.sample(StandardNormal);
                    T::from_f64(val).unwrap()
                })
                .collect();

            // Normalise to unit length
            let norm_sq: T = proj.iter().fold(T::zero(), |acc, &x| acc + x * x);
            let norm = norm_sq.sqrt();
            if norm > T::epsilon() {
                for x in &mut proj {
                    *x = *x / norm;
                }
            }

            random_projections.extend(proj);
        }

        Self {
            random_projections,
            projection_mins: Vec::new(),
            projection_scales: Vec::new(),
            bucket_centres: Vec::new(),
            n_projections,
            dim,
            metric,
            fitted: false,
        }
    }

    /// Fit quantiser on training data
    ///
    /// Computes per-projection min/max and bucket centres from data.
    ///
    /// ### Params
    ///
    /// * `data` - Iterator over vectors (each vector is a slice of length dim)
    /// * `n_samples` - Number of samples (for pre-allocation)
    pub fn fit<'a, I>(&mut self, data: I, n_samples: usize)
    where
        I: Iterator<Item = &'a [T]>,
        T: 'a,
    {
        let mut mins: Vec<T> = vec![T::infinity(); self.n_projections];
        let mut maxs: Vec<T> = vec![T::neg_infinity(); self.n_projections];

        let mut count = 0;
        for vec in data {
            let vec = self.maybe_normalise(vec);
            for proj_idx in 0..self.n_projections {
                let proj_value = self.compute_projection(&vec, proj_idx);
                if proj_value < mins[proj_idx] {
                    mins[proj_idx] = proj_value;
                }
                if proj_value > maxs[proj_idx] {
                    maxs[proj_idx] = proj_value;
                }
            }
            count += 1;
        }

        assert!(count > 0, "Cannot fit on empty data");

        // Compute quantisation parameters
        let margin = T::from_f64(0.001).unwrap();
        let n_buckets = T::from_usize(16).unwrap();
        let half = T::from_f64(0.5).unwrap();

        self.projection_mins = Vec::with_capacity(self.n_projections);
        self.projection_scales = Vec::with_capacity(self.n_projections);
        self.bucket_centres = Vec::with_capacity(self.n_projections * 16);

        for proj_idx in 0..self.n_projections {
            let min_val = mins[proj_idx] - margin;
            let max_val = maxs[proj_idx] + margin;
            let range = max_val - min_val;

            println!(
                "Projection {}: min={:.3}, max={:.3}, range={:.3}",
                proj_idx,
                min_val.to_f64().unwrap(),
                max_val.to_f64().unwrap(),
                range.to_f64().unwrap()
            );

            let scale = if range > T::epsilon() {
                n_buckets / range
            } else {
                T::one()
            };

            self.projection_mins.push(min_val);
            self.projection_scales.push(scale);

            // Bucket centres
            let bucket_width = range / n_buckets;
            for b in 0..16 {
                let b_t = T::from_usize(b).unwrap();
                self.bucket_centres
                    .push(min_val + (b_t + half) * bucket_width);
            }
        }

        self.fitted = true;
    }

    /// Encode a vector to 4-bit codes
    ///
    /// ### Params
    ///
    /// * `vec` - Input vector (length must equal dim)
    ///
    /// ### Returns
    ///
    /// Packed nibbles (2 codes per byte), length = ceil(n_projections / 2)
    pub fn encode(&self, vec: &[T]) -> Vec<u8> {
        assert_eq!(vec.len(), self.dim, "Vector dimension mismatch");
        assert!(self.fitted, "Quantiser not fitted");

        let vec = self.maybe_normalise(vec);

        let n_bytes = self.n_projections.div_ceil(2);
        let mut packed = vec![0u8; n_bytes];

        for proj_idx in 0..self.n_projections {
            let proj_value = self.compute_projection(&vec, proj_idx);
            let bucket = self.quantise_value(proj_value, proj_idx);

            let byte_idx = proj_idx / 2;
            if proj_idx % 2 == 0 {
                packed[byte_idx] |= bucket;
            } else {
                packed[byte_idx] |= bucket << 4;
            }
        }

        packed
    }

    /// Build lookup table for a query vector
    ///
    /// For asymmetric distance: query stays full precision, we precompute
    /// squared distance from query projection to each bucket centre.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector (length must equal dim)
    ///
    /// ### Returns
    ///
    /// Flattened LUT of shape (n_projections * 16)
    pub fn build_query_lut(&self, query: &[T]) -> Vec<T> {
        assert_eq!(query.len(), self.dim, "Query dimension mismatch");
        assert!(self.fitted, "Quantiser not fitted");

        let query = self.maybe_normalise(query);

        let mut lut = Vec::with_capacity(self.n_projections * 16);

        for proj_idx in 0..self.n_projections {
            let query_proj = self.compute_projection(&query, proj_idx);

            for b in 0..16 {
                let centre = self.bucket_centres[proj_idx * 16 + b];
                let diff = query_proj - centre;
                lut.push(diff * diff);
            }
        }

        lut
    }

    /// Returns bytes per encoded vector
    pub fn code_size(&self) -> usize {
        self.n_projections.div_ceil(2)
    }

    /// Returns memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.random_projections.capacity() * std::mem::size_of::<T>()
            + self.projection_mins.capacity() * std::mem::size_of::<T>()
            + self.projection_scales.capacity() * std::mem::size_of::<T>()
            + self.bucket_centres.capacity() * std::mem::size_of::<T>()
    }

    /// Normalise vector if using Cosine metric
    fn maybe_normalise(&self, vec: &[T]) -> Vec<T> {
        match self.metric {
            Dist4Bit::Euclidean => vec.to_vec(),
            Dist4Bit::Cosine => {
                let norm_sq: T = vec.iter().fold(T::zero(), |acc, &x| acc + x * x);
                let norm = norm_sq.sqrt();
                if norm > T::epsilon() {
                    vec.iter().map(|&x| x / norm).collect()
                } else {
                    vec.to_vec()
                }
            }
        }
    }

    #[inline]
    fn compute_projection(&self, vec: &[T], proj_idx: usize) -> T {
        let base = proj_idx * self.dim;
        vec.iter().enumerate().fold(T::zero(), |acc, (d, &v)| {
            acc + v * self.random_projections[base + d]
        })
    }

    #[inline]
    fn quantise_value(&self, value: T, proj_idx: usize) -> u8 {
        let normalised =
            (value - self.projection_mins[proj_idx]) * self.projection_scales[proj_idx];
        let bucket = normalised.to_usize().unwrap_or(0).min(15);

        // Only print first vector, first 5 projections
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let count = COUNTER.load(Ordering::Relaxed);
        if count < 5 {
            println!(
                "Vec 0, Proj {}: value={:.3}, min={:.3}, scale={:.3}, normalised={:.3}, bucket={}",
                proj_idx,
                value.to_f64().unwrap(),
                self.projection_mins[proj_idx].to_f64().unwrap(),
                self.projection_scales[proj_idx].to_f64().unwrap(),
                normalised.to_f64().unwrap(),
                bucket
            );
            COUNTER.store(count + 1, Ordering::Relaxed);
        }
        bucket as u8
    }
}
