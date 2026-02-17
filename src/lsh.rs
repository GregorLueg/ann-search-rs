use faer::{MatRef, RowRef};
use num_traits::Float;
use rand::{prelude::*, rng};
use rand_distr::StandardNormal;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::{cell::RefCell, collections::BinaryHeap};
use thousands::*;

use crate::prelude::*;
use crate::utils::*;

// Thread-local reusable buffers for parallel query paths. The candidates vec
// and seen set can grow large (10k+) so reusing them across queries on the
// same thread avoids allocation churn. The heap is not stored here because
// it is typed by T and small (k ~ 15).
thread_local! {
    static LSH_CANDIDATES: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static LSH_SEEN_SET: RefCell<FxHashSet<usize>> = RefCell::new(FxHashSet::default());
}

////////////////
// Main index //
////////////////

/// LSH index for approximate nearest neighbour search
///
/// Uses multiple hash tables with random hyperplane projections (SimHash) to
/// partition the space. Vectors with identical hashes are stored in the same
/// bucket. Supports multi-probe querying to improve recall without additional
/// tables.
///
/// For Euclidean distance, vectors are L2-normalised before hashing so that
/// SimHash (an angular hash) provides a reasonable proxy. Actual distance
/// computations always use the original unnormalised vectors.
///
/// ### Fields
///
/// * `vectors_flat` - Original data, flattened row-major for cache efficiency
/// * `dim` - Embedding dimensionality
/// * `n` - Number of vectors
/// * `norms` - Pre-computed L2 norms for Cosine distance (empty for Euclidean)
/// * `metric` - Distance metric (Euclidean or Cosine)
/// * `hash_tables` - Maps hash values to vector indices for each table
/// * `random_vecs` - Orthogonalised random projection vectors from N(0,1)
/// * `num_tables` - Number of hash tables
/// * `bits_per_hash` - Bits in each hash code (higher = fewer collisions)
/// * `vector_hashes` - Pre-computed hashes per vector per table, layout:
///   `[table_idx * n + vec_idx]`. Used to skip re-hashing during self-query.
pub struct LSHIndex<T> {
    // main fields
    pub vectors_flat: Vec<T>,
    pub dim: usize,
    pub n: usize,
    norms: Vec<T>,
    metric: Dist,
    // index-specific
    hash_tables: Vec<FxHashMap<u64, Vec<usize>>>,
    random_vecs: Vec<T>,
    num_tables: usize,
    bits_per_hash: usize,
    // pre-computed per-vector hashes for fast self-query
    vector_hashes: Vec<u64>,
}

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

impl<T> LSHIndex<T>
where
    T: AnnSearchFloat,
{
    //////////////////////
    // Index generation //
    //////////////////////

    /// Construct a new LSH index
    ///
    /// Builds hash tables in parallel. For each table, hashes all vectors using
    /// a different set of random hyperplane projections and groups them into
    /// buckets. Stores per-vector hashes for fast self-query (kNN graph
    /// generation).
    ///
    /// For Euclidean mode, vectors are L2-normalised once and reused across all
    /// tables (avoiding redundant per-table normalisation).
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (rows = samples, columns = dimensions)
    /// * `metric` - Distance metric (Euclidean or Cosine)
    /// * `num_tables` - Number of hash tables (with multi-probe, 4-8 is
    ///   typically sufficient)
    /// * `bits_per_hash` - Bits per hash code (more = fewer collisions, smaller
    ///   buckets)
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
        seed: usize,
    ) -> Self {
        let (vectors_flat, n, dim) = matrix_to_flat(data);

        let norms = if metric == Dist::Cosine {
            (0..n)
                .map(|i| {
                    let start = i * dim;
                    T::calculate_l2_norm(&vectors_flat[start..start + dim])
                })
                .collect()
        } else {
            Vec::new()
        };

        // generate random projection vectors from N(0,1)
        let mut rng = StdRng::seed_from_u64(seed as u64);
        let total_random_vecs = num_tables * bits_per_hash * dim;
        let mut random_vecs: Vec<T> = (0..total_random_vecs)
            .map(|_| {
                let val: f64 = rng.sample(StandardNormal);
                T::from_f64(val).unwrap()
            })
            .collect();

        orthogonalise_table_projections(&mut random_vecs, num_tables, bits_per_hash, dim);

        // pre-compute L2-normalised vectors for Euclidean hashing
        let normalised: Vec<T> = if metric == Dist::Euclidean {
            let mut buf = vec![T::zero(); n * dim];
            for i in 0..n {
                let start = i * dim;
                let norm = T::calculate_l2_norm(&vectors_flat[start..start + dim]);
                if norm > T::epsilon() {
                    for d in 0..dim {
                        buf[start + d] = vectors_flat[start + d] / norm;
                    }
                }
            }
            buf
        } else {
            Vec::new()
        };

        let hash_source: &[T] = if metric == Dist::Euclidean {
            &normalised
        } else {
            &vectors_flat
        };

        // compute all hashes and build tables in parallel
        let tables_and_hashes: Vec<(FxHashMap<u64, Vec<usize>>, Vec<u64>)> = (0..num_tables)
            .into_par_iter()
            .map(|table_idx| {
                let mut table = FxHashMap::default();
                let mut hashes = Vec::with_capacity(n);

                for vec_idx in 0..n {
                    let start = vec_idx * dim;
                    let vec = &hash_source[start..start + dim];
                    let hash = compute_hash(vec, table_idx, bits_per_hash, dim, &random_vecs);
                    hashes.push(hash);
                    table.entry(hash).or_insert_with(Vec::new).push(vec_idx);
                }

                (table, hashes)
            })
            .collect();

        let mut vector_hashes = vec![0u64; num_tables * n];
        let mut hash_tables = Vec::with_capacity(num_tables);
        for (table_idx, (table, hashes)) in tables_and_hashes.into_iter().enumerate() {
            hash_tables.push(table);
            let base = table_idx * n;
            vector_hashes[base..base + n].copy_from_slice(&hashes);
        }

        Self {
            vectors_flat,
            dim,
            n,
            norms,
            metric,
            hash_tables,
            random_vecs,
            num_tables,
            bits_per_hash,
            vector_hashes,
        }
    }

    ///////////
    // Query //
    ///////////

    /// Query the index for approximate nearest neighbours
    ///
    /// Hashes the query vector and retrieves candidates from matching buckets
    /// across all tables. With `n_probes > 0`, also checks neighbouring
    /// buckets by flipping the most uncertain hash bits (multi-probe LSH).
    ///
    /// If no candidates are found across all tables and probes, falls back to
    /// random sampling.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector (must match index dimensionality)
    /// * `k` - Number of neighbours to return
    /// * `max_cand` - Optional limit on total candidates examined
    /// * `n_probes` - Number of additional buckets to probe per table (0 =
    ///   exact hash only). Good default: `bits_per_hash` (all single-bit
    ///   flips).
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
    ) -> (Vec<usize>, Vec<T>, bool) {
        assert!(
            query_vec.len() == self.dim,
            "Query vector dimensionality mismatch"
        );

        // normalise query for hashing if Euclidean
        let hash_vec;
        let hash_input = if self.metric == Dist::Euclidean {
            let norm = T::calculate_l2_norm(query_vec);
            hash_vec = if norm > T::epsilon() {
                query_vec.iter().map(|&v| v / norm).collect::<Vec<T>>()
            } else {
                vec![T::zero(); self.dim]
            };
            hash_vec.as_slice()
        } else {
            query_vec
        };

        LSH_CANDIDATES.with(|cand_cell| {
            let mut candidates = cand_cell.borrow_mut();
            candidates.clear();

            let budget = max_cand.unwrap_or(self.n);

            for table_idx in 0..self.num_tables {
                if candidates.len() >= budget {
                    break;
                }

                let (hash, projections) = compute_hash_with_projections(
                    hash_input,
                    table_idx,
                    self.bits_per_hash,
                    self.dim,
                    &self.random_vecs,
                );

                // exact bucket
                if let Some(bucket) = self.hash_tables[table_idx].get(&hash) {
                    candidates.extend_from_slice(bucket);
                }

                // multi-probe: check neighbouring buckets ordered by uncertainty
                if n_probes > 0 && candidates.len() < budget {
                    let probes =
                        generate_probes_ranked(hash, &projections, self.bits_per_hash, n_probes);
                    for probe_hash in probes {
                        if candidates.len() >= budget {
                            break;
                        }
                        if let Some(bucket) = self.hash_tables[table_idx].get(&probe_hash) {
                            candidates.extend_from_slice(bucket);
                        }
                    }
                }
            }

            let fallback_triggered = candidates.is_empty();
            if fallback_triggered {
                let mut rng = rng();
                let sample_size = 1000.min(self.n);
                candidates.extend((0..self.n).choose_multiple(&mut rng, sample_size));
            }

            let (indices, dists) = self.rank_candidates(query_vec, &candidates, k);
            (indices, dists, fallback_triggered)
        })
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
    ) -> (Vec<usize>, Vec<T>, bool) {
        assert!(
            query_row.ncols() == self.dim,
            "Query row dimensionality mismatch"
        );

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
    /// Uses pre-computed per-vector hashes to avoid re-hashing, which is the
    /// main performance advantage over calling `query()` N times. Multi-probe
    /// uses uniform bit flipping (all single-bit flips, then pairs) since
    /// projection magnitudes are not stored per vector.
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

        #[allow(unused_variables)]
        let mut missed: usize = 0;

        for (_, _, fallback) in &results {
            if *fallback {
                missed += 1;
            }
        }

        if (missed as f32) / (self.n as f32) >= 0.01 {
            println!("More than 1% of samples were not represented in the buckets.");
            println!("Please verify underlying data");
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

    /// Returns the size of the index in bytes
    ///
    /// ### Returns
    ///
    /// Number of bytes used by the index
    pub fn memory_usage_bytes(&self) -> usize {
        let mut total = std::mem::size_of_val(self);

        total += self.vectors_flat.capacity() * std::mem::size_of::<T>();
        total += self.norms.capacity() * std::mem::size_of::<T>();
        total += self.random_vecs.capacity() * std::mem::size_of::<T>();
        total += self.vector_hashes.capacity() * std::mem::size_of::<u64>();

        // hash_tables outer Vec
        total += self.hash_tables.capacity() * std::mem::size_of::<FxHashMap<u64, Vec<usize>>>();

        for table in &self.hash_tables {
            total +=
                table.capacity() * (std::mem::size_of::<u64>() + std::mem::size_of::<Vec<usize>>());

            for indices in table.values() {
                total += indices.capacity() * std::mem::size_of::<usize>();
            }
        }

        total
    }

    /// Returns the number of bits used for each hash.
    pub fn num_bits(&self) -> usize {
        self.bits_per_hash
    }

    /////////////////////
    // Private helpers //
    /////////////////////

    /// Optimised self-query using pre-computed hashes
    ///
    /// Looks up the stored hash for `vec_idx` in each table and collects
    /// candidates from matching + probed buckets. Avoids all hashing and
    /// normalisation work.
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
        LSH_CANDIDATES.with(|cand_cell| {
            let mut candidates = cand_cell.borrow_mut();
            candidates.clear();

            let budget = max_cand.unwrap_or(self.n);

            for table_idx in 0..self.num_tables {
                if candidates.len() >= budget {
                    break;
                }

                let hash = self.vector_hashes[table_idx * self.n + vec_idx];

                // exact bucket
                if let Some(bucket) = self.hash_tables[table_idx].get(&hash) {
                    candidates.extend_from_slice(bucket);
                }

                // Multi-probe with uniform bit flipping (no projection magnitudes
                // available for stored hashes)
                if n_probes > 0 && candidates.len() < budget {
                    let probes = generate_probes_uniform(hash, self.bits_per_hash, n_probes);
                    for probe_hash in probes {
                        if candidates.len() >= budget {
                            break;
                        }
                        if let Some(bucket) = self.hash_tables[table_idx].get(&probe_hash) {
                            candidates.extend_from_slice(bucket);
                        }
                    }
                }
            }

            let fallback_triggered = candidates.is_empty();
            if fallback_triggered {
                let mut rng = rng();
                let sample_size = 1000.min(self.n);
                candidates.extend((0..self.n).choose_multiple(&mut rng, sample_size));
            }

            let start = vec_idx * self.dim;
            let query_vec = &self.vectors_flat[start..start + self.dim];

            let (indices, dists) = self.rank_candidates(query_vec, &candidates, k);
            (indices, dists, fallback_triggered)
        })
    }

    /// Deduplicate candidates, compute distances, and return top-k sorted by
    /// ascending distance
    ///
    /// ### Params
    ///
    /// * `query_vec`- The query vector to compare against.
    /// * `candidates` - The candidate indices to consider.
    /// * `k` - The number of nearest neighbors to return.
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    fn rank_candidates(
        &self,
        query_vec: &[T],
        candidates: &[usize],
        k: usize,
    ) -> (Vec<usize>, Vec<T>) {
        LSH_SEEN_SET.with(|seen_cell| {
            let mut seen = seen_cell.borrow_mut();
            seen.clear();

            let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);

            match self.metric {
                Dist::Euclidean => {
                    for &idx in candidates {
                        if seen.insert(idx) {
                            let d = self.euclidean_distance_to_query(idx, query_vec);
                            let item = (OrderedFloat(d), idx);

                            if heap.len() < k {
                                heap.push(item);
                            } else if item.0 < heap.peek().unwrap().0 {
                                heap.pop();
                                heap.push(item);
                            }
                        }
                    }
                }
                Dist::Cosine => {
                    let query_norm = T::calculate_l2_norm(query_vec);
                    for &idx in candidates {
                        if seen.insert(idx) {
                            let d = self.cosine_distance_to_query(idx, query_vec, query_norm);
                            let item = (OrderedFloat(d), idx);

                            if heap.len() < k {
                                heap.push(item);
                            } else if item.0 < heap.peek().unwrap().0 {
                                heap.pop();
                                heap.push(item);
                            }
                        }
                    }
                }
            }

            let mut results: Vec<_> = heap.into_vec();
            results.sort_unstable_by(|a, b| a.0.cmp(&b.0));

            let indices = results.iter().map(|&(_, idx)| idx).collect();
            let dists = results.iter().map(|&(OrderedFloat(d), _)| d).collect();

            (indices, dists)
        })
    }
}

/////////////
// Helpers //
/////////////

/// Compute SimHash code for a vector
///
/// Each bit is the sign of the dot product with a random hyperplane. The
/// random projections are orthogonalised per table for better bucket
/// separation.
///
/// ### Params
///
/// * `vec` - Vector to hash (should be L2-normalised for Euclidean mode)
/// * `table_idx` - Which hash table (selects projection set)
/// * `bits_per_hash` - Number of bits in output hash
/// * `dim` - Dimensionality
/// * `random_vecs` - Pool of random projection vectors
///
/// ### Returns
///
/// Hash code as u64 (only lower `bits_per_hash` bits used)
#[inline]
fn compute_hash<T>(
    vec: &[T],
    table_idx: usize,
    bits_per_hash: usize,
    dim: usize,
    random_vecs: &[T],
) -> u64
where
    T: AnnSearchFloat,
{
    let mut hash: u64 = 0;
    let random_base = table_idx * bits_per_hash * dim;

    for bit_idx in 0..bits_per_hash {
        let offset = random_base + bit_idx * dim;
        let proj_vec = &random_vecs[offset..offset + dim];
        let dot = T::dot_simd(vec, proj_vec);

        if dot >= T::zero() {
            hash |= 1u64 << bit_idx;
        }
    }

    hash
}

/// Compute SimHash code and raw projection magnitudes for multi-probe
///
/// Returns both the hash and the absolute value of each projection dot
/// product, which is used to determine probe order (bits with smallest
/// |projection| are flipped first as they are most uncertain).
///
/// ### Params
///
/// * `vec` - Vector to hash
/// * `table_idx` - Which hash table
/// * `bits_per_hash` - Number of bits in output hash
/// * `dim` - Dimensionality
/// * `random_vecs` - Pool of random projection vectors
///
/// ### Returns
///
/// Tuple of `(hash, projection_magnitudes)` where magnitudes has length
/// `bits_per_hash`
#[inline]
fn compute_hash_with_projections<T>(
    vec: &[T],
    table_idx: usize,
    bits_per_hash: usize,
    dim: usize,
    random_vecs: &[T],
) -> (u64, Vec<T>)
where
    T: AnnSearchFloat,
{
    let mut hash: u64 = 0;
    let mut projections = Vec::with_capacity(bits_per_hash);
    let random_base = table_idx * bits_per_hash * dim;

    for bit_idx in 0..bits_per_hash {
        let offset = random_base + bit_idx * dim;
        let proj_vec = &random_vecs[offset..offset + dim];
        let dot = T::dot_simd(vec, proj_vec);

        projections.push(dot.abs());
        if dot >= T::zero() {
            hash |= 1u64 << bit_idx;
        }
    }

    (hash, projections)
}

/// Generate multi-probe hashes ordered by projection uncertainty
///
/// Flips bits with smallest absolute projection value first (these are the
/// most uncertain hash bits). Generates Hamming distance-1 probes first,
/// then distance-2, up to `max_probes` total.
///
/// ### Params
///
/// * `base_hash` - Original hash code
/// * `projections` - Absolute projection magnitudes per bit
/// * `bits_per_hash` - Total number of hash bits
/// * `max_probes` - Maximum number of additional bucket lookups
///
/// ### Returns
///
/// Vector of probe hashes, length <= `max_probes`
fn generate_probes_ranked<T: Float>(
    base_hash: u64,
    projections: &[T],
    bits_per_hash: usize,
    max_probes: usize,
) -> Vec<u64> {
    // Sort bit indices by ascending |projection| (most uncertain first)
    let mut bit_order: Vec<usize> = (0..bits_per_hash).collect();
    bit_order.sort_unstable_by(|&a, &b| {
        projections[a]
            .partial_cmp(&projections[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut probes = Vec::with_capacity(max_probes);

    // Hamming distance 1
    for &bit in &bit_order {
        if probes.len() >= max_probes {
            return probes;
        }
        probes.push(base_hash ^ (1u64 << bit));
    }

    // Hamming distance 2
    for (i, &bit_i) in bit_order.iter().enumerate() {
        for &bit_j in &bit_order[i + 1..] {
            if probes.len() >= max_probes {
                return probes;
            }
            probes.push(base_hash ^ (1u64 << bit_i) ^ (1u64 << bit_j));
        }
    }

    probes
}

/// Generate multi-probe hashes with uniform bit ordering
///
/// Used for self-query where projection magnitudes are not available. Flips
/// bits in index order rather than ranked by uncertainty. Still effective,
/// just slightly less targeted than the ranked variant.
///
/// ### Params
///
/// * `base_hash` - Original hash code
/// * `bits_per_hash` - Total number of hash bits
/// * `max_probes` - Maximum number of additional bucket lookups
///
/// ### Returns
///
/// Vector of probe hashes, length <= `max_probes`
fn generate_probes_uniform(base_hash: u64, bits_per_hash: usize, max_probes: usize) -> Vec<u64> {
    let mut probes = Vec::with_capacity(max_probes);

    // Hamming distance 1
    for bit in 0..bits_per_hash {
        if probes.len() >= max_probes {
            return probes;
        }
        probes.push(base_hash ^ (1u64 << bit));
    }

    // Hamming distance 2
    for i in 0..bits_per_hash {
        for j in (i + 1)..bits_per_hash {
            if probes.len() >= max_probes {
                return probes;
            }
            probes.push(base_hash ^ (1u64 << i) ^ (1u64 << j));
        }
    }

    probes
}

/// Orthogonalise random projections within each table via modified
/// Gram-Schmidt
///
/// Orthogonal projections produce more evenly distributed hash bits,
/// improving bucket balance and query performance.
///
/// ### Params
///
/// * `vecs` - The random projection vectors to orthogonalise (mutated in
///   place)
/// * `num_tables` - Number of tables for the LSH index
/// * `bits_per_hash` - Bits per hash to use
/// * `dim` - Number of dimensions
fn orthogonalise_table_projections<T>(
    vecs: &mut [T],
    num_tables: usize,
    bits_per_hash: usize,
    dim: usize,
) where
    T: Float,
{
    for table_idx in 0..num_tables {
        let base = table_idx * bits_per_hash * dim;

        for i in 0..bits_per_hash {
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

//////////////////////
// Validation trait //
//////////////////////

impl<T> KnnValidation<T> for LSHIndex<T>
where
    T: AnnSearchFloat,
{
    fn query_for_validation(&self, query_vec: &[T], k: usize) -> (Vec<usize>, Vec<T>) {
        let (indices, dist, _) = self.query(query_vec, k, None, self.bits_per_hash);
        (indices, dist)
    }

    fn n(&self) -> usize {
        self.n
    }

    fn metric(&self) -> Dist {
        self.metric
    }
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

    #[test]
    fn test_index_creation_euclidean() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.num_tables, 4);
        assert_eq!(index.bits_per_hash, 8);
        assert_eq!(index.vectors_flat.len(), 15);
        assert_eq!(index.vector_hashes.len(), 4 * 5);
    }

    #[test]
    fn test_index_creation_cosine() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Cosine, 4, 8, 42);

        assert_eq!(index.n, 5);
        assert_eq!(index.dim, 3);
        assert_eq!(index.norms.len(), 5);
        assert_eq!(index.vector_hashes.len(), 4 * 5);
    }

    #[test]
    fn test_stored_hashes_match_recomputed() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        // Recompute hash for vector 0, table 0 and compare with stored
        let vec = &index.vectors_flat[0..index.dim];
        let norm: f32 = vec.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let normalised: Vec<f32> = vec.iter().map(|&v| v / norm).collect();

        let recomputed = compute_hash(
            &normalised,
            0,
            index.bits_per_hash,
            index.dim,
            &index.random_vecs,
        );
        let stored = index.vector_hashes[0]; // table 0, vec 0

        assert_eq!(stored, recomputed);
    }

    #[test]
    fn test_basic_query_no_probes() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 3, None, 0);

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
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 3, None, 8);

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
        // With sparse high-dim data, multi-probe should find candidates that
        // exact hashing misses
        let n = 100;
        let dim = 50;
        let mat = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 100) as f32 / 100.0);

        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 2, 12, 42);

        let query = vec![0.5; dim];
        let (idx_no_probe, _, _) = index.query(&query, 10, None, 0);
        let (idx_probed, _, _) = index.query(&query, 10, None, 12);

        // Multi-probe should find at least as many candidates
        assert!(idx_probed.len() >= idx_no_probe.len());
    }

    #[test]
    fn test_query_cosine() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Cosine, 4, 8, 42);

        let query = vec![2.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 2, None, 0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 2);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_query_row() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query_mat = Mat::from_fn(1, 3, |_, j| [1.0, 0.0, 0.0][j]);
        let (indices, distances, _) = index.query_row(query_mat.row(0), 3, None, 0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_max_cand_limit() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 10, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _, _) = index.query(&query, 2, Some(3), 0);

        assert!(indices.len() <= 2);
    }

    #[test]
    fn test_k_larger_than_n() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 100, None, 0);

        assert!(indices.len() <= 5);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    #[should_panic(expected = "Query vector dimensionality mismatch")]
    fn test_dimension_mismatch() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0];
        index.query(&query, 3, None, 0);
    }

    #[test]
    #[should_panic(expected = "Query row dimensionality mismatch")]
    fn test_query_row_dimension_mismatch() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query_mat = Mat::from_fn(1, 2, |_, j| [1.0, 0.0][j]);
        index.query_row(query_mat.row(0), 3, None, 0);
    }

    #[test]
    fn test_fallback_mechanism() {
        let mat = Mat::from_fn(10, 100, |i, j| if j == i * 10 { 1.0 } else { 0.0 });

        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 2, 16, 42);

        let query = vec![1.0; 100];
        let (indices, distances, _) = index.query(&query, 3, None, 0);

        assert!(!indices.is_empty());
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_deterministic_with_seed() {
        let mat = simple_test_data();

        let index1 = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);
        let index2 = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices1, _, _) = index1.query(&query, 3, None, 4);
        let (indices2, _, _) = index2.query(&query, 3, None, 4);

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_f64_query() {
        let mat = Mat::from_fn(3, 3, |i, j| if i == j { 1.0f64 } else { 0.0f64 });
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, distances, _) = index.query(&query, 2, None, 0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 2);
        assert_eq!(indices.len(), distances.len());
    }

    #[test]
    fn test_distances_sorted() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (_, distances, _) = index.query(&query, 5, None, 8);

        for i in 1..distances.len() {
            assert!(distances[i - 1] <= distances[i]);
        }
    }

    #[test]
    fn test_query_returns_k_or_fewer() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 8, 6, 42);

        let query = vec![1.0, 0.0, 0.0];

        for k in 1..=5 {
            let (indices, distances, _) = index.query(&query, k, None, 0);
            assert!(indices.len() <= k);
            assert_eq!(indices.len(), distances.len());
        }
    }

    #[test]
    fn test_no_duplicate_results() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 8, 6, 42);

        let query = vec![1.0, 0.0, 0.0];
        let (indices, _, _) = index.query(&query, 5, None, 6);

        let mut sorted = indices.clone();
        sorted.sort_unstable();
        sorted.dedup();

        assert_eq!(indices.len(), sorted.len(), "Results contain duplicates");
    }

    #[test]
    fn test_no_duplicate_results_with_probes() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 8, 6, 42);

        let query = vec![0.5, 0.5, 0.5];
        let (indices, _, _) = index.query(&query, 5, None, 10);

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
    fn test_self_query_uses_stored_hashes() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        // Self-query for vector 0
        let (indices, distances, _) = index.self_query_at(0, 3, None, 0);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 3);
        assert_eq!(indices.len(), distances.len());
        // Vector 0 should be its own nearest neighbour
        assert!(indices.contains(&0));
    }

    #[test]
    fn test_generate_knn() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

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
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let (knn_indices, knn_dists) = index.generate_knn(2, None, 0, false, false);

        assert_eq!(knn_indices.len(), 5);
        assert!(knn_dists.is_none());
    }

    #[test]
    fn test_larger_dataset() {
        let n = 1000;
        let dim = 50;
        let mat = Mat::from_fn(n, dim, |i, j| ((i * 7 + j * 13) % 100) as f32 / 100.0);

        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 6, 10, 42);

        let query = vec![0.5; dim];
        let (indices, distances, _) = index.query(&query, 10, None, 10);

        assert!(!indices.is_empty());
        assert!(indices.len() <= 10);
        assert_eq!(indices.len(), distances.len());

        for &idx in &indices {
            assert!(idx < n);
        }
    }

    #[test]
    fn test_probe_generation_ranked() {
        let projections = vec![0.1f32, 0.9, 0.3, 0.5];
        let base_hash = 0b1010u64;

        let probes = generate_probes_ranked(base_hash, &projections, 4, 6);

        // Should flip bit 0 first (smallest projection 0.1), then bit 2
        // (0.3), etc.
        assert_eq!(probes[0], base_hash ^ (1u64 << 0));
        assert_eq!(probes[1], base_hash ^ (1u64 << 2));
        assert!(probes.len() <= 6);
    }

    #[test]
    fn test_probe_generation_uniform() {
        let base_hash = 0b101u64;

        let probes = generate_probes_uniform(base_hash, 3, 10);

        // Hamming distance 1: flip each of 3 bits
        assert_eq!(probes[0], base_hash ^ (1u64 << 0));
        assert_eq!(probes[1], base_hash ^ (1u64 << 1));
        assert_eq!(probes[2], base_hash ^ (1u64 << 2));

        // Hamming distance 2: flip pairs
        assert_eq!(probes[3], base_hash ^ (1u64 << 0) ^ (1u64 << 1));
        assert_eq!(probes[4], base_hash ^ (1u64 << 0) ^ (1u64 << 2));
        assert_eq!(probes[5], base_hash ^ (1u64 << 1) ^ (1u64 << 2));

        // 3 + 3 = 6 total probes for 3 bits
        assert_eq!(probes.len(), 6);
    }

    #[test]
    fn test_probe_respects_max() {
        let probes = generate_probes_uniform(0u64, 16, 5);
        assert_eq!(probes.len(), 5);
    }

    #[test]
    fn test_memory_usage_includes_hashes() {
        let mat = simple_test_data();
        let index = LSHIndex::new(mat.as_ref(), Dist::Euclidean, 4, 8, 42);

        let mem = index.memory_usage_bytes();
        // Should at least account for vector_hashes (4 tables * 5 vecs * 8 bytes)
        assert!(mem >= 4 * 5 * std::mem::size_of::<u64>());
    }
}
