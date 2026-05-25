//! Exhaustive TurboQuant index.
//!
//! Flat (non-clustered) index: every vector is TurboQuant-encoded once, and
//! each query scans all of them. There is no IVF routing here — TurboQuant's
//! data-oblivious rotation means a query touches the whole set, so the hot
//! path is the block-fused SIMD scan in `tq_search` rather than a per-vector
//! loop. 2-bit and 4-bit codes use that scan; 3-bit has no SIMD kernel and
//! falls back to the scalar bit-plane scorer.
//!
//! An optional on-disk vector store keeps the original (un-quantised) floats
//! so candidates can be re-ranked with exact distances.

use bytemuck::Pod;
use faer::{MatRef, RowRef};
use faer_traits::ComplexField;
use num_traits::{Float, FromPrimitive, ToPrimitive};
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::iter::Sum;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use thousands::*;

use crate::binary::turboquant::{
    dists::{prepare_scalar_scoring, VectorDistanceTurboQuant},
    pack::{repack, BlockedLayout},
    quantiser::{TurboQuantEncoder, TurboQuantQuantiser, TurboQuantStorage},
    search::{build_query_lut, topk_blocked, QueryLut},
};
use crate::binary::vec_store::*;
use crate::prelude::*;

/// Exhaustive TurboQuant index.
///
/// Build once, then query or generate a full kNN graph. Immutable after
/// construction — no add/remove.
pub struct TurboQuantExhaustive<T> {
    /// Encoder + flat bit-plane storage.
    quantiser: TurboQuantQuantiser<T>,
    /// SIMD-blocked codes for 2-bit / 4-bit, as `(data, n_blocks)`. `None`
    /// for 3-bit, which scores through the scalar bit-plane path instead.
    blocked: Option<(Vec<u8>, usize)>,
    /// Per-vector L2 norms pre-cast to f32 for the SIMD kernels.
    norms_f32: Vec<f32>,
    /// Per-vector debias factors `1 / <u, x̂>` pre-cast to f32.
    corrections_f32: Vec<f32>,
    /// Number of stored vectors.
    n: usize,
    /// Optional on-disk original vectors for exact reranking.
    vector_store: Option<MmapVectorStore<T>>,
}

//////////////////////////////
// VectorDistanceTurboQuant //
//////////////////////////////

impl<T> VectorDistanceTurboQuant<T> for TurboQuantExhaustive<T>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    fn storage(&self) -> &TurboQuantStorage<T> {
        &self.quantiser.storage
    }

    fn encoder(&self) -> &TurboQuantEncoder<T> {
        &self.quantiser.encoder
    }
}

impl<T> TurboQuantExhaustive<T>
where
    T: Float + FromPrimitive + ToPrimitive + Send + Sync + Sum + ComplexField + SimdDistance + Pod,
{
    /// Build an exhaustive TurboQuant index.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n × dim)
    /// * `metric` - Distance metric (`Cosine` or `SquaredEuclidean`)
    /// * `bits` - Bits per coordinate (2, 3, or 4)
    /// * `seed` - Random seed for the rotation
    ///
    /// ### Returns
    ///
    /// Initialised index, or `DistanceNotSupported` for `Manhattan`.
    pub fn new(
        data: MatRef<T>,
        metric: &Dist,
        bits: usize,
        seed: usize,
    ) -> Result<Self, AnnSearchErrors> {
        if *metric == Dist::Manhattan {
            return Err(AnnSearchErrors::DistanceNotSupported(*metric));
        }

        let quantiser = TurboQuantQuantiser::new(data, metric, bits, seed as u64)?;
        let n = quantiser.n_vectors();
        let bits = quantiser.storage.bits;

        // 2/4-bit -> SIMD-blocked layout
        // 3-bit stays bit-plane.
        let blocked = if bits == 3 {
            None
        } else {
            match repack(&quantiser.storage)? {
                BlockedLayout::Standard(b) => Some((b.data, b.n_blocks)),
                BlockedLayout::ThreeBit(_) => {
                    unreachable!("repack returned ThreeBit layout for non-3-bit codes")
                }
            }
        };

        let norms_f32 = quantiser
            .storage
            .norms
            .iter()
            .map(|x| x.to_f32().unwrap())
            .collect();
        let corrections_f32 = quantiser
            .storage
            .corrections
            .iter()
            .map(|x| x.to_f32().unwrap())
            .collect();

        Ok(Self {
            quantiser,
            blocked,
            norms_f32,
            corrections_f32,
            n,
            vector_store: None,
        })
    }

    /// Build an exhaustive TurboQuant index with an on-disk vector store for
    /// exact reranking.
    ///
    /// ### Params
    ///
    /// * `data` - Data matrix (n × dim)
    /// * `metric` - Distance metric (`Cosine` or `SquaredEuclidean`)
    /// * `bits` - Bits per coordinate (2, 3, or 4)
    /// * `seed` - Random seed for the rotation
    /// * `save_path` - Directory for the persisted vectors
    ///
    /// ### Returns
    ///
    /// Initialised index with reranking enabled.
    pub fn new_with_vector_store(
        data: MatRef<T>,
        metric: &Dist,
        bits: usize,
        seed: usize,
        save_path: impl AsRef<Path>,
    ) -> Result<Self, AnnSearchErrors> {
        let mut index = Self::new(data, metric, bits, seed)?;

        let (vectors_flat, n, dim) = matrix_to_flat(data);
        let norms: Vec<T> = (0..n)
            .map(|i| compute_l2_norm(&vectors_flat[i * dim..(i + 1) * dim]))
            .collect();

        std::fs::create_dir_all(&save_path)?;
        let vectors_path = save_path.as_ref().join("vectors_flat.bin");
        let norms_path = save_path.as_ref().join("norms.bin");

        MmapVectorStore::save(&vectors_flat, &norms, dim, n, &vectors_path, &norms_path)?;
        index.vector_store = Some(MmapVectorStore::new(vectors_path, norms_path, dim, n)?);

        Ok(index)
    }

    /// Approximate k nearest neighbours for a query.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted nearest-first. Distances are the
    /// TurboQuant-reconstructed estimates (cosine or squared-Euclidean).
    pub fn query(
        &self,
        query_vec: &[T],
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let k = k.min(self.n);
        if k == 0 {
            return Ok((Vec::new(), Vec::new()));
        }

        let bits = self.quantiser.storage.bits;
        let dim = self.quantiser.storage.dim;
        let metric = self.quantiser.encoder.metric;

        let eq = self.quantiser.encode_query(query_vec)?;
        let (q_rot_f32, levels_f32) = prepare_scalar_scoring(&eq, &self.quantiser.encoder);

        if bits == 3 {
            // Scalar bit-plane scan: max-heap of size k keyed by distance.
            let mut heap: BinaryHeap<(OrderedFloat<T>, usize)> = BinaryHeap::with_capacity(k + 1);
            for idx in 0..self.n {
                let dist = self.turboquant_dist(&eq, &q_rot_f32, &levels_f32, idx);
                if heap.len() < k {
                    heap.push((OrderedFloat(dist), idx));
                } else if dist < heap.peek().unwrap().0 .0 {
                    heap.pop();
                    heap.push((OrderedFloat(dist), idx));
                }
            }
            let mut results: Vec<_> = heap.into_iter().collect();
            results.sort_unstable();
            let (distances, indices): (Vec<T>, Vec<usize>) =
                results.into_iter().map(|(d, i)| (d.0, i)).unzip();
            return Ok((indices, distances));
        }

        // 2-bit / 4-bit: single-query fused scan over the blocked layout.
        let lut = build_query_lut(&q_rot_f32, &levels_f32, bits, dim)?;
        let (blocked, n_blocks) = self
            .blocked
            .as_ref()
            .expect("blocked layout present for 2/4-bit");
        let luts: [&QueryLut; 4] = [&lut, &lut, &lut, &lut];
        let qn = eq.query_norm.to_f32().unwrap();

        let mut res = topk_blocked(
            &luts,
            blocked,
            self.n,
            *n_blocks,
            &self.norms_f32,
            &self.corrections_f32,
            [qn; 4],
            metric,
            k,
            1,
        );
        let (idx_u32, dist_f32) = res.remove(0);
        let indices = idx_u32.iter().map(|&x| x as usize).collect();
        let distances = dist_f32.iter().map(|&d| T::from_f32(d).unwrap()).collect();
        Ok((indices, distances))
    }

    /// Approximate k nearest neighbours for a query row.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted nearest-first.
    pub fn query_row(
        &self,
        query_row: RowRef<T>,
        k: usize,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query(slice, k);
        }
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query(&query_vec, k)
    }

    /// Re-rank exact-distance candidates pulled from the approximate scan.
    ///
    /// ### Params
    ///
    /// * `vector_store` - On-disk original vectors
    /// * `query_vec` - Query vector
    /// * `candidates` - Candidate indices from the approximate scan
    /// * `k` - Number of neighbours to keep
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted nearest-first by exact distance.
    fn rerank_candidates(
        &self,
        vector_store: &MmapVectorStore<T>,
        query_vec: &[T],
        candidates: &[usize],
        k: usize,
    ) -> (Vec<usize>, Vec<T>) {
        let metric = self.quantiser.encoder.metric;
        let query_norm = match metric {
            Dist::Cosine => compute_l2_norm(query_vec),
            _ => T::one(),
        };

        let mut scored: Vec<(T, usize)> = candidates
            .iter()
            .map(|&idx| {
                let dist = match metric {
                    Dist::Cosine => {
                        vector_store.cosine_distance_to_query(idx, query_vec, query_norm)
                    }
                    Dist::SquaredEuclidean => {
                        vector_store.euclidean_distance_to_query(idx, query_vec)
                    }
                    Dist::Manhattan => unreachable!("Manhattan rejected at construction"),
                };
                (dist, idx)
            })
            .collect();

        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(k);

        let indices = scored.iter().map(|p| p.1).collect();
        let dists = scored.iter().map(|p| p.0).collect();
        (indices, dists)
    }

    /// k nearest neighbours with exact-distance reranking.
    ///
    /// Pulls `k * rerank_factor` approximate candidates, re-scores them with
    /// exact distances from the vector store, and keeps the top `k`.
    ///
    /// ### Params
    ///
    /// * `query_vec` - Query vector
    /// * `k` - Number of neighbours
    /// * `rerank_factor` - Candidate multiplier (defaults to 20)
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted nearest-first by exact distance.
    pub fn query_reranking(
        &self,
        query_vec: &[T],
        k: usize,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        let rerank_factor = rerank_factor.unwrap_or(20);
        let vector_store = self
            .vector_store
            .as_ref()
            .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?;

        let (candidates, _) = self.query(query_vec, k * rerank_factor)?;
        Ok(self.rerank_candidates(vector_store, query_vec, &candidates, k))
    }

    /// k nearest neighbours with reranking, from a query row.
    ///
    /// ### Params
    ///
    /// * `query_row` - Query row reference
    /// * `k` - Number of neighbours
    /// * `rerank_factor` - Candidate multiplier (defaults to 20)
    ///
    /// ### Returns
    ///
    /// `(indices, distances)` sorted nearest-first by exact distance.
    pub fn query_row_reranking(
        &self,
        query_row: RowRef<T>,
        k: usize,
        rerank_factor: Option<usize>,
    ) -> Result<(Vec<usize>, Vec<T>), AnnSearchErrors> {
        if query_row.col_stride() == 1 {
            let slice =
                unsafe { std::slice::from_raw_parts(query_row.as_ptr(), query_row.ncols()) };
            return self.query_reranking(slice, k, rerank_factor);
        }
        let query_vec: Vec<T> = query_row.iter().cloned().collect();
        self.query_reranking(&query_vec, k, rerank_factor)
    }

    /// k nearest neighbours for a batch of queries, with 4-query fusion.
    ///
    /// For 2/4-bit, query rows are grouped into batches of 4 so each block's
    /// codes are loaded once and scored against all four queries (the fused
    /// SIMD path), parallelised across batches. 3-bit has no fused kernel and
    /// runs per-query. When `rerank` is set, candidates are re-scored with
    /// exact distances from the vector store (which must be present); the
    /// query vectors themselves come from `queries`, so no store lookup is
    /// needed for the query side.
    ///
    /// ### Params
    ///
    /// * `queries` - Query matrix (n_queries × dim)
    /// * `k` - Number of neighbours per query
    /// * `rerank` - Whether to re-rank with exact distances
    /// * `rerank_factor` - Candidate multiplier when reranking (defaults to 20)
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// `(indices, optional distances)`, one row per query, nearest-first.
    pub fn query_batch(
        &self,
        queries: MatRef<T>,
        k: usize,
        rerank: bool,
        rerank_factor: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        let nq = queries.nrows();
        let bits = self.quantiser.storage.bits;
        let dim = self.quantiser.storage.dim;
        let metric = self.quantiser.encoder.metric;
        let n = self.n;

        if queries.ncols() != dim {
            return Err(AnnSearchErrors::DimensionMismatch {
                index_dim: dim,
                query_dim: queries.ncols(),
            });
        }

        // Reranking needs the stored originals; the query side does not.
        let vector_store = if rerank {
            Some(
                self.vector_store
                    .as_ref()
                    .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?,
            )
        } else {
            None
        };

        let k_eff = k.min(n);
        let k_search = if rerank {
            (k_eff * rerank_factor.unwrap_or(20)).max(1)
        } else {
            k_eff
        };

        let counter = Arc::new(AtomicUsize::new(0));
        let report = |counter: &Arc<AtomicUsize>, added: usize| {
            if verbose {
                let count = counter.fetch_add(added, Ordering::Relaxed) + added;
                if count % 100_000 < added {
                    println!(
                        "  Processed {} / {} queries.",
                        count.separate_with_underscores(),
                        nq.separate_with_underscores()
                    );
                }
            }
        };

        let results: Vec<(Vec<usize>, Vec<T>)> = if bits == 3 {
            // No fused kernel for 3-bit: score each query independently.
            (0..nq)
                .into_par_iter()
                .map(|i| {
                    let qvec: Vec<T> = queries.row(i).iter().cloned().collect();
                    let out = if rerank {
                        self.query_reranking(&qvec, k, rerank_factor)?
                    } else {
                        self.query(&qvec, k)?
                    };
                    report(&counter, 1);
                    Ok(out)
                })
                .collect::<Result<Vec<_>, AnnSearchErrors>>()?
        } else {
            let (blocked, n_blocks) = self
                .blocked
                .as_ref()
                .expect("blocked layout present for 2/4-bit");

            let batch_starts: Vec<usize> = (0..nq).step_by(4).collect();
            let nested: Vec<Vec<(Vec<usize>, Vec<T>)>> = batch_starts
                .into_par_iter()
                .map(|start| {
                    let batch_nq = (nq - start).min(4);

                    // Materialise the batch's query rows (faer rows are not
                    // contiguous in column-major storage, so copy them).
                    let qrows: Vec<Vec<T>> = (0..batch_nq)
                        .map(|qi| queries.row(start + qi).iter().cloned().collect())
                        .collect();

                    // Encode each query and build its LUT.
                    let mut luts_owned: Vec<QueryLut> = Vec::with_capacity(batch_nq);
                    let mut qnorms = [0.0f32; 4];
                    for qi in 0..batch_nq {
                        let eq = self.quantiser.encode_query(&qrows[qi])?;
                        let (q_rot_f32, levels_f32) =
                            prepare_scalar_scoring(&eq, &self.quantiser.encoder);
                        luts_owned.push(build_query_lut(&q_rot_f32, &levels_f32, bits, dim)?);
                        qnorms[qi] = eq.query_norm.to_f32().unwrap();
                    }

                    let last = batch_nq - 1;
                    let lut_refs: [&QueryLut; 4] = [
                        &luts_owned[0],
                        &luts_owned[1.min(last)],
                        &luts_owned[2.min(last)],
                        &luts_owned[3.min(last)],
                    ];

                    let approx = topk_blocked(
                        &lut_refs,
                        blocked,
                        n,
                        *n_blocks,
                        &self.norms_f32,
                        &self.corrections_f32,
                        qnorms,
                        metric,
                        k_search,
                        batch_nq,
                    );

                    let mut out = Vec::with_capacity(batch_nq);
                    for qi in 0..batch_nq {
                        if let Some(vs) = vector_store {
                            let cands: Vec<usize> =
                                approx[qi].0.iter().map(|&x| x as usize).collect();
                            out.push(self.rerank_candidates(vs, &qrows[qi], &cands, k));
                        } else {
                            let indices = approx[qi].0.iter().map(|&x| x as usize).collect();
                            let dists = approx[qi]
                                .1
                                .iter()
                                .map(|&d| T::from_f32(d).unwrap())
                                .collect();
                            out.push((indices, dists));
                        }
                    }
                    report(&counter, batch_nq);
                    Ok(out)
                })
                .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

            nested.into_iter().flatten().collect()
        };

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            Ok((indices, Some(distances)))
        } else {
            let indices = results.into_iter().map(|(idx, _)| idx).collect();
            Ok((indices, None))
        }
    }

    /// Build the full kNN graph over all stored vectors.
    ///
    /// For 2/4-bit, self-queries are batched in groups of 4 so the fused SIMD
    /// kernel shares each block's code load across the batch; each query's
    /// candidates are then re-ranked individually. 3-bit falls back to a
    /// per-vector scan. Requires a vector store.
    ///
    /// ### Params
    ///
    /// * `k` - Neighbours per vector
    /// * `rerank_factor` - Candidate multiplier (defaults to 20)
    /// * `return_dist` - Whether to return distances
    /// * `verbose` - Print progress
    ///
    /// ### Returns
    ///
    /// `(knn_indices, optional knn_distances)`.
    pub fn generate_knn(
        &self,
        k: usize,
        rerank_factor: Option<usize>,
        return_dist: bool,
        verbose: bool,
    ) -> KnnOptionResult<T> {
        let vector_store = self
            .vector_store
            .as_ref()
            .ok_or(AnnSearchErrors::VectorStoreNotAvailable)?;

        let rerank_factor = rerank_factor.unwrap_or(20);
        let bits = self.quantiser.storage.bits;
        let dim = self.quantiser.storage.dim;
        let metric = self.quantiser.encoder.metric;
        let n = self.n;
        let counter = Arc::new(AtomicUsize::new(0));

        let report = |counter: &Arc<AtomicUsize>, added: usize| {
            if verbose {
                let count = counter.fetch_add(added, Ordering::Relaxed) + added;
                if count % 100_000 < added {
                    println!(
                        "  Processed {} / {} samples.",
                        count.separate_with_underscores(),
                        n.separate_with_underscores()
                    );
                }
            }
        };

        let results: Vec<(Vec<usize>, Vec<T>)> = if bits == 3 {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let vec = vector_store.load_vector(i);
                    let out = self.query_reranking(vec, k, Some(rerank_factor))?;
                    report(&counter, 1);
                    Ok(out)
                })
                .collect::<Result<Vec<_>, AnnSearchErrors>>()?
        } else {
            let (blocked, n_blocks) = self
                .blocked
                .as_ref()
                .expect("blocked layout present for 2/4-bit");
            let k_search = (k * rerank_factor).max(1);

            let batch_starts: Vec<usize> = (0..n).step_by(4).collect();
            let nested: Vec<Vec<(Vec<usize>, Vec<T>)>> = batch_starts
                .into_par_iter()
                .map(|start| {
                    let batch_nq = (n - start).min(4);

                    // Encode the batch's queries and build their LUTs.
                    let mut luts_owned: Vec<QueryLut> = Vec::with_capacity(batch_nq);
                    let mut qnorms = [0.0f32; 4];
                    for qi in 0..batch_nq {
                        let qvec = vector_store.load_vector(start + qi);
                        let eq = self.quantiser.encode_query(qvec)?;
                        let (q_rot_f32, levels_f32) =
                            prepare_scalar_scoring(&eq, &self.quantiser.encoder);
                        luts_owned.push(build_query_lut(&q_rot_f32, &levels_f32, bits, dim)?);
                        qnorms[qi] = eq.query_norm.to_f32().unwrap();
                    }

                    // Pad the 4-wide LUT array with the last valid query.
                    let last = batch_nq - 1;
                    let lut_refs: [&QueryLut; 4] = [
                        &luts_owned[0],
                        &luts_owned[1.min(last)],
                        &luts_owned[2.min(last)],
                        &luts_owned[3.min(last)],
                    ];

                    let approx = topk_blocked(
                        &lut_refs,
                        blocked,
                        n,
                        *n_blocks,
                        &self.norms_f32,
                        &self.corrections_f32,
                        qnorms,
                        metric,
                        k_search,
                        batch_nq,
                    );

                    let mut out = Vec::with_capacity(batch_nq);
                    for qi in 0..batch_nq {
                        let qvec = vector_store.load_vector(start + qi);
                        let cands: Vec<usize> = approx[qi].0.iter().map(|&x| x as usize).collect();
                        out.push(self.rerank_candidates(vector_store, qvec, &cands, k));
                    }
                    report(&counter, batch_nq);
                    Ok(out)
                })
                .collect::<Result<Vec<_>, AnnSearchErrors>>()?;

            nested.into_iter().flatten().collect()
        };

        if return_dist {
            let (indices, distances) = results.into_iter().unzip();
            Ok((indices, Some(distances)))
        } else {
            let indices = results.into_iter().map(|(idx, _)| idx).collect();
            Ok((indices, None))
        }
    }

    /// Number of stored vectors.
    pub fn n_vectors(&self) -> usize {
        self.n
    }

    /// Memory usage in bytes (excludes the on-disk vector store).
    pub fn memory_usage_bytes(&self) -> usize {
        std::mem::size_of_val(self)
            + self.quantiser.memory_usage_bytes()
            + self.blocked.as_ref().map_or(0, |(d, _)| d.capacity())
            + self.norms_f32.capacity() * std::mem::size_of::<f32>()
            + self.corrections_f32.capacity() * std::mem::size_of::<f32>()
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;
    use tempfile::TempDir;

    /// Decorrelated pseudo-random data (splitmix64-style per cell), so the
    /// nearest neighbour to a stored vector is unambiguously itself.
    fn test_data(n: usize, dim: usize) -> Mat<f32> {
        Mat::from_fn(n, dim, |i, j| {
            let mut x = (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
                ^ (j as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
                ^ 0x1234_5678_9ABC_DEF0;
            x ^= x >> 33;
            x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
            x ^= x >> 33;
            (x as f32 / u64::MAX as f32) * 2.0 - 1.0
        })
    }

    fn row(data: &Mat<f32>, i: usize) -> Vec<f32> {
        (0..data.ncols()).map(|j| data[(i, j)]).collect()
    }

    /// Exact brute-force nearest neighbours over the original rows.
    fn exact_knn(data: &Mat<f32>, query: &[f32], k: usize, metric: Dist) -> Vec<usize> {
        let n = data.nrows();
        let dim = data.ncols();
        let qn: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let mut scored: Vec<(f32, usize)> = (0..n)
            .map(|i| {
                let v = &(0..dim).map(|j| data[(i, j)]).collect::<Vec<f32>>();
                let dist = match metric {
                    Dist::Cosine => {
                        let dot: f32 = v.iter().zip(query).map(|(a, b)| a * b).sum();
                        let vn: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                        1.0 - dot / (qn * vn)
                    }
                    Dist::SquaredEuclidean => {
                        v.iter().zip(query).map(|(a, b)| (a - b) * (a - b)).sum()
                    }
                    Dist::Manhattan => unreachable!(),
                };
                (dist, i)
            })
            .collect();
        scored.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        scored.truncate(k);
        scored.into_iter().map(|(_, i)| i).collect()
    }

    #[test]
    fn test_construction() {
        let data = test_data(100, 64);
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 4, 42).unwrap();
        assert_eq!(index.n_vectors(), 100);
    }

    #[test]
    fn test_manhattan_rejected() {
        let data = test_data(10, 64);
        let res = TurboQuantExhaustive::new(data.as_ref(), &Dist::Manhattan, 4, 42);
        assert!(matches!(
            res,
            Err(AnnSearchErrors::DistanceNotSupported(Dist::Manhattan))
        ));
    }

    #[test]
    fn test_query_returns_k_sorted() {
        let data = test_data(100, 64);
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 4, 42).unwrap();
        let query = row(&data, 3);
        let (indices, distances) = index.query(&query, 10).unwrap();
        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
        for w in distances.windows(2) {
            assert!(w[0] <= w[1], "distances not ascending");
        }
    }

    #[test]
    fn test_query_k_exceeds_n() {
        let data = test_data(40, 64);
        let index =
            TurboQuantExhaustive::new(data.as_ref(), &Dist::SquaredEuclidean, 4, 42).unwrap();
        let query = row(&data, 0);
        let (indices, _) = index.query(&query, 1000).unwrap();
        assert_eq!(indices.len(), 40);
    }

    #[test]
    fn test_query_row_matches_query() {
        let data = test_data(60, 64);
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 4, 42).unwrap();
        let (i_row, _) = index.query_row(data.as_ref().row(5), 8).unwrap();
        let (i_vec, _) = index.query(&row(&data, 5), 8).unwrap();
        assert_eq!(i_row, i_vec);
    }

    #[test]
    fn test_self_query_recovers_self_4bit() {
        // On decorrelated data the approximate top-1 of a self-query is the
        // vector itself, both metrics.
        for metric in [Dist::Cosine, Dist::SquaredEuclidean] {
            let data = test_data(50, 128);
            let index = TurboQuantExhaustive::new(data.as_ref(), &metric, 4, 42).unwrap();
            for t in 0..50 {
                let (indices, _) = index.query(&row(&data, t), 1).unwrap();
                assert_eq!(
                    indices[0], t,
                    "{metric:?} self-query {t} top-1 was {}",
                    indices[0]
                );
            }
        }
    }

    #[test]
    fn test_3bit_query_returns_k_sorted() {
        let data = test_data(60, 64);
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 3, 42).unwrap();
        let (indices, distances) = index.query(&row(&data, 1), 10).unwrap();
        assert_eq!(indices.len(), 10);
        for w in distances.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }

    #[test]
    fn test_3bit_self_query_recovers_self() {
        let data = test_data(40, 128);
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 3, 42).unwrap();
        for t in 0..40 {
            let (indices, _) = index.query(&row(&data, t), 1).unwrap();
            assert_eq!(indices[0], t);
        }
    }

    #[test]
    fn test_reranking_without_store_errors() {
        let data = test_data(30, 64);
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 4, 42).unwrap();
        let res = index.query_reranking(&row(&data, 0), 5, Some(5));
        assert!(matches!(res, Err(AnnSearchErrors::VectorStoreNotAvailable)));
    }

    #[test]
    fn test_generate_knn_without_store_errors() {
        let data = test_data(30, 64);
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 4, 42).unwrap();
        let res = index.generate_knn(5, Some(5), false, false);
        assert!(matches!(res, Err(AnnSearchErrors::VectorStoreNotAvailable)));
    }

    #[test]
    fn test_new_with_vector_store_and_reranking() {
        let data = test_data(80, 64);
        let dir = TempDir::new().unwrap();
        let index = TurboQuantExhaustive::new_with_vector_store(
            data.as_ref(),
            &Dist::Cosine,
            4,
            42,
            dir.path(),
        )
        .unwrap();

        let (indices, distances) = index.query_reranking(&row(&data, 7), 10, Some(10)).unwrap();
        assert_eq!(indices.len(), 10);
        assert_eq!(distances.len(), 10);
        for w in distances.windows(2) {
            assert!(w[0] <= w[1]);
        }
        // Reranked self-query nearest is exactly itself with zero distance.
        assert_eq!(indices[0], 7);
        assert!(distances[0].abs() < 1e-4);
    }

    #[test]
    fn test_generate_knn_shape_and_recall() {
        // Full self-kNN with reranking; compare to exact brute force. On
        // decorrelated data with 4-bit + reranking, recall should be high.
        let n = 80;
        let dim = 128;
        let k = 10;
        let data = test_data(n, dim);
        let dir = TempDir::new().unwrap();
        let index = TurboQuantExhaustive::new_with_vector_store(
            data.as_ref(),
            &Dist::Cosine,
            4,
            42,
            dir.path(),
        )
        .unwrap();

        let (knn, dists) = index.generate_knn(k, Some(20), true, false).unwrap();
        assert_eq!(knn.len(), n);
        assert!(dists.is_some());
        assert_eq!(dists.as_ref().unwrap().len(), n);
        for neigh in knn.iter() {
            assert_eq!(neigh.len(), k);
        }

        // Recall@k vs exact, averaged over all queries.
        let mut hits = 0usize;
        for t in 0..n {
            let exact: std::collections::HashSet<usize> =
                exact_knn(&data, &row(&data, t), k, Dist::Cosine)
                    .into_iter()
                    .collect();
            hits += knn[t].iter().filter(|i| exact.contains(i)).count();
        }
        let recall = hits as f32 / (n * k) as f32;
        assert!(recall > 0.9, "mean recall@{k} = {recall} below 0.9");
    }

    #[test]
    fn test_generate_knn_no_dist() {
        let data = test_data(40, 64);
        let dir = TempDir::new().unwrap();
        let index = TurboQuantExhaustive::new_with_vector_store(
            data.as_ref(),
            &Dist::SquaredEuclidean,
            4,
            42,
            dir.path(),
        )
        .unwrap();
        let (knn, dists) = index.generate_knn(5, Some(10), false, false).unwrap();
        assert_eq!(knn.len(), 40);
        assert!(dists.is_none());
    }

    #[test]
    fn test_2bit_self_query_recovers_self() {
        // 2-bit is coarse; keep dim generous so self stays top-1.
        let data = test_data(40, 256);
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 2, 42).unwrap();
        for t in 0..40 {
            let (indices, _) = index.query(&row(&data, t), 1).unwrap();
            assert_eq!(indices[0], t);
        }
    }

    #[test]
    fn test_query_batch_matches_single_no_rerank() {
        // Fused batch (no rerank) must reproduce per-row `query` exactly:
        // fusion only shares loads, it does not change results.
        for (bits, dim) in [(4usize, 128usize), (2, 256)] {
            for metric in [Dist::Cosine, Dist::SquaredEuclidean] {
                let data = test_data(120, dim);
                let queries = test_data(7, dim); // 7 -> a full batch of 4 + 3
                let index = TurboQuantExhaustive::new(data.as_ref(), &metric, bits, 42).unwrap();

                let (batch_idx, batch_dist) = index
                    .query_batch(queries.as_ref(), 10, false, None, true, false)
                    .unwrap();
                let batch_dist = batch_dist.unwrap();

                assert_eq!(batch_idx.len(), 7);
                for i in 0..7 {
                    let (si, sd) = index.query(&row(&queries, i), 10).unwrap();
                    assert_eq!(batch_idx[i], si, "{metric:?} bits {bits} query {i} indices");
                    assert_eq!(batch_dist[i], sd, "{metric:?} bits {bits} query {i} dists");
                }
            }
        }
    }

    #[test]
    fn test_query_batch_matches_single_rerank() {
        let data = test_data(120, 128);
        let queries = test_data(6, 128);
        let dir = TempDir::new().unwrap();
        let index = TurboQuantExhaustive::new_with_vector_store(
            data.as_ref(),
            &Dist::Cosine,
            4,
            42,
            dir.path(),
        )
        .unwrap();

        let (batch_idx, batch_dist) = index
            .query_batch(queries.as_ref(), 10, true, Some(10), true, false)
            .unwrap();
        let batch_dist = batch_dist.unwrap();

        for i in 0..6 {
            let (si, sd) = index
                .query_reranking(&row(&queries, i), 10, Some(10))
                .unwrap();
            assert_eq!(batch_idx[i], si, "query {i} indices");
            assert_eq!(batch_dist[i], sd, "query {i} dists");
        }
    }

    #[test]
    fn test_query_batch_3bit() {
        let data = test_data(60, 128);
        let queries = test_data(5, 128);
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 3, 42).unwrap();

        let (batch_idx, _) = index
            .query_batch(queries.as_ref(), 8, false, None, false, false)
            .unwrap();
        assert_eq!(batch_idx.len(), 5);
        for i in 0..5 {
            let (si, _) = index.query(&row(&queries, i), 8).unwrap();
            assert_eq!(batch_idx[i], si, "3-bit query {i} indices");
        }
    }

    #[test]
    fn test_query_batch_rerank_without_store_errors() {
        let data = test_data(40, 64);
        let queries = test_data(4, 64);
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 4, 42).unwrap();
        let res = index.query_batch(queries.as_ref(), 5, true, Some(5), false, false);
        assert!(matches!(res, Err(AnnSearchErrors::VectorStoreNotAvailable)));
    }

    #[test]
    fn test_query_batch_dimension_mismatch() {
        let data = test_data(40, 64);
        let queries = test_data(4, 32); // wrong dim
        let index = TurboQuantExhaustive::new(data.as_ref(), &Dist::Cosine, 4, 42).unwrap();
        let res = index.query_batch(queries.as_ref(), 5, false, None, false, false);
        assert!(matches!(
            res,
            Err(AnnSearchErrors::DimensionMismatch {
                index_dim: 64,
                query_dim: 32
            })
        ));
    }
}
