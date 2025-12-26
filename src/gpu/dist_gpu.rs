use cubecl::prelude::*;

////////////////////////////
// GPU distance functions //
////////////////////////////

/// Compute squared Euclidean distances on GPU
///
/// Each GPU thread computes the distance between one query vector and one
/// database vector. Vectors are processed in LINE_SIZE chunks for vectorised
/// memory access.
///
/// ### Params
///
/// * `query_vectors` - Query vectors [n_queries, dim / LINE_SIZE] as Line<F>
/// * `db_chunk` - Database vectors [n_db, dim / LINE_SIZE] as Line<F>
/// * `distances` - Output matrix [n_queries, n_db] of squared distances
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` → database vector index
/// * `ABSOLUTE_POS_Y` → query vector index
#[cube(launch_unchecked)]
pub fn euclidean_distances_gpu_chunk<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_chunk: &Tensor<Line<F>>,
    distances: &mut Tensor<F>,
) {
    let query_idx = ABSOLUTE_POS_Y;
    let db_idx = ABSOLUTE_POS_X;

    if query_idx < query_vectors.shape(0) && db_idx < db_chunk.shape(0) {
        let dim_lines = query_vectors.shape(1);

        let mut sum = F::new(0.0);

        for i in 0..dim_lines {
            let q_line = query_vectors[query_idx * query_vectors.stride(0) + i];
            let d_line = db_chunk[db_idx * db_chunk.stride(0) + i];
            let diff = q_line - d_line;
            let sq = diff * diff;

            // Manual unroll for LINE_SIZE = 4
            sum += sq[0];
            sum += sq[1];
            sum += sq[2];
            sum += sq[3];
        }

        distances[query_idx * distances.stride(0) + db_idx] = sum;
    }
}

/// Compute cosine distances on GPU
///
/// Each GPU thread computes the cosine distance (1 - cosine similarity)
/// between one query vector and one database vector. Requires pre-computed
/// L2 norms for both query and database vectors.
///
/// ### Params
///
/// * `query_vectors` - Query vectors [n_queries, dim / LINE_SIZE] as Line<F>
/// * `db_chunk` - Database vectors [n_db, dim / LINE_SIZE] as Line<F>
/// * `query_norms` - Pre-computed L2 norms [n_queries]
/// * `db_norms` - Pre-computed L2 norms [n_db]
/// * `distances` - Output matrix [n_queries, n_db] of cosine distances
///
/// ### Grid mapping
///
/// * `ABSOLUTE_POS_X` → database vector index
/// * `ABSOLUTE_POS_Y` → query vector index
#[cube(launch_unchecked)]
pub fn cosine_distances_gpu_chunk<F: Float>(
    query_vectors: &Tensor<Line<F>>,
    db_chunk: &Tensor<Line<F>>,
    query_norms: &Tensor<F>,
    db_norms: &Tensor<F>,
    distances: &mut Tensor<F>,
) {
    let query_idx = ABSOLUTE_POS_Y;
    let db_idx = ABSOLUTE_POS_X;

    if query_idx < query_vectors.shape(0) && db_idx < db_chunk.shape(0) {
        let dim_lines = query_vectors.shape(1);

        let mut dot = F::new(0.0);

        for i in 0..dim_lines {
            let q_line = query_vectors[query_idx * query_vectors.stride(0) + i];
            let d_line = db_chunk[db_idx * db_chunk.stride(0) + i];
            let prod = q_line * d_line;

            // Manual unroll for LINE_SIZE = 4
            dot += prod[0];
            dot += prod[1];
            dot += prod[2];
            dot += prod[3];
        }

        let q_norm = query_norms[query_idx];
        let d_norm = db_norms[db_idx];

        distances[query_idx * distances.stride(0) + db_idx] =
            F::new(1.0) - (dot / (q_norm * d_norm));
    }
}
