//! Every index returns non-negative distances, on every query path.
//!
//! Cosine is computed as `1 - dot / (|x| |y|)`, and for a point against itself
//! the ratio rounds just above 1, so the raw value lands one `f32` ulp under
//! zero. That is harmless for ordering and fatal for interop: one negative
//! entry makes scikit-learn reject an entire precomputed distance matrix, so a
//! kNN graph handed to `DBSCAN` or a `KNeighborsTransformer` is invalid.
//!
//! There are two result-packing shapes in the crate and the fix differs:
//!
//! - build `Vec<(Vec<usize>, Vec<T>)>` and unzip -> call
//!   `pack_knn_results`, which clamps on the way through
//! - scatter rows back into original positions -> call `fix_neg_dist` on the
//!   `Option<Vec<Vec<T>>>` before returning
//!
//! Cross-set queries funnel through `query_parallel` and are already covered.
//! The self-query fast paths are the ones that need auditing: each index has
//! its own `generate_knn`, 32 of them, with no shared trait.

use ann_search_rs::*;

const N_PER_BLOB: usize = 300;
const DIM: usize = 16;
const K: usize = 10;

/// Three separated blobs, deterministic, no rng dependency.
///
/// The offset matters: cancellation in `1 - dot / (|x| |y|)` scales with how
/// large the mean is relative to the spread, so centred data would hide the
/// very thing under test.
fn blobs() -> (Vec<f32>, usize, usize) {
    blobs_dim(DIM)
}

/// [`blobs`] at a caller-chosen dimensionality, for indices with their own
/// minimum (PQ/OPQ need `dim >= 32` to form subspaces).
fn blobs_dim(dim: usize) -> (Vec<f32>, usize, usize) {
    let mut data = Vec::with_capacity(3 * N_PER_BLOB * dim);
    let mut state: u32 = 0x2545_F491;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) - 0.5
    };
    for centre in [0.0_f32, 6.0, -6.0] {
        for _ in 0..N_PER_BLOB {
            for _ in 0..dim {
                data.push(centre + next());
            }
        }
    }
    let n = 3 * N_PER_BLOB;
    (data, n, dim)
}

/// Fail with the offending value rather than a bare assert.
fn assert_non_negative(label: &str, distances: &Option<Vec<Vec<f32>>>) {
    let rows = distances
        .as_ref()
        .unwrap_or_else(|| panic!("{label}: asked for distances, got None"));
    let mut worst = 0.0_f32;
    let mut count = 0usize;
    for row in rows {
        for &d in row {
            assert!(!d.is_nan(), "{label}: produced NaN");
            if d < 0.0 {
                count += 1;
                worst = worst.min(d);
            }
        }
    }
    assert_eq!(
        count, 0,
        "{label}: {count} negative distances, most negative {worst:e}"
    );
}

//////////////////////////////////////////////////////////////////////
// Template. Both packing shapes are represented; extend from these. //
//////////////////////////////////////////////////////////////////////

#[test]
fn hnsw_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_hnsw_index((&data[..], n, dim), 16, 100, "cosine", 42, false);

    let (_, self_d) = query_hnsw_self(&index, K, 50, true, false).unwrap();
    assert_non_negative("hnsw self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) = query_hnsw_index((queries, 50, dim), &index, K, 50, true, false).unwrap();
    assert_non_negative("hnsw cross", &cross_d);
}

#[test]
fn ivf_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_ivf_index((&data[..], n, dim), Some(16), None, "cosine", 42, false).unwrap();

    let (_, self_d) = query_ivf_self(&index, K, Some(8), true, false).unwrap();
    assert_non_negative("ivf self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_ivf_index((queries, 50, dim), &index, K, Some(8), true, false).unwrap();
    assert_non_negative("ivf cross", &cross_d);
}

#[test]
fn exhaustive_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_exhaustive_index((&data[..], n, dim), "cosine");

    let (_, self_d) = query_exhaustive_self(&index, K, true, false).unwrap();
    assert_non_negative("exhaustive self", &self_d);
}

/// The clamp runs before the square root on the Euclidean paths, so it must
/// not disturb a real distance.
#[test]
fn euclidean_distances_are_unchanged() {
    let (data, n, dim) = blobs();
    let index = build_exhaustive_index((&data[..], n, dim), "euclidean");
    let (idx, dist) = query_exhaustive_self(&index, K, true, false).unwrap();
    let dist = dist.expect("distances requested");

    // Row i starts with i at distance zero, and the rest ascend.
    for (i, (ids, ds)) in idx.iter().zip(dist.iter()).enumerate() {
        assert_eq!(ids[0], i, "row {i}: self-edge missing");
        assert!(ds[0] <= 1e-5, "row {i}: self distance {} not ~0", ds[0]);
        assert!(
            ds.windows(2).all(|w| w[0] <= w[1]),
            "row {i}: distances not ascending"
        );
    }
}

///////////////////////
// Remaining CPU (13) //
///////////////////////

#[test]
fn kmknn_cosine_is_non_negative() {
    // Never actually negative (kmknn L2-normalises at build time in Cosine
    // mode), kept here as a regression guard alongside the other 12.
    let (data, n, dim) = blobs();
    let index =
        build_kmknn_index((&data[..], n, dim), "cosine", Some(16), None, 42, false).unwrap();

    let (_, self_d) = query_kmknn_self(&index, K, true, false).unwrap();
    assert_non_negative("kmknn self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) = query_kmknn_index((queries, 50, dim), &index, K, true, false).unwrap();
    assert_non_negative("kmknn cross", &cross_d);
}

#[test]
fn annoy_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_annoy_index((&data[..], n, dim), "cosine", 8, 42).unwrap();

    let (_, self_d) = query_annoy_self(&index, K, None, true, false).unwrap();
    assert_non_negative("annoy self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) = query_annoy_index((queries, 50, dim), &index, K, None, true, false).unwrap();
    assert_non_negative("annoy cross", &cross_d);
}

#[test]
fn balltree_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_balltree_index((&data[..], n, dim), "cosine", 42).unwrap();

    let (_, self_d) = query_balltree_self(&index, K, None, true, false).unwrap();
    assert_non_negative("balltree self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_balltree_index((queries, 50, dim), &index, K, None, true, false).unwrap();
    assert_non_negative("balltree cross", &cross_d);
}

#[test]
fn kd_tree_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_kd_tree_index((&data[..], n, dim), "cosine", 4, 42);

    let (_, self_d) = query_kd_tree_self(&index, K, None, true, false).unwrap();
    assert_non_negative("kd_tree self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_kd_tree_index((queries, 50, dim), &index, K, None, true, false).unwrap();
    assert_non_negative("kd_tree cross", &cross_d);
}

#[test]
fn lsh_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_lsh_index((&data[..], n, dim), "cosine", 4, 8, None, 42).unwrap();

    let (_, self_d) = query_lsh_self(&index, K, None, None, true, false).unwrap();
    assert_non_negative("lsh self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_lsh_index((queries, 50, dim), &index, K, 4, None, true, false).unwrap();
    assert_non_negative("lsh cross", &cross_d);
}

#[test]
fn nndescent_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_nndescent_index(
        (&data[..], n, dim),
        "cosine",
        0.001,
        0.0,
        Some(20),
        None,
        None,
        None,
        42,
        false,
    )
    .unwrap();

    let (_, self_d) = query_nndescent_self(&index, K, None, true, false).unwrap();
    assert_non_negative("nndescent self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_nndescent_index((queries, 50, dim), &index, K, None, true, false).unwrap();
    assert_non_negative("nndescent cross", &cross_d);

    // The raw-graph extraction path (`unpack_knn_graph`) is a separate
    // packing boundary from `generate_knn`, fed straight from the build.
    let (_, extract_d) = extract_nndescent_knn(&index, Some(K), true, true).unwrap();
    assert_non_negative("nndescent extract", &extract_d);
}

#[test]
fn nsg_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_nsg_index((&data[..], n, dim), 16, 40, 100, 20, "cosine", 42, false).unwrap();

    let (_, self_d) = query_nsg_self(&index, K, None, true, false).unwrap();
    assert_non_negative("nsg self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) = query_nsg_index((queries, 50, dim), &index, K, None, true, false).unwrap();
    assert_non_negative("nsg cross", &cross_d);
}

#[test]
fn rnn_descent_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index =
        build_rnn_descent_index((&data[..], n, dim), 20, 32, 3, 8, "cosine", None, 42, false)
            .unwrap();

    let (_, self_d) = query_rnn_descent_self(&index, K, None, None, true, false).unwrap();
    assert_non_negative("rnn_descent self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_rnn_descent_index((queries, 50, dim), &index, K, None, None, true, false).unwrap();
    assert_non_negative("rnn_descent cross", &cross_d);
}

#[test]
fn soar_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_soar_index(
        (&data[..], n, dim),
        Some(16),
        None,
        None,
        "cosine",
        42,
        false,
    )
    .unwrap();

    let (_, self_d) = query_soar_self(&index, K, Some(8), true, false).unwrap();
    assert_non_negative("soar self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_soar_index((queries, 50, dim), &index, K, Some(8), true, false).unwrap();
    assert_non_negative("soar cross", &cross_d);
}

#[test]
fn vamana_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_vamana_index((&data[..], n, dim), 16, 60, 1.0, 1.2, "cosine", 42);

    let (_, self_d) = query_vamana_self(&index, K, None, true, false).unwrap();
    assert_non_negative("vamana self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_vamana_index((queries, 50, dim), &index, K, None, true, false).unwrap();
    assert_non_negative("vamana cross", &cross_d);
}

///////////////
// Quantised //
///////////////

#[cfg(feature = "quantised")]
#[test]
fn exhaustive_bf16_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_exhaustive_bf16_index((&data[..], n, dim), "cosine", false).unwrap();

    let (_, self_d) = query_exhaustive_bf16_self(&index, K, true, false).unwrap();
    assert_non_negative("exhaustive_bf16 self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_exhaustive_bf16_index((queries, 50, dim), &index, K, true, false).unwrap();
    assert_non_negative("exhaustive_bf16 cross", &cross_d);
}

#[cfg(feature = "quantised")]
#[test]
fn exhaustive_sq8_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_exhaustive_sq8_index((&data[..], n, dim), "cosine", None, false).unwrap();

    let (_, self_d) = query_exhaustive_sq8_self(&index, K, true, false).unwrap();
    assert_non_negative("exhaustive_sq8 self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_exhaustive_sq8_index((queries, 50, dim), &index, K, true, false).unwrap();
    assert_non_negative("exhaustive_sq8 cross", &cross_d);
}

#[cfg(feature = "quantised")]
#[test]
fn exhaustive_pq_cosine_is_non_negative() {
    let (data, n, dim) = blobs_dim(32);
    let index = build_exhaustive_pq_index(
        (&data[..], n, dim),
        8,
        Some(10),
        Some(4),
        "cosine",
        42,
        false,
    )
    .unwrap();

    let (_, self_d) = query_exhaustive_pq_index_self(&index, K, true, false).unwrap();
    assert_non_negative("exhaustive_pq self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_exhaustive_pq_index((queries, 50, dim), &index, K, true, false).unwrap();
    assert_non_negative("exhaustive_pq cross", &cross_d);
}

#[cfg(feature = "quantised")]
#[test]
fn exhaustive_opq_cosine_is_non_negative() {
    let (data, n, dim) = blobs_dim(32);
    let index = build_exhaustive_opq_index(
        (&data[..], n, dim),
        8,
        Some(10),
        Some(4),
        "cosine",
        42,
        false,
    )
    .unwrap();

    let (_, self_d) = query_exhaustive_opq_index_self(&index, K, true, false).unwrap();
    assert_non_negative("exhaustive_opq self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_exhaustive_opq_index((queries, 50, dim), &index, K, true, false).unwrap();
    assert_non_negative("exhaustive_opq cross", &cross_d);
}

#[cfg(feature = "quantised")]
#[test]
fn ivf_bf16_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index =
        build_ivf_bf16_index((&data[..], n, dim), Some(16), None, "cosine", 42, false).unwrap();

    let (_, self_d) = query_ivf_bf16_self(&index, K, Some(8), true, false).unwrap();
    assert_non_negative("ivf_bf16 self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_ivf_bf16_index((queries, 50, dim), &index, K, Some(8), true, false).unwrap();
    assert_non_negative("ivf_bf16 cross", &cross_d);
}

#[cfg(feature = "quantised")]
#[test]
fn ivf_sq8_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index = build_ivf_sq8_index(
        (&data[..], n, dim),
        Some(16),
        None,
        "cosine",
        42,
        None,
        false,
    )
    .unwrap();

    let (_, self_d) = query_ivf_sq8_self(&index, K, Some(8), true, false).unwrap();
    assert_non_negative("ivf_sq8 self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_ivf_sq8_index((queries, 50, dim), &index, K, Some(8), true, false).unwrap();
    assert_non_negative("ivf_sq8 cross", &cross_d);
}

#[cfg(feature = "quantised")]
#[test]
fn ivf_pq_cosine_is_non_negative() {
    let (data, n, dim) = blobs_dim(32);
    let index = build_ivf_pq_index(
        (&data[..], n, dim),
        Some(16),
        8,
        None,
        Some(4),
        "cosine",
        42,
        false,
    )
    .unwrap();

    let (_, self_d) = query_ivf_pq_index_self(&index, K, Some(8), true, false).unwrap();
    assert_non_negative("ivf_pq self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_ivf_pq_index((queries, 50, dim), &index, K, Some(8), true, false).unwrap();
    assert_non_negative("ivf_pq cross", &cross_d);
}

#[cfg(feature = "quantised")]
#[test]
fn ivf_opq_cosine_is_non_negative() {
    let (data, n, dim) = blobs_dim(32);
    let index = build_ivf_opq_index(
        (&data[..], n, dim),
        Some(16),
        8,
        None,
        Some(4),
        Some(1),
        "cosine",
        42,
        false,
    )
    .unwrap();

    let (_, self_d) = query_ivf_opq_index_self(&index, K, Some(8), true, false).unwrap();
    assert_non_negative("ivf_opq self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_ivf_opq_index((queries, 50, dim), &index, K, Some(8), true, false).unwrap();
    assert_non_negative("ivf_opq cross", &cross_d);
}

#[cfg(feature = "quantised")]
#[test]
fn soar_pq_cosine_is_non_negative() {
    let (data, n, dim) = blobs_dim(32);
    let index = build_soar_pq_index(
        (&data[..], n, dim),
        Some(16),
        8,
        None,
        None,
        Some(4),
        "cosine",
        42,
        false,
    )
    .unwrap();

    let (_, self_d) = query_soar_pq_index_self(&index, K, Some(8), true, false).unwrap();
    assert_non_negative("soar_pq self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_soar_pq_index((queries, 50, dim), &index, K, Some(8), true, false).unwrap();
    assert_non_negative("soar_pq cross", &cross_d);
}

#[cfg(feature = "quantised")]
#[test]
fn soar_opq_cosine_is_non_negative() {
    let (data, n, dim) = blobs_dim(32);
    let index = build_soar_opq_index(
        (&data[..], n, dim),
        Some(16),
        8,
        None,
        None,
        Some(4),
        Some(1),
        "cosine",
        42,
        false,
    )
    .unwrap();

    let (_, self_d) = query_soar_opq_index_self(&index, K, Some(8), true, false).unwrap();
    assert_non_negative("soar_opq self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_soar_opq_index((queries, 50, dim), &index, K, Some(8), true, false).unwrap();
    assert_non_negative("soar_opq cross", &cross_d);
}

#[cfg(feature = "quantised")]
#[test]
fn hnsw_quantised_cosine_is_non_negative() {
    let (data, n, dim) = blobs();
    let index =
        build_hnsw_sq8u_index((&data[..], n, dim), 16, 100, "cosine", 42, None, false).unwrap();

    let (_, self_d) = query_hnsw_sq8u_self(&index, K, 64, true, false).unwrap();
    assert_non_negative("hnsw_quantised self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_hnsw_sq8u_index((queries, 50, dim), &index, K, 64, true, false).unwrap();
    assert_non_negative("hnsw_quantised cross", &cross_d);
}

////////////
// Binary //
////////////
//
// Binary/RaBitQ/TurboQuant only carry a real (float) distance once a vector
// store is attached for reranking; without one they return Hamming counts,
// which are non-negative by construction. So `save_store` and `rerank` are
// both required here to exercise the path this bug lives in.

#[cfg(feature = "binary")]
#[test]
fn exhaustive_binary_cosine_is_non_negative() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let (data, n, dim) = blobs();
    let index = build_exhaustive_index_binary(
        (&data[..], n, dim),
        64,
        42,
        "random",
        "cosine",
        true,
        Some(temp_dir.path()),
    )
    .unwrap();

    let (_, self_d) = query_exhaustive_index_binary_self(&index, K, Some(10), true, false).unwrap();
    assert_non_negative("exhaustive_binary self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_exhaustive_index_binary((queries, 50, dim), &index, K, true, Some(10), true, false)
            .unwrap();
    assert_non_negative("exhaustive_binary cross", &cross_d);
}

#[cfg(feature = "binary")]
#[test]
fn ivf_binary_cosine_is_non_negative() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let (data, n, dim) = blobs();
    let index = build_ivf_index_binary(
        (&data[..], n, dim),
        "random",
        64,
        Some(16),
        None,
        "cosine",
        42,
        true,
        Some(temp_dir.path()),
        false,
    )
    .unwrap();

    let (_, self_d) =
        query_ivf_index_binary_self(&index, K, Some(8), Some(10), true, false).unwrap();
    assert_non_negative("ivf_binary self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) = query_ivf_index_binary(
        (queries, 50, dim),
        &index,
        K,
        Some(8),
        true,
        Some(10),
        true,
        false,
    )
    .unwrap();
    assert_non_negative("ivf_binary cross", &cross_d);
}

#[cfg(feature = "binary")]
#[test]
fn exhaustive_rabitq_cosine_is_non_negative() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let (data, n, dim) = blobs();
    let index = build_exhaustive_index_rabitq(
        (&data[..], n, dim),
        None,
        "cosine",
        42,
        true,
        Some(temp_dir.path()),
    )
    .unwrap();

    let (_, self_d) =
        query_exhaustive_index_rabitq_self(&index, K, None, Some(10), true, false).unwrap();
    assert_non_negative("exhaustive_rabitq self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) = query_exhaustive_index_rabitq(
        (queries, 50, dim),
        &index,
        K,
        None,
        true,
        Some(10),
        true,
        false,
    )
    .unwrap();
    assert_non_negative("exhaustive_rabitq cross", &cross_d);
}

#[cfg(feature = "binary")]
#[test]
fn ivf_rabitq_cosine_is_non_negative() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let (data, n, dim) = blobs();
    let index = build_ivf_index_rabitq(
        (&data[..], n, dim),
        Some(16),
        None,
        "cosine",
        42,
        true,
        Some(temp_dir.path()),
        false,
    )
    .unwrap();

    let (_, self_d) =
        query_ivf_index_rabitq_self(&index, K, Some(8), Some(10), true, false).unwrap();
    assert_non_negative("ivf_rabitq self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) = query_ivf_index_rabitq(
        (queries, 50, dim),
        &index,
        K,
        Some(8),
        true,
        Some(10),
        true,
        false,
    )
    .unwrap();
    assert_non_negative("ivf_rabitq cross", &cross_d);
}

#[cfg(feature = "binary")]
#[test]
fn exhaustive_tq_cosine_is_non_negative() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let (data, n, dim) = blobs();
    let index = build_exhaustive_index_turboquant(
        (&data[..], n, dim),
        "cosine",
        4,
        42,
        true,
        Some(temp_dir.path()),
    )
    .unwrap();

    let (_, self_d) =
        query_exhaustive_index_turboquant_self(&index, K, Some(10), true, false).unwrap();
    assert_non_negative("exhaustive_tq self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) = query_exhaustive_index_turboquant(
        (queries, 50, dim),
        &index,
        K,
        true,
        Some(10),
        true,
        false,
    )
    .unwrap();
    assert_non_negative("exhaustive_tq cross", &cross_d);
}

#[cfg(feature = "binary")]
#[test]
fn ivf_tq_cosine_is_non_negative() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let (data, n, dim) = blobs();
    let index = build_ivf_index_turboquant(
        (&data[..], n, dim),
        Some(16),
        None,
        "cosine",
        4,
        42,
        true,
        Some(temp_dir.path()),
        false,
    )
    .unwrap();

    let (_, self_d) =
        query_ivf_index_turboquant_self(&index, K, Some(8), Some(10), true, false).unwrap();
    assert_non_negative("ivf_tq self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) = query_ivf_index_turboquant(
        (queries, 50, dim),
        &index,
        K,
        Some(8),
        true,
        Some(10),
        true,
        false,
    )
    .unwrap();
    assert_non_negative("ivf_tq cross", &cross_d);
}

/////////
// GPU //
/////////
//
// Compile-only coverage: these are not run here (no GPU guaranteed on the
// build machine), but they must build cleanly under `gpu,gpu-tests`. Skips
// at runtime rather than panicking if no backend is available, matching the
// `try_device` pattern used by the crate's own GPU unit tests.

#[cfg(all(feature = "gpu", feature = "gpu-tests"))]
#[test]
fn exhaustive_gpu_cosine_is_non_negative() {
    use cubecl::prelude::Runtime;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    let device = WgpuDevice::DefaultDevice;
    if std::panic::catch_unwind(|| WgpuRuntime::client(&device)).is_err() {
        eprintln!("Skipping: no wgpu backend");
        return;
    }

    let (data, n, dim) = blobs();
    let index =
        build_exhaustive_index_gpu::<f32, WgpuRuntime>((&data[..], n, dim), "cosine", device)
            .unwrap();

    let (_, self_d) = query_exhaustive_index_gpu_self(&index, K, true, false).unwrap();
    assert_non_negative("exhaustive_gpu self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_exhaustive_index_gpu((queries, 50, dim), &index, K, true, false).unwrap();
    assert_non_negative("exhaustive_gpu cross", &cross_d);
}

#[cfg(all(feature = "gpu", feature = "gpu-tests"))]
#[test]
fn ivf_gpu_cosine_is_non_negative() {
    use cubecl::prelude::Runtime;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    let device = WgpuDevice::DefaultDevice;
    if std::panic::catch_unwind(|| WgpuRuntime::client(&device)).is_err() {
        eprintln!("Skipping: no wgpu backend");
        return;
    }

    let (data, n, dim) = blobs();
    let index = build_ivf_index_gpu::<f32, WgpuRuntime>(
        (&data[..], n, dim),
        Some(16),
        None,
        "cosine",
        42,
        false,
        device,
    )
    .unwrap();

    let (_, self_d) = query_ivf_index_gpu_self(&index, K, Some(8), None, true, false).unwrap();
    assert_non_negative("ivf_gpu self", &self_d);

    let queries = &data[..50 * dim];
    let (_, cross_d) =
        query_ivf_index_gpu((queries, 50, dim), &index, K, Some(8), None, true, false).unwrap();
    assert_non_negative("ivf_gpu cross", &cross_d);
}
