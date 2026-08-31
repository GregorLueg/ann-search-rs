//! Gridsearch for SOAR-OPQ against plain IVF-OPQ.
//!
//! Spilling loses on exact full-vector search, because probing one more cell
//! costs nothing fixed there. Product quantisation changes that: codes are
//! residuals against a cell centroid, so every probed cell must rebuild the ADC
//! lookup table at `n_pq_centroids * dim` operations before scoring a single
//! candidate. At 512 dimensions that table build is over an order of magnitude
//! more expensive than scanning the cell it serves, so getting twice the
//! candidates out of *one* cell should beat getting them out of two.
//!
//! Spilling costs `2 * n * m` code bytes instead of `n * m`, so the comparison
//! that matters is against an IVF-OPQ index with **twice the subspaces**, which
//! is the third column here. Beating IVF-OPQ at the same `m` while using twice
//! the memory would prove nothing.
//!
//! OPQ adds a learned rotation on top. Codes are `PQ(R * r)`, so the lookup
//! table is built from the *rotated* query residual; `R` is orthogonal, so
//! distances are preserved only when both sides are rotated.
//!
//! Default target: 50k cell embeddings at 512 dimensions, the
//! foundation-model regime where PQ actually earns its place.
//!
//!   cargo run --example gridsearch_soar_opq --release --features quantised -- \
//!     --data cell --n-samples 50000 --dim 512

mod commons;

use ann_search_rs::prelude::SoarRule;
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::collections::HashSet;
use std::time::Instant;
use thousands::*;

/// Subvector width for the base quantiser. `m = dim / SUBVEC_DIM`, and the
/// equal-memory comparison uses `2 * m`, which stays a divisor of `dim` for any
/// `dim` that is a multiple of `SUBVEC_DIM`.
const SUBVEC_DIM: usize = 16;

/// Multipliers on `sqrt(n)` for the `nlist` sweep.
///
/// Skewed lower than the exact-search sweep. Per-query cost here is
/// `nprobe * (LUT + (n/nlist) * m)` with `LUT = n_pq_centroids * dim` fixed, so
/// at a fixed coverage fraction `nprobe/nlist` the candidate term is constant
/// and the table term grows with `nlist`. PQ therefore wants coarser cells than
/// exact search does, and the interesting region sits below `sqrt(n)`.
const NLIST_MULTIPLIERS: [f32; 4] = [0.25, 0.5, 1.0, 2.0];

/// `nlist` multiplier used for the rule comparison.
const RULE_SWEEP_MULTIPLIER: f32 = 0.5;

/// Rules compared in sweep B.
///
/// Under quantisation `||r'||^2` controls error on the spilled copy, not just
/// routing, which suggested the rules would separate here where they do not on
/// exact search. Measured on the PQ equivalent they still do not: all four land
/// within ~0.005 recall and converge to an identical plateau. Kept as a check
/// that the choice still does not matter on your data.
const RULES: [SoarRule; 4] = [
    SoarRule::Nearest,
    SoarRule::Shifted { mu: 0.3 },
    SoarRule::Shifted { mu: 0.7 },
    SoarRule::Orthogonal { lambda: 1.0 },
];

/// Short tag for a rule, for the results table
fn rule_tag(rule: &SoarRule) -> String {
    match rule {
        SoarRule::Nearest => "near".to_string(),
        SoarRule::Shifted { mu } => format!("shift{}", mu),
        SoarRule::Orthogonal { lambda } => format!("orth{}", lambda),
    }
}

/// Candidate `nprobe` values for a given `nlist`
///
/// `nprobe = 1` is the fairest head-to-head: a spilled index at np1 scans about
/// the same candidate count as a plain one at np2, off half as many lookup
/// tables.
fn nprobe_values(nlist: usize) -> Vec<usize> {
    let candidates = [
        1,
        2,
        4,
        8,
        (nlist as f32).sqrt() as usize,
        (0.05 * nlist as f32) as usize,
    ];
    let mut out: Vec<usize> = candidates
        .into_iter()
        .filter(|&p| p > 0 && p <= nlist)
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    out.sort();
    out
}

fn main() {
    let cli = Cli::parse();

    // PQ rejects dim < 32, and the builders unwrap, so guard both here rather
    // than panicking out of a documented CLI flag.
    if !cli.dim.is_multiple_of(SUBVEC_DIM) || cli.dim < 32 {
        println!(
            "[ERROR] --dim must be at least 32 and a multiple of {} for this example.",
            SUBVEC_DIM
        );
        return;
    }
    let m = cli.dim / SUBVEC_DIM;
    let m2 = m * 2;

    println!("-----------------------------");
    println!(
        "Generating synthetic data: {} samples, {} dimensions, {} clusters, {} dist.",
        cli.n_samples.separate_with_underscores(),
        cli.dim,
        cli.n_clusters,
        cli.distance
    );
    println!(
        "PQ subspaces: m={} (SOAR-OPQ and IVF-OPQ), m={} (IVF-OPQ at equal memory)",
        m, m2
    );
    println!("-----------------------------");

    let (data, _): (Mat<f32>, _) = generate_data(&cli);
    let query_data = subsample_with_noise(&data, DEFAULT_N_QUERY, cli.seed + 1);

    println!("Building exhaustive index for ground truth...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let exhaustive_build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let exhaustive_size_mb = exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data.as_ref(), &exhaustive_idx, cli.k, true, false).unwrap();
    let exhaustive_query_ms = start.elapsed().as_secs_f64() * 1000.0;

    let baseline_row = || BenchmarkResultSize {
        method: "Exhaustive (query)".to_string(),
        build_time_ms: exhaustive_build_ms,
        query_time_ms: exhaustive_query_ms,
        total_time_ms: exhaustive_build_ms + exhaustive_query_ms,
        recall_at_k: 1.0,
        mean_dist_rat: 1.0,
        median_dist_rat: 1.0,
        index_size_mb: exhaustive_size_mb,
    };

    /////////////////////////////////////////////////
    // Sweep A: SOAR-OPQ vs IVF-OPQ at m and at 2m //
    /////////////////////////////////////////////////

    let mut results_a = vec![baseline_row()];

    println!("-----------------------------");
    println!("Sweep A: IVF-OPQ(m), IVF-OPQ(2m), SOAR-OPQ(m) across nlist");
    println!("-----------------------------");

    let nlist_values: Vec<usize> = NLIST_MULTIPLIERS
        .iter()
        .map(|mult| ((cli.n_samples as f32 * mult).sqrt() as usize).max(1))
        .collect();

    for &nlist in &nlist_values {
        println!("Building IVF-OPQ (nlist={}, m={})...", nlist, m);
        let start = Instant::now();
        let pq_idx = build_ivf_opq_index(
            data.as_ref(),
            Some(nlist),
            m,
            None,
            None,
            None,
            &cli.distance,
            cli.seed as usize,
            false,
        )
        .unwrap();
        let pq_build_ms = start.elapsed().as_secs_f64() * 1000.0;
        let pq_size_mb = pq_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        println!("Building IVF-OPQ (nlist={}, m={})...", nlist, m2);
        let start = Instant::now();
        let pq2_idx = build_ivf_opq_index(
            data.as_ref(),
            Some(nlist),
            m2,
            None,
            None,
            None,
            &cli.distance,
            cli.seed as usize,
            false,
        )
        .unwrap();
        let pq2_build_ms = start.elapsed().as_secs_f64() * 1000.0;
        let pq2_size_mb = pq2_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        println!("Building SOAR-OPQ (nlist={}, m={})...", nlist, m);
        let start = Instant::now();
        let soar_idx = build_soar_opq_index(
            data.as_ref(),
            Some(nlist),
            m,
            None,
            None,
            None,
            None,
            &cli.distance,
            cli.seed as usize,
            false,
        )
        .unwrap();
        let soar_build_ms = start.elapsed().as_secs_f64() * 1000.0;
        let soar_size_mb = soar_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);
        let soar_tag = rule_tag(&soar_idx.rule());

        for nprobe in nprobe_values(nlist) {
            println!("  Querying nlist={}, nprobe={}...", nlist, nprobe);

            let start = Instant::now();
            let (nb, _) = query_ivf_opq_index(
                query_data.as_ref(),
                &pq_idx,
                cli.k,
                Some(nprobe),
                false,
                false,
            )
            .unwrap();
            let q_ms = start.elapsed().as_secs_f64() * 1000.0;
            results_a.push(BenchmarkResultSize {
                method: format!("IVFOPQ-m{}-nl{}-np{}", m, nlist, nprobe),
                build_time_ms: pq_build_ms,
                query_time_ms: q_ms,
                total_time_ms: pq_build_ms + q_ms,
                recall_at_k: calculate_recall(&true_neighbors, &nb, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    &exact_distances(&data, &query_data, &nb, &cli.distance),
                    cli.k,
                ),
                median_dist_rat: calculate_median_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    &exact_distances(&data, &query_data, &nb, &cli.distance),
                    cli.k,
                ),
                index_size_mb: pq_size_mb,
            });

            let start = Instant::now();
            let (nb, _) = query_ivf_opq_index(
                query_data.as_ref(),
                &pq2_idx,
                cli.k,
                Some(nprobe),
                false,
                false,
            )
            .unwrap();
            let q_ms = start.elapsed().as_secs_f64() * 1000.0;
            results_a.push(BenchmarkResultSize {
                method: format!("IVFOPQ-m{}-nl{}-np{}", m2, nlist, nprobe),
                build_time_ms: pq2_build_ms,
                query_time_ms: q_ms,
                total_time_ms: pq2_build_ms + q_ms,
                recall_at_k: calculate_recall(&true_neighbors, &nb, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    &exact_distances(&data, &query_data, &nb, &cli.distance),
                    cli.k,
                ),
                median_dist_rat: calculate_median_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    &exact_distances(&data, &query_data, &nb, &cli.distance),
                    cli.k,
                ),
                index_size_mb: pq2_size_mb,
            });

            let start = Instant::now();
            let (nb, _) = query_soar_opq_index(
                query_data.as_ref(),
                &soar_idx,
                cli.k,
                Some(nprobe),
                false,
                false,
            )
            .unwrap();
            let q_ms = start.elapsed().as_secs_f64() * 1000.0;
            results_a.push(BenchmarkResultSize {
                method: format!("SOAROPQ-{}-m{}-nl{}-np{}", soar_tag, m, nlist, nprobe),
                build_time_ms: soar_build_ms,
                query_time_ms: q_ms,
                total_time_ms: soar_build_ms + q_ms,
                recall_at_k: calculate_recall(&true_neighbors, &nb, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    &exact_distances(&data, &query_data, &nb, &cli.distance),
                    cli.k,
                ),
                median_dist_rat: calculate_median_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    &exact_distances(&data, &query_data, &nb, &cli.distance),
                    cli.k,
                ),
                index_size_mb: soar_size_mb,
            });
        }
    }

    print_results_size(
        &format!(
            "Sweep A: SOAR-OPQ vs IVF-OPQ, {}k samples, {}D",
            cli.n_samples / 1000,
            cli.dim
        ),
        &results_a,
    );

    /////////////////////////////////////////////
    // Sweep B: rule comparison at fixed nlist //
    /////////////////////////////////////////////

    let nlist = ((cli.n_samples as f32 * RULE_SWEEP_MULTIPLIER).sqrt() as usize).max(1);

    println!("-----------------------------");
    println!("Sweep B: rule comparison at nlist={}", nlist);
    println!("-----------------------------");

    let mut results_b = vec![baseline_row()];

    for rule in RULES {
        let tag = rule_tag(&rule);
        println!("Building SOAR-OPQ (rule={}, nlist={})...", tag, nlist);
        let start = Instant::now();
        let idx = build_soar_opq_index(
            data.as_ref(),
            Some(nlist),
            m,
            Some(rule),
            None,
            None,
            None,
            &cli.distance,
            cli.seed as usize,
            false,
        )
        .unwrap();
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;
        let size_mb = idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        for nprobe in nprobe_values(nlist) {
            println!("  Querying rule={}, nprobe={}...", tag, nprobe);
            let start = Instant::now();
            let (nb, _) =
                query_soar_opq_index(query_data.as_ref(), &idx, cli.k, Some(nprobe), false, false)
                    .unwrap();
            let q_ms = start.elapsed().as_secs_f64() * 1000.0;

            results_b.push(BenchmarkResultSize {
                method: format!("SOAROPQ-{}-np{}", tag, nprobe),
                build_time_ms: build_ms,
                query_time_ms: q_ms,
                total_time_ms: build_ms + q_ms,
                recall_at_k: calculate_recall(&true_neighbors, &nb, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    &exact_distances(&data, &query_data, &nb, &cli.distance),
                    cli.k,
                ),
                median_dist_rat: calculate_median_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    &exact_distances(&data, &query_data, &nb, &cli.distance),
                    cli.k,
                ),
                index_size_mb: size_mb,
            });
        }
    }

    print_results_size(
        &format!(
            "Sweep B: rules at nlist={}, {}k samples, {}D",
            nlist,
            cli.n_samples / 1000,
            cli.dim
        ),
        &results_b,
    );
}
