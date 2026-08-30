//! Gridsearch for the SOAR index, run as two separate sweeps.
//!
//! Sweep A holds the rule fixed and varies `nlist` against a plain IVF index at
//! the same `nlist`. That is the question of whether spilling moves the
//! recall/query-time curve at all, and where on the `nlist` axis it does it.
//!
//! Sweep B holds `nlist` fixed and varies the rule. That is the question of
//! which secondary-assignment cost is right for the metric, with
//! `SoarRule::Nearest` in as the plain second-nearest baseline the other two
//! have to beat.
//!
//! Read the tables by recall against *query time*, not against `nprobe`. At a
//! fixed `nprobe` a spilled index scans roughly twice the candidates, so a
//! recall-versus-nprobe reading flatters it and answers nothing.
//!
//! Unlike `gridsearch_ivf`, this one does not benchmark the self-query path.
//! Ground truth for it is an `O(n^2 * dim)` exhaustive self-query, and it does
//! not discriminate between the rules, which is what both sweeps here are for.
//! `query_soar_self` is covered by the unit tests instead.

mod commons;

use ann_search_rs::prelude::SoarRule;
use ann_search_rs::*;
use clap::Parser;
use commons::*;
use faer::Mat;
use std::collections::HashSet;
use std::time::Instant;
use thousands::*;

/// Multipliers on `sqrt(n)` for the `nlist` sweep.
///
/// Wider than the plain IVF gridsearch, which stops at `sqrt(2n)`. Modelling
/// query cost as `nlist*dim + nprobe*(n/nlist)*dim` puts the compute optimum at
/// `nlist* = sqrt(nprobe * n)`, which lands around 4x `sqrt(n)` at the sizes
/// here. Spilling helps in the fine-cell regime, so the interesting region is
/// at and above that optimum, not below it. 8x is where the flat centroid scan
/// should start dominating; add it here if you want to watch that happen.
const NLIST_MULTIPLIERS: [f32; 4] = [0.5, 1.0, 2.0, 4.0];

/// `nlist` multiplier used for the rule comparison in sweep B.
const RULE_SWEEP_MULTIPLIER: f32 = 2.0;

/// Rules compared in sweep B.
///
/// `Nearest` is the control. Under squared Euclidean the shifted rule is the
/// derived one and `Orthogonal` is the published loss applied off its premise;
/// under cosine the published loss is on home ground. Measured so far, the
/// three land within ~0.005 recall of each other on every dataset tried, so
/// treat this sweep as a check that the choice still does not matter on your
/// data rather than as a tuning knob expected to pay.
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
/// Skewed low relative to the plain IVF gridsearch: spilling raises recall per
/// probed cell, so any win shows up at the aggressive end of the curve.
fn nprobe_values(nlist: usize) -> Vec<usize> {
    let candidates = [
        // nprobe = 1 is the fairest head-to-head: a spilled index at np1 scans
        // about the same candidate count as a plain one at np2, but covers a
        // different set of points. Leaving it out flatters plain IVF.
        1,
        2,
        4,
        (nlist as f32).sqrt() as usize,
        (nlist as f32 * 2.0).sqrt() as usize,
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

    println!("-----------------------------");
    println!(
        "Generating synthetic data: {} samples, {} dimensions, {} clusters, {} dist.",
        cli.n_samples.separate_with_underscores(),
        cli.dim,
        cli.n_clusters,
        cli.distance
    );
    println!("-----------------------------");

    let (data, _): (Mat<f32>, _) = generate_data(&cli);
    let query_data = subsample_with_noise(&data, DEFAULT_N_QUERY, cli.seed + 1);

    // Ground truth
    println!("Building exhaustive index...");
    let start = Instant::now();
    let exhaustive_idx = build_exhaustive_index(data.as_ref(), &cli.distance);
    let exhaustive_build_ms = start.elapsed().as_secs_f64() * 1000.0;
    let exhaustive_size_mb = exhaustive_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

    println!("Querying exhaustive index...");
    let start = Instant::now();
    let (true_neighbors, true_distances) =
        query_exhaustive_index(query_data.as_ref(), &exhaustive_idx, cli.k, true, false).unwrap();
    let exhaustive_query_ms = start.elapsed().as_secs_f64() * 1000.0;

    let baseline_row = |name: &str, query_ms: f64| BenchmarkResultSize {
        method: name.to_string(),
        build_time_ms: exhaustive_build_ms,
        query_time_ms: query_ms,
        total_time_ms: exhaustive_build_ms + query_ms,
        recall_at_k: 1.0,
        mean_dist_rat: 1.0,
        median_dist_rat: 1.0,
        index_size_mb: exhaustive_size_mb,
    };

    ////////////////////////////////////////////////
    // Sweep A: SOAR against IVF across nlist      //
    ////////////////////////////////////////////////

    let mut results_a = vec![baseline_row("Exhaustive (query)", exhaustive_query_ms)];

    println!("-----------------------------");
    println!("Sweep A: default rule, IVF vs SOAR across nlist");
    println!("-----------------------------");

    let nlist_values: Vec<usize> = NLIST_MULTIPLIERS
        .iter()
        .map(|m| ((cli.n_samples as f32 * m).sqrt() as usize).max(1))
        .collect();

    for &nlist in &nlist_values {
        println!("Building IVF index (nlist={})...", nlist);
        let start = Instant::now();
        let ivf_idx = build_ivf_index(
            data.as_ref(),
            Some(nlist),
            None,
            &cli.distance,
            cli.seed as usize,
            false,
        )
        .unwrap();
        let ivf_build_ms = start.elapsed().as_secs_f64() * 1000.0;
        let ivf_size_mb = ivf_idx.memory_usage_bytes() as f64 / (1024.0 * 1024.0);

        println!("Building SOAR index (nlist={})...", nlist);
        let start = Instant::now();
        let soar_idx = build_soar_index(
            data.as_ref(),
            Some(nlist),
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
            let (ivf_nb, ivf_d) = query_ivf_index(
                query_data.as_ref(),
                &ivf_idx,
                cli.k,
                Some(nprobe),
                true,
                false,
            )
            .unwrap();
            let ivf_query_ms = start.elapsed().as_secs_f64() * 1000.0;

            results_a.push(BenchmarkResultSize {
                method: format!("IVF-nl{}-np{}", nlist, nprobe),
                build_time_ms: ivf_build_ms,
                query_time_ms: ivf_query_ms,
                total_time_ms: ivf_build_ms + ivf_query_ms,
                recall_at_k: calculate_recall(&true_neighbors, &ivf_nb, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    ivf_d.as_ref().unwrap(),
                    cli.k,
                ),
                median_dist_rat: calculate_median_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    ivf_d.as_ref().unwrap(),
                    cli.k,
                ),
                index_size_mb: ivf_size_mb,
            });

            let start = Instant::now();
            let (soar_nb, soar_d) = query_soar_index(
                query_data.as_ref(),
                &soar_idx,
                cli.k,
                Some(nprobe),
                true,
                false,
            )
            .unwrap();
            let soar_query_ms = start.elapsed().as_secs_f64() * 1000.0;

            results_a.push(BenchmarkResultSize {
                method: format!("SOAR-{}-nl{}-np{}", soar_tag, nlist, nprobe),
                build_time_ms: soar_build_ms,
                query_time_ms: soar_query_ms,
                total_time_ms: soar_build_ms + soar_query_ms,
                recall_at_k: calculate_recall(&true_neighbors, &soar_nb, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    soar_d.as_ref().unwrap(),
                    cli.k,
                ),
                median_dist_rat: calculate_median_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    soar_d.as_ref().unwrap(),
                    cli.k,
                ),
                index_size_mb: soar_size_mb,
            });
        }
    }

    print_results_size(
        &format!(
            "Sweep A: IVF vs SOAR, {}k samples, {}D",
            cli.n_samples / 1000,
            cli.dim
        ),
        &results_a,
    );

    ////////////////////////////////////////////////
    // Sweep B: rule comparison at fixed nlist     //
    ////////////////////////////////////////////////

    let nlist = ((cli.n_samples as f32 * RULE_SWEEP_MULTIPLIER).sqrt() as usize).max(1);

    println!("-----------------------------");
    println!("Sweep B: rule comparison at nlist={}", nlist);
    println!("-----------------------------");

    let mut results_b = vec![baseline_row("Exhaustive (query)", exhaustive_query_ms)];

    for rule in RULES {
        let tag = rule_tag(&rule);
        println!("Building SOAR index (rule={}, nlist={})...", tag, nlist);
        let start = Instant::now();
        let idx = build_soar_index(
            data.as_ref(),
            Some(nlist),
            Some(rule),
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
            let (nb, d) =
                query_soar_index(query_data.as_ref(), &idx, cli.k, Some(nprobe), true, false)
                    .unwrap();
            let query_ms = start.elapsed().as_secs_f64() * 1000.0;

            results_b.push(BenchmarkResultSize {
                method: format!("SOAR-{}-np{}", tag, nprobe),
                build_time_ms: build_ms,
                query_time_ms: query_ms,
                total_time_ms: build_ms + query_ms,
                recall_at_k: calculate_recall(&true_neighbors, &nb, cli.k),
                mean_dist_rat: calculate_mean_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    d.as_ref().unwrap(),
                    cli.k,
                ),
                median_dist_rat: calculate_median_distance_ratio(
                    true_distances.as_ref().unwrap(),
                    d.as_ref().unwrap(),
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

    println!();
    println!("Note: the Orthogonal rule scans all centroids per point at build");
    println!("time with no GEMM path, so its build_time column is expected to be");
    println!("the worst of the four. That is an implementation gap, not a");
    println!("property of the rule.");
}
