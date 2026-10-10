//! Random-projection forest initialisation for the MLX NN-Descent build.
//!
//! A port of `gpu_forest_init` without the query router. The projections of
//! every tree come off one MLX GEMM and one readback; medians and partition
//! scatters run on the host under rayon, as on the wgpu path; the leaf
//! all-pairs proposals are a Metal kernel, merged into the graph with the
//! NN-Descent merge, five trees per batch.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::time::Instant;

use crate::mlx::ffi::*;
use crate::mlx::nndescent_mlx::*;
use crate::prelude::*;

////////////
// Consts //
////////////

/// Trees whose leaves are proposed and merged per batch.
const TREES_PER_BATCH: usize = 5;

/// Target points per leaf; the tree depth is chosen for it.
const TARGET_LEAF: usize = 64;

/// Upper bound on staged points per leaf.
const MAX_LEAF_CAP: usize = 256;

/// All-pairs proposals within each leaf, one SIMD group per leaf. Leaf vectors,
/// pids, norms and thresholds staged in threadgroup memory for up to `LEAF`
/// points; a longer leaf (an unsplittable partition) is truncated rather than
/// overrun. Atomic outputs. `params = [n, n_leaves]`.
const LEAF_SOURCE: &str = r#"
    uint leaf = threadgroup_position_in_grid.y;
    uint lane = thread_position_in_threadgroup.x;
    if (leaf >= params[1]) {
        return;
    }
    const device float4* v = (const device float4*)vecs;
    uint start = leaf_offsets[leaf];
    uint size = min(leaf_offsets[leaf + 1] - start, (uint)LEAF);
    if (size < 2) {
        return;
    }
    threadgroup float4 sv[LEAF * D4];
    threadgroup uint sp[LEAF];
    threadgroup float sn[LEAF];
    threadgroup float st[LEAF];
    for (uint i = lane; i < size; i += 32) {
        uint p = leaf_points[start + i];
        sp[i] = p;
        st[i] = g_dist[(ulong)p * K + K - 1];
        if (COS) {
            sn[i] = norms[p];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lane; idx < size * D4; idx += 32) {
        uint r = idx / D4;
        sv[idx] = v[(ulong)sp[r] * D4 + (idx - r * D4)];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint ii = lane; ii < size; ii += 32) {
        uint pi = sp[ii];
        float thi = st[ii];
        for (uint jj = ii + 1; jj < size; jj++) {
            uint pj = sp[jj];
            if (pi == pj) {
                continue;
            }
            float d = nnd_pair<D4, COS>(sv + ii * D4, sv + jj * D4);
            if (COS) {
                d = 1.0f - d / (sn[ii] * sn[jj]);
            }
            if (d < thi) {
                nnd_emit(p_idx, p_dist, p_cnt, pi, pj, d, MP);
            }
            if (d < st[jj]) {
                nnd_emit(p_idx, p_dist, p_cnt, pj, pi, d, MP);
            }
        }
    }
"#;

/////////////
// Helpers //
/////////////

/// Points per leaf whose staging fits the threadgroup budget: a padded row
/// plus pid, norm and threshold per point.
///
/// ### Params
///
/// * `dim_padded` - Padded dimensionality
///
/// ### Returns
///
/// Capacity, at most [`MAX_LEAF_CAP`], or `DimTooHighForSharedMemory` below 2
pub(crate) fn max_leaf_size_mlx(dim_padded: usize) -> Result<usize, AnnSearchErrors> {
    let per_point = dim_padded * 4 + 12;
    let fits = MLX_TG_BYTES / per_point;
    if fits < 2 {
        return Err(AnnSearchErrors::DimTooHighForSharedMemory {
            chosen_dim: dim_padded,
            required: 2 * per_point,
            available: MLX_TG_BYTES,
        });
    }
    Ok(fits.min(MAX_LEAF_CAP))
}

/// Group points by final partition into CSR leaves.
///
/// ### Params
///
/// * `partition_ids` - Partition per point
///
/// ### Returns
///
/// `(leaf_points, leaf_offsets)`, offsets of length `n_leaves + 1`
fn build_leaf_structure(partition_ids: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let mut sorted: Vec<(u32, u32)> = partition_ids
        .iter()
        .enumerate()
        .map(|(i, &pid)| (pid, i as u32))
        .collect();
    sorted.par_sort_unstable_by_key(|&(pid, _)| pid);
    let leaf_points = sorted.iter().map(|&(_, i)| i).collect();
    let mut leaf_offsets = vec![0u32];
    for i in 1..sorted.len() {
        if sorted[i].0 != sorted[i - 1].0 {
            leaf_offsets.push(i as u32);
        }
    }
    leaf_offsets.push(sorted.len() as u32);
    (leaf_points, leaf_offsets)
}

/// Median projection per partition.
///
/// ### Params
///
/// * `partition_ids` - Current partition per point
/// * `dots` - Projection per point
/// * `n_partitions` - Partitions at this level
///
/// ### Returns
///
/// One median per partition, zero for an empty one
fn partition_medians(partition_ids: &[u32], dots: &[f32], n_partitions: usize) -> Vec<f32> {
    let cap = dots.len() / n_partitions * 3 / 2;
    let mut buckets: Vec<Vec<f32>> = vec![Vec::with_capacity(cap); n_partitions];
    for (&pid, &d) in partition_ids.iter().zip(dots) {
        buckets[pid as usize].push(d);
    }
    buckets
        .into_par_iter()
        .map(|mut b| {
            if b.is_empty() {
                return 0.0;
            }
            let mid = b.len() / 2;
            b.select_nth_unstable_by(mid, |x, y| x.total_cmp(y));
            b[mid]
        })
        .collect()
}

///////////////////
// Forest driver //
///////////////////

/// Seed the graph from a random-projection forest.
///
/// ### Params
///
/// * `ctx` - NN-Descent device context
/// * `g` - Current (random) graph
/// * `dim` - Unpadded dimensionality; the projections live in it
/// * `n_trees` - Trees to build
/// * `seed` - Base seed; trees and levels derive theirs from it exactly as on
///   the wgpu path
/// * `verbose` - Print phase timings
///
/// ### Returns
///
/// The lazy graph after every batch's proposals are merged
pub(crate) fn forest_init_mlx(
    ctx: &NndMlx,
    mut g: GraphPair,
    dim: usize,
    n_trees: usize,
    seed: usize,
    verbose: bool,
) -> Result<GraphPair, AnnSearchErrors> {
    let n = ctx.n;
    let dim_padded = ctx.d4 * 4;
    let max_leaf = max_leaf_size_mlx(dim_padded)?;
    let target = TARGET_LEAF.min(max_leaf) as f64;
    let max_depth = if n as f64 <= target {
        0
    } else {
        (n as f64 / target).log2().ceil() as usize
    };
    if verbose {
        println!("  MLX forest init: {n_trees} trees, max_depth={max_depth}, max_leaf={max_leaf}");
    }
    let start = Instant::now();

    let mut projections = vec![0.0f32; n_trees * max_depth * dim_padded];
    for tree in 0..n_trees {
        let tree_seed = (seed as u64).wrapping_add((tree as u64).wrapping_mul(0x9E3779B97F4A7C15));
        for level in 0..max_depth {
            let level_seed =
                tree_seed.wrapping_add((level as u64).wrapping_mul(0x517CC1B727220A95));
            let mut rng = SmallRng::seed_from_u64(level_seed);
            let off = (tree * max_depth + level) * dim_padded;
            let row = &mut projections[off..off + dim];
            row.iter_mut()
                .for_each(|x| *x = rng.random_range(-1.0f64..1.0) as f32);
            let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                row.iter_mut().for_each(|x| *x /= norm);
            }
        }
    }

    // `[n_trees * max_depth, n]`: every level's projections contiguous.
    let dots = if max_depth > 0 {
        let proj = Array::from_f32(
            &projections,
            &[(n_trees * max_depth) as i32, dim_padded as i32],
        );
        let vecs_t = ctx.vecs.transpose(&ctx.stream)?;
        let d = Array::addmm(
            &Array::scalar_f32(0.0),
            &proj,
            &vecs_t,
            1.0,
            0.0,
            &ctx.stream,
        )?;
        eval_all(&[&d], false)?;
        Some(d)
    } else {
        None
    };
    let all_dots: &[f32] = match &dots {
        Some(d) => d.as_f32()?,
        None => &[],
    };

    let leaves: Vec<(Vec<u32>, Vec<u32>)> = (0..n_trees)
        .into_par_iter()
        .map(|tree| {
            let mut pids = vec![0u32; n];
            for level in 0..max_depth {
                let off = (tree * max_depth + level) * n;
                let dl = &all_dots[off..off + n];
                let medians = partition_medians(&pids, dl, 1 << level);
                pids.par_iter_mut().zip(dl.par_iter()).for_each(|(p, &d)| {
                    *p = if d <= medians[*p as usize] {
                        *p * 2
                    } else {
                        *p * 2 + 1
                    };
                });
            }
            build_leaf_structure(&pids)
        })
        .collect();
    drop(dots);
    if verbose {
        println!("    Tree construction: {:.2?}", start.elapsed());
    }

    let leaf_k = MetalKernel::with_header(
        "nnd_leaf_pairs",
        &[
            "vecs",
            "norms",
            "leaf_points",
            "leaf_offsets",
            "g_dist",
            "params",
        ],
        &["p_idx", "p_dist", "p_cnt"],
        NND_HEADER,
        LEAF_SOURCE,
        true,
    );
    let merge_params = ctx.params(0, 0, false);
    let p_shape = [n as i32, MLX_MAX_PROPOSALS as i32];

    for batch in leaves.chunks(TREES_PER_BATCH) {
        let mut points: Vec<u32> = Vec::with_capacity(batch.len() * n);
        let mut offsets: Vec<u32> = Vec::new();
        for (lp, lo) in batch {
            let base = points.len() as u32;
            offsets.extend(lo[..lo.len() - 1].iter().map(|&o| o + base));
            points.extend_from_slice(lp);
        }
        offsets.push(points.len() as u32);
        let n_leaves = offsets.len() - 1;
        if n_leaves == 0 {
            continue;
        }
        // Stage for the leaves this batch has, rounded to a power of two so
        // only a few variants compile.
        let batch_max = offsets
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        let stage = batch_max
            .next_power_of_two()
            .clamp((SIMD as usize).min(max_leaf), max_leaf);

        let pts = Array::from_u32(&points, &[points.len() as i32]);
        let offs = Array::from_u32(&offsets, &[offsets.len() as i32]);
        let params = Array::from_u32(&[n as u32, n_leaves as u32], &[2]);
        let [pi, pd, pc] = outs(leaf_k.apply_zeroed(
            &[&ctx.vecs, &ctx.norms, &pts, &offs, &g.1, &params],
            &[u32_out(&p_shape), u32_out(&p_shape), u32_out(&[n as i32])],
            [SIMD, n_leaves as i32, 1],
            [SIMD, 1, 1],
            &[
                ("K", ctx.build_k as i32),
                ("D4", ctx.d4 as i32),
                ("COS", ctx.use_cosine as i32),
                ("MP", MLX_MAX_PROPOSALS as i32),
                ("LEAF", stage as i32),
            ],
            &ctx.stream,
        )?);
        let (next, _) = ctx.merge(&g, &(pi, pd, pc), &merge_params)?;
        g = next;
    }
    if verbose {
        eval_all(&[&g.0, &g.1], false)?;
        println!("  MLX forest init: {:.2?}", start.elapsed());
    }
    Ok(g)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlx_max_leaf_size_fits_budget() {
        for dim in [4usize, 32, 128, 784, 2048] {
            let cap = max_leaf_size_mlx(dim).unwrap();
            assert!(cap * (dim * 4 + 12) <= MLX_TG_BYTES, "dim {dim}");
            assert!(cap >= 2);
        }
        assert_eq!(max_leaf_size_mlx(32).unwrap(), 234);
        assert!(max_leaf_size_mlx(8192).is_err());
    }

    #[test]
    fn test_mlx_leaf_structure_is_csr() {
        let (pts, offs) = build_leaf_structure(&[2, 0, 2, 1, 0]);
        assert_eq!(offs, vec![0, 2, 3, 5]);
        let mut first: Vec<u32> = pts[0..2].to_vec();
        first.sort_unstable();
        assert_eq!(first, vec![1, 4]);
        assert_eq!(pts[2], 3);
    }
}
