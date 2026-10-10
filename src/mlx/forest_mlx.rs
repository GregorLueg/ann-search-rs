//! Random-projection forest initialisation for the MLX NN-Descent build.
//!
//! A port of `gpu_forest_init` without the query router. The projections of
//! every tree come off one MLX GEMM and one readback; medians and partition
//! scatters run on the host under rayon, as on the wgpu path; the leaf
//! all-pairs proposals are a Metal kernel, merged into the graph with the
//! NN-Descent merge, five trees per batch.

use std::time::Instant;

use crate::mlx::ffi::*;
use crate::mlx::nndescent_mlx::*;
use crate::prelude::*;
use crate::utils::rp_forest::*;

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
/// The lazy graph after every batch's proposals are merged, and the query
/// router over the first trees
pub(crate) fn forest_init_mlx(
    ctx: &NndMlx,
    mut g: GraphPair,
    dim: usize,
    n_trees: usize,
    seed: usize,
    verbose: bool,
) -> Result<(GraphPair, ForestRouter<f32>), AnnSearchErrors> {
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

    let (projections, level_vecs) =
        forest_projections::<f32>(n_trees, max_depth, dim, dim_padded, seed);

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

    let (leaves, router) = partition_forest(all_dots, level_vecs, n, max_depth, dim);
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
        for (lp, lo, _) in batch {
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
    Ok((g, router))
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
}
