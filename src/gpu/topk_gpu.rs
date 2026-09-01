//! Radix-select top-k for the GPU indices.
//!
//! The insertion-sort reducers in `dist_gpu.rs` scale badly in `k`:
//! their per-thread register arrays and their shared-memory footprint are both
//! proportional to `k`, so residency collapses and the arrays spill. Radix
//! select replaces both with a fixed 256-bin histogram, so the shared-memory
//! cost is constant in `k`.
//!
//! ### Ordering
//!
//! Selection is defined over the total order `(distance_bits, candidate_index)`.
//! Making the tie-break part of the specification rather than an artefact of the
//! scan order means the result is unique and reproducible, and lets the tests
//! assert exact index equality against the serial reference instead of only
//! distance equality.
//!
//! ### CubeCL 0.10 codegen workarounds
//!
//! The four rules in the header of `nndescent_gpu.rs` apply here in full, and
//! one of them bites harder than usual: this code manipulates float bits as
//! `u32` deliberately, which is exactly what a rule-3 miscompile looks like when
//! a distance leaks into an index slot. A wrong answer and a correct radix key
//! are indistinguishable by inspection, so everything here is asserted against a
//! host mirror.

#![allow(missing_docs)]
// Nested `if`s here are deliberate and must not be collapsed. Merging them
// introduces `&&` into the condition, and compound boolean conditions are
// precisely what miscompiles on cubecl 0.10 (rule 3 in the `nndescent_gpu`
// header, where the recorded symptom is a float bit pattern surfacing as an
// array index). Every guard in this file is written as a nested statement-form
// `if` for that reason.
#![allow(clippy::collapsible_if)]

use cubecl::frontend::Atomic;
use cubecl::ir::features::AtomicUsage;
use cubecl::ir::{ElemType, StorageType, Type, UIntKind};
use cubecl::prelude::*;
use cubecl_utils_rs::prelude::GpuLimits;

/// Radix digit width. Eight bits means 256 bins, a 1 KiB histogram, and four
/// passes to cover a 32-bit word.
///
/// Wider digits mean fewer passes but more shared memory, and shared memory is
/// what sets residency. At 11 bits the histogram alone is 8 KiB, which caps
/// residency at 4 against a 32 KiB budget, so 8 bits is the starting point.
pub const RADIX_BITS: u32 = 8;

/// Number of histogram bins, `2^RADIX_BITS`.
pub const RADIX_BINS: usize = 1 << RADIX_BITS;

/// Smallest `k` at which radix select beats `extract_topk` on the *chunked*
/// exhaustive path.
///
/// Measured end to end on an M1 Max, 50k x 50k, dim 32, four DB chunks:
///
/// | k | `extract_topk` | radix | |
/// |---|---|---|---|
/// | 15 | 383 ms | 555 ms | 0.69x |
/// | 50 | 520 | 559 | 0.93x |
/// | 75 | 719 | 731 | 0.98x |
/// | 100 | 985 | 747 | 1.32x |
/// | 150 | 1743 | 755 | 2.31x |
///
/// The threshold exists because of chunking, and it cannot be read off the
/// isolated reducer benchmark, which reports radix winning at every `k`. The
/// isolated bench always measures a *cold* chunk, where the running top-k is all
/// sentinels. In the real pipeline only the first of `n_db / DB_CHUNK_SIZE`
/// chunks is cold: after that the running top-k is tight and `extract_topk`'s
/// `dist < local[k - 1]` test rejects almost every candidate, so its later
/// chunks cost little more than a scan. Radix select has no such early-out and
/// pays the same four-plus passes on every chunk.
///
/// So this is a property of `k` *and* the chunk count. 80 is taken from the
/// four-chunk shape above, which is the conservative direction: single-chunk
/// workloads would tolerate a much lower threshold (radix wins there from
/// `k = 10`), so this leaves some gain unclaimed rather than risking a
/// regression. Re-measure if `DB_CHUNK_SIZE` changes.
///
/// ### Portability
///
/// This is a *ratio between two kernels on one device*, not a property of the
/// algorithm, so it is the most hardware-specific constant in this module. Both
/// terms it balances vary by vendor: how much radix select gains from coalescing,
/// and how fast contended shared-memory atomics are. Expect the crossover to move
/// on Vulkan or DX12 hardware, where nothing here has been measured.
///
/// Being conservative is the mitigation. At 80 the radix path is only taken where
/// the measured margin is 2.3x, so a device where radix does noticeably worse than
/// Apple Silicon still has headroom before it regresses, and the failure mode is a
/// slower arm rather than a wrong answer.
///
/// The IVF reducer needs no equivalent: it is one-shot over a complete candidate
/// buffer, so there is no warm-chunk effect and radix wins at every `k`.
pub const RADIX_SELECT_MIN_K: usize = 80;

/// Shared-memory footprint of the radix reducers, in bytes.
///
/// Counts every `SharedMemory` the kernels allocate: the histogram, the per-lane
/// scan scratch, the bucket broadcast, the descent state, the survivor key and
/// position buffers and the survivor counter. The histogram term is the constant
/// 1 KiB that makes this independent of `k`; only the two survivor buffers grow,
/// at eight bytes per neighbour.
///
/// Compare against a per-lane merge reducer, which stages one `k`-element list
/// per lane and so needs `lanes * k * 8` bytes: that hits a 32 KiB budget from
/// `k` around 100.
///
/// ### Params
///
/// * `k` - Neighbours per query
/// * `wg` - Workgroup width
/// * `surv_arrays` - Per-survivor `u32` arrays the kernel stages: two for
///   [`radix_select_ivf_topk`] (key and position), three for
///   [`radix_select_topk`], which also stages the id because its output buffer
///   aliases its input
///
/// ### Returns
///
/// Bytes of shared memory one workgroup needs.
pub fn radix_select_smem_bytes(k: usize, wg: usize, surv_arrays: usize) -> usize {
    RADIX_BINS * 4 + wg * 4 + 2 * 4 + ST_SLOTS * 4 + surv_arrays * k * 4 + 4
}

/// Survivor arrays staged by [`radix_select_ivf_topk`].
pub const SURV_ARRAYS_IVF: usize = 2;

/// Survivor arrays staged by [`radix_select_topk`].
pub const SURV_ARRAYS_DENSE: usize = 3;

/// Whether the runtime implements the `u32` shared-memory atomics the histogram
/// needs.
///
/// The CubeCL CPU runtime does not: it raises `not yet implemented: atomic<u32>`
/// while compiling the kernel, on a background thread, so the launch quietly
/// does nothing and the caller reads back whatever was in the buffer. That is
/// the same silent-failure shape as busting a device limit, and it is why this
/// is a capability query rather than an assumption. Metal and Vulkan both
/// register `AtomicUsage::Add` for `u32`.
///
/// ### Params
///
/// * `client` - Compute client to interrogate
///
/// ### Returns
///
/// `true` when `u32` atomic add is available.
pub fn atomic_u32_add_supported<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let ty = Type::new(StorageType::Atomic(ElemType::UInt(UIntKind::U32)));
    client
        .properties()
        .features
        .types
        .atomic
        .get(&ty)
        .is_some_and(|uses| uses.contains(AtomicUsage::Add))
}

/// Whether the radix reducers can serve this configuration.
///
/// Four conditions, all of which are real rather than defensive:
///
/// - **32-bit elements only.** The key transform goes through
///   `u32::reinterpret`, whose size check is an *expand-time* assert, not a
///   compile error, so an `f64` instantiation panics while the kernel is being
///   compiled rather than failing to build. WGSL has no `f64` at all, but the
///   crate also enables the CPU runtime, so the guard has to be a runtime one.
/// - **The workgroup must divide the bin count**, since each lane owns a
///   contiguous slice of the histogram.
/// - **The footprint must fit**, checked against the device rather than a
///   hardcoded budget.
/// - **The workgroup must fit the device**, both its unit budget and the x
///   extent of a cube.
///
/// This half is a pure function of [`GpuLimits`], so it is host-testable at
/// synthetic budgets with no device present, which is what the rest of the
/// crate's staging plans do. The capability half lives in
/// [`atomic_u32_add_supported`] because it needs the client.
///
/// ### Caller obligation
///
/// `wg` must equal the x extent the kernel is actually launched with. Nothing
/// here can check that: `wg` is a comptime argument that sizes the histogram
/// slice each lane owns, while the launch geometry is a separate `CubeDim`. If
/// they disagree the kernel silently returns a mostly-wrong result, sorted and
/// sentinel-free, so it looks entirely plausible. Every call site passes
/// [`WORKGROUP_SIZE_X`](crate::gpu::WORKGROUP_SIZE_X) for both.
///
/// ### Params
///
/// * `k` - Neighbours per query
/// * `elem_bytes` - Size of the float element type in bytes
/// * `wg` - Workgroup width, and the launched cube x extent
/// * `limits` - Device limits from `GpuLimits::from_client`
///
/// ### Returns
///
/// `true` when the shape fits. The footprint is checked against the larger of
/// the two kernels so one answer covers both.
pub fn radix_select_fits(k: usize, elem_bytes: usize, wg: usize, limits: &GpuLimits) -> bool {
    elem_bytes == 4
        && wg > 0
        && RADIX_BINS.is_multiple_of(wg)
        && wg <= limits.max_units_per_cube as usize
        && wg <= limits.max_cube_dim.0 as usize
        && radix_select_smem_bytes(k, wg, SURV_ARRAYS_DENSE) <= limits.max_shared_bytes
}

/// Whether the radix reducers can serve this configuration.
///
/// [`radix_select_fits`] plus [`atomic_u32_add_supported`]. Callers fall back to
/// the insertion-sort reducers in `dist_gpu.rs` when this returns `false`.
///
/// ### Params
///
/// * `client` - Compute client, interrogated for atomic support
/// * `k` - Neighbours per query
/// * `elem_bytes` - Size of the float element type in bytes
/// * `wg` - Workgroup width, and the launched cube x extent
/// * `limits` - Device limits from `GpuLimits::from_client`
///
/// ### Returns
///
/// `true` when the radix path is safe to launch.
pub fn radix_select_usable<R: Runtime>(
    client: &ComputeClient<R>,
    k: usize,
    elem_bytes: usize,
    wg: usize,
    limits: &GpuLimits,
) -> bool {
    radix_select_fits(k, elem_bytes, wg, limits) && atomic_u32_add_supported(client)
}

///////////////////
// Key transform //
///////////////////

/// Order-preserving map from an IEEE-754 binary32 value to a `u32` sort key.
///
/// For non-negative floats the raw bit pattern is already monotone; the sign
/// arm exists so the map stays a total order if a negative distance ever
/// appears, and so the `f32::MAX` sentinel can be reasoned about honestly. The
/// mask is built with bit arithmetic rather than a comparison, per rule 2.
///
/// ### Params
///
/// * `x` - Value to convert
///
/// ### Returns
///
/// A `u32` whose unsigned ordering matches the float ordering of `x`.
#[cube]
pub fn radix_key<F: Float>(x: F) -> u32 {
    let b = u32::reinterpret(x);
    let s = b >> 31u32;
    let mask = (s * 0x7FFF_FFFFu32) | 0x8000_0000u32;
    b ^ mask
}

/// Invert [`radix_key`], recovering the float from its sort key.
///
/// The chunk-accumulating exhaustive reducer reads the running top-k out of the
/// same buffer it later writes, so re-reading a selected distance during the
/// write stage would race against lanes that have already written. Keeping the
/// key in shared memory and inverting it here removes the second read entirely.
///
/// ### Params
///
/// * `key` - Sort key produced by [`radix_key`]
///
/// ### Returns
///
/// The original value, bit for bit.
#[cube]
pub fn radix_key_inv<F: Float>(key: u32) -> F {
    // The transform sets the top bit for non-negative inputs and flips every bit
    // for negative ones, so the top bit of the key says which mask to undo.
    let s = key >> 31u32;
    let mask = (s * 0x8000_0000u32) | ((1u32 - s) * 0xFFFF_FFFFu32);
    F::reinterpret(key ^ mask)
}

/// Host mirror of [`radix_key`], used to assert the kernel agrees.
///
/// ### Params
///
/// * `x` - Value to convert
///
/// ### Returns
///
/// The same `u32` sort key the device computes.
pub fn radix_key_host(x: f32) -> u32 {
    let b = x.to_bits();
    let s = b >> 31;
    let mask = (s * 0x7FFF_FFFF) | 0x8000_0000;
    b ^ mask
}

/////////////////////
// Radix machinery //
/////////////////////

/// Where an element sits relative to the current radix prefix.
///
/// The prefix is carried as a masked value per word: `pk`/`mk` fix the already
/// resolved bits of the distance key, `pp`/`mp` the resolved bits of the
/// position. Both masks start at zero, which makes every element active on the
/// first pass. Using a mask rather than a shift width keeps every shift in the
/// kernel a comptime constant below 32, so the undefined shift-by-32 case never
/// arises.
///
/// ### Params
///
/// * `key` - Sort key of the element
/// * `pos` - Candidate position, the tie-break
/// * `pk` - Resolved key bits
/// * `mk` - Mask of resolved key bits
/// * `pp` - Resolved position bits
/// * `mp` - Mask of resolved position bits
///
/// ### Returns
///
/// `0` when the element sorts below the prefix, `1` when it still matches the
/// prefix and is in play, `2` when it sorts above.
#[cube]
pub fn cmp_prefix(key: u32, pos: u32, pk: u32, mk: u32, pp: u32, mp: u32) -> u32 {
    let key_bits = key & mk;
    let pos_bits = pos & mp;

    // Statement-form assignments only: an `if` expression selecting a value
    // miscompiles on cubecl 0.10, per rule 1.
    let mut res: u32 = 2u32;
    if key_bits < pk {
        res = 0u32;
    }
    if key_bits == pk {
        if pos_bits < pp {
            res = 0u32;
        }
        if pos_bits == pp {
            res = 1u32;
        }
    }
    res
}

/// Extract the radix digit an element contributes on one pass.
///
/// ### Params
///
/// * `key` - Sort key of the element
/// * `pos` - Candidate position
/// * `use_pos` - Whether this pass resolves position bits rather than key bits
/// * `shift` - Bit offset of the digit, always a comptime constant in `0..32`
///
/// ### Returns
///
/// The digit, in `0..RADIX_BINS`.
#[cube]
pub fn digit_at(key: u32, pos: u32, #[comptime] use_pos: bool, #[comptime] shift: u32) -> u32 {
    let mut src = key;
    if use_pos {
        src = pos;
    }
    (src >> shift) & 0xFFu32
}

/// Zero the shared histogram.
///
/// ### Params
///
/// * `hist` - Histogram of [`RADIX_BINS`] atomic counters
/// * `tx` - Lane index
/// * `bins_per_lane` - Bins each lane owns, `RADIX_BINS / workgroup width`
#[cube]
pub fn hist_clear(hist: &SharedMemory<Atomic<u32>>, tx: u32, #[comptime] bins_per_lane: usize) {
    let base = tx * comptime!(bins_per_lane as u32);
    for j in 0..bins_per_lane {
        hist[(base + j as u32) as usize].store(0u32);
    }
}

/// Find the histogram bucket holding rank `remaining`, and how many elements
/// sort below it.
///
/// Every lane owns a contiguous slice of the histogram and reduces it, the
/// per-lane totals are exclusive-scanned redundantly by all lanes (32 adds, far
/// cheaper than a barrier), and the single lane whose slice straddles the target
/// rank walks its own bins to pick the bucket out. Only that lane writes, so the
/// broadcast needs no atomics.
///
/// ### Params
///
/// * `hist` - Populated histogram
/// * `partials` - Scratch for the per-lane totals, one slot per lane
/// * `bcast` - Two slots: the chosen digit and the count below it
/// * `tx` - Lane index
/// * `remaining` - Rank being located, `k - n_less`, always `>= 1`
/// * `wg` - Workgroup width
/// * `bins_per_lane` - Bins each lane owns
#[cube]
pub fn find_bucket(
    hist: &SharedMemory<Atomic<u32>>,
    partials: &mut SharedMemory<u32>,
    bcast: &mut SharedMemory<u32>,
    tx: u32,
    remaining: u32,
    #[comptime] wg: usize,
    #[comptime] bins_per_lane: usize,
) {
    let base = tx * comptime!(bins_per_lane as u32);

    let mut lane_total: u32 = 0u32;
    for j in 0..bins_per_lane {
        lane_total += hist[(base + j as u32) as usize].load();
    }
    partials[tx as usize] = lane_total;
    sync_cube();

    // Exclusive scan over the lane totals, recomputed per lane so the result
    // needs no second barrier. Comptime-unrolled rather than a `while` over a
    // u32 counter, per rule 3.
    let mut below: u32 = 0u32;
    for i in 0..wg {
        if (i as u32) < tx {
            below += partials[i];
        }
    }

    // Exactly one lane's slice contains the target rank.
    if below < remaining {
        if below + lane_total >= remaining {
            let mut cum = below;
            let mut chosen: u32 = 0u32;
            let mut settled: u32 = 0u32;
            for j in 0..bins_per_lane {
                let c = hist[(base + j as u32) as usize].load();
                if settled == 0u32 {
                    if cum + c >= remaining {
                        chosen = base + j as u32;
                        settled = 1u32;
                    }
                }
                if settled == 0u32 {
                    cum += c;
                }
            }
            bcast[0usize] = chosen;
            bcast[1usize] = cum;
        }
    }
    sync_cube();
}

/////////////////////////
// Descent state slots //
/////////////////////////

/// Resolved bits of the distance key.
pub const ST_PK: usize = 0;

/// Mask of resolved distance-key bits.
pub const ST_MK: usize = 1;

/// Resolved bits of the position tie-break.
pub const ST_PP: usize = 2;

/// Mask of resolved position bits.
pub const ST_MP: usize = 3;

/// Candidates known to sort below the prefix.
pub const ST_N_LESS: usize = 4;

/// Set to 1 once the boundary is resolved and later passes can be skipped.
pub const ST_DONE: usize = 5;

/// Slots in the descent state block.
pub const ST_SLOTS: usize = 8;

/// Advance the radix descent by one digit over the IVF candidate buffer.
///
/// Does nothing once `ST_DONE` is set, which is what makes the tie-break passes
/// free in the common case. The branch is uniform across the cube because every
/// lane reads the same shared slot, so the barriers inside are safe.
///
/// ### Params
///
/// * `dists` - Candidate distances
/// * `in_base` - Row offset for this query
/// * `count` - Valid candidates in the row
/// * `kr` - Neighbours requested
/// * `hist` - Shared histogram
/// * `partials` - Per-lane scan scratch
/// * `bcast` - Bucket broadcast slots
/// * `st` - Descent state
/// * `tx` - Lane index
/// * `use_pos` - Whether this pass resolves position bits
/// * `shift` - Bit offset of the digit
/// * `wg` - Workgroup width
#[cube]
pub fn radix_pass_ivf<F: Float>(
    dists: &Tensor<F>,
    in_base: u32,
    count: u32,
    kr: u32,
    hist: &SharedMemory<Atomic<u32>>,
    partials: &mut SharedMemory<u32>,
    bcast: &mut SharedMemory<u32>,
    st: &mut SharedMemory<u32>,
    tx: u32,
    #[comptime] use_pos: bool,
    #[comptime] shift: u32,
    #[comptime] wg: usize,
) {
    if st[ST_DONE] == 0u32 {
        hist_clear(hist, tx, comptime!(RADIX_BINS / wg));
        sync_cube();

        let pk = st[ST_PK];
        let mk = st[ST_MK];
        let pp = st[ST_PP];
        let mp = st[ST_MP];

        let mut c = tx;
        while c < count {
            let key = radix_key::<F>(dists[(in_base + c) as usize]);
            if cmp_prefix(key, c, pk, mk, pp, mp) == 1u32 {
                let d = digit_at(key, c, use_pos, shift);
                hist[d as usize].fetch_add(1u32);
            }
            c += CUBE_DIM_X;
        }
        sync_cube();

        find_bucket(
            hist,
            partials,
            bcast,
            tx,
            kr - st[ST_N_LESS],
            wg,
            comptime!(RADIX_BINS / wg),
        );

        let chosen = bcast[0usize];
        let cum = bcast[1usize];
        let bucket = hist[chosen as usize].load();

        if tx == 0u32 {
            // Fold the digit into the prefix so the next pass narrows further.
            if use_pos {
                st[ST_PP] = pp | (chosen << shift);
                st[ST_MP] = mp | (0xFFu32 << shift);
            } else {
                st[ST_PK] = pk | (chosen << shift);
                st[ST_MK] = mk | (0xFFu32 << shift);
            }
            st[ST_N_LESS] = st[ST_N_LESS] + cum;

            // The chosen bucket holds the rank being chased, so `n_less +
            // bucket` is at least `k`. Once it is exactly `k` the boundary is
            // resolved and every remaining pass is skipped.
            if st[ST_N_LESS] + bucket == kr {
                st[ST_DONE] = 1u32;
            }
        }
        sync_cube();
    }
}

////////////////
// IVF select //
////////////////

/// Radix-select top-k over the IVF candidate buffer.
///
/// One cube per query. Selects the `k` smallest candidates under the total order
/// `(distance_bits, position)`, which is exactly the order the serial
/// `reduce_ivf_topk` produces: it accepts on strict `<` while scanning positions
/// upwards, so among bit-identical distances the earliest position wins and
/// later equals sort after it. Making that a defined total order rather than an
/// artefact of scan order is what lets the tests assert exact index equality.
///
/// Shared memory is [`radix_select_smem_bytes`] at [`SURV_ARRAYS_IVF`]: constant
/// apart from the `8 * k` survivor buffer, so it stays far below a per-lane
/// merge reducer at every `k`.
///
/// ### Params
///
/// * `candidate_dists` - Candidate distances `[n_queries, max_candidates]`
/// * `candidate_indices` - Candidate database ids, same shape
/// * `candidates_per_query` - Valid prefix length per query `[n_queries]`
/// * `out_dists` - Selected distances `[n_queries, k]`, ascending
/// * `out_indices` - Selected database ids `[n_queries, k]`
/// * `k_param` - Runtime mirror of `k`
/// * `k` - Neighbours per query
/// * `wg` - Workgroup width, must divide [`RADIX_BINS`]
#[cube(launch_unchecked)]
pub fn radix_select_ivf_topk<F: Float>(
    candidate_dists: &Tensor<F>,
    candidate_indices: &Tensor<u32>,
    candidates_per_query: &Tensor<u32>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    k_param: u32,
    #[comptime] k: usize,
    #[comptime] wg: usize,
) {
    let q_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;

    if q_idx >= candidate_dists.shape(0) {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let kr = k_param;
    let count = candidates_per_query[q_idx];
    let in_base = q_idx * candidate_dists.stride(0);
    let out_base = q_idx * out_dists.stride(0);

    // All shared allocations at kernel scope, never inside a branch (rule 4).
    let hist = SharedMemory::<Atomic<u32>>::new(RADIX_BINS);
    let mut partials = SharedMemory::<u32>::new(wg);
    let mut bcast = SharedMemory::<u32>::new(2usize);
    let mut st = SharedMemory::<u32>::new(ST_SLOTS);
    let mut surv_key = SharedMemory::<u32>::new(k);
    let mut surv_pos = SharedMemory::<u32>::new(k);
    let found = SharedMemory::<Atomic<u32>>::new(1usize);

    if tx == 0u32 {
        st[ST_PK] = 0u32;
        st[ST_MK] = 0u32;
        st[ST_PP] = 0u32;
        st[ST_MP] = 0u32;
        st[ST_N_LESS] = 0u32;
        // With no more candidates than slots every element is selected, so the
        // descent has nothing to resolve. u32 sentinel, never a bool (rule 3).
        st[ST_DONE] = 0u32;
        if count <= kr {
            st[ST_DONE] = 1u32;
        }
        found[0usize].store(0u32);
    }
    sync_cube();

    // Four passes fix the distance key, four more break ties on position. The
    // shifts are written out rather than derived from a loop counter: an
    // `#[unroll]` index is not usable in comptime position on cubecl 0.10.
    let ib = in_base as u32;
    radix_pass_ivf::<F>(
        candidate_dists,
        ib,
        count,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        false,
        24u32,
        wg,
    );
    radix_pass_ivf::<F>(
        candidate_dists,
        ib,
        count,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        false,
        16u32,
        wg,
    );
    radix_pass_ivf::<F>(
        candidate_dists,
        ib,
        count,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        false,
        8u32,
        wg,
    );
    radix_pass_ivf::<F>(
        candidate_dists,
        ib,
        count,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        false,
        0u32,
        wg,
    );
    radix_pass_ivf::<F>(
        candidate_dists,
        ib,
        count,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        true,
        24u32,
        wg,
    );
    radix_pass_ivf::<F>(
        candidate_dists,
        ib,
        count,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        true,
        16u32,
        wg,
    );
    radix_pass_ivf::<F>(
        candidate_dists,
        ib,
        count,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        true,
        8u32,
        wg,
    );
    radix_pass_ivf::<F>(
        candidate_dists,
        ib,
        count,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        true,
        0u32,
        wg,
    );

    // Emit. Everything below the prefix, plus everything still matching it,
    // is exactly the selected set.
    let pk = st[ST_PK];
    let mk = st[ST_MK];
    let pp = st[ST_PP];
    let mp = st[ST_MP];

    let mut c = tx;
    while c < count {
        let key = radix_key::<F>(candidate_dists[(ib + c) as usize]);
        if cmp_prefix(key, c, pk, mk, pp, mp) < 2u32 {
            let slot = found[0usize].fetch_add(1u32);
            if slot < kr {
                surv_key[slot as usize] = key;
                surv_pos[slot as usize] = c;
            }
        }
        c += CUBE_DIM_X;
    }
    sync_cube();

    // Positions are unique, so the `(key, position)` composite is unique and
    // the descent must converge on exactly `k` within eight passes. The clamp
    // is a guard against that reasoning being wrong: without it a non-converged
    // descent would read past the survivor buffer and corrupt neighbouring
    // shared memory rather than simply returning a wrong answer.
    let mut n_found = found[0usize].load();
    if n_found > kr {
        n_found = kr;
    }

    // Rank each survivor by counting how many survivors precede it under the
    // total order. O(k^2 / wg) and independent of the candidate count, against
    // the O(k) insert per accepted candidate the merge reducers pay.
    let mut s = tx;
    while s < n_found {
        let my_key = surv_key[s as usize];
        let my_pos = surv_pos[s as usize];

        let mut rank: u32 = 0u32;
        let mut t = 0u32;
        while t < n_found {
            let mut smaller: u32 = 0u32;
            if surv_key[t as usize] < my_key {
                smaller = 1u32;
            }
            if surv_key[t as usize] == my_key {
                if surv_pos[t as usize] < my_pos {
                    smaller = 1u32;
                }
            }
            rank += smaller;
            t += 1u32;
        }

        out_dists[out_base + rank as usize] = candidate_dists[(ib + my_pos) as usize];
        out_indices[out_base + rank as usize] = candidate_indices[(ib + my_pos) as usize];
        s += CUBE_DIM_X;
    }

    // Pad the tail when the query saw fewer than `k` candidates. The host filters
    // IVF rows on this sentinel, so it has to survive.
    let mut p = n_found + tx;
    while p < kr {
        out_dists[out_base + p as usize] = F::new(f32::MAX);
        out_indices[out_base + p as usize] = 0u32;
        p += CUBE_DIM_X;
    }
}

///////////////////////
// Exhaustive select //
///////////////////////

// The exhaustive reducer is chunk-accumulating: the host launches it once per DB
// chunk into the same output buffer, and each launch must fold the running top-k
// in. Rather than adding a merge step after the selection, the running top-k is
// fed in as `k` extra virtual candidates occupying positions `0..k`, with the
// chunk's own candidates at `k..k + chunk`. That falls out exactly right:
//
// - Running entries sort before chunk candidates on equal distances, which is
//   what `extract_topk` does (its acceptance test is a strict `<`, so an equal
//   new candidate never displaces an existing one).
// - On the first chunk the running entries are all `f32::MAX` sentinels, whose
//   keys sort above every real distance, so they are squeezed out when the chunk
//   supplies enough candidates and survive as padding when it does not.

/// Fetch a virtual candidate's sort key.
///
/// Positions below `kr` address the running top-k, positions at or above it
/// address the current chunk.
///
/// ### Params
///
/// * `distances` - Chunk distance matrix
/// * `out_dists` - Running top-k buffer
/// * `in_base` - Row offset into the chunk matrix
/// * `out_base` - Row offset into the running top-k
/// * `p` - Virtual position
/// * `kr` - Neighbours requested
///
/// ### Returns
///
/// The sort key of the candidate at position `p`.
#[cube]
pub fn dense_key<F: Float>(
    distances: &Tensor<F>,
    out_dists: &Tensor<F>,
    in_base: u32,
    out_base: u32,
    p: u32,
    kr: u32,
) -> u32 {
    let mut v = F::new(f32::MAX);
    if p < kr {
        v = out_dists[(out_base + p) as usize];
    }
    if p >= kr {
        v = distances[(in_base + p - kr) as usize];
    }
    radix_key::<F>(v)
}

/// Advance the radix descent by one digit over the exhaustive candidate set.
///
/// ### Params
///
/// * `distances` - Chunk distance matrix
/// * `out_dists` - Running top-k buffer, read only here
/// * `in_base` - Row offset into the chunk matrix
/// * `out_base` - Row offset into the running top-k
/// * `chunk` - Valid columns in this chunk
/// * `kr` - Neighbours requested
/// * `hist` - Shared histogram
/// * `partials` - Per-lane scan scratch
/// * `bcast` - Bucket broadcast slots
/// * `st` - Descent state
/// * `tx` - Lane index
/// * `use_pos` - Whether this pass resolves position bits
/// * `shift` - Bit offset of the digit
/// * `wg` - Workgroup width
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn radix_pass_dense<F: Float>(
    distances: &Tensor<F>,
    out_dists: &Tensor<F>,
    in_base: u32,
    out_base: u32,
    chunk: u32,
    kr: u32,
    hist: &SharedMemory<Atomic<u32>>,
    partials: &mut SharedMemory<u32>,
    bcast: &mut SharedMemory<u32>,
    st: &mut SharedMemory<u32>,
    tx: u32,
    #[comptime] use_pos: bool,
    #[comptime] shift: u32,
    #[comptime] wg: usize,
) {
    if st[ST_DONE] == 0u32 {
        hist_clear(hist, tx, comptime!(RADIX_BINS / wg));
        sync_cube();

        let pk = st[ST_PK];
        let mk = st[ST_MK];
        let pp = st[ST_PP];
        let mp = st[ST_MP];
        let total = kr + chunk;

        let mut p = tx;
        while p < total {
            let key = dense_key::<F>(distances, out_dists, in_base, out_base, p, kr);
            if cmp_prefix(key, p, pk, mk, pp, mp) == 1u32 {
                let d = digit_at(key, p, use_pos, shift);
                hist[d as usize].fetch_add(1u32);
            }
            p += CUBE_DIM_X;
        }
        sync_cube();

        find_bucket(
            hist,
            partials,
            bcast,
            tx,
            kr - st[ST_N_LESS],
            wg,
            comptime!(RADIX_BINS / wg),
        );

        let chosen = bcast[0usize];
        let cum = bcast[1usize];
        let bucket = hist[chosen as usize].load();

        if tx == 0u32 {
            if use_pos {
                st[ST_PP] = pp | (chosen << shift);
                st[ST_MP] = mp | (0xFFu32 << shift);
            } else {
                st[ST_PK] = pk | (chosen << shift);
                st[ST_MK] = mk | (0xFFu32 << shift);
            }
            st[ST_N_LESS] = st[ST_N_LESS] + cum;
            if st[ST_N_LESS] + bucket == kr {
                st[ST_DONE] = 1u32;
            }
        }
        sync_cube();
    }
}

/// Radix-select top-k for the exhaustive path, accumulating across DB chunks.
///
/// Drop-in replacement for `extract_topk`: same buffers, same chunk-accumulating
/// contract, same `init_topk` seeding requirement, and the same tie-break. One
/// cube per query rather than one thread, which also fixes coalescing, since the
/// serial kernel has adjacent threads reading addresses a full row apart.
///
/// ### Params
///
/// * `distances` - Chunk distance matrix `[n_queries, dist_stride]`
/// * `out_dists` - Running and final top-k distances `[n_queries, k]`
/// * `out_indices` - Running and final top-k ids `[n_queries, k]`
/// * `chunk_offset` - Global id of column zero of this chunk
/// * `actual_chunk_size` - Valid columns in this chunk
/// * `k_param` - Runtime mirror of `k`
/// * `k` - Neighbours per query
/// * `wg` - Workgroup width, must divide [`RADIX_BINS`]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn radix_select_topk<F: Float>(
    distances: &Tensor<F>,
    out_dists: &mut Tensor<F>,
    out_indices: &mut Tensor<u32>,
    chunk_offset: u32,
    actual_chunk_size: u32,
    k_param: u32,
    #[comptime] k: usize,
    #[comptime] wg: usize,
) {
    let q_idx = (CUBE_POS_Y * CUBE_COUNT_X + CUBE_POS_X) as usize;

    if q_idx >= distances.shape(0) {
        terminate!();
    }

    let tx = UNIT_POS_X;
    let kr = k_param;
    let in_base = (q_idx * distances.stride(0)) as u32;
    let out_base = (q_idx * out_dists.stride(0)) as u32;
    let total = kr + actual_chunk_size;

    let hist = SharedMemory::<Atomic<u32>>::new(RADIX_BINS);
    let mut partials = SharedMemory::<u32>::new(wg);
    let mut bcast = SharedMemory::<u32>::new(2usize);
    let mut st = SharedMemory::<u32>::new(ST_SLOTS);
    let mut surv_key = SharedMemory::<u32>::new(k);
    let mut surv_pos = SharedMemory::<u32>::new(k);
    let mut surv_idx = SharedMemory::<u32>::new(k);
    let found = SharedMemory::<Atomic<u32>>::new(1usize);

    if tx == 0u32 {
        st[ST_PK] = 0u32;
        st[ST_MK] = 0u32;
        st[ST_PP] = 0u32;
        st[ST_MP] = 0u32;
        st[ST_N_LESS] = 0u32;
        st[ST_DONE] = 0u32;
        if total <= kr {
            st[ST_DONE] = 1u32;
        }
        found[0usize].store(0u32);
    }
    sync_cube();

    radix_pass_dense::<F>(
        distances,
        out_dists,
        in_base,
        out_base,
        actual_chunk_size,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        false,
        24u32,
        wg,
    );
    radix_pass_dense::<F>(
        distances,
        out_dists,
        in_base,
        out_base,
        actual_chunk_size,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        false,
        16u32,
        wg,
    );
    radix_pass_dense::<F>(
        distances,
        out_dists,
        in_base,
        out_base,
        actual_chunk_size,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        false,
        8u32,
        wg,
    );
    radix_pass_dense::<F>(
        distances,
        out_dists,
        in_base,
        out_base,
        actual_chunk_size,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        false,
        0u32,
        wg,
    );
    radix_pass_dense::<F>(
        distances,
        out_dists,
        in_base,
        out_base,
        actual_chunk_size,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        true,
        24u32,
        wg,
    );
    radix_pass_dense::<F>(
        distances,
        out_dists,
        in_base,
        out_base,
        actual_chunk_size,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        true,
        16u32,
        wg,
    );
    radix_pass_dense::<F>(
        distances,
        out_dists,
        in_base,
        out_base,
        actual_chunk_size,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        true,
        8u32,
        wg,
    );
    radix_pass_dense::<F>(
        distances,
        out_dists,
        in_base,
        out_base,
        actual_chunk_size,
        kr,
        &hist,
        &mut partials,
        &mut bcast,
        &mut st,
        tx,
        true,
        0u32,
        wg,
    );

    // Stage the winners, including their ids, before anything writes back. The
    // distance itself is recovered from the key rather than re-read, because
    // positions below `kr` alias the output buffer.
    let pk = st[ST_PK];
    let mk = st[ST_MK];
    let pp = st[ST_PP];
    let mp = st[ST_MP];

    let mut p = tx;
    while p < total {
        let key = dense_key::<F>(distances, out_dists, in_base, out_base, p, kr);
        if cmp_prefix(key, p, pk, mk, pp, mp) < 2u32 {
            let slot = found[0usize].fetch_add(1u32);
            if slot < kr {
                surv_key[slot as usize] = key;
                surv_pos[slot as usize] = p;
                // Guarded both ways, same reason as `dense_key`: an
                // unconditional `out_indices` load here runs past the row.
                let mut id = 0u32;
                if p < kr {
                    id = out_indices[(out_base + p) as usize];
                }
                if p >= kr {
                    id = chunk_offset + p - kr;
                }
                surv_idx[slot as usize] = id;
            }
        }
        p += CUBE_DIM_X;
    }
    sync_cube();

    let mut n_found = found[0usize].load();
    if n_found > kr {
        n_found = kr;
    }

    // Safe to write back over the running top-k now, but note *why*, because it
    // is not the barrier on its own: `sync_cube()` lowers to `workgroupBarrier()`,
    // which orders workgroup memory, not storage. What retires the reads is that
    // every `out_*` value the emit loop loaded was consumed into `surv_key` /
    // `surv_idx`, which are workgroup memory the barrier does order. If a future
    // change makes the emit loop carry an `out_*` value past this point in a
    // register instead, that reasoning breaks and a `storage` barrier is needed.
    let mut s = tx;
    while s < n_found {
        let my_key = surv_key[s as usize];
        let my_pos = surv_pos[s as usize];

        let mut rank: u32 = 0u32;
        let mut t = 0u32;
        while t < n_found {
            let mut smaller: u32 = 0u32;
            if surv_key[t as usize] < my_key {
                smaller = 1u32;
            }
            if surv_key[t as usize] == my_key {
                if surv_pos[t as usize] < my_pos {
                    smaller = 1u32;
                }
            }
            rank += smaller;
            t += 1u32;
        }

        out_dists[(out_base + rank) as usize] = radix_key_inv::<F>(my_key);
        out_indices[(out_base + rank) as usize] = surv_idx[s as usize];
        s += CUBE_DIM_X;
    }
}

///////////////////////////
// Mechanism smoke tests //
///////////////////////////

/// S3: a `#[cube]` helper that writes through a shared-memory reference.
///
/// The crate has no `#[cube]` device functions at all today, so sharing helpers
/// between the reducers is unproven ground.
///
/// ### Params
///
/// * `mem` - Shared buffer to write into
/// * `idx` - Slot to write
/// * `value` - Value to store
#[cfg(all(test, feature = "gpu-tests"))]
#[cube]
fn shared_set(mem: &mut SharedMemory<u32>, idx: u32, value: u32) {
    mem[idx as usize] = value;
}

/// S3: a `#[cube]` helper that reads through a shared-memory reference.
///
/// ### Params
///
/// * `mem` - Shared buffer to read from
/// * `idx` - Slot to read
///
/// ### Returns
///
/// The stored value.
#[cfg(all(test, feature = "gpu-tests"))]
#[cube]
fn shared_get(mem: &SharedMemory<u32>, idx: u32) -> u32 {
    mem[idx as usize]
}

/// S1: does `SharedMemory<Atomic<u32>>` work on the wgpu backend?
///
/// Nothing in this crate, and nothing in cubecl's own runtime tests, allocates
/// an atomic in shared memory: every existing use is `&Tensor<Atomic<u32>>` in
/// global memory. The 256-bin histogram rests entirely on this working.
///
/// One cube of 32 lanes. Every bin is incremented exactly once by the spread
/// phase, then bin 0 takes 32 x 100 contended increments, so a correct run
/// leaves bin 0 at 3201 and every other bin at 1.
///
/// ### Params
///
/// * `out` - Destination for the 256 bin counts
#[cfg(all(test, feature = "gpu-tests"))]
#[cube(launch_unchecked)]
fn smoke_atomic_hist(out: &mut Tensor<u32>) {
    let hist = SharedMemory::<Atomic<u32>>::new(256usize);
    let tx = UNIT_POS_X;

    for j in 0..8u32 {
        hist[(j * 32u32 + tx) as usize].store(0u32);
    }
    sync_cube();

    // Spread: each of the 256 bins is touched by exactly one lane.
    for j in 0..8u32 {
        hist[(j * 32u32 + tx) as usize].fetch_add(1u32);
    }

    // Contention: all 32 lanes hammer the same bin.
    for _j in 0..100u32 {
        hist[0usize].fetch_add(1u32);
    }
    sync_cube();

    for j in 0..8u32 {
        let b = j * 32u32 + tx;
        out[b as usize] = hist[b as usize].load();
    }
}

/// S2: does `u32::reinterpret` give the host bit pattern inside a kernel that
/// is generic over `F: Float`?
///
/// `FloatBits` in cubecl 0.10 exposes only expand-side methods, so `x.to_bits()`
/// does not typecheck for a generic `F`. `Reinterpret` is the route, and this
/// pins it against `f32::to_bits`.
///
/// ### Params
///
/// * `input` - Values to convert `[n]`
/// * `out` - Destination for the sort keys `[n]`
#[cfg(all(test, feature = "gpu-tests"))]
#[cube(launch_unchecked)]
fn smoke_radix_key<F: Float>(input: &Tensor<F>, out: &mut Tensor<u32>) {
    let i = ABSOLUTE_POS_X as usize;
    if i >= input.shape(0) {
        terminate!();
    }
    out[i] = radix_key::<F>(input[i]);
}

/// S3: drive the shared-memory helpers from a kernel.
///
/// Each lane writes `tx * 7 + 1` through [`shared_set`] and reads its
/// left neighbour's slot back through [`shared_get`], so a broken reference
/// round-trip shows up as a shifted or zeroed row rather than as a plausible
/// one.
///
/// ### Params
///
/// * `out` - Destination `[32]`
#[cfg(all(test, feature = "gpu-tests"))]
#[cube(launch_unchecked)]
fn smoke_shared_helper(out: &mut Tensor<u32>) {
    let mut scratch = SharedMemory::<u32>::new(32usize);
    let tx = UNIT_POS_X;

    shared_set(&mut scratch, tx, tx * 7u32 + 1u32);
    sync_cube();

    let neighbour = (tx + 31u32) % 32u32;
    out[tx as usize] = shared_get(&scratch, neighbour);
}

///////////
// Tests //
///////////

#[cfg(test)]
mod host_tests {
    use super::*;
    use cubecl_utils_rs::prelude::GpuLimits;

    /// S4: the key transform must be a strictly monotone map on the values that
    /// can actually reach it, and the sentinel must sort above every finite
    /// non-negative distance.
    #[test]
    fn test_radix_key_is_monotone() {
        let values = [
            0.0f32,
            f32::MIN_POSITIVE,
            1e-30,
            1e-6,
            0.5,
            1.0,
            2.0,
            1e6,
            3.4e38,
            f32::MAX,
        ];

        for w in values.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            assert!(
                radix_key_host(lo) < radix_key_host(hi),
                "key order broke between {lo} and {hi}: {} vs {}",
                radix_key_host(lo),
                radix_key_host(hi)
            );
        }
    }

    /// Negative distances are not produced by any metric here, but the map has
    /// a sign arm and an untested arm is a liability.
    #[test]
    fn test_radix_key_handles_negatives() {
        let values = [-1e6f32, -1.0, -0.5, -f32::MIN_POSITIVE, 0.0, 0.5, 1.0];

        for w in values.windows(2) {
            assert!(
                radix_key_host(w[0]) < radix_key_host(w[1]),
                "key order broke between {} and {}",
                w[0],
                w[1]
            );
        }
    }

    /// The reducers pad unfilled slots with `f32::MAX`, so the sentinel has to
    /// rank last among everything a distance metric can produce.
    #[test]
    fn test_sentinel_sorts_last() {
        let sentinel = radix_key_host(f32::MAX);
        for d in [0.0f32, 1e-30, 1.0, 1e6, 3.4e38] {
            assert!(
                radix_key_host(d) < sentinel,
                "distance {d} did not sort below the f32::MAX sentinel"
            );
        }
        // Infinity and NaN rank above the sentinel, which is the safe direction:
        // they can only ever be selected after every real candidate.
        assert!(radix_key_host(f32::INFINITY) > sentinel);
        assert!(radix_key_host(f32::NAN) > sentinel);
    }

    /// `radix_key_inv` must round-trip bit for bit.
    ///
    /// The exhaustive reducer writes its selected distances by inverting the key
    /// rather than re-reading them, so a lossy inverse would corrupt every output
    /// distance while leaving the ordering, the indices and the sentinel padding
    /// all perfectly plausible.
    #[test]
    fn test_radix_key_inverse_round_trips() {
        let values = [
            0.0f32,
            f32::MIN_POSITIVE,
            1e-30,
            1e-6,
            0.25,
            1.0,
            2.0,
            3.5,
            1e6,
            3.4e38,
            f32::MAX,
            -0.0,
            -1.0,
            -1e6,
        ];

        for v in values {
            let key = radix_key_host(v);
            // Host mirror of the device inverse.
            let s = key >> 31;
            let mask = (s * 0x8000_0000) | ((1 - s) * 0xFFFF_FFFF);
            let back = f32::from_bits(key ^ mask);
            assert_eq!(
                back.to_bits(),
                v.to_bits(),
                "round trip lost {v}: got {back}"
            );
        }
    }

    /// Host-only budget tests for the shape guard.
    ///
    /// These replace the coverage `test_plan_topk_merge_boundary` used to give:
    /// pure functions of `GpuLimits`, exercised at synthetic budgets with no
    /// device present, which is the pattern every other staging plan in this
    /// crate follows.
    #[test]
    fn test_radix_select_fits_budgets() {
        let mut limits = GpuLimits {
            max_shared_bytes: 32_768,
            max_cube_count: (65_535, 65_535, 65_535),
            max_units_per_cube: 1024,
            max_cube_dim: (1024, 1024, 1024),
            max_binding_bytes: 4_294_967_292,
            plane_size_min: 32,
            plane_size_max: 32,
        };

        // Non-f32 elements can never take this path: the key transform asserts
        // at expand time, which panics during kernel compilation.
        assert!(!radix_select_fits(50, 8, 32, &limits));

        // The workgroup has to divide the bin count and fit the device.
        assert!(radix_select_fits(50, 4, 32, &limits));
        assert!(!radix_select_fits(50, 4, 0, &limits));
        assert!(
            !radix_select_fits(50, 4, 24, &limits),
            "24 does not divide 256"
        );
        assert!(radix_select_fits(50, 4, 256, &limits));
        limits.max_units_per_cube = 64;
        assert!(
            !radix_select_fits(50, 4, 256, &limits),
            "over the unit budget"
        );
        limits.max_units_per_cube = 1024;
        limits.max_cube_dim.0 = 64;
        assert!(
            !radix_select_fits(50, 4, 256, &limits),
            "over the cube x extent"
        );
        limits.max_cube_dim.0 = 1024;

        // Footprint boundary against a 32 KiB budget, both sides.
        let wg = 32usize;
        let mut k = 1usize;
        while radix_select_fits(k + 1, 4, wg, &limits) {
            k += 1;
        }
        assert!(radix_select_smem_bytes(k, wg, SURV_ARRAYS_DENSE) <= 32_768);
        assert!(radix_select_smem_bytes(k + 1, wg, SURV_ARRAYS_DENSE) > 32_768);

        // Smaller devices must shrink the admissible k, larger ones grow it.
        for budget in [16_384usize, 32_768, 49_152, 65_536] {
            limits.max_shared_bytes = budget;
            let mut fits = 1usize;
            while radix_select_fits(fits + 1, 4, wg, &limits) {
                fits += 1;
            }
            assert!(
                radix_select_smem_bytes(fits, wg, SURV_ARRAYS_DENSE) <= budget,
                "budget {budget}: admitted a footprint it cannot hold"
            );
        }
    }

    /// The two kernels' footprints must match what they actually allocate.
    ///
    /// Over-allocating shared memory fails silently on this backend, so a
    /// formula that drifts from the `SharedMemory::new` calls is the worst kind
    /// of bug available here.
    #[test]
    fn test_smem_formula_matches_allocations() {
        let (k, wg) = (250usize, 32usize);

        // hist 256, partials wg, bcast 2, st ST_SLOTS, found 1, plus survivors.
        let fixed = (RADIX_BINS + wg + 2 + ST_SLOTS + 1) * 4;
        assert_eq!(
            radix_select_smem_bytes(k, wg, SURV_ARRAYS_IVF),
            fixed + 2 * k * 4,
            "IVF kernel stages surv_key and surv_pos"
        );
        assert_eq!(
            radix_select_smem_bytes(k, wg, SURV_ARRAYS_DENSE),
            fixed + 3 * k * 4,
            "dense kernel also stages surv_idx"
        );
        assert!(
            radix_select_smem_bytes(k, wg, SURV_ARRAYS_DENSE)
                > radix_select_smem_bytes(k, wg, SURV_ARRAYS_IVF),
            "the guard checks the larger of the two, so dense must be larger"
        );
    }

    /// The digit extraction the passes rely on has to agree with the key.
    #[test]
    fn test_digit_extraction_orders_buckets() {
        let a = radix_key_host(1.0);
        let b = radix_key_host(2.0);
        assert!((a >> 24) <= (b >> 24), "top digit is not monotone");
    }
}

#[cfg(test)]
#[cfg(feature = "gpu-tests")]
mod gpu_smoke {
    use super::*;
    use crate::gpu::dist_gpu::{extract_topk, init_topk, reduce_ivf_topk};
    use crate::gpu::WORKGROUP_SIZE_X;
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
    use cubecl_utils_rs::prelude::*;

    /// The capability query must agree with what each runtime actually does.
    ///
    /// The CubeCL CPU runtime raises `not yet implemented: atomic<u32>` while
    /// compiling the kernel, on a background thread, so the launch silently does
    /// nothing and the caller reads back stale memory. That is why the guard is
    /// a capability query and not an assumption, and why it is asserted here
    /// rather than trusted: this test failing is the difference between the
    /// exhaustive path falling back correctly and returning garbage.
    // Compares the capability query across runtimes, so it needs the CubeCL
    // software CPU backend. The rest of this module is host-only and stays on
    // the plain `gpu` feature.
    #[cfg(feature = "gpu-cpu")]
    #[test]
    fn test_atomic_capability_query_matches_runtimes() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};

        let wgpu_client = WgpuRuntime::client(&WgpuDevice::default());
        assert!(
            atomic_u32_add_supported::<WgpuRuntime>(&wgpu_client),
            "wgpu should report u32 atomic add"
        );

        let cpu_client = CpuRuntime::client(&CpuDevice);
        assert!(
            !atomic_u32_add_supported::<CpuRuntime>(&cpu_client),
            "the CPU runtime does not implement atomic<u32>; if this now passes, \
             the fallback in radix_select_usable can be relaxed"
        );

        // And the composite guard must follow it, so the reducers fall back.
        let cpu_limits = GpuLimits::from_client(&cpu_client);
        assert!(
            !radix_select_usable::<CpuRuntime>(&cpu_client, 15, 4, 32, &cpu_limits),
            "radix select must not be selected on a runtime without u32 atomics"
        );
    }

    /// S1: shared-memory atomics must count exactly, both spread and contended.
    #[test]
    fn test_shared_memory_atomics_count_exactly() {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let out = GpuTensor::<WgpuRuntime, u32>::empty(vec![256], &client).unwrap();

        unsafe {
            smoke_atomic_hist::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(32, 1),
                out.clone().into_tensor_arg(),
            );
        }

        let got = out.read(&client).unwrap();

        assert_eq!(
            got[0], 3201,
            "contended bin is wrong: shared-memory atomics do not accumulate correctly"
        );
        for (bin, count) in got.iter().enumerate().skip(1) {
            assert_eq!(
                *count, 1,
                "bin {bin} counted {count}, expected exactly 1 from the spread phase"
            );
        }
    }

    /// S2: the device key transform must match the host mirror bit for bit.
    #[test]
    fn test_radix_key_matches_host() {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let values: Vec<f32> = vec![
            0.0,
            f32::MIN_POSITIVE,
            1e-30,
            1e-6,
            0.25,
            0.5,
            1.0,
            2.0,
            3.5,
            1e6,
            3.4e38,
            f32::MAX,
            -1.0,
            -0.5,
            -1e6,
        ];
        let n = values.len();

        let input = GpuTensor::<WgpuRuntime, f32>::from_slice(&values, vec![n], &client).unwrap();
        let out = GpuTensor::<WgpuRuntime, u32>::empty(vec![n], &client).unwrap();

        unsafe {
            smoke_radix_key::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static((n as u32).div_ceil(32), 1, 1),
                CubeDim::new_2d(32, 1),
                input.into_tensor_arg(),
                out.clone().into_tensor_arg(),
            );
        }

        let got = out.read(&client).unwrap();

        for (i, v) in values.iter().enumerate() {
            assert_eq!(
                got[i],
                radix_key_host(*v),
                "device key for {v} was {:#010x}, host says {:#010x}",
                got[i],
                radix_key_host(*v)
            );
        }
    }

    /// Ground truth: select the `k` smallest under the total order
    /// `(distance, position)`, padding with the sentinel.
    ///
    /// This is the specification, not a reimplementation of the kernel. It is
    /// also exactly what `reduce_ivf_topk` produces: that kernel accepts on
    /// strict `<` while scanning positions upwards, so the earliest position
    /// wins among bit-identical distances.
    ///
    /// ### Params
    ///
    /// * `dists` - Candidate distances for one query
    /// * `indices` - Candidate database ids for one query
    /// * `count` - Valid prefix length
    /// * `k` - Neighbours to select
    ///
    /// ### Returns
    ///
    /// Selected distances and ids, both length `k`.
    fn cpu_select(dists: &[f32], indices: &[u32], count: usize, k: usize) -> (Vec<f32>, Vec<u32>) {
        let mut order: Vec<usize> = (0..count).collect();
        order.sort_by(|&a, &b| {
            dists[a]
                .partial_cmp(&dists[b])
                .expect("no NaN in the reference data")
                .then(a.cmp(&b))
        });

        let mut od = vec![f32::MAX; k];
        let mut oi = vec![0u32; k];
        for (slot, &pos) in order.iter().take(k).enumerate() {
            od[slot] = dists[pos];
            oi[slot] = indices[pos];
        }
        (od, oi)
    }

    /// Launch the radix reducer over synthetic candidate buffers.
    ///
    /// ### Params
    ///
    /// * `candidate_dists` - Distances `[n_queries, max_candidates]`
    /// * `candidate_indices` - Ids, same shape
    /// * `candidates_per_query` - Valid prefix per query
    /// * `n_queries` - Query count
    /// * `max_candidates` - Row stride
    /// * `k` - Neighbours
    ///
    /// ### Returns
    ///
    /// Flattened output distances and ids, both `[n_queries, k]`.
    fn run_radix_ivf(
        candidate_dists: &[f32],
        candidate_indices: &[u32],
        candidates_per_query: &[u32],
        n_queries: usize,
        max_candidates: usize,
        k: usize,
    ) -> (Vec<f32>, Vec<u32>) {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let limits = GpuLimits::from_client(&client);

        let cd = GpuTensor::<WgpuRuntime, f32>::from_slice(
            candidate_dists,
            vec![n_queries, max_candidates],
            &client,
        )
        .unwrap();
        let ci = GpuTensor::<WgpuRuntime, u32>::from_slice(
            candidate_indices,
            vec![n_queries, max_candidates],
            &client,
        )
        .unwrap();
        let cpq = GpuTensor::<WgpuRuntime, u32>::from_slice(
            candidates_per_query,
            vec![n_queries],
            &client,
        )
        .unwrap();

        let od = GpuTensor::<WgpuRuntime, f32>::empty(vec![n_queries, k], &client).unwrap();
        let oi = GpuTensor::<WgpuRuntime, u32>::empty(vec![n_queries, k], &client).unwrap();

        let wg = WORKGROUP_SIZE_X as usize;
        let (gx, gy) = grid_2d(n_queries as u32, &limits).unwrap();
        unsafe {
            radix_select_ivf_topk::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                cd.into_tensor_arg(),
                ci.into_tensor_arg(),
                cpq.into_tensor_arg(),
                od.clone().into_tensor_arg(),
                oi.clone().into_tensor_arg(),
                k as u32,
                k,
                wg,
            );
        }

        (od.read(&client).unwrap(), oi.read(&client).unwrap())
    }

    /// Compare the kernel against the specification over every output slot.
    ///
    /// ### Params
    ///
    /// * `dists` - Candidate distances
    /// * `indices` - Candidate ids
    /// * `counts` - Valid prefix per query
    /// * `nq` - Query count
    /// * `mc` - Row stride
    /// * `k` - Neighbours
    /// * `label` - Context for panic messages
    fn assert_matches_reference(
        dists: &[f32],
        indices: &[u32],
        counts: &[u32],
        nq: usize,
        mc: usize,
        k: usize,
        label: &str,
    ) {
        let (gd, gi) = run_radix_ivf(dists, indices, counts, nq, mc, k);

        for q in 0..nq {
            let (ed, ei) = cpu_select(
                &dists[q * mc..(q + 1) * mc],
                &indices[q * mc..(q + 1) * mc],
                counts[q] as usize,
                k,
            );
            for s in 0..k {
                assert_eq!(
                    gd[q * k + s].to_bits(),
                    ed[s].to_bits(),
                    "{label}: query {q} slot {s} distance, got {} want {}",
                    gd[q * k + s],
                    ed[s]
                );
                assert_eq!(
                    gi[q * k + s],
                    ei[s],
                    "{label}: query {q} slot {s} index, got {} want {}",
                    gi[q * k + s],
                    ei[s]
                );
            }
        }
    }

    /// The serial `reduce_ivf_topk` fallback must satisfy the same
    /// specification as the radix kernel.
    ///
    /// It is what the IVF path takes for non-f32 elements and for runtimes
    /// without `u32` atomics, so it is production code, not just a reference.
    /// Unlike the radix path it inserts straight into the output buffer and so
    /// requires `init_topk` seeding first, which is easy to forget at a call
    /// site and produces garbage rather than an error when omitted.
    #[test]
    fn test_serial_ivf_fallback_matches_reference() {
        let (nq, mc, k) = (16usize, 256usize, 30usize);
        let dists: Vec<f32> = (0..nq * mc)
            .map(|i| ((i * 7919 + 13) % 100_003) as f32 * 0.001)
            .collect();
        let indices: Vec<u32> = (0..nq * mc).map(|i| (i % mc) as u32 * 3 + 1).collect();
        let counts = vec![mc as u32; nq];

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let limits = GpuLimits::from_client(&client);

        let cd = GpuTensor::<WgpuRuntime, f32>::from_slice(&dists, vec![nq, mc], &client).unwrap();
        let ci =
            GpuTensor::<WgpuRuntime, u32>::from_slice(&indices, vec![nq, mc], &client).unwrap();
        let cpq = GpuTensor::<WgpuRuntime, u32>::from_slice(&counts, vec![nq], &client).unwrap();
        let od = GpuTensor::<WgpuRuntime, f32>::empty(vec![nq, k], &client).unwrap();
        let oi = GpuTensor::<WgpuRuntime, u32>::empty(vec![nq, k], &client).unwrap();

        let (init_gy, init_gz) = grid_2d((nq as u32).div_ceil(4), &limits).unwrap();
        unsafe {
            init_topk::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static((k as u32).div_ceil(WORKGROUP_SIZE_X), init_gy, init_gz),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
                od.clone().into_tensor_arg(),
                oi.clone().into_tensor_arg(),
                4,
            );
        }

        let (gx, gy) = grid_2d((nq as u32).div_ceil(WORKGROUP_SIZE_X), &limits).unwrap();
        unsafe {
            reduce_ivf_topk::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static(gx, gy, 1),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                cd.into_tensor_arg(),
                ci.into_tensor_arg(),
                cpq.into_tensor_arg(),
                od.clone().into_tensor_arg(),
                oi.clone().into_tensor_arg(),
            );
        }

        let gd = od.read(&client).unwrap();
        let gi = oi.read(&client).unwrap();

        for q in 0..nq {
            let (ed, ei) = cpu_select(
                &dists[q * mc..(q + 1) * mc],
                &indices[q * mc..(q + 1) * mc],
                mc,
                k,
            );
            for s in 0..k {
                assert_eq!(
                    gd[q * k + s].to_bits(),
                    ed[s].to_bits(),
                    "serial fallback, query {q} slot {s}"
                );
                assert_eq!(
                    gi[q * k + s],
                    ei[s],
                    "serial fallback index, query {q} slot {s}"
                );
            }
        }
    }

    /// The ordinary case: distinct distances, full rows.
    #[test]
    fn test_radix_ivf_matches_reference() {
        let (nq, mc, k) = (64usize, 512usize, 50usize);
        let dists: Vec<f32> = (0..nq * mc)
            .map(|i| ((i * 7919 + 13) % 100_003) as f32 * 0.001)
            .collect();
        let indices: Vec<u32> = (0..nq * mc).map(|i| (i % mc) as u32 * 3 + 1).collect();
        let counts = vec![mc as u32; nq];

        assert_matches_reference(&dists, &indices, &counts, nq, mc, k, "distinct");
    }

    /// Every distance identical, so the whole selection is decided by the
    /// position tie-break passes. This is the case the four key passes alone
    /// cannot resolve, and the one most likely to expose a wrong descent.
    #[test]
    fn test_radix_ivf_all_duplicates() {
        let (nq, mc, k) = (16usize, 256usize, 30usize);
        let dists = vec![0.25f32; nq * mc];
        let indices: Vec<u32> = (0..nq * mc).map(|i| (i % mc) as u32 * 7 + 2).collect();
        let counts = vec![mc as u32; nq];

        assert_matches_reference(&dists, &indices, &counts, nq, mc, k, "all duplicates");
    }

    /// Heavy but partial duplication: many ties, only some at the boundary.
    #[test]
    fn test_radix_ivf_heavy_ties() {
        let (nq, mc, k) = (16usize, 512usize, 40usize);
        // Only eight distinct values across 512 candidates, so the k-th and
        // (k+1)-th are always equal and the boundary always needs the tie pass.
        let dists: Vec<f32> = (0..nq * mc).map(|i| (i % 8) as f32 * 0.5).collect();
        let indices: Vec<u32> = (0..nq * mc).map(|i| (i % mc) as u32).collect();
        let counts = vec![mc as u32; nq];

        assert_matches_reference(&dists, &indices, &counts, nq, mc, k, "heavy ties");
    }

    /// Fewer candidates than slots: the tail must stay sentinel-padded, which is
    /// what the IVF host filter keys off.
    #[test]
    fn test_radix_ivf_fewer_than_k() {
        let (nq, mc, k) = (8usize, 128usize, 40usize);
        let dists: Vec<f32> = (0..nq * mc)
            .map(|i| ((i * 31 + 5) % 997) as f32 * 0.01)
            .collect();
        let indices: Vec<u32> = (0..nq * mc).map(|i| (i % mc) as u32).collect();
        // Query q sees q * 3 candidates, so several rows fall short of k.
        let counts: Vec<u32> = (0..nq).map(|q| (q * 3) as u32).collect();

        assert_matches_reference(&dists, &indices, &counts, nq, mc, k, "short rows");

        let (gd, _) = run_radix_ivf(&dists, &indices, &counts, nq, mc, k);
        for q in 0..nq {
            let valid = (q * 3).min(k);
            for s in valid..k {
                assert_eq!(
                    gd[q * k + s],
                    f32::MAX,
                    "query {q} slot {s} should be sentinel-padded"
                );
            }
        }
    }

    /// High k, well past where a per-lane merge reducer runs out of shared
    /// memory budget.
    #[test]
    fn test_radix_ivf_high_k() {
        for k in [250usize, 1024] {
            let (nq, mc) = (8usize, 2048usize);
            let dists: Vec<f32> = (0..nq * mc)
                .map(|i| ((i * 7919 + 13) % 100_003) as f32 * 0.001)
                .collect();
            let indices: Vec<u32> = (0..nq * mc).map(|i| (i % mc) as u32).collect();
            let counts = vec![mc as u32; nq];

            assert_matches_reference(&dists, &indices, &counts, nq, mc, k, &format!("k={k}"));
        }
    }

    /// Same input twice must give byte-identical output. Radix select compacts
    /// survivors through an atomic counter, so slot order is nondeterministic;
    /// only the total order makes the final ranking reproducible.
    #[test]
    fn test_radix_ivf_is_deterministic() {
        let (nq, mc, k) = (32usize, 512usize, 50usize);
        let dists: Vec<f32> = (0..nq * mc).map(|i| (i % 64) as f32 * 0.125).collect();
        let indices: Vec<u32> = (0..nq * mc).map(|i| (i % mc) as u32).collect();
        let counts = vec![mc as u32; nq];

        let (d1, i1) = run_radix_ivf(&dists, &indices, &counts, nq, mc, k);
        let (d2, i2) = run_radix_ivf(&dists, &indices, &counts, nq, mc, k);

        assert_eq!(i1, i2, "indices differed between two identical runs");
        assert_eq!(
            d1.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
            d2.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
            "distances differed between two identical runs"
        );
    }

    /// Run one of the exhaustive reducers over a chunked DB, mirroring what
    /// `query_batch_gpu` does: a reused `[nq, chunk]` buffer, one launch per
    /// chunk, `chunk_offset` mapping columns to global ids.
    ///
    /// ### Params
    ///
    /// * `full` - Full distance matrix `[nq, ndb]`, row-major
    /// * `nq` - Query count
    /// * `ndb` - Database size
    /// * `k` - Neighbours
    /// * `chunk` - Columns per launch
    /// * `radix` - Whether to drive the radix kernel or the serial reference
    ///
    /// ### Returns
    ///
    /// Selected distances and global ids, both `[nq, k]`.
    fn run_dense_chunked(
        full: &[f32],
        nq: usize,
        ndb: usize,
        k: usize,
        chunk: usize,
        radix: bool,
    ) -> (Vec<f32>, Vec<u32>) {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let limits = GpuLimits::from_client(&client);

        let od = GpuTensor::<WgpuRuntime, f32>::empty(vec![nq, k], &client).unwrap();
        let oi = GpuTensor::<WgpuRuntime, u32>::empty(vec![nq, k], &client).unwrap();

        let (init_gy, init_gz) = grid_2d((nq as u32).div_ceil(4), &limits).unwrap();
        unsafe {
            init_topk::launch_unchecked::<f32, WgpuRuntime>(
                &client,
                CubeCount::Static((k as u32).div_ceil(WORKGROUP_SIZE_X), init_gy, init_gz),
                CubeDim::new_2d(WORKGROUP_SIZE_X, 4),
                od.clone().into_tensor_arg(),
                oi.clone().into_tensor_arg(),
                4,
            );
        }

        let wg = WORKGROUP_SIZE_X as usize;
        let mut start = 0usize;
        while start < ndb {
            let this = chunk.min(ndb - start);

            // The host reuses one buffer per chunk; the test rebuilds it, which
            // is equivalent and keeps the slicing obvious.
            let mut block = vec![0f32; nq * this];
            for q in 0..nq {
                block[q * this..(q + 1) * this]
                    .copy_from_slice(&full[q * ndb + start..q * ndb + start + this]);
            }
            let blk =
                GpuTensor::<WgpuRuntime, f32>::from_slice(&block, vec![nq, this], &client).unwrap();

            if radix {
                let (gx, gy) = grid_2d(nq as u32, &limits).unwrap();
                unsafe {
                    radix_select_topk::launch_unchecked::<f32, WgpuRuntime>(
                        &client,
                        CubeCount::Static(gx, gy, 1),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                        blk.into_tensor_arg(),
                        od.clone().into_tensor_arg(),
                        oi.clone().into_tensor_arg(),
                        start as u32,
                        this as u32,
                        k as u32,
                        k,
                        wg,
                    );
                }
            } else {
                let (gx, gy) = grid_2d((nq as u32).div_ceil(WORKGROUP_SIZE_X), &limits).unwrap();
                unsafe {
                    extract_topk::launch_unchecked::<f32, WgpuRuntime>(
                        &client,
                        CubeCount::Static(gx, gy, 1),
                        CubeDim::new_2d(WORKGROUP_SIZE_X, 1),
                        blk.into_tensor_arg(),
                        od.clone().into_tensor_arg(),
                        oi.clone().into_tensor_arg(),
                        start as u32,
                        this as u32,
                        k as u32,
                        k,
                    );
                }
            }

            start += this;
        }

        (od.read(&client).unwrap(), oi.read(&client).unwrap())
    }

    /// Check the chunked radix reducer against both the specification and the
    /// serial kernel it replaces.
    ///
    /// ### Params
    ///
    /// * `full` - Full distance matrix
    /// * `nq` - Query count
    /// * `ndb` - Database size
    /// * `k` - Neighbours
    /// * `chunk` - Columns per launch
    /// * `label` - Context for panic messages
    fn assert_dense_matches(
        full: &[f32],
        nq: usize,
        ndb: usize,
        k: usize,
        chunk: usize,
        label: &str,
    ) {
        let (rd, ri) = run_dense_chunked(full, nq, ndb, k, chunk, true);
        let (sd, si) = run_dense_chunked(full, nq, ndb, k, chunk, false);

        for q in 0..nq {
            let ids: Vec<u32> = (0..ndb as u32).collect();
            let (ed, ei) = cpu_select(&full[q * ndb..(q + 1) * ndb], &ids, ndb, k);

            for s in 0..k {
                assert_eq!(
                    rd[q * k + s].to_bits(),
                    ed[s].to_bits(),
                    "{label}: radix vs spec, query {q} slot {s}: got {} want {}",
                    rd[q * k + s],
                    ed[s]
                );
                assert_eq!(
                    ri[q * k + s],
                    ei[s],
                    "{label}: radix vs spec index, query {q} slot {s}: got {} want {}",
                    ri[q * k + s],
                    ei[s]
                );
                // The serial kernel is the shipped reference, so any divergence
                // from it is a behaviour change regardless of what the spec says.
                assert_eq!(
                    rd[q * k + s].to_bits(),
                    sd[q * k + s].to_bits(),
                    "{label}: radix vs extract_topk, query {q} slot {s}"
                );
                assert_eq!(
                    ri[q * k + s],
                    si[q * k + s],
                    "{label}: radix vs extract_topk index, query {q} slot {s}"
                );
            }
        }
    }

    /// Single chunk, distinct distances.
    #[test]
    fn test_radix_dense_single_chunk() {
        let (nq, ndb, k) = (32usize, 1024usize, 40usize);
        let full: Vec<f32> = (0..nq * ndb)
            .map(|i| ((i * 7919 + 13) % 100_003) as f32 * 0.001)
            .collect();
        assert_dense_matches(&full, nq, ndb, k, ndb, "single chunk");
    }

    /// Several chunks, which is where the accumulation contract lives. The
    /// running top-k enters each launch as `k` virtual candidates, so a wrong
    /// tie-break or a lost fold shows up here and nowhere else.
    #[test]
    fn test_radix_dense_multi_chunk() {
        let (nq, ndb, k) = (32usize, 1024usize, 40usize);
        let full: Vec<f32> = (0..nq * ndb)
            .map(|i| ((i * 7919 + 13) % 100_003) as f32 * 0.001)
            .collect();
        for chunk in [128usize, 256, 333] {
            assert_dense_matches(&full, nq, ndb, k, chunk, &format!("chunk={chunk}"));
        }
    }

    /// Chunked and tie-heavy at once: only eight distinct distances, so nearly
    /// every selection decision is made by the position tie-break, and it has to
    /// agree with the serial kernel across chunk boundaries.
    #[test]
    fn test_radix_dense_chunked_heavy_ties() {
        let (nq, ndb, k) = (16usize, 512usize, 30usize);
        let full: Vec<f32> = (0..nq * ndb).map(|i| (i % 8) as f32 * 0.25).collect();
        for chunk in [64usize, 128, 512] {
            assert_dense_matches(&full, nq, ndb, k, chunk, &format!("ties chunk={chunk}"));
        }
    }

    /// Fewer database vectors than `k`: the sentinel seeding must survive as
    /// padding, exactly as the serial kernel leaves it.
    #[test]
    fn test_radix_dense_fewer_than_k() {
        let (nq, ndb, k) = (8usize, 20usize, 50usize);
        let full: Vec<f32> = (0..nq * ndb)
            .map(|i| ((i * 31 + 5) % 997) as f32 * 0.01)
            .collect();
        assert_dense_matches(&full, nq, ndb, k, 8, "short db");

        let (rd, _) = run_dense_chunked(&full, nq, ndb, k, 8, true);
        for q in 0..nq {
            for s in ndb..k {
                assert_eq!(
                    rd[q * k + s],
                    f32::MAX,
                    "query {q} slot {s} should still be sentinel"
                );
            }
        }
    }

    /// High k over multiple chunks, the regime the whole exercise is about.
    #[test]
    fn test_radix_dense_high_k() {
        let (nq, ndb, k) = (8usize, 2048usize, 250usize);
        let full: Vec<f32> = (0..nq * ndb)
            .map(|i| ((i * 7919 + 13) % 100_003) as f32 * 0.001)
            .collect();
        assert_dense_matches(&full, nq, ndb, k, 512, "k=250 chunked");
    }

    /// End to end through the public exhaustive index at a `k` above
    /// [`RADIX_SELECT_MIN_K`], so the production dispatch actually selects the
    /// radix path.
    ///
    /// The kernel-level tests drive `radix_select_topk` directly, which says
    /// nothing about whether `query_batch_gpu` routes to it. With the threshold
    /// in place the existing integration tests all sit below it, so without this
    /// the production radix path would have no end-to-end coverage at all.
    #[test]
    fn test_exhaustive_index_uses_radix_above_threshold() {
        use crate::gpu::exhaustive_gpu::ExhaustiveIndexGpu;
        use crate::utils::dist::Dist;
        use faer::Mat;

        let (n, dim, k) = (600usize, 8usize, 100usize);
        assert!(k >= RADIX_SELECT_MIN_K, "test must exercise the radix path");

        let data = Mat::from_fn(n, dim, |i, j| {
            let h = ((i * dim + j) as u64).wrapping_mul(2_654_435_761) % 10_007;
            h as f32 * 1e-3
        });

        let index = ExhaustiveIndexGpu::<f32, WgpuRuntime>::new(
            data.as_ref(),
            Dist::SquaredEuclidean,
            WgpuDevice::default(),
        )
        .unwrap();

        let (indices, dists) = index.query_batch(data.as_ref(), k, false).unwrap();

        for q in 0..n {
            // Every query is its own nearest neighbour at distance zero.
            assert_eq!(indices[q][0], q, "query {q} did not find itself first");
            assert!(
                dists[q][0] < 1e-6,
                "query {q} self-distance was {}",
                dists[q][0]
            );

            assert_eq!(indices[q].len(), k);
            assert!(
                dists[q].windows(2).all(|w| w[0] <= w[1]),
                "query {q} row is not sorted ascending"
            );
            assert!(
                dists[q].iter().all(|d| *d < f32::MAX),
                "query {q} kept a sentinel, so the reducer did not fill the row"
            );
        }

        // Brute-force the first few rows against the spec to catch a wrong
        // selection rather than merely a non-empty one.
        for q in 0..8 {
            let mut all: Vec<(f32, usize)> = (0..n)
                .map(|d| {
                    let s: f32 = (0..dim)
                        .map(|j| {
                            let diff = data[(q, j)] - data[(d, j)];
                            diff * diff
                        })
                        .sum();
                    (s, d)
                })
                .collect();
            all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));

            for slot in 0..k {
                assert!(
                    (dists[q][slot] - all[slot].0).abs() <= 1e-5 * all[slot].0.max(1.0),
                    "query {q} slot {slot}: got {} want {}",
                    dists[q][slot],
                    all[slot].0
                );
            }
        }
    }

    /// S3: a `#[cube]` helper taking a shared-memory reference round-trips.
    #[test]
    fn test_shared_memory_helper_round_trips() {
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let out = GpuTensor::<WgpuRuntime, u32>::empty(vec![32], &client).unwrap();

        unsafe {
            smoke_shared_helper::launch_unchecked::<WgpuRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_2d(32, 1),
                out.clone().into_tensor_arg(),
            );
        }

        let got = out.read(&client).unwrap();

        for tx in 0..32u32 {
            let neighbour = (tx + 31) % 32;
            assert_eq!(
                got[tx as usize],
                neighbour * 7 + 1,
                "lane {tx} read the wrong slot back through the helper"
            );
        }
    }
}
