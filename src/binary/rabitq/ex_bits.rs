//! Multi-bit RaBitQ+ codes.
//!
//! The one-bit encoder in [`crate::binary::rabitq::quantiser`] keeps only the
//! sign of each rotated residual coordinate. RaBitQ+ adds `ex_bits` magnitude
//! bits below that sign, giving a `total_bits = ex_bits + 1` code whose
//! estimate converges on the true distance as the width grows. That is what
//! lets a graph index drop the float vectors entirely and still answer
//! accurately: the multi-bit code replaces the exact distance rather than
//! merely screening for it.
//!
//! ### References
//!
//! Gao et al., "Practical and Asymptotically Optimal Quantization of
//! High-Dimensional Vectors in Euclidean Space for Approximate Nearest Neighbor
//! Search", SIGMOD 2025.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::prelude::*;

////////////
// Consts //
////////////

/// Widest extended code supported, matching the reference's `[1, 8]`.
pub const MAX_EX_BITS: usize = 8;

/// Slack factor in the error bound, `kConstEpsilon` in the reference.
///
/// Chosen there so the bound holds with high probability rather than
/// absolutely; it is not derived from the data.
const ERROR_EPSILON: f64 = 1.9;

/// Fraction of the scale interval the sweep starts from, per `ex_bits`.
///
/// Index `b` is the start for `ex_bits = b`. Below these fractions the cosine
/// is never competitive, so the sweep would only burn events getting there.
/// Copied from the reference's `kTightStart`; retuning it changes which scale
/// the search settles on.
const TIGHT_START: [f64; MAX_EX_BITS + 1] = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81];

/// Rounding slack in the `t_const` path, `kEps` in the reference.
const FAST_QUANT_EPS: f64 = 1e-5;

/// Vectors probed when averaging a dimension-wide scale.
const CONST_SCALE_PROBES: usize = 100;

////////////
// ExCode //
////////////

/// A multi-bit RaBitQ+ encoding of one vector.
///
/// Sign and magnitude are stored apart rather than as one `total_bits` code.
/// That is not a layout preference: at the widest setting the combined code
/// needs nine bits, and keeping them apart is also what lets a search score the
/// cheap sign-only estimate first and reach for the magnitude bits only when
/// the candidate survives.
#[cfg_attr(
    feature = "serialise",
    derive(serde::Serialize, serde::Deserialize),
    serde(bound = "T: AnnSearchFloat")
)]
#[derive(Clone, Debug)]
pub struct ExEncoding<T> {
    /// One sign bit per coordinate, `dim / 8` bytes, set when the residual is
    /// positive
    pub sign_code: Vec<u8>,
    /// Packed magnitude levels, `dim * ex_bits / 8` bytes, empty at `ex_bits` 0
    pub ex_code: Vec<u8>,
    /// Query-independent additive term
    pub f_add: T,
    /// Multiplier on the query inner product
    pub f_rescale: T,
    /// Error bound coefficient, multiplied by `||q - c||` at query time
    pub f_error: T,
}

/////////////
// Scaling //
/////////////

/// Level a coordinate takes at a given scale.
///
/// The comparison is written as a division rather than a multiplication because
/// the sweep's breakpoints are divisions: multiplying instead can round to the
/// far side of an event and desynchronise the incremental sums from the levels
/// they are supposed to describe.
///
/// ### Params
///
/// * `magnitude` - Absolute residual coordinate, non-negative and finite
/// * `t` - Scale, non-negative and finite
/// * `max_level` - Largest legal level, `2^ex_bits - 1`
///
/// ### Returns
///
/// The level in `0..=max_level`
#[inline]
fn level_at_scale(magnitude: f64, t: f64, max_level: u32) -> u32 {
    if magnitude == 0.0 {
        return 0;
    }

    let mut level = (t * magnitude).min(max_level as f64) as u32;
    if level < max_level && (level as f64 + 1.0) / magnitude <= t {
        level += 1;
    } else if level > 0 && (level as f64) / magnitude > t {
        level -= 1;
    }
    level
}

/// One coordinate's next level-changing scale.
#[derive(Clone, Copy, PartialEq)]
struct Event {
    /// Scale at which this coordinate steps up
    t: f64,
    /// Coordinate index
    coord: usize,
}

impl Eq for Event {}

impl Ord for Event {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed so `BinaryHeap` yields the smallest scale first. Ties break
        // on the index purely to keep the order total and reproducible.
        other
            .t
            .partial_cmp(&self.t)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.coord.cmp(&self.coord))
    }
}

impl PartialOrd for Event {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Scale maximising the cosine between a residual and its quantised direction.
///
/// With `a` the normalised absolute residual and `q_d = level_d + 0.5`, the
/// cosine is `sum(a_d q_d) / sqrt(sum(q_d^2))`. Both sums change by a closed
/// form each time one coordinate steps up a level, and levels only step at
/// `(level + 1) / a_d`, so sweeping those breakpoints in order evaluates every
/// distinct quantisation the interval contains without revisiting a coordinate.
///
/// ### Params
///
/// * `abs_residual` - Absolute residual coordinates, normalised to unit length
/// * `ex_bits` - Magnitude bits, `1..=MAX_EX_BITS`
///
/// ### Returns
///
/// The scale, or zero when every coordinate is zero
pub fn best_rescale_factor(abs_residual: &[f64], ex_bits: usize) -> f64 {
    let dim = abs_residual.len();
    if dim == 0 {
        return 0.0;
    }

    let max_o = abs_residual.iter().cloned().fold(0.0f64, f64::max);
    if max_o == 0.0 {
        return 0.0;
    }

    let max_level = (1u32 << ex_bits) - 1;
    let t_end = (max_level as f64 + 10.0) / max_o;
    let t_start = t_end * TIGHT_START[ex_bits];

    let mut levels = vec![0u32; dim];
    let mut denominator = dim as f64 * 0.25;
    let mut numerator = 0.0f64;
    let mut events = BinaryHeap::with_capacity(dim);

    for (i, &magnitude) in abs_residual.iter().enumerate() {
        let level = level_at_scale(magnitude, t_start, max_level);
        levels[i] = level;
        denominator += (level as f64 * level as f64) + level as f64;
        numerator += (level as f64 + 0.5) * magnitude;

        // A saturated coordinate never steps again, and a zero one has no
        // finite breakpoint at all.
        if magnitude > 0.0 && level < max_level {
            let next = (level as f64 + 1.0) / magnitude;
            if next < t_end {
                events.push(Event { t: next, coord: i });
            }
        }
    }

    let mut max_ip = numerator / denominator.sqrt();
    let mut best_t = t_start;

    while let Some(&Event { t: current_t, .. }) = events.peek() {
        // Every coordinate sharing this breakpoint steps together; evaluating
        // between them would score a quantisation that no scale produces.
        while let Some(&Event { t, coord }) = events.peek() {
            if t != current_t {
                break;
            }
            events.pop();

            levels[coord] += 1;
            // Going from level `k - 1` to `k` moves `q` from `k - 0.5` to
            // `k + 0.5`, so the squared sum gains exactly `2k`.
            denominator += 2.0 * levels[coord] as f64;
            numerator += abs_residual[coord];

            if levels[coord] < max_level {
                let next = (levels[coord] as f64 + 1.0) / abs_residual[coord];
                if next < t_end {
                    events.push(Event { t: next, coord });
                }
            }
        }

        let current_ip = numerator / denominator.sqrt();
        if current_ip > max_ip {
            max_ip = current_ip;
            best_t = current_t;
        }
    }

    best_t
}

/// Average scale over random unit vectors of this shape.
///
/// [`best_rescale_factor`] costs a heap sweep per vector. For a large build the
/// reference instead averages the scale over a sample of random directions once
/// and reuses it, which is what [`quantise_ex`] takes as `t_const`.
///
/// ### Params
///
/// * `dim` - Dimensionality the scale is for
/// * `ex_bits` - Magnitude bits, `1..=MAX_EX_BITS`
/// * `seed` - Random seed, for reproducibility
///
/// ### Returns
///
/// The averaged scale
pub fn const_scaling_factor(dim: usize, ex_bits: usize, seed: u64) -> f64 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };

    let mut total = 0.0f64;
    let mut probe = vec![0.0f64; dim];

    for _ in 0..CONST_SCALE_PROBES {
        for value in probe.iter_mut() {
            *value = next();
        }
        let norm = probe.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm == 0.0 {
            continue;
        }
        for value in probe.iter_mut() {
            *value = (*value / norm).abs();
        }
        total += best_rescale_factor(&probe, ex_bits);
    }

    total / CONST_SCALE_PROBES as f64
}

/// Quantise normalised absolute residual magnitudes into levels.
///
/// ### Params
///
/// * `abs_residual` - Absolute residual coordinates, normalised to unit length
/// * `ex_bits` - Magnitude bits, `1..=MAX_EX_BITS`
/// * `t_const` - Precomputed scale from [`const_scaling_factor`], or `None` to
///   search per vector with [`best_rescale_factor`]
///
/// ### Returns
///
/// `(levels, ipnorm_inv)`, the second being
/// `1 / sum((level + 0.5) * magnitude)`
pub fn quantise_ex(abs_residual: &[f64], ex_bits: usize, t_const: Option<f64>) -> (Vec<u8>, f64) {
    let max_level = (1u32 << ex_bits) - 1;
    let mut levels = vec![0u8; abs_residual.len()];
    let mut ipnorm = 0.0f64;

    match t_const {
        Some(t) => {
            for (i, &magnitude) in abs_residual.iter().enumerate() {
                let level = (((t * magnitude) + FAST_QUANT_EPS) as u32).min(max_level);
                levels[i] = level as u8;
                ipnorm += (level as f64 + 0.5) * magnitude;
            }
        }
        None => {
            let t = best_rescale_factor(abs_residual, ex_bits);
            for (i, &magnitude) in abs_residual.iter().enumerate() {
                let level = level_at_scale(magnitude, t, max_level);
                levels[i] = level as u8;
                ipnorm += (level as f64 + 0.5) * magnitude;
            }
        }
    }

    if ipnorm == 0.0 {
        return (levels, 1.0);
    }
    let ipnorm_inv = 1.0 / ipnorm;
    if ipnorm_inv.is_normal() {
        (levels, ipnorm_inv)
    } else {
        (levels, 1.0)
    }
}

/////////////
// Packing //
/////////////

/// Bytes a packed code occupies.
///
/// ### Params
///
/// * `dim` - Number of coordinates, a multiple of 8
/// * `total_bits` - Bits per coordinate, `1..=MAX_EX_BITS + 1`
///
/// ### Returns
///
/// Byte count
#[inline]
pub fn excode_bytes(dim: usize, total_bits: usize) -> usize {
    dim * total_bits / 8
}

/// Pack per-coordinate levels into a little-endian bit stream.
///
/// ### Params
///
/// * `levels` - One level per coordinate, each below `2^total_bits`
/// * `total_bits` - Bits per coordinate
///
/// ### Returns
///
/// The packed bytes, `levels.len() * total_bits / 8` of them
pub fn pack_excode(levels: &[u8], total_bits: usize) -> Vec<u8> {
    let mut packed = vec![0u8; excode_bytes(levels.len(), total_bits)];

    for (d, &level) in levels.iter().enumerate() {
        let bit = d * total_bits;
        for b in 0..total_bits {
            if level >> b & 1 == 1 {
                let at = bit + b;
                packed[at / 8] |= 1 << (at % 8);
            }
        }
    }

    packed
}

/// Unpack a bit stream written by [`pack_excode`].
///
/// ### Params
///
/// * `packed` - Packed bytes
/// * `dim` - Number of coordinates
/// * `total_bits` - Bits per coordinate
///
/// ### Returns
///
/// One level per coordinate
pub fn unpack_excode(packed: &[u8], dim: usize, total_bits: usize) -> Vec<u8> {
    let mut levels = vec![0u8; dim];

    for (d, level) in levels.iter_mut().enumerate() {
        let bit = d * total_bits;
        for b in 0..total_bits {
            let at = bit + b;
            if packed[at / 8] >> (at % 8) & 1 == 1 {
                *level |= 1 << b;
            }
        }
    }

    levels
}

//////////////
// Encoding //
//////////////

/// Encode a vector against a centroid at `ex_bits + 1` total bits.
///
/// Both inputs are in the rotated frame. The sign occupies the top bit of each
/// level and the magnitude the `ex_bits` below it, so the one-bit code is
/// recoverable from the wide one by taking that top bit.
///
/// ### Params
///
/// * `rotated` - Rotated vector, length `dim`
/// * `centroid_rotated` - Rotated centroid, length `dim`
/// * `ex_bits` - Magnitude bits, `0..=MAX_EX_BITS`. Zero gives the plain
///   one-bit code
/// * `t_const` - Precomputed scale, or `None` to search per vector
///
/// ### Returns
///
/// The encoding, or an error when `ex_bits` is out of range
pub fn encode_ex_bits<T>(
    rotated: &[T],
    centroid_rotated: &[T],
    ex_bits: usize,
    t_const: Option<f64>,
) -> Result<ExEncoding<T>, AnnSearchErrors>
where
    T: AnnSearchFloat,
{
    if ex_bits > MAX_EX_BITS {
        return Err(AnnSearchErrors::RaBitQInvalidExBits {
            ex_bits,
            max: MAX_EX_BITS,
        });
    }

    let dim = rotated.len();

    let residual: Vec<f64> = rotated
        .iter()
        .zip(centroid_rotated)
        .map(|(&v, &c)| (v - c).to_f64().unwrap_or(0.0))
        .collect();

    let l2_sqr: f64 = residual.iter().map(|x| x * x).sum();

    if l2_sqr == 0.0 {
        return Ok(ExEncoding {
            sign_code: vec![0u8; dim / 8],
            ex_code: vec![0u8; excode_bytes(dim, ex_bits)],
            f_add: T::zero(),
            f_rescale: T::zero(),
            f_error: T::zero(),
        });
    }

    let l2_norm = l2_sqr.sqrt();

    let ex_levels = if ex_bits == 0 {
        vec![0u8; dim]
    } else {
        let abs_normalised: Vec<f64> = residual.iter().map(|x| x.abs() / l2_norm).collect();
        let (mut magnitude_levels, _) = quantise_ex(&abs_normalised, ex_bits, t_const);

        let mask = ((1u32 << ex_bits) - 1) as u8;
        for (level, &r) in magnitude_levels.iter_mut().zip(&residual) {
            if r <= 0.0 {
                *level = !*level & mask;
            }
        }
        magnitude_levels
    };

    let mut sign_code = vec![0u8; dim / 8];
    for (d, &r) in residual.iter().enumerate() {
        if r > 0.0 {
            sign_code[d / 8] |= 1 << (d % 8);
        }
    }

    let sign_step = (1u64 << ex_bits) as f64;
    let cb = -(sign_step - 0.5);
    let xu_cb: Vec<f64> = ex_levels
        .iter()
        .zip(&residual)
        .map(|(&u, &r)| u as f64 + if r > 0.0 { sign_step } else { 0.0 } + cb)
        .collect();

    let centroid: Vec<f64> = centroid_rotated
        .iter()
        .map(|&c| c.to_f64().unwrap_or(0.0))
        .collect();

    let ip_resi_xucb: f64 = residual.iter().zip(&xu_cb).map(|(a, b)| a * b).sum();
    let ip_cent_xucb: f64 = centroid.iter().zip(&xu_cb).map(|(a, b)| a * b).sum();
    let xu_cb_sqr: f64 = xu_cb.iter().map(|x| x * x).sum();

    if ip_resi_xucb <= 0.0 || dim < 2 {
        return Ok(ExEncoding {
            sign_code,
            ex_code: pack_excode(&ex_levels, ex_bits),
            f_add: T::from_f64(l2_sqr).unwrap_or_else(T::zero),
            f_rescale: T::zero(),
            f_error: T::zero(),
        });
    }

    let error = l2_norm
        * ERROR_EPSILON
        * (((l2_sqr * xu_cb_sqr) / (ip_resi_xucb * ip_resi_xucb) - 1.0) / (dim as f64 - 1.0))
            .max(0.0)
            .sqrt();

    let f_add = l2_sqr + (2.0 * l2_sqr * ip_cent_xucb / ip_resi_xucb);
    let f_rescale = -2.0 * l2_sqr / ip_resi_xucb;

    Ok(ExEncoding {
        sign_code,
        ex_code: pack_excode(&ex_levels, ex_bits),
        f_add: T::from_f64(f_add).unwrap_or_else(T::zero),
        f_rescale: T::from_f64(f_rescale).unwrap_or_else(T::zero),
        f_error: T::from_f64(2.0 * error).unwrap_or_else(T::zero),
    })
}

/// Estimate the squared distance from a prepared query to an encoded vector.
///
/// The sign and magnitude halves contribute separately, which is what lets the
/// combined code exceed a byte without ever being materialised:
///
/// ```text
/// est = f_add + g_add + f_rescale * (2^ex_bits * <q, sign> + <q, ex> + cb * sum(q))
/// ```
///
/// ### Params
///
/// * `encoding` - The vector's encoding
/// * `query_rotated` - Rotated query, length `dim`
/// * `g_add` - `||q - c||^2` for the centroid the vector was encoded against
/// * `ex_bits` - Magnitude bits the encoding was made at
///
/// ### Returns
///
/// The estimated squared Euclidean distance
pub fn estimate_ex<T>(encoding: &ExEncoding<T>, query_rotated: &[T], g_add: T, ex_bits: usize) -> T
where
    T: AnnSearchFloat,
{
    let dim = query_rotated.len();

    let mut ip_sign = T::zero();
    for (d, &q) in query_rotated.iter().enumerate() {
        if encoding.sign_code[d / 8] >> (d % 8) & 1 == 1 {
            ip_sign = ip_sign + q;
        }
    }

    let mut ip_ex = T::zero();
    if ex_bits > 0 {
        let levels = unpack_excode(&encoding.ex_code, dim, ex_bits);
        for (&level, &q) in levels.iter().zip(query_rotated) {
            ip_ex = ip_ex + q * T::from_u8(level).unwrap_or_else(T::zero);
        }
    }

    let sum_q = query_rotated.iter().fold(T::zero(), |a, &b| a + b);
    let sign_step = T::from_f64((1u64 << ex_bits) as f64).unwrap_or_else(T::one);
    let cb = T::from_f64(-((1u64 << ex_bits) as f64 - 0.5)).unwrap_or_else(T::zero);

    encoding.f_add + g_add + encoding.f_rescale * (sign_step * ip_sign + ip_ex + cb * sum_q)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random floats in `[-0.5, 0.5)`.
    fn rng(seed: u64) -> impl FnMut() -> f64 {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        }
    }

    fn random_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut next = rng(seed);
        (0..dim).map(|_| next() as f32).collect()
    }

    /// Ground truth from the C++ reference, `ex_bits_code_with_factor` at
    /// `METRIC_L2` with `t_const = -1`, over the vectors `random_vec(64, 5)`
    /// and `random_vec(64, 6)`.
    ///
    /// `(ex_bits, scale, f_add, f_rescale, f_error, magnitude levels)`.
    #[allow(clippy::type_complexity)]
    const REFERENCE: &[(usize, f64, f64, f64, f64, &[u8])] = &[
        (
            1,
            8.55066283,
            1.14453793,
            -0.749480486,
            0.438388795,
            &[
                0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1,
                1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0,
                0, 0, 1, 1, 1, 0, 0, 1,
            ],
        ),
        (
            2,
            15.2680078,
            0.905673027,
            -0.39984417,
            0.234583095,
            &[
                0, 2, 3, 1, 0, 1, 1, 0, 1, 0, 2, 1, 2, 2, 1, 1, 0, 2, 1, 2, 3, 2, 1, 0, 2, 1, 2, 2,
                3, 0, 0, 0, 2, 0, 0, 2, 3, 1, 0, 3, 2, 2, 3, 1, 0, 2, 0, 2, 1, 1, 0, 2, 0, 1, 3, 1,
                1, 0, 2, 3, 3, 1, 1, 3,
            ],
        ),
        (
            3,
            30.5360156,
            0.85550499,
            -0.197509438,
            0.127020881,
            &[
                0, 4, 6, 2, 1, 2, 3, 1, 3, 1, 4, 2, 5, 4, 2, 3, 0, 4, 2, 5, 7, 5, 2, 0, 5, 3, 5, 4,
                7, 0, 1, 1, 4, 0, 1, 5, 7, 3, 0, 6, 4, 5, 7, 2, 0, 4, 0, 5, 2, 3, 1, 5, 0, 2, 6, 2,
                2, 1, 4, 6, 7, 3, 2, 7,
            ],
        ),
        (
            4,
            50.8149105,
            0.947724342,
            -0.119602561,
            0.065033026,
            &[
                0, 10, 10, 4, 2, 3, 5, 3, 5, 2, 10, 4, 8, 9, 7, 8, 0, 7, 3, 11, 15, 12, 4, 0, 8, 8,
                9, 10, 15, 0, 2, 5, 10, 0, 2, 9, 14, 6, 0, 11, 6, 8, 15, 3, 1, 7, 3, 12, 7, 9, 1,
                8, 0, 7, 10, 6, 3, 2, 7, 13, 15, 9, 7, 14,
            ],
        ),
        (
            6,
            215.510655,
            0.856853485,
            -0.0282542128,
            0.0148209594,
            &[
                0, 42, 44, 18, 11, 14, 22, 13, 25, 8, 41, 18, 36, 37, 28, 31, 4, 31, 15, 46, 62,
                47, 19, 0, 37, 30, 38, 42, 60, 2, 9, 21, 41, 4, 11, 40, 57, 25, 2, 47, 28, 36, 60,
                16, 4, 31, 9, 47, 27, 34, 7, 35, 1, 26, 45, 24, 15, 12, 30, 51, 60, 35, 28, 58,
            ],
        ),
        (
            8,
            846.915175,
            0.871282578,
            -0.00718827685,
            0.00337654864,
            &[
                3, 170, 176, 72, 45, 57, 88, 52, 99, 33, 168, 74, 142, 151, 115, 127, 16, 125, 59,
                188, 250, 191, 75, 4, 146, 126, 150, 172, 242, 9, 36, 87, 167, 16, 46, 158, 231,
                101, 8, 186, 112, 144, 240, 65, 19, 124, 41, 191, 113, 140, 28, 139, 5, 110, 179,
                102, 60, 47, 120, 208, 241, 145, 116, 232,
            ],
        ),
    ];

    #[test]
    fn test_matches_the_reference_implementation() {
        // The property tests below would pass on a port that converged but got
        // the levels or the factors subtly wrong, so this pins both against
        // numbers taken from the C++ reference itself.
        let dim = 64;
        let data = random_vec(dim, 5);
        let centroid = random_vec(dim, 6);

        let residual: Vec<f64> = data
            .iter()
            .zip(&centroid)
            .map(|(&v, &c)| (v - c) as f64)
            .collect();
        let l2 = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
        let abs_normalised: Vec<f64> = residual.iter().map(|x| x.abs() / l2).collect();

        for &(ex_bits, scale, f_add, f_rescale, f_error, levels) in REFERENCE {
            approx::assert_relative_eq!(
                best_rescale_factor(&abs_normalised, ex_bits),
                scale,
                max_relative = 1e-6
            );

            let encoded = encode_ex_bits(&data, &centroid, ex_bits, None).unwrap();
            assert_eq!(
                unpack_excode(&encoded.ex_code, dim, ex_bits),
                levels,
                "levels differ at ex_bits {ex_bits}"
            );

            approx::assert_relative_eq!(encoded.f_add as f64, f_add, max_relative = 1e-5);
            approx::assert_relative_eq!(encoded.f_rescale as f64, f_rescale, max_relative = 1e-5);

            // Looser, and deliberately so. The error factor carries
            // `sqrt((l2_sqr * ||xu_cb||^2 / ip^2 - 1) / (dim - 1))`, whose
            // subtraction cancels hard once the reconstruction is good; the
            // reference evaluates it in f32 throughout and this port
            // accumulates in f64, so the two diverge by ~6e-4 at the widest
            // settings and agree everywhere else. A real porting mistake would
            // be out by orders of magnitude, not by a twentieth of a percent.
            approx::assert_relative_eq!(encoded.f_error as f64, f_error, max_relative = 1e-2);
        }
    }

    #[test]
    fn test_sign_code_tracks_the_residual() {
        let dim = 64;
        let data = random_vec(dim, 5);
        let centroid = random_vec(dim, 6);
        let encoded = encode_ex_bits(&data, &centroid, 4, None).unwrap();

        for d in 0..dim {
            let set = encoded.sign_code[d / 8] >> (d % 8) & 1 == 1;
            assert_eq!(set, data[d] - centroid[d] > 0.0, "sign differs at {d}");
        }
    }

    #[test]
    fn test_pack_round_trips_at_every_width() {
        let dim = 64;
        for total_bits in 1..=(MAX_EX_BITS + 1) {
            let mask = if total_bits == 8 {
                u8::MAX
            } else {
                ((1u16 << total_bits) - 1) as u8
            };
            let levels: Vec<u8> = (0..dim)
                .map(|d| (d as u8).wrapping_mul(37) & mask)
                .collect();

            let packed = pack_excode(&levels, total_bits);
            assert_eq!(packed.len(), excode_bytes(dim, total_bits));
            assert_eq!(unpack_excode(&packed, dim, total_bits), levels);
        }
    }

    #[test]
    fn test_level_at_scale_respects_bounds() {
        for ex_bits in 1..=MAX_EX_BITS {
            let max_level = (1u32 << ex_bits) - 1;
            for step in 0..50 {
                let magnitude = step as f64 / 25.0;
                let level = level_at_scale(magnitude, 3.7, max_level);
                assert!(level <= max_level);
                if magnitude == 0.0 {
                    assert_eq!(level, 0);
                }
            }
        }
    }

    #[test]
    fn test_rescale_factor_beats_the_interval_start() {
        // The sweep can only improve on where it starts, and on real residuals
        // it should actually move.
        let mut next = rng(11);
        let dim = 128;
        let raw: Vec<f64> = (0..dim).map(|_| next()).collect();
        let norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let abs_residual: Vec<f64> = raw.iter().map(|x| (x / norm).abs()).collect();

        for ex_bits in 1..=4 {
            let max_o = abs_residual.iter().cloned().fold(0.0f64, f64::max);
            let t_end = (((1u32 << ex_bits) - 1) as f64 + 10.0) / max_o;
            let t = best_rescale_factor(&abs_residual, ex_bits);
            assert!(t >= t_end * TIGHT_START[ex_bits]);
            assert!(t < t_end);
        }
    }

    #[test]
    fn test_more_bits_reconstruct_the_direction_better() {
        // The whole point of the extended code: cosine between the residual and
        // its recentred reconstruction rises with the width.
        let dim = 256;
        let vector = random_vec(dim, 5);
        let centroid = random_vec(dim, 6);

        let mut previous = -1.0f64;
        for ex_bits in 0..=6 {
            let encoded = encode_ex_bits(&vector, &centroid, ex_bits, None).unwrap();
            let levels = if ex_bits == 0 {
                vec![0u8; dim]
            } else {
                unpack_excode(&encoded.ex_code, dim, ex_bits)
            };

            let step = (1u64 << ex_bits) as f64;
            let cb = -(step - 0.5);
            let recon: Vec<f64> = levels
                .iter()
                .enumerate()
                .map(|(d, &u)| {
                    let sign = encoded.sign_code[d / 8] >> (d % 8) & 1;
                    u as f64 + if sign == 1 { step } else { 0.0 } + cb
                })
                .collect();
            let residual: Vec<f64> = vector
                .iter()
                .zip(&centroid)
                .map(|(&v, &c)| (v - c) as f64)
                .collect();

            let dot: f64 = residual.iter().zip(&recon).map(|(a, b)| a * b).sum();
            let na = residual.iter().map(|x| x * x).sum::<f64>().sqrt();
            let nb = recon.iter().map(|x| x * x).sum::<f64>().sqrt();
            let cosine = dot / (na * nb);

            assert!(
                cosine > previous,
                "ex_bits {ex_bits} gave cosine {cosine}, not better than {previous}"
            );
            previous = cosine;
        }
        assert!(previous > 0.99, "6 extra bits only reached {previous}");
    }

    #[test]
    fn test_estimate_converges_on_the_true_distance() {
        // Averaged over vectors, not asserted per vector: the quantisation
        // error is a random variable, and once the estimate is within ~1e-4 the
        // f32 arithmetic in `estimate_ex` is the larger term, so a single draw
        // is not monotone even though the code is.
        let dim = 256;
        let trials = 24;
        let centroid = random_vec(dim, 21);

        let mut previous = f64::INFINITY;
        for ex_bits in [0usize, 2, 4, 6] {
            let mut total = 0.0f64;

            for trial in 0..trials {
                let vector = random_vec(dim, 200 + trial);
                let query = random_vec(dim, 900 + trial);

                let truth: f64 = vector
                    .iter()
                    .zip(&query)
                    .map(|(&v, &q)| ((v - q) as f64).powi(2))
                    .sum();
                let g_add: f32 = centroid
                    .iter()
                    .zip(&query)
                    .map(|(&c, &q)| (c - q) * (c - q))
                    .sum();

                let encoded = encode_ex_bits(&vector, &centroid, ex_bits, None).unwrap();
                let est = estimate_ex(&encoded, &query, g_add, ex_bits) as f64;
                total += (est - truth).abs() / truth;
            }

            let mean = total / trials as f64;
            assert!(
                mean < previous,
                "ex_bits {ex_bits} mean error {mean} did not beat {previous}"
            );
            previous = mean;
        }
        assert!(
            previous < 0.01,
            "6 extra bits still {previous} off on average"
        );
    }

    #[test]
    fn test_error_factor_bounds_the_estimate() {
        // `f_error * ||q - c||` is the reference's high-probability bound on
        // the estimate's absolute error.
        let dim = 128;
        let centroid = random_vec(dim, 31);

        let mut breaches = 0;
        let trials = 60;
        for trial in 0..trials {
            let vector = random_vec(dim, 100 + trial);
            let query = random_vec(dim, 500 + trial);

            let truth: f32 = vector
                .iter()
                .zip(&query)
                .map(|(&v, &q)| (v - q) * (v - q))
                .sum();
            let g_add: f32 = centroid
                .iter()
                .zip(&query)
                .map(|(&c, &q)| (c - q) * (c - q))
                .sum();

            let encoded = encode_ex_bits(&vector, &centroid, 3, None).unwrap();
            let est = estimate_ex(&encoded, &query, g_add, 3);

            if (est - truth).abs() > encoded.f_error * g_add.sqrt() {
                breaches += 1;
            }
        }
        assert!(breaches * 20 <= trials, "{breaches} of {trials} breached");
    }

    #[test]
    fn test_const_scale_matches_a_searched_one() {
        // The averaged scale stands in for the per-vector search, so it has to
        // land in the same neighbourhood on a typical vector.
        let dim = 128;
        let t_const = const_scaling_factor(dim, 4, 42);

        let mut next = rng(77);
        let raw: Vec<f64> = (0..dim).map(|_| next()).collect();
        let norm = raw.iter().map(|x| x * x).sum::<f64>().sqrt();
        let abs_residual: Vec<f64> = raw.iter().map(|x| (x / norm).abs()).collect();
        let searched = best_rescale_factor(&abs_residual, 4);

        let ratio = t_const / searched;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "t_const {t_const} vs {searched}"
        );
    }

    #[test]
    fn test_zero_residual_is_not_degenerate() {
        let dim = 64;
        let centroid = random_vec(dim, 9);
        let encoded = encode_ex_bits(&centroid, &centroid, 4, None).unwrap();

        assert_eq!(encoded.f_add, 0.0);
        assert_eq!(encoded.f_rescale, 0.0);
        assert_eq!(encoded.f_error, 0.0);

        let query = random_vec(dim, 10);
        let g_add: f32 = centroid
            .iter()
            .zip(&query)
            .map(|(&c, &q)| (c - q) * (c - q))
            .sum();
        approx::assert_relative_eq!(
            estimate_ex(&encoded, &query, g_add, 4),
            g_add,
            max_relative = 1e-6
        );
    }

    #[test]
    fn test_too_many_bits_is_rejected() {
        let dim = 32;
        let vector = random_vec(dim, 1);
        let centroid = random_vec(dim, 2);
        assert!(matches!(
            encode_ex_bits(&vector, &centroid, MAX_EX_BITS + 1, None),
            Err(AnnSearchErrors::RaBitQInvalidExBits { .. })
        ));
    }

    #[test]
    fn test_t_const_path_agrees_with_the_search() {
        // The fast path trades the sweep for an averaged scale; it must still
        // produce a usable estimate, not merely a different one.
        let dim = 128;
        let centroid = random_vec(dim, 41);
        let vector = random_vec(dim, 42);
        let query = random_vec(dim, 43);

        let truth: f32 = vector
            .iter()
            .zip(&query)
            .map(|(&v, &q)| (v - q) * (v - q))
            .sum();
        let g_add: f32 = centroid
            .iter()
            .zip(&query)
            .map(|(&c, &q)| (c - q) * (c - q))
            .sum();

        let t_const = const_scaling_factor(dim, 4, 42);
        let searched = encode_ex_bits(&vector, &centroid, 4, None).unwrap();
        let fast = encode_ex_bits(&vector, &centroid, 4, Some(t_const)).unwrap();

        let err_searched = (estimate_ex(&searched, &query, g_add, 4) - truth).abs() / truth;
        let err_fast = (estimate_ex(&fast, &query, g_add, 4) - truth).abs() / truth;

        assert!(err_fast < 0.1, "fast path error {err_fast}");
        assert!(err_searched < 0.1, "searched error {err_searched}");
    }
}
