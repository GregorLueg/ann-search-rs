#![allow(dead_code)]

use faer::RowRef;
use num_traits::Float;
use std::iter::Sum;
use std::sync::OnceLock;
use wide::{f32x4, f32x8, f64x2, f64x4};

#[cfg(feature = "quantised")]
use half::bf16;
#[cfg(feature = "quantised")]
use num_traits::{FromPrimitive, ToPrimitive};
#[cfg(all(feature = "quantised", target_arch = "aarch64"))]
use std::arch::aarch64::*;
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
use std::arch::x86_64::*;

////////////
// Helper //
////////////

/// Enum for the Distance metric to use
#[derive(Clone, Debug, Copy, PartialEq, Default)]
pub enum Dist {
    /// Euclidean distance
    #[default]
    Euclidean,
    /// Cosine distance
    Cosine,
}

/// Parsing the approximate nearest neighbour distance
///
/// Currently, only Cosine and Euclidean are supported. Longer term, others
/// shall be implemented.
///
/// ### Params
///
/// * `s` - The string that defines the tied summarisation type
///
/// ### Results
///
/// The `Dist` defining the distance metric to use for the approximate
/// neighbour search.
pub fn parse_ann_dist(s: &str) -> Option<Dist> {
    match s.to_lowercase().as_str() {
        "euclidean" => Some(Dist::Euclidean),
        "cosine" => Some(Dist::Cosine),
        _ => None,
    }
}

////////////////////
// VectorDistance //
////////////////////

//////////////////////
// SIMD for f32/f64 //
//////////////////////

// Enum for the different architectures and potential SIMD levels
#[derive(Clone, Copy, Debug)]
pub enum SimdLevel {
    /// Scalar version
    Scalar,
    /// 128-bit (also covers NEON which is used by Apple)
    Sse,
    /// 256-bit
    Avx2,
    /// 512-bit
    Avx512,
}

static SIMD_LEVEL: OnceLock<SimdLevel> = OnceLock::new();

/// Function to detect which SIMD implementation to use
pub fn detect_simd_level() -> SimdLevel {
    *SIMD_LEVEL.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                return SimdLevel::Avx512;
            }
            if is_x86_feature_detected!("avx2") {
                return SimdLevel::Avx2;
            }
            if is_x86_feature_detected!("sse4.1") {
                return SimdLevel::Sse;
            }
            return SimdLevel::Scalar;
        }

        #[cfg(target_arch = "aarch64")]
        {
            // NEON is always available on aarch64
            SimdLevel::Sse
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdLevel::Scalar
        }
    })
}

/// Trait for SIMD distance calculations
pub trait SimdDistance: Sized + Copy {
    /// Calculate Euclidean distance via SIMD
    ///
    /// ### Params
    ///
    /// * `a` - Slice of vector a
    /// * `b` - Slice of vector b
    ///
    /// ### Returns
    ///
    /// Squared Euclidean distance
    fn euclidean_simd(a: &[Self], b: &[Self]) -> Self;

    /// Calculate dot product via SIMD
    ///
    /// ### Params
    ///
    /// * `a` - Slice of vector a
    /// * `b` - Slice of vector b
    ///
    /// ### Returns
    ///
    /// Dot product
    fn dot_simd(a: &[Self], b: &[Self]) -> Self;

    /// Subtracts one vector from the other
    ///
    /// ### Params
    ///
    /// * `a` - Slice of vector a
    /// * `b` - Slice of vector b
    ///
    /// ### Returns
    ///
    /// Subtracted vector
    fn subtract_simd(a: &[Self], b: &[Self]) -> Vec<Self>;

    /// Calculate the norm
    ///
    /// ### Params
    ///
    /// * `vec` - Slice of vector for which to calculate the norm
    ///
    /// ### Returns
    ///
    /// Norm of the vector
    fn calculate_norm(vec: &[Self]) -> Self;
}

///////////////////
// f32 Euclidean //
///////////////////

/// Euclidean distance - f32, scalar
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Squared euclidean distance
#[inline(always)]
fn euclidean_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Euclidean distance - f32, optimised for 128 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Squared euclidean distance
#[inline(always)]
fn euclidean_f32_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let va = f32x4::from(*(a_ptr.add(offset) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(offset) as *const [f32; 4]));
            let diff = va - vb;
            acc += diff * diff;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Euclidean distance - f32, optimised for 256 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Squared euclidean distance
#[inline(always)]
fn euclidean_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from(*(a_ptr.add(offset) as *const [f32; 8]));
            let vb = f32x8::from(*(b_ptr.add(offset) as *const [f32; 8]));
            let diff = va - vb;
            acc += diff * diff;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Euclidean distance - f32, optimised for 512 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Squared euclidean distance
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn euclidean_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            let diff = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);

        // Remainder
        for i in (chunks * 16)..len {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn euclidean_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    // Fallback - shouldn't be called but needed for compilation
    euclidean_f32_avx2(a, b)
}

///////////////////
// f64 Euclidean //
///////////////////

/// Euclidean distance - f64, scalar
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Squared euclidean distance
#[inline(always)]
fn euclidean_f64_scalar(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Euclidean distance - f64, optimised for 128 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Squared euclidean distance
#[inline(always)]
fn euclidean_f64_sse(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let chunks = len / 2;
    let mut acc = f64x2::ZERO;

    // to avoid trait bound blabla -> unsafe
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 2;
            let va = f64x2::from(*(a_ptr.add(offset) as *const [f64; 2]));
            let vb = f64x2::from(*(b_ptr.add(offset) as *const [f64; 2]));
            let diff = va - vb;
            acc += diff * diff;
        }
    }

    let mut sum = acc.reduce_add();
    if len % 2 == 1 {
        let diff = a[len - 1] - b[len - 1];
        sum += diff * diff;
    }
    sum
}

/// Euclidean distance - f64, optimised for 256 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Squared euclidean distance
#[inline(always)]
fn euclidean_f64_avx2(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f64x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let va = f64x4::from(*(a_ptr.add(offset) as *const [f64; 4]));
            let vb = f64x4::from(*(b_ptr.add(offset) as *const [f64; 4]));
            let diff = va - vb;
            acc += diff * diff;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        let diff = a[i] - b[i];
        sum += diff * diff;
    }
    sum
}

/// Euclidean distance - f64, optimised for 512 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Squared euclidean distance
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn euclidean_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm512_setzero_pd();

        for i in 0..chunks {
            let va = _mm512_loadu_pd(a.as_ptr().add(i * 8));
            let vb = _mm512_loadu_pd(b.as_ptr().add(i * 8));
            let diff = _mm512_sub_pd(va, vb);
            acc = _mm512_fmadd_pd(diff, diff, acc);
        }

        let mut sum = _mm512_reduce_add_pd(acc);

        for i in (chunks * 8)..len {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }
        sum
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn euclidean_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
    euclidean_f64_avx2(a, b)
}

/////////////////////
// f32 dot product //
/////////////////////

/// Dot product - f32, scalar
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Dot product - f32, optimised for 128-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_f32_sse(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let va = f32x4::from(*(a_ptr.add(offset) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(offset) as *const [f32; 4]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Dot product - f32, optimised for 256-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from(*(a_ptr.add(offset) as *const [f32; 8]));
            let vb = f32x8::from(*(b_ptr.add(offset) as *const [f32; 8]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Dot product - f32, optimised for 512-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            sum += a[i] * b[i];
        }
        sum
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn dot_f32_avx512(a: &[f32], b: &[f32]) -> f32 {
    dot_f32_avx2(a, b)
}

/////////////////////
// f64 dot product //
/////////////////////

/// Dot product - f64, scalar
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_f64_scalar(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Dot product - f64, optimised for 128-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_f64_sse(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let chunks = len / 2;
    let mut acc = f64x2::ZERO;

    // unsafe again to avoid trait errors
    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 2;
            let va = f64x2::from(*(a_ptr.add(offset) as *const [f64; 2]));
            let vb = f64x2::from(*(b_ptr.add(offset) as *const [f64; 2]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    if len % 2 == 1 {
        sum += a[len - 1] * b[len - 1];
    }
    sum
}

/// Dot product - f64, optimised for 256-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[inline(always)]
fn dot_f64_avx2(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len();
    let chunks = len / 4;
    let mut acc = f64x4::ZERO;

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let va = f64x4::from(*(a_ptr.add(offset) as *const [f64; 4]));
            let vb = f64x4::from(*(b_ptr.add(offset) as *const [f64; 4]));
            acc += va * vb;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Dot product - f64, optimised for 512-bit
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// Dot product
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn dot_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm512_setzero_pd();

        for i in 0..chunks {
            let va = _mm512_loadu_pd(a.as_ptr().add(i * 8));
            let vb = _mm512_loadu_pd(b.as_ptr().add(i * 8));
            acc = _mm512_fmadd_pd(va, vb, acc);
        }

        let mut sum = _mm512_reduce_add_pd(acc);
        for i in (chunks * 8)..len {
            sum += a[i] * b[i];
        }
        sum
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn dot_f64_avx512(a: &[f64], b: &[f64]) -> f64 {
    dot_f64_avx2(a, b)
}

///////////////////
// f32 Subtract  //
///////////////////

/// Vector subtraction - f32, scalar
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// `Vec<a - b>`
#[inline(always)]
fn subtract_f32_scalar(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// Vector subtraction - f32, optimised for 128 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// `Vec<a - b>`
#[inline(always)]
fn subtract_f32_sse(a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len();
    let chunks = len / 4;
    let mut result = Vec::with_capacity(len);

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr: *mut f32 = result.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let va = f32x4::from(*(a_ptr.add(offset) as *const [f32; 4]));
            let vb = f32x4::from(*(b_ptr.add(offset) as *const [f32; 4]));
            let diff = va - vb;
            *(result_ptr.add(offset) as *mut [f32; 4]) = diff.into();
        }

        for i in (chunks * 4)..len {
            *result_ptr.add(i) = a[i] - b[i];
        }

        result.set_len(len);
    }
    result
}

/// Vector subtraction - f32, optimised for 256 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// `Vec<a - b>`
#[inline(always)]
fn subtract_f32_avx2(a: &[f32], b: &[f32]) -> Vec<f32> {
    let len = a.len();
    let chunks = len / 8;
    let mut result = Vec::with_capacity(len);

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr: *mut f32 = result.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let va = f32x8::from(*(a_ptr.add(offset) as *const [f32; 8]));
            let vb = f32x8::from(*(b_ptr.add(offset) as *const [f32; 8]));
            let diff = va - vb;
            *(result_ptr.add(offset) as *mut [f32; 8]) = diff.into();
        }

        for i in (chunks * 8)..len {
            *result_ptr.add(i) = a[i] - b[i];
        }

        result.set_len(len);
    }
    result
}

/// Vector subtraction - f32, optimised for 512 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// `Vec<a - b>`
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn subtract_f32_avx512(a: &[f32], b: &[f32]) -> Vec<f32> {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;
    let mut result = Vec::with_capacity(len);

    unsafe {
        let result_ptr = result.as_mut_ptr();

        for i in 0..chunks {
            let va = _mm512_loadu_ps(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            let diff = _mm512_sub_ps(va, vb);
            _mm512_storeu_ps(result_ptr.add(i * 16), diff);
        }

        for i in (chunks * 16)..len {
            *result_ptr.add(i) = a[i] - b[i];
        }

        result.set_len(len);
    }
    result
}

/// Vector subtraction - f32, fall back version
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// `Vec<a - b>`
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn subtract_f32_avx512(a: &[f32], b: &[f32]) -> Vec<f32> {
    subtract_f32_avx2(a, b)
}

///////////////////
// f64 Subtract  //
///////////////////

/// Vector subtraction - f64, scalar
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// `Vec<a - b>`
#[inline(always)]
fn subtract_f64_scalar(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x - y).collect()
}

/// Vector subtraction - f64, optimised for 128 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// `Vec<a - b>`
#[inline(always)]
fn subtract_f64_sse(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len();
    let chunks = len / 2;
    let mut result = Vec::with_capacity(len);

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr: *mut f64 = result.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 2;
            let va = f64x2::from(*(a_ptr.add(offset) as *const [f64; 2]));
            let vb = f64x2::from(*(b_ptr.add(offset) as *const [f64; 2]));
            let diff = va - vb;
            *(result_ptr.add(offset) as *mut [f64; 2]) = diff.into();
        }

        if len % 2 == 1 {
            *result_ptr.add(len - 1) = a[len - 1] - b[len - 1];
        }

        result.set_len(len);
    }
    result
}

/// Vector subtraction - f64, optimised for 256 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// `Vec<a - b>`
#[inline(always)]
fn subtract_f64_avx2(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len();
    let chunks = len / 4;
    let mut result = Vec::with_capacity(len);

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr: *mut f64 = result.as_mut_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let va = f64x4::from(*(a_ptr.add(offset) as *const [f64; 4]));
            let vb = f64x4::from(*(b_ptr.add(offset) as *const [f64; 4]));
            let diff = va - vb;
            *(result_ptr.add(offset) as *mut [f64; 4]) = diff.into();
        }

        for i in (chunks * 4)..len {
            *result_ptr.add(i) = a[i] - b[i];
        }

        result.set_len(len);
    }
    result
}

/// Vector subtraction - f64, optimised for 512 bits
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// `Vec<a - b>`
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn subtract_f64_avx512(a: &[f64], b: &[f64]) -> Vec<f64> {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 8;
    let mut result = Vec::with_capacity(len);

    unsafe {
        let result_ptr = result.as_mut_ptr();

        for i in 0..chunks {
            let va = _mm512_loadu_pd(a.as_ptr().add(i * 8));
            let vb = _mm512_loadu_pd(b.as_ptr().add(i * 8));
            let diff = _mm512_sub_pd(va, vb);
            _mm512_storeu_pd(result_ptr.add(i * 8), diff);
        }

        for i in (chunks * 8)..len {
            *result_ptr.add(i) = a[i] - b[i];
        }

        result.set_len(len);
    }
    result
}

/// Vector subtraction - f64, fall back version for AVX512
///
/// ### Params
///
/// * `a` - Slice of vector a
/// * `b` - Slice of vector b
///
/// ### Returns
///
/// `Vec<a - b>`
#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn subtract_f64_avx512(a: &[f64], b: &[f64]) -> Vec<f64> {
    subtract_f64_avx2(a, b)
}

///////////////
// f32, Norm //
///////////////

/// Norm - f32, scalar
///
/// ### Params
///
/// * `vec` - Slice of vector a
///
/// ### Returns
///
/// Returns the norm
#[inline(always)]
fn compute_norm_f32_scalar(vec: &[f32]) -> f32 {
    let mut sum = 0.0_f32;
    for &x in vec {
        sum += x * x;
    }
    sum.sqrt()
}

/// Norm - f32, SSE
///
/// ### Params
///
/// * `vec` - Slice of vector a
///
/// ### Returns
///
/// Returns the norm
#[inline(always)]
fn compute_norm_f32_sse(vec: &[f32]) -> f32 {
    let len = vec.len();
    let chunks = len / 4;
    let mut acc = f32x4::ZERO;

    unsafe {
        let vec_ptr = vec.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let v_chunk = f32x4::from(*(vec_ptr.add(offset) as *const [f32; 4]));
            acc += v_chunk * v_chunk;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += vec[i] * vec[i];
    }
    sum.sqrt()
}

/// Norm - f32, AVX2
///
/// ### Params
///
/// * `vec` - Slice of vector a
///
/// ### Returns
///
/// Returns the norm
#[inline(always)]
fn compute_norm_f32_avx2(vec: &[f32]) -> f32 {
    let len = vec.len();
    let chunks = len / 8;
    let mut acc = f32x8::ZERO;

    unsafe {
        let vec_ptr = vec.as_ptr();

        for i in 0..chunks {
            let offset = i * 8;
            let v_chunk = f32x8::from(*(vec_ptr.add(offset) as *const [f32; 8]));
            acc += v_chunk * v_chunk;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 8)..len {
        sum += vec[i] * vec[i];
    }

    sum.sqrt()
}

/// Norm - f32, AVX512
///
/// ### Params
///
/// * `vec` - Slice of vector a
///
/// ### Returns
///
/// Returns the norm
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn compute_norm_f32_avx512(vec: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let v_chunk = _mm512_loadu_ps(vec.as_ptr().add(i * 16));
            acc = _mm512_fmadd_ps(v_chunk, v_chunk, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            sum += vec[i] * vec[i];
        }
        sum.sqrt()
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn compute_norm_f32_avx512(vec: &[f32]) -> f32 {
    compute_norm_f32_avx2(vec)
}

///////////////
// f64, Norm //
///////////////

/// Norm - f32, scalar
///
/// ### Params
///
/// * `vec` - Slice of vector a
///
/// ### Returns
///
/// Returns the norm
#[inline(always)]
fn compute_norm_f64_scalar(vec: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    for &x in vec {
        sum += x * x;
    }
    sum.sqrt()
}

/// Norm - f64, SSE
///
/// ### Params
///
/// * `vec` - Slice of vector a
///
/// ### Returns
///
/// Returns the norm
#[inline(always)]
fn compute_norm_f64_sse(vec: &[f64]) -> f64 {
    let len = vec.len();
    let chunks = len / 2;
    let mut acc = f64x2::ZERO;

    // unsafe again to avoid trait errors
    unsafe {
        let vec_ptr = vec.as_ptr();

        for i in 0..chunks {
            let offset = i * 2;
            let v_chunk = f64x2::from(*(vec_ptr.add(offset) as *const [f64; 2]));
            acc += v_chunk * v_chunk;
        }
    }

    let mut sum = acc.reduce_add();
    if len % 2 == 1 {
        sum += vec[len - 1] * vec[len - 1];
    }
    sum.sqrt()
}

/// Norm - f64, AVX2
///
/// ### Params
///
/// * `vec` - Slice of vector a
///
/// ### Returns
///
/// Returns the norm
#[inline(always)]
fn compute_norm_f64_avx2(vec: &[f64]) -> f64 {
    let len = vec.len();
    let chunks = len / 4;
    let mut acc = f64x4::ZERO;

    unsafe {
        let vec_ptr = vec.as_ptr();

        for i in 0..chunks {
            let offset = i * 4;
            let v_chunk = f64x4::from(*(vec_ptr.add(offset) as *const [f64; 4]));
            acc += v_chunk * v_chunk;
        }
    }

    let mut sum = acc.reduce_add();
    for i in (chunks * 4)..len {
        sum += vec[i] * vec[i];
    }

    sum.sqrt()
}

/// Norm - f64, AVX512
///
/// ### Params
///
/// * `vec` - Slice of vector a
///
/// ### Returns
///
/// Returns the norm
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
#[inline(always)]
fn compute_norm_f64_avx512(vec: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let len = vec.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm512_setzero_pd();

        for i in 0..chunks {
            let v_chunk = _mm512_loadu_pd(vec.as_ptr().add(i * 8));
            acc = _mm512_fmadd_pd(v_chunk, v_chunk, acc);
        }

        let mut sum = _mm512_reduce_add_pd(acc);
        for i in (chunks * 8)..len {
            sum += vec[i] * vec[i];
        }
        sum.sqrt()
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx512f")))]
#[inline(always)]
fn compute_norm_f64_avx512(vec: &[f64]) -> f64 {
    compute_norm_f64_avx2(vec)
}

//////////////////////////////////
// SimdDistance implementations //
//////////////////////////////////

impl SimdDistance for f32 {
    #[inline]
    fn euclidean_simd(a: &[f32], b: &[f32]) -> f32 {
        match detect_simd_level() {
            SimdLevel::Avx512 => euclidean_f32_avx512(a, b),
            SimdLevel::Avx2 => euclidean_f32_avx2(a, b),
            SimdLevel::Sse => euclidean_f32_sse(a, b),
            SimdLevel::Scalar => euclidean_f32_scalar(a, b),
        }
    }

    #[inline]
    fn dot_simd(a: &[f32], b: &[f32]) -> f32 {
        match detect_simd_level() {
            SimdLevel::Avx512 => dot_f32_avx512(a, b),
            SimdLevel::Avx2 => dot_f32_avx2(a, b),
            SimdLevel::Sse => dot_f32_sse(a, b),
            SimdLevel::Scalar => dot_f32_scalar(a, b),
        }
    }

    #[inline]
    fn subtract_simd(a: &[f32], b: &[f32]) -> Vec<f32> {
        match detect_simd_level() {
            SimdLevel::Avx512 => subtract_f32_avx512(a, b),
            SimdLevel::Avx2 => subtract_f32_avx2(a, b),
            SimdLevel::Sse => subtract_f32_sse(a, b),
            SimdLevel::Scalar => subtract_f32_scalar(a, b),
        }
    }

    #[inline]
    fn calculate_norm(vec: &[Self]) -> Self {
        match detect_simd_level() {
            SimdLevel::Avx512 => compute_norm_f32_avx512(vec),
            SimdLevel::Avx2 => compute_norm_f32_avx2(vec),
            SimdLevel::Sse => compute_norm_f32_sse(vec),
            SimdLevel::Scalar => compute_norm_f32_scalar(vec),
        }
    }
}

impl SimdDistance for f64 {
    #[inline]
    fn euclidean_simd(a: &[f64], b: &[f64]) -> f64 {
        match detect_simd_level() {
            SimdLevel::Avx512 => euclidean_f64_avx512(a, b),
            SimdLevel::Avx2 => euclidean_f64_avx2(a, b),
            SimdLevel::Sse => euclidean_f64_sse(a, b),
            SimdLevel::Scalar => euclidean_f64_scalar(a, b),
        }
    }

    #[inline]
    fn dot_simd(a: &[f64], b: &[f64]) -> f64 {
        match detect_simd_level() {
            SimdLevel::Avx512 => dot_f64_avx512(a, b),
            SimdLevel::Avx2 => dot_f64_avx2(a, b),
            SimdLevel::Sse => dot_f64_sse(a, b),
            SimdLevel::Scalar => dot_f64_scalar(a, b),
        }
    }

    #[inline]
    fn subtract_simd(a: &[f64], b: &[f64]) -> Vec<f64> {
        match detect_simd_level() {
            SimdLevel::Avx512 => subtract_f64_avx512(a, b),
            SimdLevel::Avx2 => subtract_f64_avx2(a, b),
            SimdLevel::Sse => subtract_f64_sse(a, b),
            SimdLevel::Scalar => subtract_f64_scalar(a, b),
        }
    }

    #[inline]
    fn calculate_norm(vec: &[Self]) -> Self {
        match detect_simd_level() {
            SimdLevel::Avx512 => compute_norm_f64_avx512(vec),
            SimdLevel::Avx2 => compute_norm_f64_avx2(vec),
            SimdLevel::Sse => compute_norm_f64_sse(vec),
            SimdLevel::Scalar => compute_norm_f64_scalar(vec),
        }
    }
}

//////////////////////////
// VectorDistance Trait //
//////////////////////////

/// Trait for computing distances between Floats
pub trait VectorDistance<T>
where
    T: Float + Sum + SimdDistance,
{
    /// Get the internal flat vector representation
    fn vectors_flat(&self) -> &[T];

    /// Get the internal dimensions
    fn dim(&self) -> usize;

    /// Get the normalised values
    fn norms(&self) -> &[T];

    ///////////////
    // Euclidean //
    ///////////////

    /// Euclidean distance between two internal vectors (squared)
    ///
    /// ### Params
    ///
    /// * `i` - Sample index i
    /// * `j` - Sample index j
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance between the two samples
    #[inline(always)]
    fn euclidean_distance(&self, i: usize, j: usize) -> T {
        let start_i = i * self.dim();
        let start_j = j * self.dim();
        let vec_i = &self.vectors_flat()[start_i..start_i + self.dim()];
        let vec_j = &self.vectors_flat()[start_j..start_j + self.dim()];
        T::euclidean_simd(vec_i, vec_j)
    }

    /// Euclidean distance between query vector and internal vector (squared)
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query` - Query vector slice
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance
    #[inline(always)]
    fn euclidean_distance_to_query(&self, internal_idx: usize, query: &[T]) -> T {
        let start = internal_idx * self.dim();
        let vec = &self.vectors_flat()[start..start + self.dim()];
        T::euclidean_simd(vec, query)
    }

    ////////////
    // Cosine //
    ////////////

    /// Cosine distance between two internal vectors
    ///
    /// Uses pre-computed norms.
    ///
    /// ### Params
    ///
    /// * `i` - Sample index i
    /// * `j` - Sample index j
    ///
    /// ### Returns
    ///
    /// The Cosine distance between the two samples
    #[inline(always)]
    fn cosine_distance(&self, i: usize, j: usize) -> T {
        let start_i = i * self.dim();
        let start_j = j * self.dim();
        let vec_i = &self.vectors_flat()[start_i..start_i + self.dim()];
        let vec_j = &self.vectors_flat()[start_j..start_j + self.dim()];
        let dot = T::dot_simd(vec_i, vec_j);
        T::one() - (dot / (self.norms()[i] * self.norms()[j]))
    }

    /// Cosine distance between query vector and internal vector
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query` - Query vector slice
    /// * `query_norm` - Pre-computed norm of query vector
    ///
    /// ### Returns
    ///
    /// The Cosine distance
    #[inline(always)]
    fn cosine_distance_to_query(&self, internal_idx: usize, query: &[T], query_norm: T) -> T {
        let start = internal_idx * self.dim();
        let vec = &self.vectors_flat()[start..start + self.dim()];
        let dot = T::dot_simd(vec, query);
        T::one() - (dot / (query_norm * self.norms()[internal_idx]))
    }
}

////////////////////////
// VectorDistanceBf16 //
////////////////////////

//////////
// SIMD //
//////////

/////////////////////////
// Bit shift functions //
/////////////////////////

/// Convert 4 bf16 values to 4 f32 values using SSE
///
/// bf16 is the upper 16 bits of f32, so conversion is just a left shift by 16.
///
/// ### Params
///
/// * `ptr` - Pointer to 4 consecutive bf16 values
///
/// ### Safety
///
/// Caller must ensure `ptr` points to at least 4 valid bf16 values.
///
/// ### Returns
///
/// 128-bit register containing 4 f32 values
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn bf16x4_to_f32x4_sse(ptr: *const bf16) -> __m128 {
    // Load 64 bits (4 x bf16) into lower half of 128-bit register
    let raw = _mm_loadl_epi64(ptr as *const __m128i);
    // Zero-extend each 16-bit value to 32-bit
    let extended = _mm_cvtepu16_epi32(raw);
    // Shift left by 16 to place bf16 bits in correct f32 position
    let shifted = _mm_slli_epi32(extended, 16);
    // Reinterpret as f32
    _mm_castsi128_ps(shifted)
}

/// Horizontal sum of 4 f32 values in SSE register
///
/// ### Params
///
/// * `v` - 128-bit register containing 4 f32 values
///
/// ### Safety
///
/// None beyond normal SSE requirements.
///
/// ### Returns
///
/// Sum of all 4 f32 values
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum_f32_sse(v: __m128) -> f32 {
    // [a, b, c, d] -> [b, b, d, d]
    let shuf = _mm_movehdup_ps(v);
    // [a+b, b+b, c+d, d+d]
    let sums = _mm_add_ps(v, shuf);
    // [c+d, d+d, c+d, d+d]
    let shuf2 = _mm_movehl_ps(sums, sums);
    // [a+b+c+d, ...]
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

/// Convert 8 bf16 values to 8 f32 values using AVX2
///
/// ### Params
///
/// * `ptr` - Pointer to 8 consecutive bf16 values
///
/// ### Safety
///
/// Caller must ensure `ptr` points to at least 8 valid bf16 values.
///
/// ### Returns
///
/// 256-bit register containing 8 f32 values
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn bf16x8_to_f32x8_avx2(ptr: *const bf16) -> __m256 {
    // Load 128 bits (8 x bf16)
    let raw = _mm_loadu_si128(ptr as *const __m128i);
    // Zero-extend each 16-bit value to 32-bit (produces 256-bit result)
    let extended = _mm256_cvtepu16_epi32(raw);
    // Shift left by 16
    let shifted = _mm256_slli_epi32(extended, 16);
    // Reinterpret as f32
    _mm256_castsi256_ps(shifted)
}

/// Horizontal sum of 8 f32 values in AVX2 register
///
/// ### Params
///
/// * `v` - 256-bit register containing 8 f32 values
///
/// ### Safety
///
/// None beyond normal AVX2 requirements.
///
/// ### Returns
///
/// Sum of all 8 f32 values
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
unsafe fn hsum_f32_avx2(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let low = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps(v, 1);
    // Sum the lanes
    let sum128 = _mm_add_ps(low, high);
    // Use SSE horizontal sum for the rest
    hsum_f32_sse(sum128)
}

/// Convert 16 bf16 values to 16 f32 values using AVX-512
///
/// ### Params
///
/// * `ptr` - Pointer to 16 consecutive bf16 values
///
/// ### Safety
///
/// Caller must ensure `ptr` points to at least 16 valid bf16 values.
///
/// ### Returns
///
/// 512-bit register containing 16 f32 values
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
#[inline(always)]
unsafe fn bf16x16_to_f32x16_avx512(ptr: *const bf16) -> __m512 {
    // Load 256 bits (16 x bf16)
    let raw = _mm256_loadu_si256(ptr as *const __m256i);
    // Zero-extend each 16-bit value to 32-bit (produces 512-bit result)
    let extended = _mm512_cvtepu16_epi32(raw);
    // Shift left by 16
    let shifted = _mm512_slli_epi32(extended, 16);
    // Reinterpret as f32
    _mm512_castsi512_ps(shifted)
}

/// Convert 4 bf16 values to 4 f32 values using NEON
///
/// ### Params
///
/// * `ptr` - Pointer to 4 consecutive bf16 values
///
/// ### Safety
///
/// Caller must ensure `ptr` points to at least 4 valid bf16 values.
///
/// ### Returns
///
/// NEON register containing 4 f32 values
#[cfg(all(feature = "quantised", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn bf16x4_to_f32x4_neon(ptr: *const bf16) -> float32x4_t {
    // Load 4 x bf16 as u16
    let raw = vld1_u16(ptr as *const u16);
    // Zero-extend to u32
    let extended = vmovl_u16(raw);
    // Shift left by 16
    let shifted = vshlq_n_u32(extended, 16);
    // Reinterpret as f32
    vreinterpretq_f32_u32(shifted)
}

/// Horizontal sum of 4 f32 values in NEON register
///
/// ### Params
///
/// * `v` - NEON register containing 4 f32 values
///
/// ### Safety
///
/// None beyond normal NEON requirements.
///
/// ### Returns
///
/// Sum of all 4 f32 values
#[cfg(all(feature = "quantised", target_arch = "aarch64"))]
#[inline(always)]
unsafe fn hsum_f32_neon(v: float32x4_t) -> f32 {
    vaddvq_f32(v)
}

///////////////////
// Distance func //
///////////////////

///////////////
// Euclidean //
///////////////

/// Euclidean distance between bf16 vectors - scalar fallback
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(feature = "quantised")]
#[inline(always)]
fn euclidean_bf16_scalar(a: &[bf16], b: &[bf16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            // ignore rust analyser
            let d = f32::from(*x) - f32::from(*y);
            d * d
        })
        .sum()
}

/// Euclidean distance between bf16 vectors - SSE (128-bit)
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn euclidean_bf16_sse(a: &[bf16], b: &[bf16]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = _mm_setzero_ps();

        for i in 0..chunks {
            let va = bf16x4_to_f32x4_sse(a.as_ptr().add(i * 4));
            let vb = bf16x4_to_f32x4_sse(b.as_ptr().add(i * 4));
            let diff = _mm_sub_ps(va, vb);
            // acc += diff * diff
            acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
        }

        let mut sum = hsum_f32_sse(acc);

        // Handle remainder
        for i in (chunks * 4)..len {
            let diff = a[i].to_f32() - b[i].to_f32();
            sum += diff * diff;
        }
        sum
    }
}

/// Euclidean distance between bf16 vectors - AVX2 (256-bit)
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn euclidean_bf16_avx2(a: &[bf16], b: &[bf16]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = bf16x8_to_f32x8_avx2(a.as_ptr().add(i * 8));
            let vb = bf16x8_to_f32x8_avx2(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            // FMA: acc = diff * diff + acc
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        let mut sum = hsum_f32_avx2(acc);

        // Handle remainder
        for i in (chunks * 8)..len {
            let diff = a[i].to_f32() - b[i].to_f32();
            sum += diff * diff;
        }
        sum
    }
}

/// Euclidean distance between bf16 vectors - AVX-512 (512-bit)
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
#[inline(always)]
fn euclidean_bf16_avx512(a: &[bf16], b: &[bf16]) -> f32 {
    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = bf16x16_to_f32x16_avx512(a.as_ptr().add(i * 16));
            let vb = bf16x16_to_f32x16_avx512(b.as_ptr().add(i * 16));
            let diff = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);

        // Handle remainder
        for i in (chunks * 16)..len {
            let diff = a[i].to_f32() - b[i].to_f32();
            sum += diff * diff;
        }
        sum
    }
}

/// Euclidean distance - AVX-512 fallback for non-AVX512 compilation
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
fn euclidean_bf16_avx512(a: &[bf16], b: &[bf16]) -> f32 {
    euclidean_bf16_avx2(a, b)
}

/// Euclidean distance between bf16 vectors - NEON (128-bit, aarch64)
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(all(feature = "quantised", target_arch = "aarch64"))]
#[inline(always)]
fn euclidean_bf16_neon(a: &[bf16], b: &[bf16]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let va = bf16x4_to_f32x4_neon(a.as_ptr().add(i * 4));
            let vb = bf16x4_to_f32x4_neon(b.as_ptr().add(i * 4));
            let diff = vsubq_f32(va, vb);
            // FMA: acc = diff * diff + acc
            acc = vfmaq_f32(acc, diff, diff);
        }

        let mut sum = hsum_f32_neon(acc);

        // Handle remainder
        for i in (chunks * 4)..len {
            let diff = a[i].to_f32() - b[i].to_f32();
            sum += diff * diff;
        }
        sum
    }
}

/// Dispatch Euclidean distance calculation to appropriate SIMD implementation
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(feature = "quantised")]
#[inline]
pub fn euclidean_bf16_simd(a: &[bf16], b: &[bf16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match crate::detect_simd_level() {
            crate::SimdLevel::Avx512 => euclidean_bf16_avx512(a, b),
            crate::SimdLevel::Avx2 => euclidean_bf16_avx2(a, b),
            crate::SimdLevel::Sse => euclidean_bf16_sse(a, b),
            crate::SimdLevel::Scalar => euclidean_bf16_scalar(a, b),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        euclidean_bf16_neon(a, b)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        euclidean_bf16_scalar(a, b)
    }
}

//////////////////////////
// bf16 vs f32 routines //
//////////////////////////

/// Euclidean distance: bf16 vs f32 - scalar fallback
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(feature = "quantised")]
#[inline(always)]
fn euclidean_bf16_f32_scalar(a: &[bf16], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = f32::from(*x) - y;
            d * d
        })
        .sum()
}

/// Euclidean distance: bf16 vs f32 - SSE (128-bit, x86_64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn euclidean_bf16_f32_sse(a: &[bf16], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = _mm_setzero_ps();

        for i in 0..chunks {
            let va = bf16x4_to_f32x4_sse(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            let diff = _mm_sub_ps(va, vb);
            acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
        }

        let mut sum = hsum_f32_sse(acc);
        for i in (chunks * 4)..len {
            let diff = a[i].to_f32() - b[i];
            sum += diff * diff;
        }
        sum
    }
}

/// Euclidean distance: bf16 vs f32 - AVX2 (256-bit, x86_64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn euclidean_bf16_f32_avx2(a: &[bf16], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = bf16x8_to_f32x8_avx2(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let diff = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        let mut sum = hsum_f32_avx2(acc);
        for i in (chunks * 8)..len {
            let diff = a[i].to_f32() - b[i];
            sum += diff * diff;
        }
        sum
    }
}

/// Euclidean distance: bf16 vs f32 - AVX512 (512-bit, x86_64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
#[inline(always)]
fn euclidean_bf16_f32_avx512(a: &[bf16], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = bf16x16_to_f32x16_avx512(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            let diff = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            let diff = a[i].to_f32() - b[i];
            sum += diff * diff;
        }
        sum
    }
}

/// Euclidean distance: bf16 vs f32 - AVX512 fallback to AVX2
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
fn euclidean_bf16_f32_avx512(a: &[bf16], b: &[f32]) -> f32 {
    euclidean_bf16_f32_avx2(a, b)
}

/// Euclidean distance: bf16 vs f32 - NEON (128-bit, aarch64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance
#[cfg(all(feature = "quantised", target_arch = "aarch64"))]
#[inline(always)]
fn euclidean_bf16_f32_neon(a: &[bf16], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let va = bf16x4_to_f32x4_neon(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            let diff = vsubq_f32(va, vb);
            acc = vfmaq_f32(acc, diff, diff);
        }

        let mut sum = hsum_f32_neon(acc);
        for i in (chunks * 4)..len {
            let diff = a[i].to_f32() - b[i];
            sum += diff * diff;
        }
        sum
    }
}

//////////////////////////
// bf16 vs f64 routines //
//////////////////////////

/// Euclidean distance: bf16 vs f64 - scalar fallback
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance (f64 converted to f32)
#[cfg(feature = "quantised")]
#[inline(always)]
fn euclidean_bf16_f64_scalar(a: &[bf16], b: &[f64]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = f32::from(*x) - (*y as f32);
            d * d
        })
        .sum()
}

/// Euclidean distance: bf16 vs f64 - SSE (128-bit, x86_64)
///
/// Processes 4 elements per iteration: bf16x4 → f32x4, f64x2+f64x2 → f32x4
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance (f64 converted to f32)
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn euclidean_bf16_f64_sse(a: &[bf16], b: &[f64]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = _mm_setzero_ps();

        for i in 0..chunks {
            let offset = i * 4;

            // Load and convert bf16 → f32
            let va = bf16x4_to_f32x4_sse(a.as_ptr().add(offset));

            // Load f64x2 twice, convert each to f32x2, combine into f32x4
            let b_lo = _mm_loadu_pd(b.as_ptr().add(offset)); // 2 f64
            let b_hi = _mm_loadu_pd(b.as_ptr().add(offset + 2)); // 2 f64
            let b_lo_f32 = _mm_cvtpd_ps(b_lo); // lower 2 floats valid
            let b_hi_f32 = _mm_cvtpd_ps(b_hi); // lower 2 floats valid
                                               // Combine: [lo0, lo1, hi0, hi1]
            let vb = _mm_movelh_ps(b_lo_f32, b_hi_f32);

            let diff = _mm_sub_ps(va, vb);
            acc = _mm_add_ps(acc, _mm_mul_ps(diff, diff));
        }

        let mut sum = hsum_f32_sse(acc);
        for i in (chunks * 4)..len {
            let diff = a[i].to_f32() - (b[i] as f32);
            sum += diff * diff;
        }
        sum
    }
}

/// Euclidean distance: bf16 vs f64 - AVX2 (256-bit, x86_64)
///
/// Processes 8 elements per iteration: bf16x8 → f32x8, f64x4+f64x4 → f32x8
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance (f64 converted to f32)
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn euclidean_bf16_f64_avx2(a: &[bf16], b: &[f64]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;

            // bf16x8 → f32x8
            let va = bf16x8_to_f32x8_avx2(a.as_ptr().add(offset));

            // f64x4 → f32x4 (returns 128-bit), do twice
            let b_lo = _mm256_loadu_pd(b.as_ptr().add(offset));
            let b_hi = _mm256_loadu_pd(b.as_ptr().add(offset + 4));
            let b_lo_f32 = _mm256_cvtpd_ps(b_lo); // __m128
            let b_hi_f32 = _mm256_cvtpd_ps(b_hi); // __m128

            // Combine two __m128 into __m256
            let vb = _mm256_insertf128_ps(_mm256_castps128_ps256(b_lo_f32), b_hi_f32, 1);

            let diff = _mm256_sub_ps(va, vb);
            acc = _mm256_fmadd_ps(diff, diff, acc);
        }

        let mut sum = hsum_f32_avx2(acc);
        for i in (chunks * 8)..len {
            let diff = a[i].to_f32() - (b[i] as f32);
            sum += diff * diff;
        }
        sum
    }
}

/// Euclidean distance: bf16 vs f64 - AVX512 (512-bit, x86_64)
///
/// Processes 16 elements per iteration: bf16x16 → f32x16, f64x8+f64x8 → f32x16
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance (f64 converted to f32)
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
#[inline(always)]
fn euclidean_bf16_f64_avx512(a: &[bf16], b: &[f64]) -> f32 {
    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let offset = i * 16;

            // bf16x16 → f32x16
            let va = bf16x16_to_f32x16_avx512(a.as_ptr().add(offset));

            // f64x8 → f32x8 (returns __m256), do twice
            let b_lo = _mm512_loadu_pd(b.as_ptr().add(offset));
            let b_hi = _mm512_loadu_pd(b.as_ptr().add(offset + 8));
            let b_lo_f32 = _mm512_cvtpd_ps(b_lo); // __m256
            let b_hi_f32 = _mm512_cvtpd_ps(b_hi); // __m256

            // Combine two __m256 into __m512
            let vb = _mm512_insertf32x8(_mm512_castps256_ps512(b_lo_f32), b_hi_f32, 1);

            let diff = _mm512_sub_ps(va, vb);
            acc = _mm512_fmadd_ps(diff, diff, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            let diff = a[i].to_f32() - (b[i] as f32);
            sum += diff * diff;
        }
        sum
    }
}

/// Euclidean distance: bf16 vs f64 - AVX512 fallback to AVX2
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance (f64 converted to f32)
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
fn euclidean_bf16_f64_avx512(a: &[bf16], b: &[f64]) -> f32 {
    euclidean_bf16_f64_avx2(a, b)
}

/// Euclidean distance: bf16 vs f64 - NEON (128-bit, aarch64)
///
/// Processes 4 elements per iteration: bf16x4 → f32x4, f64x2+f64x2 → f32x4
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Squared Euclidean distance (f64 converted to f32)
#[cfg(all(feature = "quantised", target_arch = "aarch64"))]
#[inline(always)]
fn euclidean_bf16_f64_neon(a: &[bf16], b: &[f64]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;

            // bf16x4 → f32x4
            let va = bf16x4_to_f32x4_neon(a.as_ptr().add(offset));

            // f64x2 → f32x2, twice
            let b_lo = vld1q_f64(b.as_ptr().add(offset)); // float64x2_t
            let b_hi = vld1q_f64(b.as_ptr().add(offset + 2)); // float64x2_t
            let b_lo_f32 = vcvt_f32_f64(b_lo); // float32x2_t
            let b_hi_f32 = vcvt_f32_f64(b_hi); // float32x2_t

            // Combine two float32x2_t into float32x4_t
            let vb = vcombine_f32(b_lo_f32, b_hi_f32);

            let diff = vsubq_f32(va, vb);
            acc = vfmaq_f32(acc, diff, diff);
        }

        let mut sum = hsum_f32_neon(acc);
        for i in (chunks * 4)..len {
            let diff = a[i].to_f32() - (b[i] as f32);
            sum += diff * diff;
        }
        sum
    }
}

/////////////////
// Dispatchers //
/////////////////

/// Euclidean distance: bf16 storage vs f32 query - SIMD dispatcher
///
/// ### Params
///
/// * `a` - bf16 vector slice (storage)
/// * `b` - f32 vector slice (query)
///
/// ### Returns
///
/// Squared Euclidean distance as f32
#[cfg(feature = "quantised")]
#[inline]
pub fn euclidean_bf16_f32_simd(a: &[bf16], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match detect_simd_level() {
            SimdLevel::Avx512 => euclidean_bf16_f32_avx512(a, b),
            SimdLevel::Avx2 => euclidean_bf16_f32_avx2(a, b),
            SimdLevel::Sse => euclidean_bf16_f32_sse(a, b),
            SimdLevel::Scalar => euclidean_bf16_f32_scalar(a, b),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        euclidean_bf16_f32_neon(a, b)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        euclidean_bf16_f32_scalar(a, b)
    }
}

/// Euclidean distance: bf16 storage vs f64 query - SIMD dispatcher
///
/// ### Params
///
/// * `a` - bf16 vector slice (storage)
/// * `b` - f64 vector slice (query)
///
/// ### Returns
///
/// Squared Euclidean distance as f32 (f64 converted on-the-fly)
#[cfg(feature = "quantised")]
#[inline]
pub fn euclidean_bf16_f64_simd(a: &[bf16], b: &[f64]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match detect_simd_level() {
            SimdLevel::Avx512 => euclidean_bf16_f64_avx512(a, b),
            SimdLevel::Avx2 => euclidean_bf16_f64_avx2(a, b),
            SimdLevel::Sse => euclidean_bf16_f64_sse(a, b),
            SimdLevel::Scalar => euclidean_bf16_f64_scalar(a, b),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        euclidean_bf16_f64_neon(a, b)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        euclidean_bf16_f64_scalar(a, b)
    }
}

////////////
// Cosine //
////////////

/// Dot product of bf16 vectors - scalar fallback
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(feature = "quantised")]
#[inline(always)]
fn dot_bf16_scalar(a: &[bf16], b: &[bf16]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| f32::from(*x) * f32::from(*y))
        .sum()
}

/// Dot product of bf16 vectors - SSE (128-bit)
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn dot_bf16_sse(a: &[bf16], b: &[bf16]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = _mm_setzero_ps();

        for i in 0..chunks {
            let va = bf16x4_to_f32x4_sse(a.as_ptr().add(i * 4));
            let vb = bf16x4_to_f32x4_sse(b.as_ptr().add(i * 4));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
        }

        let mut sum = hsum_f32_sse(acc);

        for i in (chunks * 4)..len {
            sum += a[i].to_f32() * b[i].to_f32();
        }
        sum
    }
}

/// Dot product of bf16 vectors - AVX2 (256-bit)
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn dot_bf16_avx2(a: &[bf16], b: &[bf16]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = bf16x8_to_f32x8_avx2(a.as_ptr().add(i * 8));
            let vb = bf16x8_to_f32x8_avx2(b.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        let mut sum = hsum_f32_avx2(acc);

        for i in (chunks * 8)..len {
            sum += a[i].to_f32() * b[i].to_f32();
        }
        sum
    }
}

/// Dot product of bf16 vectors - AVX-512 (512-bit)
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
#[inline(always)]
fn dot_bf16_avx512(a: &[bf16], b: &[bf16]) -> f32 {
    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = bf16x16_to_f32x16_avx512(a.as_ptr().add(i * 16));
            let vb = bf16x16_to_f32x16_avx512(b.as_ptr().add(i * 16));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);

        for i in (chunks * 16)..len {
            sum += a[i].to_f32() * b[i].to_f32();
        }
        sum
    }
}

/// Dot product - AVX-512 fallback for non-AVX512 compilation
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
fn dot_bf16_avx512(a: &[bf16], b: &[bf16]) -> f32 {
    dot_bf16_avx2(a, b)
}

/// Dot product of bf16 vectors - NEON (128-bit, aarch64)
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(all(feature = "quantised", target_arch = "aarch64"))]
#[inline(always)]
fn dot_bf16_neon(a: &[bf16], b: &[bf16]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let va = bf16x4_to_f32x4_neon(a.as_ptr().add(i * 4));
            let vb = bf16x4_to_f32x4_neon(b.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, va, vb);
        }

        let mut sum = hsum_f32_neon(acc);

        for i in (chunks * 4)..len {
            sum += a[i].to_f32() * b[i].to_f32();
        }
        sum
    }
}

/// Dispatch dot product calculation to appropriate SIMD implementation
///
/// ### Params
///
/// * `a` - First bf16 vector slice
/// * `b` - Second bf16 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(feature = "quantised")]
#[inline]
pub fn dot_bf16_simd(a: &[bf16], b: &[bf16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match crate::detect_simd_level() {
            crate::SimdLevel::Avx512 => dot_bf16_avx512(a, b),
            crate::SimdLevel::Avx2 => dot_bf16_avx2(a, b),
            crate::SimdLevel::Sse => dot_bf16_sse(a, b),
            crate::SimdLevel::Scalar => dot_bf16_scalar(a, b),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        dot_bf16_neon(a, b)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_bf16_scalar(a, b)
    }
}

//////////////////////////
// bf16 vs f32 dot prod //
//////////////////////////

/// Dot product: bf16 vs f32 - scalar fallback
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(feature = "quantised")]
#[inline(always)]
fn dot_bf16_f32_scalar(a: &[bf16], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| f32::from(*x) * y).sum()
}

/// Dot product: bf16 vs f32 - SSE (128-bit, x86_64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn dot_bf16_f32_sse(a: &[bf16], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = _mm_setzero_ps();

        for i in 0..chunks {
            let va = bf16x4_to_f32x4_sse(a.as_ptr().add(i * 4));
            let vb = _mm_loadu_ps(b.as_ptr().add(i * 4));
            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
        }

        let mut sum = hsum_f32_sse(acc);
        for i in (chunks * 4)..len {
            sum += a[i].to_f32() * b[i];
        }
        sum
    }
}

/// Dot product: bf16 vs f32 - AVX2 (256-bit, x86_64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn dot_bf16_f32_avx2(a: &[bf16], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let va = bf16x8_to_f32x8_avx2(a.as_ptr().add(i * 8));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        let mut sum = hsum_f32_avx2(acc);
        for i in (chunks * 8)..len {
            sum += a[i].to_f32() * b[i];
        }
        sum
    }
}

/// Dot product: bf16 vs f32 - AVX512 (512-bit, x86_64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
#[inline(always)]
fn dot_bf16_f32_avx512(a: &[bf16], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let va = bf16x16_to_f32x16_avx512(a.as_ptr().add(i * 16));
            let vb = _mm512_loadu_ps(b.as_ptr().add(i * 16));
            acc = _mm512_fmadd_ps(va, vb, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            sum += a[i].to_f32() * b[i];
        }
        sum
    }
}

/// Dot product: bf16 vs f32 - AVX512 fallback to AVX2
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
fn dot_bf16_f32_avx512(a: &[bf16], b: &[f32]) -> f32 {
    dot_bf16_f32_avx2(a, b)
}

/// Dot product: bf16 vs f32 - NEON (128-bit, aarch64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f32 vector slice
///
/// ### Returns
///
/// Dot product
#[cfg(all(feature = "quantised", target_arch = "aarch64"))]
#[inline(always)]
fn dot_bf16_f32_neon(a: &[bf16], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let va = bf16x4_to_f32x4_neon(a.as_ptr().add(i * 4));
            let vb = vld1q_f32(b.as_ptr().add(i * 4));
            acc = vfmaq_f32(acc, va, vb);
        }

        let mut sum = hsum_f32_neon(acc);
        for i in (chunks * 4)..len {
            sum += a[i].to_f32() * b[i];
        }
        sum
    }
}

//////////////////////////
// bf16 vs f64 dot prod //
//////////////////////////

/// Dot product: bf16 vs f64 - scalar fallback
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Dot product (f64 converted to f32)
#[cfg(feature = "quantised")]
#[inline(always)]
fn dot_bf16_f64_scalar(a: &[bf16], b: &[f64]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| f32::from(*x) * (*y as f32))
        .sum()
}

/// Dot product: bf16 vs f64 - SSE (128-bit, x86_64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Dot product (f64 converted to f32)
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn dot_bf16_f64_sse(a: &[bf16], b: &[f64]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = _mm_setzero_ps();

        for i in 0..chunks {
            let offset = i * 4;

            let va = bf16x4_to_f32x4_sse(a.as_ptr().add(offset));

            let b_lo = _mm_loadu_pd(b.as_ptr().add(offset));
            let b_hi = _mm_loadu_pd(b.as_ptr().add(offset + 2));
            let b_lo_f32 = _mm_cvtpd_ps(b_lo);
            let b_hi_f32 = _mm_cvtpd_ps(b_hi);
            let vb = _mm_movelh_ps(b_lo_f32, b_hi_f32);

            acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
        }

        let mut sum = hsum_f32_sse(acc);
        for i in (chunks * 4)..len {
            sum += a[i].to_f32() * (b[i] as f32);
        }
        sum
    }
}

/// Dot product: bf16 vs f64 - AVX2 (256-bit, x86_64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Dot product (f64 converted to f32)
#[cfg(all(feature = "quantised", target_arch = "x86_64"))]
#[inline(always)]
fn dot_bf16_f64_avx2(a: &[bf16], b: &[f64]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    unsafe {
        let mut acc = _mm256_setzero_ps();

        for i in 0..chunks {
            let offset = i * 8;

            let va = bf16x8_to_f32x8_avx2(a.as_ptr().add(offset));

            let b_lo = _mm256_loadu_pd(b.as_ptr().add(offset));
            let b_hi = _mm256_loadu_pd(b.as_ptr().add(offset + 4));
            let b_lo_f32 = _mm256_cvtpd_ps(b_lo);
            let b_hi_f32 = _mm256_cvtpd_ps(b_hi);
            let vb = _mm256_insertf128_ps(_mm256_castps128_ps256(b_lo_f32), b_hi_f32, 1);

            acc = _mm256_fmadd_ps(va, vb, acc);
        }

        let mut sum = hsum_f32_avx2(acc);
        for i in (chunks * 8)..len {
            sum += a[i].to_f32() * (b[i] as f32);
        }
        sum
    }
}

/// Dot product: bf16 vs f64 - AVX512 (512-bit, x86_64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Dot product (f64 converted to f32)
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    target_feature = "avx512f"
))]
#[inline(always)]
fn dot_bf16_f64_avx512(a: &[bf16], b: &[f64]) -> f32 {
    let len = a.len();
    let chunks = len / 16;

    unsafe {
        let mut acc = _mm512_setzero_ps();

        for i in 0..chunks {
            let offset = i * 16;

            let va = bf16x16_to_f32x16_avx512(a.as_ptr().add(offset));

            let b_lo = _mm512_loadu_pd(b.as_ptr().add(offset));
            let b_hi = _mm512_loadu_pd(b.as_ptr().add(offset + 8));
            let b_lo_f32 = _mm512_cvtpd_ps(b_lo);
            let b_hi_f32 = _mm512_cvtpd_ps(b_hi);
            let vb = _mm512_insertf32x8(_mm512_castps256_ps512(b_lo_f32), b_hi_f32, 1);

            acc = _mm512_fmadd_ps(va, vb, acc);
        }

        let mut sum = _mm512_reduce_add_ps(acc);
        for i in (chunks * 16)..len {
            sum += a[i].to_f32() * (b[i] as f32);
        }
        sum
    }
}

/// Dot product: bf16 vs f64 - AVX512 fallback to AVX2
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Dot product (f64 converted to f32)
#[cfg(all(
    feature = "quantised",
    target_arch = "x86_64",
    not(target_feature = "avx512f")
))]
#[inline(always)]
fn dot_bf16_f64_avx512(a: &[bf16], b: &[f64]) -> f32 {
    dot_bf16_f64_avx2(a, b)
}

/// Dot product: bf16 vs f64 - NEON (128-bit, aarch64)
///
/// ### Params
///
/// * `a` - bf16 vector slice
/// * `b` - f64 vector slice
///
/// ### Returns
///
/// Dot product (f64 converted to f32)
#[cfg(all(feature = "quantised", target_arch = "aarch64"))]
#[inline(always)]
fn dot_bf16_f64_neon(a: &[bf16], b: &[f64]) -> f32 {
    let len = a.len();
    let chunks = len / 4;

    unsafe {
        let mut acc = vdupq_n_f32(0.0);

        for i in 0..chunks {
            let offset = i * 4;

            let va = bf16x4_to_f32x4_neon(a.as_ptr().add(offset));

            let b_lo = vld1q_f64(b.as_ptr().add(offset));
            let b_hi = vld1q_f64(b.as_ptr().add(offset + 2));
            let b_lo_f32 = vcvt_f32_f64(b_lo);
            let b_hi_f32 = vcvt_f32_f64(b_hi);
            let vb = vcombine_f32(b_lo_f32, b_hi_f32);

            acc = vfmaq_f32(acc, va, vb);
        }

        let mut sum = hsum_f32_neon(acc);
        for i in (chunks * 4)..len {
            sum += a[i].to_f32() * (b[i] as f32);
        }
        sum
    }
}

/////////////////
// Dispatchers //
/////////////////

/// Dot product: bf16 storage vs f32 query - SIMD dispatcher
///
/// ### Params
///
/// * `a` - bf16 vector slice (storage)
/// * `b` - f32 vector slice (query)
///
/// ### Returns
///
/// Dot product as f32
#[cfg(feature = "quantised")]
#[inline]
pub fn dot_bf16_f32_simd(a: &[bf16], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match detect_simd_level() {
            SimdLevel::Avx512 => dot_bf16_f32_avx512(a, b),
            SimdLevel::Avx2 => dot_bf16_f32_avx2(a, b),
            SimdLevel::Sse => dot_bf16_f32_sse(a, b),
            SimdLevel::Scalar => dot_bf16_f32_scalar(a, b),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        dot_bf16_f32_neon(a, b)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_bf16_f32_scalar(a, b)
    }
}

/// Dot product: bf16 storage vs f64 query - SIMD dispatcher
///
/// ### Params
///
/// * `a` - bf16 vector slice (storage)
/// * `b` - f64 vector slice (query)
///
/// ### Returns
///
/// Dot product as f32 (f64 converted on-the-fly)
#[cfg(feature = "quantised")]
#[inline]
pub fn dot_bf16_f64_simd(a: &[bf16], b: &[f64]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        match detect_simd_level() {
            SimdLevel::Avx512 => dot_bf16_f64_avx512(a, b),
            SimdLevel::Avx2 => dot_bf16_f64_avx2(a, b),
            SimdLevel::Sse => dot_bf16_f64_sse(a, b),
            SimdLevel::Scalar => dot_bf16_f64_scalar(a, b),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        dot_bf16_f64_neon(a, b)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        dot_bf16_f64_scalar(a, b)
    }
}

////////////////////
// Bf16Compatible //
////////////////////

/// Marker trait for types we can dispatch bf16 distance to
#[cfg(feature = "quantised")]
pub trait Bf16Compatible: Float + ToPrimitive {
    /// Calculate the Euclidean distance between BF16 and a Bf16-compatible
    /// float
    ///
    /// ### Params
    ///
    /// * `a` - Slice representing the bf16-encoded vector
    /// * `b` - Slice of the Bf16-compatible  float
    ///
    /// ### Returns
    ///
    /// Euclidean distance
    fn euclidean_bf16_dispatch(a: &[bf16], b: &[Self]) -> f32;

    /// Calculate the dot product between BF16 and a Bf16-compatible float
    ///
    /// ### Params
    ///
    /// * `a` - Slice representing the bf16-encoded vector
    /// * `b` - Slice of the Bf16-compatible  float
    ///
    /// ### Returns
    ///
    /// Dot product
    fn dot_bf16_dispatch(a: &[bf16], b: &[Self]) -> f32;
}

#[cfg(feature = "quantised")]
impl Bf16Compatible for f32 {
    #[inline(always)]
    fn euclidean_bf16_dispatch(a: &[bf16], b: &[Self]) -> f32 {
        euclidean_bf16_f32_simd(a, b)
    }

    #[inline(always)]
    fn dot_bf16_dispatch(a: &[bf16], b: &[Self]) -> f32 {
        dot_bf16_f32_simd(a, b)
    }
}

#[cfg(feature = "quantised")]
impl Bf16Compatible for f64 {
    #[inline(always)]
    fn euclidean_bf16_dispatch(a: &[bf16], b: &[Self]) -> f32 {
        euclidean_bf16_f64_simd(a, b)
    }

    #[inline(always)]
    fn dot_bf16_dispatch(a: &[bf16], b: &[Self]) -> f32 {
        dot_bf16_f64_simd(a, b)
    }
}

///////////
// Trait //
///////////

#[cfg(feature = "quantised")]
/// Trait for computing distances between Floats
pub trait VectorDistanceBf16<T>
where
    T: Float + Sum + FromPrimitive + ToPrimitive + SimdDistance,
{
    /// Get the internal flat vector representation
    fn vectors_flat(&self) -> &[bf16];

    /// Get the internal dimensions
    fn dim(&self) -> usize;

    /// Get the normalised values
    fn norms(&self) -> &[T];

    ///////////////
    // Euclidean //
    ///////////////

    /// Euclidean distance between two internal vectors (squared; bf16)
    ///
    /// ### Params
    ///
    /// * `i` - Sample index i
    /// * `j` - Sample index j
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance between the two samples
    #[inline(always)]
    fn euclidean_distance_bf16(&self, i: usize, j: usize) -> T {
        let start_i = i * self.dim();
        let start_j = j * self.dim();
        let vec_i = &self.vectors_flat()[start_i..start_i + self.dim()];
        let vec_j = &self.vectors_flat()[start_j..start_j + self.dim()];

        let result = euclidean_bf16_simd(vec_i, vec_j);
        T::from_f32(result).unwrap()
    }

    /// Euclidean distance between query vector and internal vector
    /// (squared; bf16)
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query` - Query vector slice
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance
    #[inline(always)]
    fn euclidean_distance_to_query_bf16<Q>(&self, internal_idx: usize, query: &[Q]) -> T
    where
        Q: Bf16Compatible,
    {
        let start = internal_idx * self.dim();
        let vec = &self.vectors_flat()[start..start + self.dim()];
        T::from_f32(Q::euclidean_bf16_dispatch(vec, query)).unwrap()
    }

    /// Euclidean distance between query vector and internal vector
    /// (squared; bf16)
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query` - Query vector slice
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance
    #[inline(always)]
    fn euclidean_distance_to_query_dual_bf16(&self, internal_idx: usize, query: &[bf16]) -> T {
        let start = internal_idx * self.dim();
        let vec = &self.vectors_flat()[start..start + self.dim()];

        let result = euclidean_bf16_simd(vec, query);
        T::from_f32(result).unwrap()
    }

    ////////////
    // Cosine //
    ////////////

    /// Cosine distance between two internal vectors
    ///
    /// Uses pre-computed norms.
    ///
    /// ### Params
    ///
    /// * `i` - Sample index i
    /// * `j` - Sample index j
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The Cosine distance between the two samples
    #[inline(always)]
    fn cosine_distance_bf16(&self, i: usize, j: usize) -> T {
        let start_i = i * self.dim();
        let start_j = j * self.dim();
        let vec_i = &self.vectors_flat()[start_i..start_i + self.dim()];
        let vec_j = &self.vectors_flat()[start_j..start_j + self.dim()];

        let dot = dot_bf16_simd(vec_i, vec_j);
        let norm_i = self.norms()[i].to_f32().unwrap();
        let norm_j = self.norms()[j].to_f32().unwrap();

        let dist = 1.0 - (dot / (norm_i * norm_j));
        T::from_f32(dist).unwrap()
    }

    /// Cosine distance between query vector and internal vector
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query` - Query vector slice
    /// * `query_norm` - Pre-computed norm of query vector
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The Cosine distance
    #[inline(always)]
    fn cosine_distance_to_query_bf16<Q>(&self, internal_idx: usize, query: &[Q], query_norm: T) -> T
    where
        Q: Bf16Compatible,
    {
        let start = internal_idx * self.dim();
        let vec = &self.vectors_flat()[start..start + self.dim()];
        let dot = Q::dot_bf16_dispatch(vec, query);
        let dist = 1.0
            - (dot / (query_norm.to_f32().unwrap() * self.norms()[internal_idx].to_f32().unwrap()));
        T::from_f32(dist).unwrap()
    }

    /// Cosine distance between query vector and internal vector
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query` - Query vector slice
    /// * `query_norm` - Pre-computed norm of query vector
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The Cosine distance
    #[inline(always)]
    fn cosine_distance_to_query_dual_bf16(
        &self,
        internal_idx: usize,
        query: &[bf16],
        query_norm: bf16,
    ) -> T {
        let start = internal_idx * self.dim();
        let vec = &self.vectors_flat()[start..start + self.dim()];

        let dot = dot_bf16_simd(vec, query);
        let norm_internal = self.norms()[internal_idx].to_f32().unwrap();

        let dist = 1.0 - (dot / (query_norm.to_f32() * norm_internal));
        T::from_f32(dist).unwrap()
    }
}

///////////////////////
// VectorDistanceSq8 //
///////////////////////

// Tests with SIMD have not yielded any benefit here...

#[cfg(feature = "quantised")]
/// Trait for computing distances between `i8`
pub trait VectorDistanceSq8<T>
where
    T: Float + FromPrimitive + ToPrimitive,
{
    /// Get the internal flat vector representation (quantised to i8)
    fn vectors_flat_quantised(&self) -> &[i8];

    /// Get the internal norms of the quantised vectors
    fn norms_quantised(&self) -> &[i32];

    /// Get the internal dimensions
    fn dim(&self) -> usize;

    ///////////////
    // Euclidean //
    ///////////////

    /// Calculate euclidean distance against quantised query
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query_i8` - Query vector slice quantised to i8
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance
    #[inline(always)]
    fn euclidean_distance_i8(&self, internal_idx: usize, query_i8: &[i8]) -> T {
        let start = internal_idx * self.dim();
        unsafe {
            let db_vec = &self
                .vectors_flat_quantised()
                .get_unchecked(start..start + self.dim());

            let sum: i32 = query_i8
                .iter()
                .zip(db_vec.iter())
                .map(|(&q, &d)| {
                    let diff = q as i32 - d as i32;
                    diff * diff
                })
                .sum();

            T::from_i32(sum).unwrap()
        }
    }

    /// Calculate cosine distance against quantised query
    ///
    /// ### Params
    ///
    /// * `internal_idx` - Index of internal vector
    /// * `query_i8` - Query vector slice quantised to i8
    /// * `query_norm_sq` - Squared norm of the query vector
    ///
    /// ### Safety
    ///
    /// Uses unsafe to retrieve the data in an unchecked manner for maximum
    /// performance.
    ///
    /// ### Returns
    ///
    /// The squared Euclidean distance
    #[inline(always)]
    fn cosine_distance_i8(&self, vec_idx: usize, query_i8: &[i8], query_norm_sq: i32) -> T {
        let start = vec_idx * self.dim();

        unsafe {
            let db_vec = &self
                .vectors_flat_quantised()
                .get_unchecked(start..start + self.dim());

            let dot: i32 = query_i8
                .iter()
                .zip(db_vec.iter())
                .map(|(&q, &d)| q as i32 * d as i32)
                .sum();

            let db_norm_sq: i32 = self.norms_quantised()[vec_idx];

            let query_norm = T::from_i32(query_norm_sq).unwrap().sqrt();
            let db_norm = T::from_i32(db_norm_sq).unwrap().sqrt();

            if query_norm > T::zero() && db_norm > T::zero() {
                T::one() - T::from_i32(dot).unwrap() / (query_norm * db_norm)
            } else {
                T::one()
            }
        }
    }
}

///////////////////////
// VectorDistanceAdc //
///////////////////////

#[cfg(feature = "quantised")]
pub trait VectorDistanceAdc<T>
where
    T: Float + FromPrimitive + ToPrimitive + Sum + SimdDistance,
{
    /// Get the m value from the codebook
    fn codebook_m(&self) -> usize;

    /// Get the number of centroids from the codebook
    fn codebook_n_centroids(&self) -> usize;

    /// Get the subvector dimensions from the codebook
    fn codebook_subvec_dim(&self) -> usize;

    /// Get the internal flat centroids representation
    fn centroids(&self) -> &[T];

    /// Get the internal dimensions
    fn dim(&self) -> usize;

    /// Return the codebooks data
    fn codebooks(&self) -> &[Vec<T>];

    /// Get the quantised codes
    fn quantised_codes(&self) -> &[u8];

    /// Build ADC lookup tables for a specific cluster
    ///
    /// ### Params
    ///
    /// * `query` - The query vector
    /// * `cluster_idx`
    ///
    /// ### Returns
    ///
    /// Lookup table as flat Vec<T> of size M * n_centroids
    fn build_lookup_tables(&self, query_vec: &[T], cluster_idx: usize) -> Vec<T> {
        let m = self.codebook_m();
        let subvec_dim = self.codebook_subvec_dim();
        let n_cents = self.codebook_n_centroids();

        let centroid = &self.centroids()[cluster_idx * self.dim()..(cluster_idx + 1) * self.dim()];

        let query_residual = T::subtract_simd(query_vec, centroid);

        let mut table = vec![T::zero(); m * n_cents];

        for subspace in 0..m {
            let query_sub = &query_residual[subspace * subvec_dim..(subspace + 1) * subvec_dim];
            let table_offset = subspace * n_cents;

            for centroid_idx in 0..n_cents {
                let centroid_start = centroid_idx * subvec_dim;
                let pq_centroid =
                    &self.codebooks()[subspace][centroid_start..centroid_start + subvec_dim];

                // squared Euclidean distance for ADC
                let dist = T::euclidean_simd(query_sub, pq_centroid);

                table[table_offset + centroid_idx] = dist;
            }
        }

        table
    }

    /// Compute distance using ADC lookup tables
    ///
    /// Optimised with manual unrolling and unsafe indexing for small m
    ///
    /// ### Params
    ///
    /// * `vec_idx` - Index of database vector
    /// * `lookup_tables` - Precomputed distance table (flat layout)
    ///
    /// ### Returns
    ///
    /// Approximate distance
    #[inline(always)]
    fn compute_distance_adc(&self, vec_idx: usize, lookup_table: &[T]) -> T {
        let m = self.codebook_m();
        let n_cents = self.codebook_n_centroids();
        let codes_start = vec_idx * m;
        let codes = &self.quantised_codes()[codes_start..codes_start + m];

        // manual unrolling for common small m values with unsafe indexing
        match m {
            8 => {
                let mut sum = T::zero();
                for i in 0..8 {
                    let code = unsafe { *codes.get_unchecked(i) } as usize;
                    let offset = i * n_cents + code;
                    sum = sum + unsafe { *lookup_table.get_unchecked(offset) };
                }
                sum
            }
            16 => {
                let mut sum = T::zero();
                for i in 0..16 {
                    let code = unsafe { *codes.get_unchecked(i) } as usize;
                    let offset = i * n_cents + code;
                    sum = sum + unsafe { *lookup_table.get_unchecked(offset) };
                }
                sum
            }
            32 => {
                let mut sum = T::zero();
                for i in 0..32 {
                    let code = unsafe { *codes.get_unchecked(i) } as usize;
                    let offset = i * n_cents + code;
                    sum = sum + unsafe { *lookup_table.get_unchecked(offset) };
                }
                sum
            }
            _ => {
                // Generic fallback for other m values
                codes
                    .iter()
                    .enumerate()
                    .map(|(subspace, &code)| {
                        let offset = subspace * n_cents + (code as usize);
                        lookup_table[offset]
                    })
                    .fold(T::zero(), |acc, x| acc + x)
            }
        }
    }
}

///////////////
// Functions //
///////////////

/////////////
// Helpers //
/////////////

/// Static Euclidean distance between two arbitrary vectors (squared)
///
/// ### Params
///
/// * `a` - Slice of vector one
/// * `b` - Slice of vector two
///
/// ### Returns
///
/// Squared euclidean distance
#[inline(always)]
pub fn euclidean_distance_static<T>(a: &[T], b: &[T]) -> T
where
    T: Float + SimdDistance,
{
    assert!(a.len() == b.len(), "Vectors a and b need to have same len!");

    T::euclidean_simd(a, b)
}

/// Static Cosine distance between two arbitrary vectors
///
/// Computes norms on the fly
///
/// ### Params
///
/// * `a` - Slice of vector one
/// * `b` - Slice of vector two
///
/// ### Returns
///
/// Squared cosine distance
#[inline(always)]
pub fn cosine_distance_static<T>(a: &[T], b: &[T]) -> T
where
    T: Float + SimdDistance,
{
    assert!(a.len() == b.len(), "Vectors a and b need to have same len!");

    let dot: T = T::dot_simd(a, b);
    let norm_a = T::calculate_norm(a);
    let norm_b = T::calculate_norm(b);

    T::one() - (dot / (norm_a * norm_b))
}

/// Static Cosine distance between two arbitrary vectors
///
/// This version accepts pre-calculated norms
///
/// Computes norms on the fly
///
/// ### Params
///
/// * `a` - Slice of vector one
/// * `b` - Slice of vector two
///
/// ### Returns
///
/// Squared cosine distance
pub fn cosine_distance_static_norm<T>(a: &[T], b: &[T], norm_a: &T, norm_b: &T) -> T
where
    T: Float + SimdDistance,
{
    assert!(a.len() == b.len(), "Vectors a and b need to have same len!");

    let dot: T = T::dot_simd(a, b);

    T::one() - (dot / (*norm_a * *norm_b))
}

/// Helper to normalise vector in place
///
/// ### Params
///
/// * `vec` - The vector to normalise
#[inline(always)]
pub fn normalise_vector<T>(vec: &mut [T])
where
    T: Float + Sum + SimdDistance,
{
    let norm = compute_norm(vec);
    if norm > T::zero() {
        vec.iter_mut().for_each(|v| *v = *v / norm);
    }
}

/// Compute the L2 norm of a slice
///
/// ### Params
///
/// * `vec` - Slice for which to calculate L2 norm
///
/// ### Returns
///
/// L2 norm
#[inline(always)]
pub fn compute_norm<T>(vec: &[T]) -> T
where
    T: Float + SimdDistance,
{
    T::calculate_norm(vec)
}

/// Compute the L2 norm of a row reference
///
/// ### Params
///
/// * `row` - Row for which to calculate L2 norm
///
/// ### Returns
///
/// L2 norm
#[inline(always)]
pub fn compute_norm_row<T>(row: RowRef<T>) -> T
where
    T: Float + SimdDistance,
{
    // optimised unsafe path
    if row.col_stride() == 1 {
        let slice = unsafe { std::slice::from_raw_parts(row.as_ptr(), row.ncols()) };
        return T::calculate_norm(slice);
    }
    // clone and use SIMD
    let vec: Vec<T> = row.iter().cloned().collect();
    T::calculate_norm(&vec)
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    struct TestVectors {
        data: Vec<f32>,
        dim: usize,
        norms: Vec<f32>,
    }

    impl VectorDistance<f32> for TestVectors {
        fn vectors_flat(&self) -> &[f32] {
            &self.data
        }

        fn dim(&self) -> usize {
            self.dim
        }

        fn norms(&self) -> &[f32] {
            &self.norms
        }
    }

    #[test]
    fn test_parse_ann_dist_euclidean() {
        assert_eq!(parse_ann_dist("euclidean"), Some(Dist::Euclidean));
        assert_eq!(parse_ann_dist("Euclidean"), Some(Dist::Euclidean));
        assert_eq!(parse_ann_dist("EUCLIDEAN"), Some(Dist::Euclidean));
    }

    #[test]
    fn test_parse_ann_dist_cosine() {
        assert_eq!(parse_ann_dist("cosine"), Some(Dist::Cosine));
        assert_eq!(parse_ann_dist("Cosine"), Some(Dist::Cosine));
        assert_eq!(parse_ann_dist("COSINE"), Some(Dist::Cosine));
    }

    #[test]
    fn test_parse_ann_dist_invalid() {
        assert_eq!(parse_ann_dist("manhattan"), None);
        assert_eq!(parse_ann_dist(""), None);
        assert_eq!(parse_ann_dist("cosine "), None); // Trailing space
        assert_eq!(parse_ann_dist(" euclidean"), None); // Leading space
    }

    #[test]
    fn test_euclidean_distance_basic() {
        let data = vec![
            1.0, 0.0, 0.0, // Vector 0: [1, 0, 0]
            0.0, 1.0, 0.0, // Vector 1: [0, 1, 0]
            1.0, 1.0, 0.0, // Vector 2: [1, 1, 0]
        ];

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![],
        };

        // Distance between [1,0,0] and [0,1,0] should be sqrt(2) squared = 2
        let dist_01 = vecs.euclidean_distance(0, 1);
        assert_relative_eq!(dist_01, 2.0, epsilon = 1e-6);

        // Distance between [1,0,0] and [1,1,0] should be 1
        let dist_02 = vecs.euclidean_distance(0, 2);
        assert_relative_eq!(dist_02, 1.0, epsilon = 1e-6);

        // Distance to itself should be 0
        let dist_00 = vecs.euclidean_distance(0, 0);
        assert_relative_eq!(dist_00, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_euclidean_distance_symmetry() {
        let data = vec![2.0, 3.0, 5.0, 1.0, 4.0, 2.0];

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![],
        };

        let dist_01 = vecs.euclidean_distance(0, 1);
        let dist_10 = vecs.euclidean_distance(1, 0);

        assert_relative_eq!(dist_01, dist_10, epsilon = 1e-6);
    }

    #[test]
    fn test_euclidean_distance_unrolled() {
        // Test with dimension not divisible by 4 to test both unrolled and remainder loops
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, // 5 dimensions
            5.0, 4.0, 3.0, 2.0, 1.0,
        ];

        let vecs = TestVectors {
            data,
            dim: 5,
            norms: vec![],
        };

        let dist = vecs.euclidean_distance(0, 1);
        // Expected: (1-5)^2 + (2-4)^2 + (3-3)^2 + (4-2)^2 + (5-1)^2 = 16 + 4 + 0 + 4 + 16 = 40
        assert_relative_eq!(dist, 40.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_basic() {
        let data = vec![
            1.0, 0.0, 0.0, // Vector 0
            0.0, 1.0, 0.0, // Vector 1
            1.0, 1.0, 0.0, // Vector 2 (45 degrees from both)
        ];

        // Pre-compute norms
        let norm0 = (1.0_f32 * 1.0 + 0.0 * 0.0 + 0.0 * 0.0).sqrt();
        let norm1 = (0.0_f32 * 0.0 + 1.0 * 1.0 + 0.0 * 0.0).sqrt();
        let norm2 = (1.0_f32 * 1.0 + 1.0 * 1.0 + 0.0 * 0.0).sqrt();

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![norm0, norm1, norm2],
        };

        // Orthogonal vectors: cosine similarity = 0, distance = 1
        let dist_01 = vecs.cosine_distance(0, 1);
        assert_relative_eq!(dist_01, 1.0, epsilon = 1e-6);

        // 45 degree angle: cosine similarity = 1/sqrt(2), distance = 1 - 1/sqrt(2)
        let dist_02 = vecs.cosine_distance(0, 2);
        assert_relative_eq!(dist_02, 1.0 - 1.0 / 2.0_f32.sqrt(), epsilon = 1e-5);

        // Same vector: cosine similarity = 1, distance = 0
        let dist_00 = vecs.cosine_distance(0, 0);
        assert_relative_eq!(dist_00, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_norm() {
        let data_1 = vec![1.5, 2.5, 2.0];
        let data_2 = vec![2.5, 0.5, 1.0];
        let data_3 = vec![1.0, 0.0, 1.0];

        let norm_1 = &data_1.iter().map(|x| *x * *x).sum::<f64>().sqrt();
        let norm_2 = &data_2.iter().map(|x| *x * *x).sum::<f64>().sqrt();
        let norm_3 = &data_3.iter().map(|x| *x * *x).sum::<f64>().sqrt();

        assert_relative_eq!(*norm_1, compute_norm(&data_1), epsilon = 1e-5);
        assert_relative_eq!(*norm_2, compute_norm(&data_2), epsilon = 1e-5);
        assert_relative_eq!(*norm_3, compute_norm(&data_3), epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_distance_symmetry() {
        let data = vec![2.0, 3.0, 5.0, 1.0, 4.0, 2.0];

        let norm0 = (2.0_f32 * 2.0 + 3.0 * 3.0 + 5.0 * 5.0).sqrt();
        let norm1 = (1.0_f32 * 1.0 + 4.0 * 4.0 + 2.0 * 2.0).sqrt();

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![norm0, norm1],
        };

        let dist_01 = vecs.cosine_distance(0, 1);
        let dist_10 = vecs.cosine_distance(1, 0);

        assert_relative_eq!(dist_01, dist_10, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_unrolled() {
        // Test with dimension not divisible by 4
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let norm0 = (1.0_f32 + 4.0 + 9.0 + 16.0 + 25.0).sqrt();
        let norm1 = (25.0_f32 + 16.0 + 9.0 + 4.0 + 1.0).sqrt();

        let vecs = TestVectors {
            data,
            dim: 5,
            norms: vec![norm0, norm1],
        };

        // Dot product: 1*5 + 2*4 + 3*3 + 4*2 + 5*1 = 5 + 8 + 9 + 8 + 5 = 35
        // Cosine similarity: 35 / (norm0 * norm1)
        let dist = vecs.cosine_distance(0, 1);
        let expected = 1.0 - (35.0 / (norm0 * norm1));
        assert_relative_eq!(dist, expected, epsilon = 1e-5);
    }

    #[test]
    fn test_parallel_vectors() {
        let data = vec![
            1.0, 2.0, 3.0, 2.0, 4.0, 6.0, // Parallel to first (scaled by 2)
        ];

        let norm0 = (1.0_f32 + 4.0 + 9.0).sqrt();
        let norm1 = (4.0_f32 + 16.0 + 36.0).sqrt();

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![norm0, norm1],
        };

        // Parallel vectors should have cosine distance ≈ 0
        let dist = vecs.cosine_distance(0, 1);
        assert_relative_eq!(dist, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_opposite_vectors() {
        let data = vec![
            1.0, 2.0, 3.0, -1.0, -2.0, -3.0, // Opposite direction
        ];

        let norm0 = (1.0_f32 + 4.0 + 9.0).sqrt();
        let norm1 = (1.0_f32 + 4.0 + 9.0).sqrt();

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![norm0, norm1],
        };

        // Opposite vectors should have cosine distance ≈ 2
        let dist = vecs.cosine_distance(0, 1);
        assert_relative_eq!(dist, 2.0, epsilon = 1e-5);
    }

    #[test]
    fn test_large_dimension() {
        // Test with larger dimension to stress the unrolling
        let dim = 100;
        let mut data = Vec::with_capacity(dim * 2);

        for i in 0..dim {
            data.push(i as f32);
        }
        for i in 0..dim {
            data.push((dim - i) as f32);
        }

        let norm0 = data[0..dim].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm1 = data[dim..].iter().map(|x| x * x).sum::<f32>().sqrt();

        let vecs = TestVectors {
            data,
            dim,
            norms: vec![norm0, norm1],
        };

        // Just verify it computes without crashing and is symmetric
        let dist_01 = vecs.euclidean_distance(0, 1);
        let dist_10 = vecs.euclidean_distance(1, 0);
        assert_relative_eq!(dist_01, dist_10, epsilon = 1e-3);

        let cos_01 = vecs.cosine_distance(0, 1);
        let cos_10 = vecs.cosine_distance(1, 0);
        assert_relative_eq!(cos_01, cos_10, epsilon = 1e-5);
    }

    #[test]
    fn test_euclidean_distance_to_query() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![],
        };

        let query = vec![1.0, 1.0, 0.0];

        // Distance from [1,0,0] to [1,1,0] should be 1
        let dist_0 = vecs.euclidean_distance_to_query(0, &query);
        assert_relative_eq!(dist_0, 1.0, epsilon = 1e-6);

        // Distance from [0,1,0] to [1,1,0] should be 1
        let dist_1 = vecs.euclidean_distance_to_query(1, &query);
        assert_relative_eq!(dist_1, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_to_query() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0];

        let norm0 = 1.0;
        let norm1 = 1.0;
        let norm2 = 2.0_f32.sqrt();

        let vecs = TestVectors {
            data,
            dim: 3,
            norms: vec![norm0, norm1, norm2],
        };

        let query = vec![1.0, 1.0, 0.0];
        let query_norm = 2.0_f32.sqrt();

        // Orthogonal: cosine distance should be 1
        let dist_0 = vecs.cosine_distance_to_query(0, &query, query_norm);
        assert_relative_eq!(dist_0, 1.0 - 1.0 / 2.0_f32.sqrt(), epsilon = 1e-6);

        // Same vector: cosine distance should be 0
        let dist_2 = vecs.cosine_distance_to_query(2, &query, query_norm);
        assert_relative_eq!(dist_2, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_euclidean_distance_static() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        // (1-4)^2 + (2-5)^2 + (3-6)^2 = 9 + 9 + 9 = 27
        let dist = euclidean_distance_static(&a, &b);
        assert_relative_eq!(dist, 27.0, epsilon = 1e-6);

        // Zero distance to self
        let dist_self = euclidean_distance_static(&a, &a);
        assert_relative_eq!(dist_self, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_static() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];

        // Orthogonal vectors
        let dist = cosine_distance_static(&a, &b);
        assert_relative_eq!(dist, 1.0, epsilon = 1e-6);

        // Parallel vectors
        let c = vec![2.0, 0.0, 0.0];
        let dist_parallel = cosine_distance_static(&a, &c);
        assert_relative_eq!(dist_parallel, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_normalise_vector() {
        let mut vec = vec![3.0, 4.0, 0.0];
        normalise_vector(&mut vec);

        // Should be [0.6, 0.8, 0.0]
        assert_relative_eq!(vec[0], 0.6, epsilon = 1e-6);
        assert_relative_eq!(vec[1], 0.8, epsilon = 1e-6);
        assert_relative_eq!(vec[2], 0.0, epsilon = 1e-6);

        // Norm should be 1
        let norm = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_normalise_vector_zero() {
        let mut vec = vec![0.0, 0.0, 0.0];
        normalise_vector(&mut vec);

        // Zero vector should remain zero
        assert_relative_eq!(vec[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vec[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(vec[2], 0.0, epsilon = 1e-6);
    }

    #[cfg(feature = "quantised")]
    mod quantised_tests {
        use super::*;

        struct TestVectorsSq8 {
            data: Vec<i8>,
            norms: Vec<i32>,
            dim: usize,
        }

        impl VectorDistanceSq8<f32> for TestVectorsSq8 {
            fn vectors_flat_quantised(&self) -> &[i8] {
                &self.data
            }

            fn norms_quantised(&self) -> &[i32] {
                &self.norms
            }

            fn dim(&self) -> usize {
                self.dim
            }
        }

        #[test]
        fn test_euclidean_distance_i8() {
            let data = vec![127, 0, 0, 0, 127, 0];

            let vecs = TestVectorsSq8 {
                data,
                norms: vec![],
                dim: 3,
            };

            let query = vec![127, 127, 0];

            // Distance from [127,0,0] to [127,127,0] should be 127^2
            let dist = vecs.euclidean_distance_i8(0, &query);
            assert_relative_eq!(dist, 16129.0, epsilon = 1e-3);
        }

        #[test]
        fn test_cosine_distance_i8() {
            let data = vec![127, 0, 0, 0, 127, 0, 127, 127, 0];

            let norm0 = 127 * 127;
            let norm1 = 127 * 127;
            let norm2 = 127 * 127 + 127 * 127;

            let vecs = TestVectorsSq8 {
                data,
                norms: vec![norm0, norm1, norm2],
                dim: 3,
            };

            let query = vec![127, 127, 0];
            let query_norm_sq = 127 * 127 + 127 * 127;

            // Orthogonal vectors
            let dist_0 = vecs.cosine_distance_i8(0, &query, query_norm_sq);
            assert_relative_eq!(dist_0, 1.0 - 1.0 / 2.0_f32.sqrt(), epsilon = 1e-5);

            // Same direction
            let dist_2 = vecs.cosine_distance_i8(2, &query, query_norm_sq);
            assert_relative_eq!(dist_2, 0.0, epsilon = 1e-5);
        }
    }
}
