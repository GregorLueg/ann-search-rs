//! Shared dense GEMM entry point. Drop-in for `faer::linalg::matmul::matmul`
//! that routes through Apple Accelerate's cblas under the `accelerate`
//! feature on macOS, and through faer everywhere else.

use faer::linalg::matmul::matmul as faer_matmul;
use faer::{Accum, MatMut, MatRef, Par};
use faer_traits::ComplexField;

#[cfg(all(feature = "accelerate", target_os = "macos"))]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn cblas_dgemm(
        order: i32,
        trans_a: i32,
        trans_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

/// Map a strided view onto a column-major BLAS operand
///
/// ### Params
///
/// * `rs` - Row stride in elements
/// * `cs` - Column stride in elements
/// * `r` - Number of rows
/// * `c` - Number of columns
///
/// ### Returns
///
/// `Some((transposed, ld))`, where `transposed` means the view is row-major
/// (a column-major operand needing the Trans flag) and `ld` is the leading
/// dimension; `None` if neither axis has unit stride with the other stride
/// covering the extent. A stride along an extent-1 axis is never read, so
/// it is ignored.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn blas_layout(rs: isize, cs: isize, r: usize, c: usize) -> Option<(bool, i32)> {
    let (r, c) = (r.max(1) as isize, c.max(1) as isize);
    let ld = if r == 1 && c == 1 {
        return Some((false, 1));
    } else if rs == 1 && cs >= r {
        (false, cs)
    } else if cs == 1 && rs >= c {
        (true, rs)
    } else if r == 1 && cs >= 1 {
        (false, cs)
    } else if c == 1 && rs >= 1 {
        (true, rs)
    } else {
        return None;
    };
    i32::try_from(ld.1).ok().map(|v| (ld.0, v))
}

/// Try the cblas path for a destination that is column-major
///
/// ### Params
///
/// * `dst` - Column-major destination
/// * `accum` - Replace or accumulate
/// * `lhs` - Left operand
/// * `rhs` - Right operand
/// * `alpha` - Scale of the product
///
/// ### Returns
///
/// Whether cblas ran; `false` leaves `dst` untouched.
#[cfg(all(feature = "accelerate", target_os = "macos"))]
fn try_cblas<T: ComplexField>(
    dst: &mut MatMut<T>,
    accum: Accum,
    lhs: MatRef<T>,
    rhs: MatRef<T>,
    alpha: T,
) -> bool {
    const COL_MAJOR: i32 = 102;
    const NO_TRANS: i32 = 111;
    const TRANS: i32 = 112;

    let (m, n, k) = (dst.nrows(), dst.ncols(), lhs.ncols());
    let (Some((false, ldc)), Some((ta, lda)), Some((tb, ldb))) = (
        blas_layout(dst.row_stride(), dst.col_stride(), m, n),
        blas_layout(lhs.row_stride(), lhs.col_stride(), m, k),
        blas_layout(rhs.row_stride(), rhs.col_stride(), k, n),
    ) else {
        return false;
    };
    let (Ok(mi), Ok(ni), Ok(ki)) = (i32::try_from(m), i32::try_from(n), i32::try_from(k)) else {
        return false;
    };
    let ta = if ta { TRANS } else { NO_TRANS };
    let tb = if tb { TRANS } else { NO_TRANS };
    let beta = matches!(accum, Accum::Add) as u8 as f64;

    // The real ComplexField types of 4 and 8 bytes are f32 and f64.
    if !T::IS_REAL {
        false
    } else if std::mem::size_of::<T>() == 4 {
        // SAFETY: T is f32, the layouts were validated above and the
        // pointers come from live views of the stated shapes.
        unsafe {
            let alpha: f32 = std::mem::transmute_copy(&alpha);
            cblas_sgemm(
                COL_MAJOR,
                ta,
                tb,
                mi,
                ni,
                ki,
                alpha,
                lhs.as_ptr() as *const f32,
                lda,
                rhs.as_ptr() as *const f32,
                ldb,
                beta as f32,
                dst.as_ptr_mut() as *mut f32,
                ldc,
            );
        }
        true
    } else if std::mem::size_of::<T>() == 8 {
        // SAFETY: as above, for f64.
        unsafe {
            let alpha: f64 = std::mem::transmute_copy(&alpha);
            cblas_dgemm(
                COL_MAJOR,
                ta,
                tb,
                mi,
                ni,
                ki,
                alpha,
                lhs.as_ptr() as *const f64,
                lda,
                rhs.as_ptr() as *const f64,
                ldb,
                beta,
                dst.as_ptr_mut() as *mut f64,
                ldc,
            );
        }
        true
    } else {
        false
    }
}

/// Dense GEMM: `dst = alpha * lhs * rhs`, or `dst += ...` for `Accum::Add`
///
/// Same contract as faer's `matmul`. Under the `accelerate` feature on macOS
/// f32 and f64 products whose operands are plain strided buffers go to
/// Accelerate's cblas, which ignores `par` and runs on its own threads;
/// anything else (other types, exotic strides, empty dimensions) goes to
/// faer with the given `par`.
///
/// ### Params
///
/// * `dst` - Destination, `m x n`
/// * `accum` - Replace or accumulate into `dst`
/// * `lhs` - Left operand, `m x k`
/// * `rhs` - Right operand, `k x n`
/// * `alpha` - Scale of the product
/// * `par` - Parallelism for the faer fallback
#[inline]
pub(crate) fn gemm<T: ComplexField>(
    #[allow(unused_mut)] mut dst: MatMut<T>,
    accum: Accum,
    lhs: MatRef<T>,
    rhs: MatRef<T>,
    alpha: T,
    par: Par,
) {
    #[cfg(all(feature = "accelerate", target_os = "macos"))]
    if dst.nrows() > 0 && dst.ncols() > 0 && lhs.ncols() > 0 {
        // A row-major destination is a column-major one of the transposed
        // product.
        if try_cblas(&mut dst, accum, lhs, rhs, alpha.clone())
            || try_cblas(
                &mut dst.as_mut().transpose_mut(),
                accum,
                rhs.transpose(),
                lhs.transpose(),
                alpha.clone(),
            )
        {
            return;
        }
    }

    faer_matmul(dst, accum, lhs, rhs, alpha, par);
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn fill<T: ComplexField + From<f32>>(r: usize, c: usize, seed: f32) -> Mat<T> {
        Mat::from_fn(r, c, |i, j| {
            T::from(((i * 31 + j * 17) as f32 * 0.37 + seed).sin())
        })
    }

    fn check<T>(tol: f64)
    where
        T: ComplexField + From<f32> + Into<f64>,
    {
        let (m, n, k) = (7, 5, 9);
        let alpha = T::from(0.5);
        // Every operand kind: col-major, row-major (transposed), subcols
        let a_cm = fill::<T>(m, k, 0.1);
        let a_rm = fill::<T>(k, m, 0.1);
        let a_wide = fill::<T>(m, k + 4, 0.1);
        let b_cm = fill::<T>(k, n, 0.2);
        let b_rm = fill::<T>(n, k, 0.2);
        let b_wide = fill::<T>(k, n + 3, 0.2);
        let lhs: Vec<MatRef<T>> = vec![
            a_cm.as_ref(),
            a_rm.as_ref().transpose(),
            a_wide.as_ref().subcols(2, k),
        ];
        let rhs: Vec<MatRef<T>> = vec![
            b_cm.as_ref(),
            b_rm.as_ref().transpose(),
            b_wide.as_ref().subcols(1, n),
        ];

        for l in &lhs {
            for r in &rhs {
                for accum in [Accum::Replace, Accum::Add] {
                    let init = fill::<T>(m, n, 0.7);
                    let mut want = init.clone();
                    faer_matmul(want.as_mut(), accum, *l, *r, alpha.clone(), Par::Seq);

                    // dst: col-major, row-major, subcols
                    let mut d_cm = init.clone();
                    gemm(d_cm.as_mut(), accum, *l, *r, alpha.clone(), Par::Seq);

                    let init_t = init.transpose().to_owned();
                    let mut d_rm = init_t.clone();
                    gemm(
                        d_rm.as_mut().transpose_mut(),
                        accum,
                        *l,
                        *r,
                        alpha.clone(),
                        Par::Seq,
                    );

                    let mut wide = fill::<T>(m, n + 3, 0.9);
                    wide.as_mut().subcols_mut(2, n).copy_from(init.as_ref());
                    let before = wide.clone();
                    gemm(
                        wide.as_mut().subcols_mut(2, n),
                        accum,
                        *l,
                        *r,
                        alpha.clone(),
                        Par::Seq,
                    );

                    for i in 0..m {
                        for j in 0..n {
                            let w: f64 = want[(i, j)].clone().into();
                            for got in [&d_cm[(i, j)], &d_rm[(j, i)], &wide[(i, j + 2)]] {
                                let g: f64 = got.clone().into();
                                assert!((g - w).abs() < tol, "{g} vs {w}");
                            }
                        }
                    }
                    // The columns outside the subcols view stay untouched
                    for i in 0..m {
                        for j in [0, 1, n + 2] {
                            assert_eq!(wide[(i, j)], before[(i, j)]);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn gemm_matches_faer() {
        check::<f32>(1e-4);
        check::<f64>(1e-12);
    }
}
