//! Calibration knobs shared by the uniformly quantised builders.
//!
//! Same rule as [`crate::kmeans`]: `None` everywhere means "let the crate pick
//! its own heuristic", which is not the same as passing the crate's defaults
//! explicitly, so a struct is only built when the caller actually asked for
//! something.

use ann_search_rs::prelude::UniformQuantParams;

/// Assemble [`UniformQuantParams`] from the two knobs worth exposing.
///
/// The seed is not a knob of its own: calibration samples rows, so it shares
/// the estimator's build seed rather than inviting two seeds that have to be
/// kept in step.
///
/// ### Params
///
/// * `drop_ratio` - Fraction trimmed from *each* tail of every dimension before
///   the range is fixed, or `None` for the crate default. Must be in
///   `[0, 0.5)`; the crate validates it and errors rather than clamping.
/// * `sample_rows` - Rows sampled for calibration, or `None` to auto-pick.
/// * `seed` - The estimator's build seed, used for the calibration sample.
///
/// ### Returns
///
/// `None` when neither knob was touched, so the builder takes its own path.
pub(crate) fn quant_params(
    drop_ratio: Option<f64>,
    sample_rows: Option<usize>,
    seed: usize,
) -> Option<UniformQuantParams> {
    if drop_ratio.is_none() && sample_rows.is_none() {
        return None;
    }
    let default = UniformQuantParams::default();
    Some(UniformQuantParams::new(
        drop_ratio.unwrap_or(default.drop_ratio),
        sample_rows,
        seed,
    ))
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_untouched_knobs_defer_to_the_crate() {
        assert!(quant_params(None, None, 42).is_none());
    }

    #[test]
    fn test_sample_rows_alone_keeps_the_default_drop_ratio() {
        let params = quant_params(None, Some(1024), 42).expect("sample_rows forces a struct");
        assert_eq!(params.drop_ratio, UniformQuantParams::default().drop_ratio);
        assert_eq!(params.sample_rows, Some(1024));
    }

    #[test]
    fn test_knobs_are_carried_through() {
        let params = quant_params(Some(0.01), Some(512), 7).expect("both knobs force a struct");
        assert_eq!(params.drop_ratio, 0.01);
        assert_eq!(params.sample_rows, Some(512));
        assert_eq!(params.seed, 7);
    }
}
