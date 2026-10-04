//! Distance codec for the quantised graph index.
//!
//! The graph walk only ever asks two questions: how far is the query from node
//! `i`, and how far is node `i` from node `j`. [`GraphCodec`] is those two
//! questions plus the per-query setup, which is all a graph index needs to know
//! about how vectors are stored.
//!
//! Scores are *monotone*, not reportable. Ranking on the raw integer distance
//! keeps the scale multiply out of the inner loop; [`GraphCodec::finalise`]
//! converts to a real distance once per returned neighbour.

use crate::prelude::*;

////////////////
// GraphCodec //
////////////////

/// How the graph index stores vectors and measures distance between them.
///
/// A single trait rather than a separate symmetric one: the uniform quantiser
/// encodes queries into the same code space as the database, so one kernel
/// answers both. A codec without that property (product quantisation, RaBitQ)
/// would need the two split apart.
pub trait GraphCodec<T>: Sync
where
    T: AnnSearchFloat,
{
    /// Per-query state, built once and reused for every node visited.
    type Query: Send + Sync;

    /// Number of stored vectors.
    ///
    /// ### Returns
    ///
    /// Vector count
    fn n(&self) -> usize;

    /// Dimensionality of the stored vectors.
    ///
    /// ### Returns
    ///
    /// Number of features
    fn dim(&self) -> usize;

    /// The distance metric this codec was built for.
    ///
    /// ### Returns
    ///
    /// The metric, see [`Dist`]
    fn metric(&self) -> Dist;

    /// Prepare a query for repeated scoring.
    ///
    /// ### Params
    ///
    /// * `query` - Query vector of length `dim`
    ///
    /// ### Returns
    ///
    /// The per-query state, or an error on a dimension mismatch
    fn encode_query(&self, query: &[T]) -> Result<Self::Query, AnnSearchErrors>;

    /// Score between a prepared query and a stored vector. Smaller is nearer.
    ///
    /// ### Params
    ///
    /// * `query` - Prepared query state
    /// * `id` - Index of the stored vector
    ///
    /// ### Returns
    ///
    /// A value monotone in the true distance
    fn score(&self, query: &Self::Query, id: usize) -> T;

    /// Cheap first-tier score and a lower bound on [`Self::score`].
    ///
    /// The walk steers its beam and the upper-layer descent by the first
    /// value and pays for the full score only where the bound could still
    /// enter the result set. A codec with one tier returns its full score
    /// twice, which is the default.
    ///
    /// ### Params
    ///
    /// * `query` - Prepared query state
    /// * `id` - Index of the stored vector
    ///
    /// ### Returns
    ///
    /// `(estimate, lower bound)` on the same scale as [`Self::score`]
    #[inline]
    fn score_coarse(&self, query: &Self::Query, id: usize) -> (T, T) {
        let d = self.score(query, id);
        (d, d)
    }

    /// Whether [`Self::score_coarse`] is cheaper than [`Self::score`], so the
    /// walk should gate the full score behind the bound.
    ///
    /// ### Returns
    ///
    /// `false` unless the codec has a distinct coarse tier
    fn two_tier(&self) -> bool {
        false
    }

    /// Score between two stored vectors. Smaller is nearer.
    ///
    /// This is the construction-time distance, so the graph is built in the
    /// same space it is searched in.
    ///
    /// ### Params
    ///
    /// * `a` - Index of the first stored vector
    /// * `b` - Index of the second stored vector
    ///
    /// ### Returns
    ///
    /// A value monotone in the true distance
    fn score_sym(&self, a: usize, b: usize) -> T;

    /// Convert a score into a reportable distance.
    ///
    /// ### Params
    ///
    /// * `score` - A value returned by [`Self::score`] or [`Self::score_sym`]
    ///
    /// ### Returns
    ///
    /// The approximate distance under this codec's metric
    fn finalise(&self, score: T) -> T;
}
