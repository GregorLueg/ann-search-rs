//! Saving indices to disk and loading them back in.
//!
//! An index is persisted as a *directory*, not a single file. The bincode
//! payload lands in `index.bin`; anything that cannot live inside that payload
//! (currently only the binary indices' memory-mapped re-ranking store) is
//! written alongside it by the [`IndexIo::save_aux`] hook.
//!
//! `index.bin` carries a small header ahead of the payload so that loading the
//! wrong file, the wrong index type, or the wrong float width fails loudly
//! instead of producing garbage. There is deliberately no checksum and no
//! compression: indices are large, and neither pays for itself here.

use bincode::config::{standard, Configuration};
use serde::{de::DeserializeOwned, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::prelude::*;

/////////////////
// Byte layout //
/////////////////

/// Magic bytes every `index.bin` opens with.
const MAGIC: &[u8; 8] = b"ANNSRS\0\0";

/// Version of the on-disk format. Bump whenever a change stops an older file
/// from decoding correctly.
const FORMAT_VERSION: u32 = 1;

/// Name of the bincode payload inside an index directory.
const INDEX_FILE: &str = "index.bin";

/// The one bincode configuration used for both directions.
///
/// Little-endian, variable-length integers, no decode limit. The varint
/// encoding is what makes the crate's pervasive `Vec<usize>` portable between
/// 32- and 64-bit targets, and it shrinks the id-heavy fields on the way: most
/// ids fit in three bytes rather than eight.
///
/// Nothing in the header describes this choice, so encoding and decoding must
/// read it from here rather than each naming a config of their own. A mismatch
/// would decode into silent garbage.
const CONFIG: Configuration = standard();

/// Write the `index.bin` header.
///
/// Layout, little-endian: 8 magic bytes, `u32` format version, one byte for the
/// float width, one byte for the length of the index-kind tag, then the tag
/// itself.
///
/// ### Params
///
/// * `writer` - Sink to write the header to
/// * `kind` - Index-kind tag, e.g. `"hnsw"`
/// * `float_width` - `size_of::<T>()` for the index's float type
///
/// ### Returns
///
/// `Ok(())`, or the underlying IO error.
fn write_header<W>(writer: &mut W, kind: &str, float_width: usize) -> Result<(), AnnSearchErrors>
where
    W: Write,
{
    writer.write_all(MAGIC)?;
    writer.write_all(&FORMAT_VERSION.to_le_bytes())?;
    writer.write_all(&[float_width as u8, kind.len() as u8])?;
    writer.write_all(kind.as_bytes())?;

    Ok(())
}

/// Read and validate the `index.bin` header.
///
/// Checks the magic bytes, the format version, the index kind and the float
/// width, in that order, so the error names the first thing that is actually
/// wrong.
///
/// ### Params
///
/// * `reader` - Source to read the header from
/// * `path` - Path being read, used only for the error message
/// * `kind` - Index-kind tag the caller expects
/// * `float_width` - `size_of::<T>()` the caller expects
///
/// ### Returns
///
/// `Ok(())` when the header matches, otherwise the specific mismatch.
fn read_header<R>(
    reader: &mut R,
    path: &Path,
    kind: &'static str,
    float_width: usize,
) -> Result<(), AnnSearchErrors>
where
    R: Read,
{
    let not_an_index = || AnnSearchErrors::NotAnIndexFile {
        path: path.display().to_string(),
    };

    let mut magic = [0u8; 8];
    reader.read_exact(&mut magic).map_err(|_| not_an_index())?;
    if &magic != MAGIC {
        return Err(not_an_index());
    }

    let mut version = [0u8; 4];
    reader
        .read_exact(&mut version)
        .map_err(|_| not_an_index())?;
    let version = u32::from_le_bytes(version);
    if version != FORMAT_VERSION {
        return Err(AnnSearchErrors::UnsupportedFormatVersion {
            found: version,
            supported: FORMAT_VERSION,
        });
    }

    let mut tags = [0u8; 2];
    reader.read_exact(&mut tags).map_err(|_| not_an_index())?;

    let mut found_kind = vec![0u8; tags[1] as usize];
    reader
        .read_exact(&mut found_kind)
        .map_err(|_| not_an_index())?;
    let found_kind = String::from_utf8_lossy(&found_kind).into_owned();
    if found_kind != kind {
        return Err(AnnSearchErrors::IndexKindMismatch {
            expected: kind,
            found: found_kind,
        });
    }

    let found_width = tags[0] as usize;
    if found_width != float_width {
        return Err(AnnSearchErrors::FloatWidthMismatch {
            expected: float_width,
            found: found_width,
        });
    }

    Ok(())
}

/////////////
// IndexIo //
/////////////

/// Persist an index to a directory and read it back.
///
/// Implementors supply an element type and a [`IndexIo::KIND`] tag; the two
/// hooks [`IndexIo::save_aux`] and [`IndexIo::load_aux`] default to doing
/// nothing and only the binary indices override them, to carry their on-disk
/// re-ranking store along.
///
/// ### Note
///
/// A saved directory is self-contained and can be moved. It is *not* portable
/// across format versions, and floats are stored at their native width, so an
/// index built over `f32` cannot be loaded as `f64`. Both are rejected by the
/// header rather than silently mis-decoded.
pub trait IndexIo: Serialize + DeserializeOwned + Sized {
    /// Float type the index is built over. Recorded in the header as its width
    /// in bytes, so that an `f32` index cannot be loaded as an `f64` one.
    type Elem: AnnSearchFloat;

    /// Tag written into the header, used to reject a load of the wrong index
    /// type. Must be unique across the crate; `test_index_kinds_are_unique`
    /// guards that.
    const KIND: &'static str;

    /// Write state that does not live inside the bincode payload.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory the index is being saved into
    ///
    /// ### Returns
    ///
    /// `Ok(())`. The default implementation does nothing.
    fn save_aux(&self, dir: &Path) -> Result<(), AnnSearchErrors> {
        let _ = dir;
        Ok(())
    }

    /// Re-attach state that the bincode payload skipped.
    ///
    /// Runs immediately after decoding, before the index is handed back.
    ///
    /// ### Params
    ///
    /// * `dir` - Directory the index is being loaded from
    ///
    /// ### Returns
    ///
    /// `Ok(())`. The default implementation does nothing.
    fn load_aux(&mut self, dir: &Path) -> Result<(), AnnSearchErrors> {
        let _ = dir;
        Ok(())
    }

    /// Save the index into `dir`, creating it if needed.
    ///
    /// ### Params
    ///
    /// * `dir` - Target directory. An index already saved there is overwritten.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or an IO / encoding error.
    fn save_index(&self, dir: impl AsRef<Path>) -> Result<(), AnnSearchErrors> {
        let dir = dir.as_ref();
        std::fs::create_dir_all(dir)?;

        let mut writer = BufWriter::new(File::create(dir.join(INDEX_FILE))?);
        write_header(&mut writer, Self::KIND, size_of::<Self::Elem>())?;
        bincode::serde::encode_into_std_write(self, &mut writer, CONFIG)
            .map_err(|e| AnnSearchErrors::EncodeError(e.to_string()))?;

        // BufWriter flushes on Drop but swallows the error; surface it instead
        writer
            .into_inner()
            .map_err(|e| AnnSearchErrors::IoError(e.into_error()))?;

        self.save_aux(dir)
    }

    /// Load an index previously written by [`IndexIo::save_index`].
    ///
    /// ### Params
    ///
    /// * `dir` - Directory holding `index.bin`
    ///
    /// ### Returns
    ///
    /// The reconstructed index, or the first header mismatch / IO / decoding
    /// error encountered.
    fn load_index(dir: impl AsRef<Path>) -> Result<Self, AnnSearchErrors> {
        let dir = dir.as_ref();
        let path = dir.join(INDEX_FILE);

        let mut reader = BufReader::new(File::open(&path)?);
        read_header(&mut reader, &path, Self::KIND, size_of::<Self::Elem>())?;

        let mut index: Self = bincode::serde::decode_from_std_read(&mut reader, CONFIG)
            .map_err(|e| AnnSearchErrors::DecodeError(e.to_string()))?;
        index.load_aux(dir)?;

        Ok(index)
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use faer::{Mat, MatRef};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tempfile::TempDir;

    use crate::cpu::{
        annoy::*, ball_tree::*, exhaustive::*, hnsw::*, ivf::*, kd_forest::*, kmknn::*, lsh::*,
        nndescent::*, nsg::*, rnn_descent::*, vamana::*,
    };
    use crate::*;

    /// Samples in the test fixtures. Enough for k-means over eight cells
    /// without degenerate empty clusters.
    const N_DATA: usize = 256;

    /// Query rows in the test fixtures.
    const N_QUERY: usize = 16;

    /// Fixture dimensionality. 32 is the minimum product quantisation accepts,
    /// is divisible by every `m` used below, and is a multiple of 8 for
    /// TurboQuant.
    const DIM: usize = 32;

    /// Voronoi cells for every IVF-flavoured fixture.
    const NLIST: usize = 8;

    /// Neighbours requested in every round trip.
    const K: usize = 5;

    /// Deterministic random matrix
    ///
    /// ### Params
    ///
    /// * `n` - Rows
    /// * `dim` - Columns
    /// * `seed` - Seed for reproducibility
    ///
    /// ### Returns
    ///
    /// An `n` x `dim` matrix of uniform values in `[0, 1)`.
    fn random_matrix(n: usize, dim: usize, seed: u64) -> Mat<f32> {
        let mut rng = StdRng::seed_from_u64(seed);

        Mat::from_fn(n, dim, |_, _| rng.random::<f32>())
    }

    /// Unwrap a query result, requiring that distances were returned
    ///
    /// ### Params
    ///
    /// * `res` - The result of a `query_*` call made with `return_dist = true`
    ///
    /// ### Returns
    ///
    /// Tuple of `(indices, distances)`
    fn unwrap_knn(res: KnnOptionResult<f32>) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) {
        let (indices, distances) = res.expect("query failed");

        (indices, distances.expect("distances were requested"))
    }

    /// Assert two query results are identical
    ///
    /// ### Params
    ///
    /// * `before` - Result from the index as built
    /// * `after` - Result from the index after a save/load round trip
    fn assert_same_knn(
        before: &(Vec<Vec<usize>>, Vec<Vec<f32>>),
        after: &(Vec<Vec<usize>>, Vec<Vec<f32>>),
    ) {
        assert_eq!(
            before.0, after.0,
            "neighbour ids changed across a round trip"
        );
        assert_eq!(before.1.len(), after.1.len());

        for (row_b, row_a) in before.1.iter().zip(after.1.iter()) {
            assert_eq!(row_b.len(), row_a.len());
            for (b, a) in row_b.iter().zip(row_a.iter()) {
                assert_relative_eq!(b, a);
            }
        }
    }

    /// Build, query, save, load, query again, and assert nothing moved.
    ///
    /// `$build` and `$query` are coerced to `fn` pointers, so they must not
    /// capture. That is deliberate: it keeps each fixture self-contained.
    macro_rules! round_trip {
        ($name:ident, $ty:ty, $build:expr, $query:expr) => {
            #[test]
            fn $name() {
                let build: fn(MatRef<f32>) -> $ty = $build;
                let query: fn(&$ty, MatRef<f32>) -> (Vec<Vec<usize>>, Vec<Vec<f32>>) = $query;

                let data = random_matrix(N_DATA, DIM, 7);
                let queries = random_matrix(N_QUERY, DIM, 11);

                let index = build(data.as_ref());
                let before = query(&index, queries.as_ref());

                let dir = TempDir::new().unwrap();
                index.save_index(dir.path()).unwrap();

                let loaded = <$ty>::load_index(dir.path()).unwrap();
                let after = query(&loaded, queries.as_ref());

                assert_same_knn(&before, &after);
            }
        };
    }

    // -- cpu --

    round_trip!(
        test_round_trip_exhaustive,
        ExhaustiveIndex<f32>,
        |m| build_exhaustive_index(m, "euclidean"),
        |i, q| unwrap_knn(query_exhaustive_index(q, i, K, true, false))
    );

    round_trip!(
        test_round_trip_kmknn,
        KmknnIndex<f32>,
        |m| build_kmknn_index(m, "euclidean", Some(NLIST), None, 1, false).unwrap(),
        |i, q| unwrap_knn(query_kmknn_index(q, i, K, true, false))
    );

    round_trip!(
        test_round_trip_annoy,
        AnnoyIndex<f32>,
        |m| build_annoy_index(m, "euclidean", 4, 1).unwrap(),
        |i, q| unwrap_knn(query_annoy_index(q, i, K, None, true, false))
    );

    round_trip!(
        test_round_trip_ball_tree,
        BallTreeIndex<f32>,
        |m| build_balltree_index(m, "euclidean", 1).unwrap(),
        |i, q| unwrap_knn(query_balltree_index(q, i, K, None, true, false))
    );

    round_trip!(
        test_round_trip_hnsw,
        HnswIndex<f32>,
        |m| build_hnsw_index(m, 8, 32, "euclidean", 1, false),
        |i, q| unwrap_knn(query_hnsw_index(q, i, K, 32, true, false))
    );

    round_trip!(
        test_round_trip_ivf,
        IvfIndex<f32>,
        |m| build_ivf_index(m, Some(NLIST), None, "euclidean", 1, false).unwrap(),
        |i, q| unwrap_knn(query_ivf_index(q, i, K, Some(NLIST), true, false))
    );

    round_trip!(
        test_round_trip_kd_forest,
        KdTreeIndex<f32>,
        |m| build_kd_tree_index(m, "euclidean", 4, 1),
        |i, q| unwrap_knn(query_kd_tree_index(q, i, K, None, true, false))
    );

    round_trip!(
        test_round_trip_lsh,
        LSHIndex<f32>,
        |m| build_lsh_index(m, "euclidean", 4, 8, 1).unwrap(),
        |i, q| unwrap_knn(query_lsh_index(q, i, K, 2, None, true, false))
    );

    round_trip!(
        test_round_trip_nndescent,
        NNDescent<f32>,
        |m| build_nndescent_index(
            m,
            "euclidean",
            0.001,
            1.0,
            Some(10),
            None,
            None,
            None,
            1,
            false
        )
        .unwrap(),
        |i, q| unwrap_knn(query_nndescent_index(q, i, K, None, true, false))
    );

    round_trip!(
        test_round_trip_vamana,
        VamanaIndex<f32>,
        |m| build_vamana_index(m, 16, 32, 1.2, 1.2, "euclidean", 1),
        |i, q| unwrap_knn(query_vamana_index(q, i, K, None, true, false))
    );

    round_trip!(
        test_round_trip_nsg,
        NsgIndex<f32>,
        |m| build_nsg_index(m, 16, 32, 64, 10, "euclidean", 1, false).unwrap(),
        |i, q| unwrap_knn(query_nsg_index(q, i, K, None, true, false))
    );

    round_trip!(
        test_round_trip_rnn_descent,
        RnnDescentIndex<f32>,
        |m| build_rnn_descent_index(m, 8, 16, 2, 4, "euclidean", None, 1, false).unwrap(),
        |i, q| unwrap_knn(query_rnn_descent_index(q, i, K, None, None, true, false))
    );

    // -- quantised --

    #[cfg(feature = "quantised")]
    mod quantised_round_trips {
        use super::*;

        use crate::quantised::{
            exhaustive_bf16::*, exhaustive_opq::*, exhaustive_pq::*, exhaustive_sq8::*,
            ivf_bf16::*, ivf_opq::*, ivf_pq::*, ivf_sq8::*,
        };

        /// Subspaces for the product-quantised fixtures. `DIM` must be
        /// divisible by it.
        const M: usize = 4;

        /// Sub-codebook size. The default of 256 would exceed `N_DATA`.
        const N_CENTROIDS: usize = 16;

        round_trip!(
            test_round_trip_exhaustive_bf16,
            ExhaustiveIndexBf16<f32>,
            |m| build_exhaustive_bf16_index(m, "euclidean", false).unwrap(),
            |i, q| unwrap_knn(query_exhaustive_bf16_index(q, i, K, true, false))
        );

        round_trip!(
            test_round_trip_ivf_bf16,
            IvfIndexBf16<f32>,
            |m| build_ivf_bf16_index(m, Some(NLIST), None, "euclidean", 1, false).unwrap(),
            |i, q| unwrap_knn(query_ivf_bf16_index(q, i, K, Some(NLIST), true, false))
        );

        round_trip!(
            test_round_trip_exhaustive_sq8,
            ExhaustiveSq8Index<f32>,
            |m| build_exhaustive_sq8_index(m, "euclidean", false).unwrap(),
            |i, q| unwrap_knn(query_exhaustive_sq8_index(q, i, K, true, false))
        );

        round_trip!(
            test_round_trip_ivf_sq8,
            IvfSq8Index<f32>,
            |m| build_ivf_sq8_index(m, Some(NLIST), None, "euclidean", 1, false).unwrap(),
            |i, q| unwrap_knn(query_ivf_sq8_index(q, i, K, Some(NLIST), true, false))
        );

        round_trip!(
            test_round_trip_exhaustive_pq,
            ExhaustivePqIndex<f32>,
            |m| build_exhaustive_pq_index(m, M, None, Some(N_CENTROIDS), "euclidean", 1, false)
                .unwrap(),
            |i, q| unwrap_knn(query_exhaustive_pq_index(q, i, K, true, false))
        );

        round_trip!(
            test_round_trip_ivf_pq,
            IvfPqIndex<f32>,
            |m| build_ivf_pq_index(
                m,
                Some(NLIST),
                M,
                None,
                Some(N_CENTROIDS),
                "euclidean",
                1,
                false
            )
            .unwrap(),
            |i, q| unwrap_knn(query_ivf_pq_index(q, i, K, Some(NLIST), true, false))
        );

        round_trip!(
            test_round_trip_exhaustive_opq,
            ExhaustiveOpqIndex<f32>,
            |m| build_exhaustive_opq_index(m, M, None, Some(N_CENTROIDS), "euclidean", 1, false)
                .unwrap(),
            |i, q| unwrap_knn(query_exhaustive_opq_index(q, i, K, true, false))
        );

        round_trip!(
            test_round_trip_ivf_opq,
            IvfOpqIndex<f32>,
            |m| build_ivf_opq_index(
                m,
                Some(NLIST),
                M,
                None,
                Some(N_CENTROIDS),
                Some(2),
                "euclidean",
                1,
                false
            )
            .unwrap(),
            |i, q| unwrap_knn(query_ivf_opq_index(q, i, K, Some(NLIST), true, false))
        );
    }

    // -- binary --

    #[cfg(feature = "binary")]
    mod binary_round_trips {
        use super::*;

        use std::path::PathBuf;

        use crate::binary::exhaustive_binary::*;
        use crate::binary::exhaustive_rabitq::*;
        use crate::binary::exhaustive_tq::*;
        use crate::binary::ivf_binary::*;
        use crate::binary::ivf_rabitq::*;
        use crate::binary::ivf_tq::*;
        use crate::binary::vec_store::{NORMS_FILE, VECTORS_FILE};

        /// Bits per binary code. Must be a multiple of 8.
        const N_BITS: usize = 64;

        /// Bits per coordinate for the TurboQuant fixtures.
        const TQ_BITS: usize = 4;

        /// Over-fetch factor for the re-ranking fixtures.
        const RERANK_FACTOR: usize = 4;

        /// Stands in for the absent `save_path` when no store is wanted.
        const NO_PATH: Option<PathBuf> = None;

        round_trip!(
            test_round_trip_exhaustive_binary,
            ExhaustiveIndexBinary<f32>,
            |m| build_exhaustive_index_binary(m, N_BITS, 1, "random", "cosine", false, NO_PATH)
                .unwrap(),
            |i, q| unwrap_knn(query_exhaustive_index_binary(
                q, i, K, false, None, true, false
            ))
        );

        round_trip!(
            test_round_trip_ivf_binary,
            IvfIndexBinary<f32>,
            |m| build_ivf_index_binary(
                m,
                "random",
                N_BITS,
                Some(NLIST),
                None,
                "cosine",
                1,
                false,
                NO_PATH,
                false
            )
            .unwrap(),
            |i, q| unwrap_knn(query_ivf_index_binary(
                q,
                i,
                K,
                Some(NLIST),
                false,
                None,
                true,
                false
            ))
        );

        round_trip!(
            test_round_trip_exhaustive_rabitq,
            ExhaustiveIndexRaBitQ<f32>,
            |m| build_exhaustive_index_rabitq(m, Some(NLIST), "euclidean", 1, false, NO_PATH)
                .unwrap(),
            |i, q| unwrap_knn(query_exhaustive_index_rabitq(
                q,
                i,
                K,
                Some(NLIST),
                false,
                None,
                true,
                false
            ))
        );

        round_trip!(
            test_round_trip_ivf_rabitq,
            IvfIndexRaBitQ<f32>,
            |m| build_ivf_index_rabitq(m, Some(NLIST), None, "euclidean", 1, false, NO_PATH, false)
                .unwrap(),
            |i, q| unwrap_knn(query_ivf_index_rabitq(
                q,
                i,
                K,
                Some(NLIST),
                false,
                None,
                true,
                false
            ))
        );

        round_trip!(
            test_round_trip_exhaustive_tq,
            TurboQuantExhaustive<f32>,
            |m| build_exhaustive_index_turboquant(m, "euclidean", TQ_BITS, 1, false, NO_PATH)
                .unwrap(),
            |i, q| unwrap_knn(query_exhaustive_index_turboquant(
                q, i, K, false, None, true, false
            ))
        );

        round_trip!(
            test_round_trip_ivf_tq,
            IvfTurboQuant<f32>,
            |m| build_ivf_index_turboquant(
                m,
                Some(NLIST),
                None,
                "euclidean",
                TQ_BITS,
                1,
                false,
                NO_PATH,
                false
            )
            .unwrap(),
            |i, q| unwrap_knn(query_ivf_index_turboquant(
                q,
                i,
                K,
                Some(NLIST),
                false,
                None,
                true,
                false
            ))
        );

        /// The re-ranking store must survive a save into a *different*
        /// directory, which is what makes a saved index relocatable.
        ///
        /// This also pins the ordering invariant: the store is written in
        /// original data order and every consumer maps back through
        /// `original_ids` before touching it. `save_aux` copies the files
        /// verbatim; rewriting them in the index's internal order would break
        /// re-ranking silently.
        #[test]
        fn test_ivf_binary_rerank_survives_relocation() {
            let data = random_matrix(N_DATA, DIM, 7);
            let queries = random_matrix(N_QUERY, DIM, 11);

            let store_dir = TempDir::new().unwrap();
            let index_dir = TempDir::new().unwrap();

            let index = build_ivf_index_binary(
                data.as_ref(),
                "random",
                N_BITS,
                Some(NLIST),
                None,
                "cosine",
                1,
                true,
                Some(store_dir.path()),
                false,
            )
            .unwrap();

            let before = unwrap_knn(query_ivf_index_binary(
                queries.as_ref(),
                &index,
                K,
                Some(NLIST),
                true,
                Some(RERANK_FACTOR),
                true,
                false,
            ));

            index.save_index(index_dir.path()).unwrap();
            assert!(index_dir.path().join(VECTORS_FILE).exists());
            assert!(index_dir.path().join(NORMS_FILE).exists());

            // Drop the originals: the saved directory must stand on its own
            store_dir.close().unwrap();

            let loaded = IvfIndexBinary::<f32>::load_index(index_dir.path()).unwrap();
            let after = unwrap_knn(query_ivf_index_binary(
                queries.as_ref(),
                &loaded,
                K,
                Some(NLIST),
                true,
                Some(RERANK_FACTOR),
                true,
                false,
            ));

            assert_same_knn(&before, &after);
        }

        /// Sign-based binarisation is the only path that reads `old_to_new`,
        /// which `optimise_memory_layout` fills in after the store is written.
        /// Nothing else exercises that field across a round trip.
        ///
        #[test]
        fn test_ivf_binary_asymmetric_query_survives_round_trip() {
            let data = random_matrix(N_DATA, DIM, 7);
            let queries = random_matrix(N_QUERY, DIM, 11);

            let index = build_ivf_index_binary(
                data.as_ref(),
                "sign",
                N_BITS,
                Some(NLIST),
                None,
                "cosine",
                1,
                false,
                NO_PATH,
                false,
            )
            .unwrap();

            let before = unwrap_knn(query_ivf_index_binary(
                queries.as_ref(),
                &index,
                K,
                Some(NLIST),
                false,
                None,
                true,
                false,
            ));

            let dir = TempDir::new().unwrap();
            index.save_index(dir.path()).unwrap();

            let loaded = IvfIndexBinary::<f32>::load_index(dir.path()).unwrap();
            let after = unwrap_knn(query_ivf_index_binary(
                queries.as_ref(),
                &loaded,
                K,
                Some(NLIST),
                false,
                None,
                true,
                false,
            ));

            assert_same_knn(&before, &after);
        }

        /// Saving into the directory the store already occupies must not
        /// rewrite the store files.
        ///
        /// This is not an IO optimisation. `MmapVectorStore` holds a live
        /// mapping of those files, and truncating a mapped file is a `SIGBUS`
        /// rather than an `Err`. The inode check is the assertion; equal
        /// contents would pass even if the file had been replaced.
        #[cfg(unix)]
        #[test]
        fn test_save_into_store_directory_leaves_files_alone() {
            use std::os::unix::fs::MetadataExt;

            let data = random_matrix(N_DATA, DIM, 7);
            let dir = TempDir::new().unwrap();

            let index = build_exhaustive_index_turboquant(
                data.as_ref(),
                "euclidean",
                TQ_BITS,
                1,
                true,
                Some(dir.path()),
            )
            .unwrap();

            let inode = |name: &str| std::fs::metadata(dir.path().join(name)).unwrap().ino();
            let (vectors_before, norms_before) = (inode(VECTORS_FILE), inode(NORMS_FILE));

            index.save_index(dir.path()).unwrap();

            assert_eq!(vectors_before, inode(VECTORS_FILE));
            assert_eq!(norms_before, inode(NORMS_FILE));

            // No temporary left behind either
            assert!(!dir.path().join("vectors_flat.bin.tmp").exists());
        }
    }

    // -- header --

    /// Take the error out of a failed load.
    ///
    /// `Result::unwrap_err` needs the `Ok` type to be `Debug`, and no index in
    /// the crate derives it.
    ///
    /// ### Params
    ///
    /// * `res` - Result of a load that is expected to have failed
    ///
    /// ### Returns
    ///
    /// The error. Panics if the load succeeded.
    fn expect_err<T>(res: Result<T, AnnSearchErrors>) -> AnnSearchErrors {
        match res {
            Ok(_) => panic!("load succeeded but should have been rejected"),
            Err(e) => e,
        }
    }

    /// Save a small exhaustive index and hand back the directory holding it.
    ///
    /// ### Returns
    ///
    /// The temporary directory. Kept alive by the caller.
    fn saved_exhaustive() -> TempDir {
        let data = random_matrix(64, DIM, 7);
        let index = build_exhaustive_index(data.as_ref(), "euclidean");

        let dir = TempDir::new().unwrap();
        index.save_index(dir.path()).unwrap();

        dir
    }

    #[test]
    fn test_rejects_a_file_that_is_not_an_index() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(INDEX_FILE), b"definitely not an index").unwrap();

        let err = expect_err(ExhaustiveIndex::<f32>::load_index(dir.path()));
        assert!(matches!(err, AnnSearchErrors::NotAnIndexFile { .. }));
    }

    #[test]
    fn test_rejects_a_truncated_file() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join(INDEX_FILE), &MAGIC[..4]).unwrap();

        let err = expect_err(ExhaustiveIndex::<f32>::load_index(dir.path()));
        assert!(matches!(err, AnnSearchErrors::NotAnIndexFile { .. }));
    }

    #[test]
    fn test_rejects_an_unknown_format_version() {
        let dir = saved_exhaustive();
        let path = dir.path().join(INDEX_FILE);

        let mut bytes = std::fs::read(&path).unwrap();
        bytes[8..12].copy_from_slice(&(FORMAT_VERSION + 1).to_le_bytes());
        std::fs::write(&path, &bytes).unwrap();

        let err = expect_err(ExhaustiveIndex::<f32>::load_index(dir.path()));
        assert!(matches!(
            err,
            AnnSearchErrors::UnsupportedFormatVersion {
                found,
                supported: FORMAT_VERSION,
            } if found == FORMAT_VERSION + 1
        ));
    }

    #[test]
    fn test_rejects_the_wrong_index_kind() {
        let dir = saved_exhaustive();

        let err = expect_err(IvfIndex::<f32>::load_index(dir.path()));
        assert!(matches!(
            err,
            AnnSearchErrors::IndexKindMismatch { expected: "ivf", ref found } if found == "exhaustive"
        ));
    }

    #[test]
    fn test_rejects_the_wrong_float_width() {
        let dir = saved_exhaustive();

        let err = expect_err(ExhaustiveIndex::<f64>::load_index(dir.path()));
        assert!(matches!(
            err,
            AnnSearchErrors::FloatWidthMismatch {
                expected: 8,
                found: 4
            }
        ));
    }

    #[test]
    fn test_missing_directory_is_an_io_error() {
        let err = expect_err(ExhaustiveIndex::<f32>::load_index(
            "/nonexistent/ann-search-rs",
        ));
        assert!(matches!(err, AnnSearchErrors::IoError(_)));
    }

    #[test]
    fn test_f64_indices_round_trip() {
        let mut rng = StdRng::seed_from_u64(7);
        let data = Mat::<f64>::from_fn(64, DIM, |_, _| rng.random::<f64>());

        let index = build_exhaustive_index(data.as_ref(), "cosine");
        let (before, _) = query_exhaustive_self(&index, K, false, false).unwrap();

        let dir = TempDir::new().unwrap();
        index.save_index(dir.path()).unwrap();

        let loaded = ExhaustiveIndex::<f64>::load_index(dir.path()).unwrap();
        let (after, _) = query_exhaustive_self(&loaded, K, false, false).unwrap();

        assert_eq!(before, after);
    }

    /// The free functions in the crate root are what the README documents, so
    /// they need to infer without a turbofish on the value being saved.
    #[test]
    fn test_free_function_wrappers_round_trip() {
        let data = random_matrix(64, DIM, 7);
        let index = build_exhaustive_index(data.as_ref(), "euclidean");

        let dir = TempDir::new().unwrap();
        crate::save_index(&index, dir.path()).unwrap();

        let loaded: ExhaustiveIndex<f32> = crate::load_index(dir.path()).unwrap();

        let (before, _) = query_exhaustive_self(&index, K, false, false).unwrap();
        let (after, _) = query_exhaustive_self(&loaded, K, false, false).unwrap();

        assert_eq!(before, after);
    }

    /// Two indices sharing a `KIND` would cross-load straight past the header
    /// check, so the tags have to stay distinct.
    #[test]
    fn test_index_kinds_are_unique() {
        let mut kinds = vec![
            <ExhaustiveIndex<f32> as IndexIo>::KIND,
            <KmknnIndex<f32> as IndexIo>::KIND,
            <AnnoyIndex<f32> as IndexIo>::KIND,
            <BallTreeIndex<f32> as IndexIo>::KIND,
            <HnswIndex<f32> as IndexIo>::KIND,
            <IvfIndex<f32> as IndexIo>::KIND,
            <KdTreeIndex<f32> as IndexIo>::KIND,
            <LSHIndex<f32> as IndexIo>::KIND,
            <NNDescent<f32> as IndexIo>::KIND,
            <NsgIndex<f32> as IndexIo>::KIND,
            <RnnDescentIndex<f32> as IndexIo>::KIND,
            <VamanaIndex<f32> as IndexIo>::KIND,
        ];

        #[cfg(feature = "quantised")]
        {
            use crate::quantised::{
                exhaustive_bf16::*, exhaustive_opq::*, exhaustive_pq::*, exhaustive_sq8::*,
                ivf_bf16::*, ivf_opq::*, ivf_pq::*, ivf_sq8::*,
            };

            kinds.extend([
                <ExhaustiveIndexBf16<f32> as IndexIo>::KIND,
                <IvfIndexBf16<f32> as IndexIo>::KIND,
                <ExhaustiveSq8Index<f32> as IndexIo>::KIND,
                <IvfSq8Index<f32> as IndexIo>::KIND,
                <ExhaustivePqIndex<f32> as IndexIo>::KIND,
                <IvfPqIndex<f32> as IndexIo>::KIND,
                <ExhaustiveOpqIndex<f32> as IndexIo>::KIND,
                <IvfOpqIndex<f32> as IndexIo>::KIND,
            ]);
        }

        #[cfg(feature = "binary")]
        {
            use crate::binary::{
                exhaustive_binary::*, exhaustive_rabitq::*, exhaustive_tq::*, ivf_binary::*,
                ivf_rabitq::*, ivf_tq::*,
            };

            kinds.extend([
                <ExhaustiveIndexBinary<f32> as IndexIo>::KIND,
                <IvfIndexBinary<f32> as IndexIo>::KIND,
                <ExhaustiveIndexRaBitQ<f32> as IndexIo>::KIND,
                <IvfIndexRaBitQ<f32> as IndexIo>::KIND,
                <TurboQuantExhaustive<f32> as IndexIo>::KIND,
                <IvfTurboQuant<f32> as IndexIo>::KIND,
            ]);
        }

        let total = kinds.len();
        kinds.sort_unstable();
        kinds.dedup();

        assert_eq!(total, kinds.len(), "duplicate KIND tag across indices");
    }
}
