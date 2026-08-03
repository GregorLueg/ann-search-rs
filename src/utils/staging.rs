//! Write-then-rename staging for files that are replaced in place.
//!
//! Two reasons this exists. A file that another process (or another index in
//! this one) has memory-mapped must never be truncated: `File::create` on a
//! mapped path is a `SIGBUS` waiting to happen, whereas a rename leaves the old
//! inode alive for whoever still holds it. And a multi-file bundle wants every
//! file to appear at once, so the writes go to temporary names first and the
//! renames happen together at the end.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::prelude::*;

/// Counter that makes temporary filenames unique within a process, so two
/// concurrent saves into one directory cannot race on the same path.
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Temporary path sitting next to `dst`
///
/// Same directory as the destination, so the later rename stays within one
/// filesystem. The name carries the process id and a counter rather than
/// replacing the extension, which keeps it unique and keeps `dst`'s own name
/// visible in a directory listing.
///
/// ### Params
///
/// * `dst` - Final path the staged file will be renamed to
///
/// ### Returns
///
/// The temporary path.
fn tmp_path(dst: &Path) -> PathBuf {
    let mut name = dst.file_name().unwrap_or_default().to_os_string();
    name.push(format!(
        ".{}.{}.tmp",
        std::process::id(),
        TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ));

    dst.with_file_name(name)
}

/// A set of files written under temporary names, waiting to be published.
///
/// [`StagedFiles::commit`] renames them in the order they were staged; anything
/// still staged when the value is dropped is deleted, so a failure part-way
/// through the writing phase leaves no litter behind.
pub struct StagedFiles {
    /// Pairs of `(temporary path, final path)`, in staging order
    pending: Vec<(PathBuf, PathBuf)>,
}

impl StagedFiles {
    /// Empty staging set
    ///
    /// ### Returns
    ///
    /// A set with nothing staged.
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Reserve a temporary path for `dst`
    ///
    /// The caller writes to the returned path. Nothing is created here, so a
    /// caller that fails before writing is still safe to drop.
    ///
    /// ### Params
    ///
    /// * `dst` - Final path the file should end up at
    ///
    /// ### Returns
    ///
    /// The temporary path to write to.
    pub fn stage(&mut self, dst: &Path) -> PathBuf {
        let tmp = tmp_path(dst);
        self.pending.push((tmp.clone(), dst.to_path_buf()));

        tmp
    }

    /// Whether anything is staged
    ///
    /// ### Returns
    ///
    /// `true` when no file is waiting to be published.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Rename every staged file onto its destination
    ///
    /// Renames run in staging order, so the caller controls which file becomes
    /// visible last. A rename that fails leaves the remaining temporaries to be
    /// cleaned up by `Drop`.
    ///
    /// ### Returns
    ///
    /// `Ok(())`, or the first rename error.
    pub fn commit(mut self) -> Result<(), AnnSearchErrors> {
        while !self.pending.is_empty() {
            let (tmp, dst) = self.pending.remove(0);
            std::fs::rename(&tmp, &dst)?;
        }

        Ok(())
    }
}

impl Default for StagedFiles {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for StagedFiles {
    fn drop(&mut self) {
        for (tmp, _) in &self.pending {
            let _ = std::fs::remove_file(tmp);
        }
    }
}

///////////
// Tests //
///////////

#[cfg(test)]
mod tests {
    use super::*;

    use tempfile::TempDir;

    #[test]
    fn test_commit_publishes_in_staging_order() {
        let dir = TempDir::new().unwrap();
        let (first, second) = (dir.path().join("a"), dir.path().join("b"));

        let mut staged = StagedFiles::new();
        let tmp_a = staged.stage(&first);
        let tmp_b = staged.stage(&second);
        std::fs::write(&tmp_a, b"a").unwrap();
        std::fs::write(&tmp_b, b"b").unwrap();

        staged.commit().unwrap();

        assert_eq!(std::fs::read(&first).unwrap(), b"a");
        assert_eq!(std::fs::read(&second).unwrap(), b"b");
        assert!(!tmp_a.exists());
        assert!(!tmp_b.exists());
    }

    #[test]
    fn test_drop_removes_uncommitted_temporaries() {
        let dir = TempDir::new().unwrap();
        let dst = dir.path().join("a");

        let tmp = {
            let mut staged = StagedFiles::new();
            let tmp = staged.stage(&dst);
            std::fs::write(&tmp, b"a").unwrap();

            tmp
        };

        assert!(!tmp.exists());
        assert!(!dst.exists());
    }

    #[test]
    fn test_temporary_names_do_not_collide() {
        let dir = TempDir::new().unwrap();
        let dst = dir.path().join("vectors_flat.bin");

        let mut staged = StagedFiles::new();
        let first = staged.stage(&dst);
        let second = staged.stage(&dst);

        assert_ne!(first, second);
        assert_eq!(first.parent(), Some(dir.path()));
    }
}
