use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use fs2::FileExt;
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::fs;
use tokio::io::AsyncWriteExt;
use tokio::sync::{Mutex, MutexGuard};

use crate::domain::{AbstractNode, DistillationState, RawNode};
use crate::error::{EngineError, Result};
use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};
use crate::model::embedding::Embedding;
use crate::storage::RawLifecyclePatch;

use super::graph::{ObjectGraphRepository, StoredGraphEdges};

const STORE_FORMAT_VERSION: u32 = 1;
/// Bumped whenever the on-disk index layout written by
/// `rebuild_indexes_unlocked` changes. `open()` compares this with the value
/// stored in `<root>/indexes/.index-version` and skips the (expensive) full
/// rebuild whenever they match.
///
/// v2: the global timeline moved from a single monolithic
/// `indexes/timeline/raw.json` (rewritten in full on every insert — O(N) per
/// write, O(N^2) over the store's life) to per-day shards under
/// `indexes/timeline/raw/<YYYYMMDD>.json`, so an insert only rewrites the
/// bounded shard for the node's day.
///
/// v3: the embedding manifests moved from a single global
/// `indexes/vector/{raw,abstract}_embeddings.json` (scanned in full on every
/// activation — every embedding ever written for the installation) to
/// per-session shards under `indexes/vector/{raw,abstract}/<session|none>.json`,
/// so a session-scoped search only reads its own session's embeddings.
const INDEX_VERSION: u32 = 5;
pub(super) const MAX_SESSION_INDEX_ENTRIES: usize = 4_096;
pub(super) const MAX_LOOP_INDEX_ENTRIES: usize = 4_097;
pub(super) const MAX_VECTOR_SHARD_ENTRIES: usize = 4_096;

#[derive(Debug, Clone)]
pub struct FileObjectStore {
    root: PathBuf,
    gate: Arc<Mutex<()>>,
}

pub(super) struct StoreGuard<'a> {
    _process_guard: MutexGuard<'a, ()>,
    lock_file: File,
}

impl Drop for StoreGuard<'_> {
    fn drop(&mut self) {
        let _ = FileExt::unlock(&self.lock_file);
    }
}

impl FileObjectStore {
    /// # Errors
    ///
    /// Returns an [`EngineError::Storage`] when the on-disk layout cannot be
    /// created or the index rebuild fails on a corrupt store.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        // Bridge async filesystem work to a sync constructor so existing
        // callers that build the store outside of a tokio task keep working.
        // We block in place on a dedicated future; the work itself is the same
        // tokio::fs path used by all other helpers.
        let root = root.as_ref().to_path_buf();
        tokio_block_on(async move { Self::open_async(root).await })
    }

    /// # Errors
    ///
    /// Returns an [`EngineError::Storage`] when the on-disk layout cannot be
    /// created or the index rebuild fails on a corrupt store.
    pub async fn open_async(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        // The agent opens a brand-new `FileObjectStore` per run on a directory
        // shared by every run for the same (space, installation). A per-instance
        // gate would NOT serialize two such runs, so their interleaved
        // read-modify-write of the shared store.json / index files would race
        // (rename ENOENT, lost index updates, rebuild wiping another run's
        // writes). Share ONE gate per canonical root across all instances so all
        // runs for a directory mutually exclude. [S1]
        let gate = shared_gate_for_root(&root).await;
        let store = Self { root, gate };
        store.validate_root_boundary().await?;
        store.ensure_layout().await?;
        store.ensure_indexes().await?;
        Ok(store)
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    pub(super) async fn lock(&self) -> Result<StoreGuard<'_>> {
        let process_guard = self.gate.lock().await;
        let lock_path = self.root.join(".store.lock");
        let lock_file = tokio::task::spawn_blocking(move || -> std::io::Result<File> {
            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .truncate(false)
                .open(&lock_path)?;
            file.lock_exclusive()?;
            Ok(file)
        })
        .await
        .map_err(|err| EngineError::Storage(format!("store lock task failed: {err}")))?
        .map_err(|err| EngineError::Storage(format!("failed to lock object store: {err}")))?;
        Ok(StoreGuard {
            _process_guard: process_guard,
            lock_file,
        })
    }

    async fn validate_root_boundary(&self) -> Result<()> {
        let metadata = fs::symlink_metadata(&self.root).await.map_err(|err| {
            EngineError::Storage(format!(
                "failed to inspect object store root {}: {err}",
                self.root.display()
            ))
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(EngineError::Storage(format!(
                "object store root must be a real directory, not a symlink: {}",
                self.root.display()
            )));
        }
        Ok(())
    }

    async fn ensure_layout(&self) -> Result<()> {
        for directory in [
            self.raw_dir(),
            self.abstract_dir(),
            self.raw_embedding_dir(),
            self.abstract_embedding_dir(),
            self.graph_dir(),
            self.checkpoint_dir(),
            self.journal_dir(),
            self.quarantine_dir(),
            self.distillation_claim_dir(),
            self.raw_operation_dir(),
            self.abstract_operation_dir(),
        ] {
            self.ensure_real_directory(&directory).await?;
        }
        self.ensure_index_layout().await
    }

    async fn ensure_index_layout(&self) -> Result<()> {
        for directory in [
            self.session_index_dir(),
            self.loop_index_dir(),
            self.timeline_index_dir(),
            self.backlog_index_dir(),
            self.vector_index_dir(),
        ] {
            self.ensure_real_directory(&directory).await?;
        }
        Ok(())
    }

    async fn ensure_real_directory(&self, directory: &Path) -> Result<()> {
        self.validate_managed_path(directory).await?;
        fs::create_dir_all(directory).await.map_err(|err| {
            EngineError::Storage(format!(
                "failed to create object store directory {}: {err}",
                directory.display()
            ))
        })?;
        let metadata = fs::symlink_metadata(directory).await.map_err(|err| {
            EngineError::Storage(format!(
                "failed to inspect object store directory {}: {err}",
                directory.display()
            ))
        })?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            return Err(EngineError::Storage(format!(
                "managed object store path is not a real directory: {}",
                directory.display()
            )));
        }
        Ok(())
    }

    async fn validate_managed_path(&self, path: &Path) -> Result<()> {
        let relative = path.strip_prefix(&self.root).map_err(|_| {
            EngineError::Storage(format!(
                "object store path escapes root: {}",
                path.display()
            ))
        })?;
        let mut current = self.root.clone();
        for component in relative.components() {
            current.push(component);
            match fs::symlink_metadata(&current).await {
                Ok(metadata) if metadata.file_type().is_symlink() => {
                    return Err(EngineError::Storage(format!(
                        "object store refuses symlink path component: {}",
                        current.display()
                    )));
                }
                Ok(_) => {}
                Err(err) if err.kind() == std::io::ErrorKind::NotFound => break,
                Err(err) => {
                    return Err(EngineError::Storage(format!(
                        "failed to inspect object store path {}: {err}",
                        current.display()
                    )));
                }
            }
        }
        Ok(())
    }

    async fn ensure_indexes(&self) -> Result<()> {
        let _guard = self.lock().await?;
        self.reap_stale_temporary_files_unlocked().await?;
        self.replay_journal_unlocked().await?;
        let mut metadata = self.ensure_metadata_unlocked().await?;
        if self.read_index_version().await? == Some(INDEX_VERSION)
            && self.indexes_pass_sanity_check().await
            && self.indexes_are_complete().await?
            && self.embedding_indexes_are_complete().await?
        {
            // The on-disk indexes were written by this code version, our
            // canonical entries parse cleanly, every persisted raw node is
            // accounted for in the global timeline, AND every persisted
            // embedding body is accounted for in its vector manifest. Trust
            // them: this is what makes repeated `open()` calls (per request)
            // cheap. A version bump, missing marker, corrupted critical index, a
            // raw node that is missing from the timeline, or an embedding body
            // missing from its manifest (e.g. a body that was written but whose
            // manifest upsert did not complete before the process was killed
            // mid-insert) forces a full rebuild on the next open.
            return Ok(());
        }
        self.rebuild_indexes_unlocked(&mut metadata).await
    }

    /// Completeness reconciliation for the mid-insert crash window (raw bodies
    /// vs the global timeline). The sibling `embedding_indexes_are_complete`
    /// covers the structurally-identical window for embedding bodies vs their
    /// vector manifests.
    ///
    /// `insert_raw` / `update_raw_lifecycle` write the durable raw node body
    /// (`raw/<id>.json`) BEFORE syncing the retrieval indexes (timeline,
    /// session, loop, backlog). The body is therefore the source of truth and
    /// `rebuild_indexes_unlocked` reconstructs every index from the bodies,
    /// pushing every raw node into the global timeline unconditionally
    /// (`global_timeline.push(entry)`). If the process is SIGKILLed/OOM-killed
    /// in that window the body exists but is absent from every index, leaving
    /// the node permanently orphaned from all retrieval paths even though the
    /// index version + sanity check still pass.
    ///
    /// This probe closes that window: it compares the set of `raw/` ids against
    /// the ids present in the global timeline index. Because the timeline is the
    /// one index guaranteed to contain *every* raw node, timeline coverage is a
    /// sound and cheap completeness invariant — any raw id missing from the
    /// timeline means the indexes are incomplete and must be rebuilt. This is a
    /// directory listing + set membership test on open (not a per-write cost),
    /// so it preserves the cheap-repeated-open property for the common
    /// (consistent) case where every body is already indexed.
    async fn indexes_are_complete(&self) -> Result<bool> {
        let raw_paths = self.list_paths(&self.raw_dir()).await?;
        if raw_paths.is_empty() {
            // No raw bodies on disk: nothing can be orphaned from the timeline.
            return Ok(true);
        }
        let timeline = self.read_all_timeline_entries_unlocked().await?;
        let indexed: HashSet<RawNodeId> = timeline.into_iter().map(|entry| entry.id).collect();
        for path in raw_paths {
            // `list_paths` already filters to `*.json`; the file stem is the
            // raw node id (see `raw_path`). Parse it back and require timeline
            // coverage. A stem that does not parse as a RawNodeId is treated as
            // a mismatch so a rebuild re-derives a clean index from the bodies.
            match path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| RawNodeId::from_str(stem).ok())
            {
                Some(id) if indexed.contains(&id) => {}
                _ => return Ok(false),
            }
        }
        Ok(true)
    }

    /// Completeness reconciliation for the embedding mid-insert crash window
    /// (embedding bodies vs their vector manifests). This is the
    /// structurally-identical sibling of `indexes_are_complete`: the latter
    /// covers raw bodies vs the timeline, this covers embedding bodies vs their
    /// manifests.
    ///
    /// `index_raw_with_session` / `index_abstract_with_session` write the
    /// durable embedding body (`embeddings/{raw,abstract}/<id>.json`) BEFORE
    /// upserting the id into its per-session vector shard
    /// (`indexes/vector/{raw,abstract}/<session|none>.json`). The body is the
    /// source of truth and `rebuild_indexes_unlocked` regenerates the shards from
    /// a body scan (`rebuild_embedding_shards_unlocked`). If the process is
    /// SIGKILLed/OOM-killed in that window the body exists but is absent from any
    /// shard, so `search_raw` /
    /// `search_abstract` (which iterate the manifest, not the directory) never
    /// visit it — the embedding is permanently invisible to vector search even
    /// though the index version + sanity check still pass.
    ///
    /// The invariant is body-anchored: every embedding BODY id must be present
    /// in its manifest. A manifest id without a body is harmless — `search_*`
    /// already `try_read_json`s each id and silently skips a missing body — so
    /// we deliberately do NOT require the reverse, nor do we require every raw
    /// node to own an embedding. Like the raw probe this is a directory listing
    /// + set membership test on open (not a per-write cost), with an empty-dir
    ///   fast path, preserving the cheap-repeated-open property for the common
    ///   (consistent) case.
    async fn embedding_indexes_are_complete(&self) -> Result<bool> {
        if !self
            .embedding_dir_covered_by_shards::<RawNodeId>(
                &self.raw_embedding_dir(),
                &self.raw_embedding_shard_dir(),
            )
            .await?
        {
            return Ok(false);
        }
        self.embedding_dir_covered_by_shards::<AbstractNodeId>(
            &self.abstract_embedding_dir(),
            &self.abstract_embedding_shard_dir(),
        )
        .await
    }

    /// Returns `true` when every `*.json` body in `embedding_dir` has its id
    /// (the file stem) present in SOME per-session shard under `shard_dir`.
    /// Empty body directories are vacuously complete. An unparseable body stem
    /// is treated as a mismatch so a rebuild re-derives clean shards from the
    /// bodies. A wrong-shape (but valid-JSON) shard is likewise treated as not
    /// covering the bodies (`Ok(false)`), forcing a rebuild rather than
    /// surfacing a Storage error.
    async fn embedding_dir_covered_by_shards<Id>(
        &self,
        embedding_dir: &Path,
        shard_dir: &Path,
    ) -> Result<bool>
    where
        Id: DeserializeOwned + FromStr + Eq + std::hash::Hash,
    {
        let body_paths = self.list_paths(embedding_dir).await?;
        if body_paths.is_empty() {
            // No embedding bodies on disk: nothing can be orphaned.
            return Ok(true);
        }
        let mut shard_ids: HashSet<Id> = HashSet::new();
        for shard_path in self.list_embedding_shard_paths(shard_dir).await? {
            match self.try_read_json::<IdManifest<Id>>(&shard_path).await {
                Ok(Some(manifest)) => shard_ids.extend(manifest.ids),
                Ok(None) => {}
                Err(_) => return Ok(false),
            }
        }
        for path in body_paths {
            // `list_paths` already filters to `*.json` (so `.tmp` staging files
            // are ignored); the file stem is the embedding id (see
            // `raw_embedding_path` / `abstract_embedding_path`). Parse it back
            // and require manifest coverage.
            match path
                .file_stem()
                .and_then(|stem| stem.to_str())
                .and_then(|stem| Id::from_str(stem).ok())
            {
                Some(id) if shard_ids.contains(&id) => {}
                _ => return Ok(false),
            }
        }
        Ok(true)
    }

    async fn indexes_pass_sanity_check(&self) -> bool {
        // Touch the canonical materialized indexes that downstream callers
        // depend on. If any of them fail to parse, fall through to a rebuild
        // rather than surfacing a Storage error for what's a recoverable
        // on-disk inconsistency.
        for path in [
            self.undistilled_index_path(),
            self.pushed_undistilled_index_path(),
        ] {
            if !path_index_is_parseable_or_missing(&path).await {
                return false;
            }
        }
        // Every sharded index file (timeline + per-session embeddings) must
        // parse too; a corrupt shard forces a rebuild.
        for shard_dir in [
            self.raw_timeline_shard_dir(),
            self.raw_embedding_shard_dir(),
            self.abstract_embedding_shard_dir(),
        ] {
            let Ok(shard_paths) = self.list_paths(&shard_dir).await else {
                return false;
            };
            for path in shard_paths {
                if !path_index_is_parseable_or_missing(&path).await {
                    return false;
                }
            }
        }
        true
    }

    async fn read_index_version(&self) -> Result<Option<u32>> {
        let path = self.index_version_path();
        match fs::read(&path).await {
            Ok(payload) => {
                let text = String::from_utf8_lossy(&payload);
                Ok(text.trim().parse::<u32>().ok())
            }
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(EngineError::Storage(format!(
                "failed to read object store index version {}: {err}",
                path.display()
            ))),
        }
    }

    async fn write_index_version(&self) -> Result<()> {
        let path = self.index_version_path();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.map_err(|err| {
                EngineError::Storage(format!(
                    "failed to create object store index version parent {}: {err}",
                    parent.display()
                ))
            })?;
        }
        fs::write(&path, INDEX_VERSION.to_string().as_bytes())
            .await
            .map_err(|err| {
                EngineError::Storage(format!(
                    "failed to write object store index version {}: {err}",
                    path.display()
                ))
            })
    }

    async fn rebuild_indexes_unlocked(&self, metadata: &mut StoreMetadata) -> Result<()> {
        if fs::try_exists(&self.index_root()).await.map_err(|err| {
            EngineError::Storage(format!(
                "failed to probe object store index root {}: {err}",
                self.index_root().display()
            ))
        })? {
            fs::remove_dir_all(self.index_root()).await.map_err(|err| {
                EngineError::Storage(format!(
                    "failed to reset object store index root {}: {err}",
                    self.index_root().display()
                ))
            })?;
        }
        self.ensure_index_layout().await?;

        let raw_nodes = self.list_raw_nodes_scan_unlocked().await?;
        let abstract_nodes = self.list_abstract_nodes_scan_unlocked().await?;

        let mut timeline_shards: HashMap<String, Vec<RawIndexEntry>> = HashMap::new();
        let mut session_timelines: HashMap<SessionId, Vec<RawIndexEntry>> = HashMap::new();
        let mut loop_timelines: HashMap<LoopId, Vec<RawIndexEntry>> = HashMap::new();
        let mut undistilled = Vec::new();
        let mut pushed_undistilled = Vec::new();

        for node in raw_nodes {
            let entry = RawIndexEntry::from(&node);
            timeline_shards
                .entry(timeline_bucket(node.timestamp))
                .or_default()
                .push(entry.clone());
            if let Some(session_id) = node.session_id {
                session_timelines
                    .entry(session_id)
                    .or_default()
                    .push(entry.clone());
            }
            if let Some(loop_id) = node.loop_id {
                loop_timelines
                    .entry(loop_id)
                    .or_default()
                    .push(entry.clone());
            }
            if node.distillation_state != DistillationState::Distilled {
                undistilled.push(entry.clone());
                if node.overflow.was_pushed_out_of_session {
                    pushed_undistilled.push(entry.clone());
                }
            }
            if let Some(operation_key) = &node.operation_key {
                self.write_json(
                    &self.raw_operation_path(operation_key),
                    &StoredId {
                        id: node.id,
                        operation_key: operation_key.clone(),
                    },
                )
                .await?;
            }
        }

        for (bucket, mut entries) in timeline_shards {
            sort_raw_index_entries(&mut entries);
            self.write_json(&self.raw_timeline_shard_path(&bucket), &entries)
                .await?;
        }
        sort_raw_index_entries(&mut undistilled);
        self.write_json(&self.undistilled_index_path(), &undistilled)
            .await?;
        sort_raw_index_entries(&mut pushed_undistilled);
        self.write_json(&self.pushed_undistilled_index_path(), &pushed_undistilled)
            .await?;

        for (session_id, mut entries) in session_timelines {
            sort_raw_index_entries(&mut entries);
            if entries.len() > MAX_SESSION_INDEX_ENTRIES {
                entries.drain(..entries.len() - MAX_SESSION_INDEX_ENTRIES);
            }
            self.write_json(&self.session_index_path(&session_id), &entries)
                .await?;
        }
        for (loop_id, mut entries) in loop_timelines {
            sort_raw_index_entries(&mut entries);
            if entries.len() > MAX_LOOP_INDEX_ENTRIES {
                entries.drain(..entries.len() - MAX_LOOP_INDEX_ENTRIES);
            }
            self.write_json(&self.loop_index_path(&loop_id), &entries)
                .await?;
        }

        for node in abstract_nodes {
            if let Some(operation_key) = &node.operation_key {
                self.write_json(
                    &self.abstract_operation_path(operation_key),
                    &StoredId {
                        id: node.id,
                        operation_key: operation_key.clone(),
                    },
                )
                .await?;
            }
        }

        self.rebuild_embedding_shards_unlocked::<RawNodeId>(
            &self.raw_embedding_dir(),
            &self.raw_embedding_shard_dir(),
        )
        .await?;
        self.rebuild_embedding_shards_unlocked::<AbstractNodeId>(
            &self.abstract_embedding_dir(),
            &self.abstract_embedding_shard_dir(),
        )
        .await?;

        self.mark_indexes_rebuilt_unlocked(metadata).await?;
        self.write_index_version().await
    }

    fn metadata_path(&self) -> PathBuf {
        self.root.join("store.json")
    }

    fn index_root(&self) -> PathBuf {
        self.root.join("indexes")
    }

    fn index_version_path(&self) -> PathBuf {
        self.index_root().join(".index-version")
    }

    fn raw_dir(&self) -> PathBuf {
        self.root.join("raw")
    }

    fn abstract_dir(&self) -> PathBuf {
        self.root.join("abstract")
    }

    fn raw_embedding_dir(&self) -> PathBuf {
        self.root.join("embeddings").join("raw")
    }

    fn abstract_embedding_dir(&self) -> PathBuf {
        self.root.join("embeddings").join("abstract")
    }

    fn graph_dir(&self) -> PathBuf {
        self.root.join("graph")
    }

    fn checkpoint_dir(&self) -> PathBuf {
        self.root.join("checkpoints")
    }

    fn journal_dir(&self) -> PathBuf {
        self.root.join("journal")
    }

    fn journal_path(&self) -> PathBuf {
        self.journal_dir().join("current.json")
    }

    fn quarantine_dir(&self) -> PathBuf {
        self.root.join("quarantine")
    }

    fn distillation_claim_dir(&self) -> PathBuf {
        self.root.join("claims").join("distillation")
    }

    pub(super) fn distillation_claim_path(
        &self,
        session_id: &SessionId,
        loop_id: &LoopId,
    ) -> PathBuf {
        self.distillation_claim_dir()
            .join(format!("{session_id}--{loop_id}.json"))
    }

    fn raw_operation_dir(&self) -> PathBuf {
        self.root.join("receipts").join("raw_operation")
    }

    fn abstract_operation_dir(&self) -> PathBuf {
        self.root.join("receipts").join("abstract_operation")
    }

    fn session_index_dir(&self) -> PathBuf {
        self.index_root().join("session")
    }

    fn loop_index_dir(&self) -> PathBuf {
        self.index_root().join("loop")
    }

    fn timeline_index_dir(&self) -> PathBuf {
        self.index_root().join("timeline")
    }

    fn backlog_index_dir(&self) -> PathBuf {
        self.index_root().join("backlog")
    }

    fn vector_index_dir(&self) -> PathBuf {
        self.index_root().join("vector")
    }

    pub(super) fn raw_path(&self, id: &RawNodeId) -> PathBuf {
        self.raw_dir().join(format!("{id}.json"))
    }

    pub(super) fn abstract_path(&self, id: &AbstractNodeId) -> PathBuf {
        self.abstract_dir().join(format!("{id}.json"))
    }

    pub(super) fn raw_embedding_path(&self, id: &RawNodeId) -> PathBuf {
        self.raw_embedding_dir().join(format!("{id}.json"))
    }

    pub(super) fn abstract_embedding_path(&self, id: &AbstractNodeId) -> PathBuf {
        self.abstract_embedding_dir().join(format!("{id}.json"))
    }

    pub(super) fn graph_path(&self, id: &AbstractNodeId) -> PathBuf {
        self.graph_dir().join(format!("{id}.json"))
    }

    pub(super) fn checkpoint_path(&self, session_id: &SessionId, loop_id: &LoopId) -> PathBuf {
        self.checkpoint_dir()
            .join(format!("{session_id}--{loop_id}.json"))
    }

    async fn ensure_metadata_unlocked(&self) -> Result<StoreMetadata> {
        if let Some(metadata) = self
            .try_read_json::<StoreMetadata>(&self.metadata_path())
            .await?
        {
            if metadata.format_version != STORE_FORMAT_VERSION {
                return Err(EngineError::Storage(format!(
                    "unsupported object store format version {} in {}",
                    metadata.format_version,
                    self.metadata_path().display()
                )));
            }
            return Ok(metadata);
        }

        let now = Utc::now();
        let metadata = StoreMetadata {
            format_version: STORE_FORMAT_VERSION,
            created_at: now,
            updated_at: now,
            last_index_rebuild_at: None,
        };
        self.write_json(&self.metadata_path(), &metadata).await?;
        Ok(metadata)
    }

    async fn write_metadata_unlocked(&self, metadata: &StoreMetadata) -> Result<()> {
        self.write_json(&self.metadata_path(), metadata).await
    }

    pub(super) async fn touch_metadata_unlocked(&self) -> Result<()> {
        let mut metadata = self.ensure_metadata_unlocked().await?;
        metadata.updated_at = Utc::now();
        self.write_metadata_unlocked(&metadata).await
    }

    async fn mark_indexes_rebuilt_unlocked(&self, metadata: &mut StoreMetadata) -> Result<()> {
        let now = Utc::now();
        metadata.updated_at = now;
        metadata.last_index_rebuild_at = Some(now);
        self.write_metadata_unlocked(metadata).await
    }

    pub(super) fn raw_operation_path(&self, operation_key: &str) -> PathBuf {
        self.raw_operation_dir()
            .join(format!("{}.json", encode_operation_key(operation_key)))
    }

    pub(super) fn abstract_operation_path(&self, operation_key: &str) -> PathBuf {
        self.abstract_operation_dir()
            .join(format!("{}.json", encode_operation_key(operation_key)))
    }

    pub(super) fn session_index_path(&self, session_id: &SessionId) -> PathBuf {
        self.session_index_dir().join(format!("{session_id}.json"))
    }

    pub(super) fn loop_index_path(&self, loop_id: &LoopId) -> PathBuf {
        self.loop_index_dir().join(format!("{loop_id}.json"))
    }

    /// Directory holding the per-day global-timeline shards
    /// (`indexes/timeline/raw/<YYYYMMDD>.json`). Sharding by day keeps each
    /// insert's read-modify-write bounded to a single day's entries instead of
    /// the entire never-pruned timeline.
    fn raw_timeline_shard_dir(&self) -> PathBuf {
        self.timeline_index_dir().join("raw")
    }

    pub(super) fn raw_timeline_shard_path(&self, bucket: &str) -> PathBuf {
        self.raw_timeline_shard_dir().join(format!("{bucket}.json"))
    }

    /// List the timeline shard buckets (file stems, i.e. `YYYYMMDD`) currently
    /// on disk, ascending. `list_paths` already sorts and filters to `*.json`.
    pub(super) async fn list_timeline_shard_buckets(&self) -> Result<Vec<String>> {
        let mut buckets = Vec::new();
        for path in self.list_paths(&self.raw_timeline_shard_dir()).await? {
            if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
                buckets.push(stem.to_string());
            }
        }
        Ok(buckets)
    }

    /// Read every global-timeline entry across all shards. Used only on open by
    /// the completeness probe and by full rebuilds — never on the per-insert
    /// hot path.
    async fn read_all_timeline_entries_unlocked(&self) -> Result<Vec<RawIndexEntry>> {
        let mut all = Vec::new();
        for bucket in self.list_timeline_shard_buckets().await? {
            let mut shard = self
                .read_raw_index_entries_unlocked(&self.raw_timeline_shard_path(&bucket))
                .await?;
            all.append(&mut shard);
        }
        Ok(all)
    }

    /// Merge the global timeline shards newest-day-first into the most-recent
    /// `limit` entries (timestamp DESC, id ASC tiebreak), applying the optional
    /// `from`/`to` window. Because shards partition by whole UTC days, once we
    /// have collected `limit` entries from the newest days, every older shard
    /// can only contain strictly-older entries, so we stop early — bounding the
    /// read to a few shards instead of the whole store.
    pub(super) async fn read_global_timeline_entries_unlocked(
        &self,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<RawIndexEntry>> {
        let buckets = self.list_timeline_shard_buckets().await?;
        let mut collected: Vec<RawIndexEntry> = Vec::new();
        // `list_timeline_shard_buckets` is ascending; iterate newest first.
        for bucket in buckets.into_iter().rev() {
            let mut shard = self
                .read_raw_index_entries_unlocked(&self.raw_timeline_shard_path(&bucket))
                .await?;
            shard.retain(|entry| from.is_none_or(|value| entry.timestamp >= value));
            shard.retain(|entry| to.is_none_or(|value| entry.timestamp <= value));
            collected.append(&mut shard);
            if collected.len() >= limit {
                break;
            }
        }
        collected.sort_by(|left, right| {
            right
                .timestamp
                .cmp(&left.timestamp)
                .then_with(|| left.id.cmp(&right.id))
        });
        collected.truncate(limit);
        Ok(collected)
    }

    pub(super) fn undistilled_index_path(&self) -> PathBuf {
        self.backlog_index_dir().join("undistilled_raw.json")
    }

    pub(super) fn pushed_undistilled_index_path(&self) -> PathBuf {
        self.backlog_index_dir().join("pushed_undistilled_raw.json")
    }

    fn raw_embedding_shard_dir(&self) -> PathBuf {
        self.vector_index_dir().join("raw")
    }

    fn abstract_embedding_shard_dir(&self) -> PathBuf {
        self.vector_index_dir().join("abstract")
    }

    /// Per-session embedding shard path. `None` (session-less / legacy entries)
    /// maps to the reserved `none` bucket. A session-scoped search reads only
    /// its own shard, bounding the scan to a single session's embeddings instead
    /// of every embedding in the installation.
    pub(super) fn raw_embedding_shard_path(&self, session_id: Option<&SessionId>) -> PathBuf {
        self.raw_embedding_shard_dir()
            .join(format!("{}.json", embedding_shard_bucket(session_id)))
    }

    pub(super) fn abstract_embedding_shard_path(&self, session_id: Option<&SessionId>) -> PathBuf {
        self.abstract_embedding_shard_dir()
            .join(format!("{}.json", embedding_shard_bucket(session_id)))
    }

    /// List the embedding shard files in `shard_dir` (`*.json`, sorted).
    pub(super) async fn list_embedding_shard_paths(
        &self,
        shard_dir: &Path,
    ) -> Result<Vec<PathBuf>> {
        self.list_paths(shard_dir).await
    }

    #[allow(clippy::unused_self)] // grouped with FileObjectStore for cohesion
    pub(super) async fn write_json<T: Serialize>(&self, path: &Path, value: &T) -> Result<()> {
        self.validate_managed_path(path).await?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.map_err(|err| {
                EngineError::Storage(format!(
                    "failed to create parent directory {}: {err}",
                    parent.display()
                ))
            })?;
            self.validate_managed_path(parent).await?;
        }
        let payload = serde_json::to_vec_pretty(value).map_err(|err| {
            EngineError::Storage(format!(
                "failed to serialize object store payload {}: {err}",
                path.display()
            ))
        })?;
        // Unique temp name (not a deterministic `<stem>.tmp`) so two writers
        // staging the same target never clobber each other's staging file — the
        // rename is then always tmp -> target with no cross-writer ENOENT. The
        // `.tmp` extension keeps it out of every `*.json` index/body listing.
        // [S1]
        let temporary = path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
        let mut file = fs::File::create(&temporary).await.map_err(|err| {
            EngineError::Storage(format!(
                "failed to create temporary object {}: {err}",
                temporary.display()
            ))
        })?;
        file.write_all(&payload).await.map_err(|err| {
            EngineError::Storage(format!(
                "failed to write temporary object {}: {err}",
                temporary.display()
            ))
        })?;
        file.sync_all().await.map_err(|err| {
            EngineError::Storage(format!(
                "failed to sync temporary object {}: {err}",
                temporary.display()
            ))
        })?;
        drop(file);
        if let Err(err) = fs::rename(&temporary, path).await {
            // Best-effort cleanup so a failed rename does not leak the staged
            // temp file (a hard kill mid-write can still leave one, which is
            // harmless — readers only ever consider `*.json`).
            let _ = fs::remove_file(&temporary).await;
            return Err(EngineError::Storage(format!(
                "failed to move object {} into place: {err}",
                path.display()
            )));
        }
        if let Some(parent) = path.parent() {
            self.sync_directory(parent).await?;
        }
        Ok(())
    }

    #[allow(clippy::unused_self)] // grouped with FileObjectStore for cohesion
    async fn read_json<T: DeserializeOwned>(&self, path: &Path) -> Result<T> {
        self.validate_managed_path(path).await?;
        let payload = fs::read(path).await.map_err(|err| {
            EngineError::Storage(format!("failed to read object {}: {err}", path.display()))
        })?;
        match serde_json::from_slice(&payload) {
            Ok(value) => Ok(value),
            Err(err) if path == self.journal_path() => Err(EngineError::Storage(format!(
                "failed to deserialize recovery journal {}; manual recovery is required: {err}",
                path.display()
            ))),
            Err(err) => {
                let filename = path
                    .file_name()
                    .and_then(|value| value.to_str())
                    .unwrap_or("object");
                let quarantine_path = self
                    .quarantine_dir()
                    .join(format!("{filename}.{}.corrupt", uuid::Uuid::new_v4()));
                self.ensure_real_directory(&self.quarantine_dir()).await?;
                fs::rename(path, &quarantine_path).await.map_err(|move_err| {
                    EngineError::Storage(format!(
                        "failed to deserialize object {} ({err}) and failed to quarantine it: {move_err}",
                        path.display()
                    ))
                })?;
                self.sync_directory(&self.quarantine_dir()).await?;
                Err(EngineError::Storage(format!(
                    "corrupt object {} was quarantined at {}: {err}",
                    path.display(),
                    quarantine_path.display()
                )))
            }
        }
    }

    pub(super) async fn try_read_json<T: DeserializeOwned>(
        &self,
        path: &Path,
    ) -> Result<Option<T>> {
        match fs::try_exists(path).await {
            Ok(true) => self.read_json(path).await.map(Some),
            Ok(false) => Ok(None),
            Err(err) => Err(EngineError::Storage(format!(
                "failed to probe object {}: {err}",
                path.display()
            ))),
        }
    }

    #[allow(clippy::unused_self)] // grouped with FileObjectStore for cohesion
    pub(super) async fn remove_file_if_exists(&self, path: &Path) -> Result<()> {
        self.validate_managed_path(path).await?;
        match fs::try_exists(path).await {
            Ok(true) => fs::remove_file(path).await.map_err(|err| {
                EngineError::Storage(format!("failed to remove object {}: {err}", path.display()))
            }),
            Ok(false) => Ok(()),
            Err(err) => Err(EngineError::Storage(format!(
                "failed to probe object {} for removal: {err}",
                path.display()
            ))),
        }
    }

    #[allow(clippy::unused_self)] // grouped with FileObjectStore for cohesion
    async fn list_paths(&self, directory: &Path) -> Result<Vec<PathBuf>> {
        self.validate_managed_path(directory).await?;
        match fs::try_exists(directory).await {
            Ok(false) => return Ok(Vec::new()),
            Ok(true) => {}
            Err(err) => {
                return Err(EngineError::Storage(format!(
                    "failed to probe object store directory {}: {err}",
                    directory.display()
                )));
            }
        }
        let mut paths = Vec::new();
        let mut entries = fs::read_dir(directory).await.map_err(|err| {
            EngineError::Storage(format!(
                "failed to read object store directory {}: {err}",
                directory.display()
            ))
        })?;
        loop {
            match entries.next_entry().await {
                Ok(Some(entry)) => {
                    let path = entry.path();
                    if path.extension().and_then(|value| value.to_str()) == Some("json") {
                        paths.push(path);
                    }
                }
                Ok(None) => break,
                Err(err) => {
                    return Err(EngineError::Storage(format!(
                        "failed to enumerate object store directory {}: {err}",
                        directory.display()
                    )));
                }
            }
        }
        paths.sort();
        Ok(paths)
    }

    async fn list_raw_nodes_scan_unlocked(&self) -> Result<Vec<RawNode>> {
        let mut nodes = Vec::new();
        for path in self.list_paths(&self.raw_dir()).await? {
            nodes.push(self.read_json(&path).await?);
        }
        nodes.sort_by(|left: &RawNode, right: &RawNode| {
            left.timestamp
                .cmp(&right.timestamp)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(nodes)
    }

    async fn reap_stale_temporary_files_unlocked(&self) -> Result<()> {
        let root = self.root.clone();
        tokio::task::spawn_blocking(move || -> std::io::Result<()> {
            let mut pending = vec![root];
            while let Some(directory) = pending.pop() {
                for entry in std::fs::read_dir(directory)? {
                    let entry = entry?;
                    let file_type = entry.file_type()?;
                    if file_type.is_symlink() {
                        continue;
                    }
                    let path = entry.path();
                    if file_type.is_dir() {
                        pending.push(path);
                        continue;
                    }
                    if path.extension().and_then(|value| value.to_str()) != Some("tmp") {
                        continue;
                    }
                    // The store-wide advisory lock is held while this sweep
                    // runs, so no live writer can own a staging file.
                    let _ = std::fs::remove_file(path);
                }
            }
            Ok(())
        })
        .await
        .map_err(|err| EngineError::Storage(format!("temporary-file reaper failed: {err}")))?
        .map_err(|err| EngineError::Storage(format!("failed to reap stale temporary files: {err}")))
    }

    async fn list_abstract_nodes_scan_unlocked(&self) -> Result<Vec<AbstractNode>> {
        let mut nodes = Vec::new();
        for path in self.list_paths(&self.abstract_dir()).await? {
            nodes.push(self.read_json(&path).await?);
        }
        nodes.sort_by(|left: &AbstractNode, right: &AbstractNode| {
            left.timestamp
                .cmp(&right.timestamp)
                .then_with(|| left.id.cmp(&right.id))
        });
        Ok(nodes)
    }

    pub(super) async fn read_raw_index_entries_unlocked(
        &self,
        path: &Path,
    ) -> Result<Vec<RawIndexEntry>> {
        Ok(self
            .try_read_json::<Vec<RawIndexEntry>>(path)
            .await?
            .unwrap_or_default())
    }

    async fn write_manifest_unlocked<Id>(&self, path: &Path, ids: &[Id]) -> Result<()>
    where
        Id: Serialize + Clone + PartialEq,
    {
        let mut unique = Vec::with_capacity(ids.len());
        for id in ids {
            if !unique.contains(id) {
                unique.push(id.clone());
            }
        }
        self.write_json(path, &IdManifest { ids: unique }).await
    }

    pub(super) async fn read_manifest_unlocked<Id>(&self, path: &Path) -> Result<Vec<Id>>
    where
        Id: DeserializeOwned,
    {
        Ok(self
            .try_read_json::<IdManifest<Id>>(path)
            .await?
            .map(|manifest| manifest.ids)
            .unwrap_or_default())
    }

    pub(super) async fn upsert_manifest_id_unlocked<Id>(
        &self,
        path: &Path,
        id: Id,
    ) -> Result<Vec<Id>>
    where
        Id: Serialize + DeserializeOwned + Copy + PartialEq,
    {
        let mut ids = self.read_manifest_unlocked::<Id>(path).await?;
        ids.retain(|existing| *existing != id);
        ids.push(id);
        let overflow = ids.len().saturating_sub(MAX_VECTOR_SHARD_ENTRIES);
        let evicted = if overflow == 0 {
            Vec::new()
        } else {
            ids.drain(..overflow).collect()
        };
        self.write_manifest_unlocked(path, &ids).await?;
        Ok(evicted)
    }

    /// Regenerate the per-session embedding shard manifests for one body
    /// directory from a full body scan. Each `StoredEmbedding` body carries the
    /// session it was indexed for, so we group ids into the matching shard
    /// bucket (`<session>` or `none`) and write one `IdManifest` per bucket.
    /// Only used on rebuild (never the per-insert hot path).
    async fn rebuild_embedding_shards_unlocked<Id>(
        &self,
        body_dir: &Path,
        shard_dir: &Path,
    ) -> Result<()>
    where
        Id: DeserializeOwned + Serialize + Ord + Copy + std::fmt::Display,
    {
        let mut shards: HashMap<String, Vec<(DateTime<Utc>, Id)>> = HashMap::new();
        for path in self.list_paths(body_dir).await? {
            let record: StoredEmbedding<Id> = self.read_json(&path).await?;
            shards
                .entry(embedding_shard_bucket(record.session_id.as_ref()))
                .or_default()
                .push((record.indexed_at, record.id));
        }
        for (bucket, mut entries) in shards {
            entries.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
            let overflow = entries.len().saturating_sub(MAX_VECTOR_SHARD_ENTRIES);
            for (_, evicted) in entries.drain(..overflow) {
                self.remove_file_if_exists(&body_dir.join(format!("{evicted}.json")))
                    .await?;
            }
            let ids = entries.into_iter().map(|(_, id)| id).collect::<Vec<_>>();
            self.write_manifest_unlocked(&shard_dir.join(format!("{bucket}.json")), &ids)
                .await?;
        }
        Ok(())
    }

    async fn upsert_raw_index_entry_unlocked(
        &self,
        path: &Path,
        entry: RawIndexEntry,
    ) -> Result<()> {
        let mut entries = self.read_raw_index_entries_unlocked(path).await?;
        entries.retain(|existing| existing.id != entry.id);
        entries.push(entry);
        sort_raw_index_entries(&mut entries);
        self.write_json(path, &entries).await
    }

    async fn upsert_bounded_raw_index_entry_unlocked(
        &self,
        path: &Path,
        entry: RawIndexEntry,
        limit: usize,
    ) -> Result<()> {
        let mut entries = self.read_raw_index_entries_unlocked(path).await?;
        entries.retain(|existing| existing.id != entry.id);
        entries.push(entry);
        sort_raw_index_entries(&mut entries);
        if entries.len() > limit {
            entries.drain(..entries.len() - limit);
        }
        self.write_json(path, &entries).await
    }

    async fn remove_raw_index_entry_unlocked(&self, path: &Path, id: RawNodeId) -> Result<()> {
        let mut entries = self.read_raw_index_entries_unlocked(path).await?;
        let before = entries.len();
        entries.retain(|entry| entry.id != id);
        if before != entries.len() {
            self.write_json(path, &entries).await?;
        }
        Ok(())
    }

    pub(super) async fn sync_raw_indexes_unlocked(&self, node: &RawNode) -> Result<()> {
        let entry = RawIndexEntry::from(node);
        // Write the secondary indexes (session, loop, backlog) FIRST and the
        // global timeline LAST. Combined with the body-before-indexes ordering
        // in `insert_raw`/`update_raw_lifecycle`, this makes timeline coverage a
        // sound completeness invariant: if a node is present in the timeline,
        // every secondary index for it was already written, so a mid-insert
        // crash that the timeline-anchored `indexes_are_complete` probe accepts
        // can never leave the node missing from session/loop/backlog retrieval
        // (it would instead be missing from the timeline -> probe fails ->
        // rebuild). [C5]
        if let Some(session_id) = node.session_id {
            self.upsert_bounded_raw_index_entry_unlocked(
                &self.session_index_path(&session_id),
                entry.clone(),
                MAX_SESSION_INDEX_ENTRIES,
            )
            .await?;
        }
        if let Some(loop_id) = node.loop_id {
            self.upsert_bounded_raw_index_entry_unlocked(
                &self.loop_index_path(&loop_id),
                entry.clone(),
                MAX_LOOP_INDEX_ENTRIES,
            )
            .await?;
        }

        let is_undistilled = node.distillation_state != DistillationState::Distilled;
        if is_undistilled {
            self.upsert_raw_index_entry_unlocked(&self.undistilled_index_path(), entry.clone())
                .await?;
        } else {
            self.remove_raw_index_entry_unlocked(&self.undistilled_index_path(), node.id)
                .await?;
        }

        let pushed = is_undistilled && node.overflow.was_pushed_out_of_session;
        if pushed {
            self.upsert_raw_index_entry_unlocked(
                &self.pushed_undistilled_index_path(),
                entry.clone(),
            )
            .await?;
        } else {
            self.remove_raw_index_entry_unlocked(&self.pushed_undistilled_index_path(), node.id)
                .await?;
        }

        // Timeline last, into the bounded per-day shard. The shard's
        // read-modify-write touches only this node's day, not the whole
        // never-pruned timeline (C2).
        self.upsert_raw_index_entry_unlocked(
            &self.raw_timeline_shard_path(&timeline_bucket(node.timestamp)),
            entry,
        )
        .await?;

        Ok(())
    }

    pub(super) async fn commit_raw_unlocked(
        &self,
        node: &RawNode,
        embedding: &Embedding,
    ) -> Result<()> {
        let mutation = MutationJournal::CommitRaw {
            node: node.clone(),
            embedding: embedding.clone(),
        };
        self.write_journal_unlocked(&mutation).await?;
        self.apply_raw_commit_unlocked(node, embedding).await?;
        self.clear_journal_unlocked().await
    }

    pub(super) async fn commit_abstract_unlocked(
        &self,
        node: &AbstractNode,
        embedding: &Embedding,
    ) -> Result<()> {
        let mutation = MutationJournal::CommitAbstract {
            node: node.clone(),
            embedding: embedding.clone(),
        };
        self.write_journal_unlocked(&mutation).await?;
        self.apply_abstract_commit_unlocked(node, embedding).await?;
        self.clear_journal_unlocked().await
    }

    pub(super) async fn update_raw_lifecycle_unlocked(
        &self,
        ids: &[RawNodeId],
        patch: &RawLifecyclePatch,
    ) -> Result<()> {
        let mutation = MutationJournal::UpdateRawLifecycle {
            ids: ids.to_vec(),
            patch: patch.clone(),
        };
        self.write_journal_unlocked(&mutation).await?;
        self.apply_raw_lifecycle_unlocked(ids, patch).await?;
        self.clear_journal_unlocked().await
    }

    async fn write_journal_unlocked(&self, mutation: &MutationJournal) -> Result<()> {
        self.write_json(&self.journal_path(), mutation).await
    }

    async fn clear_journal_unlocked(&self) -> Result<()> {
        self.remove_file_if_exists(&self.journal_path()).await?;
        self.sync_directory(&self.journal_dir()).await
    }

    async fn replay_journal_unlocked(&self) -> Result<()> {
        let Some(mutation) = self
            .try_read_json::<MutationJournal>(&self.journal_path())
            .await?
        else {
            return Ok(());
        };
        match &mutation {
            MutationJournal::CommitRaw { node, embedding } => {
                self.apply_raw_commit_unlocked(node, embedding).await?;
            }
            MutationJournal::CommitAbstract { node, embedding } => {
                self.apply_abstract_commit_unlocked(node, embedding).await?;
            }
            MutationJournal::UpdateRawLifecycle { ids, patch } => {
                self.apply_raw_lifecycle_unlocked(ids, patch).await?;
            }
        }
        self.clear_journal_unlocked().await
    }

    async fn apply_raw_commit_unlocked(&self, node: &RawNode, embedding: &Embedding) -> Result<()> {
        self.write_json(&self.raw_path(&node.id), node).await?;
        if let Some(operation_key) = &node.operation_key {
            self.write_json(
                &self.raw_operation_path(operation_key),
                &StoredId {
                    id: node.id,
                    operation_key: operation_key.clone(),
                },
            )
            .await?;
        }
        self.sync_raw_indexes_unlocked(node).await?;
        self.write_json(
            &self.raw_embedding_path(&node.id),
            &StoredEmbedding {
                id: node.id,
                embedding: embedding.clone(),
                indexed_at: node.timestamp,
                session_id: node.session_id,
            },
        )
        .await?;
        let evicted = self
            .upsert_manifest_id_unlocked(
                &self.raw_embedding_shard_path(node.session_id.as_ref()),
                node.id,
            )
            .await?;
        for id in evicted {
            self.remove_file_if_exists(&self.raw_embedding_path(&id))
                .await?;
        }
        self.touch_metadata_unlocked().await
    }

    async fn apply_abstract_commit_unlocked(
        &self,
        node: &AbstractNode,
        embedding: &Embedding,
    ) -> Result<()> {
        self.write_json(&self.abstract_path(&node.id), node).await?;
        if let Some(operation_key) = &node.operation_key {
            self.write_json(
                &self.abstract_operation_path(operation_key),
                &StoredId {
                    id: node.id,
                    operation_key: operation_key.clone(),
                },
            )
            .await?;
        }
        self.write_json(
            &self.abstract_embedding_path(&node.id),
            &StoredEmbedding {
                id: node.id,
                embedding: embedding.clone(),
                indexed_at: node.timestamp,
                session_id: node.session_id,
            },
        )
        .await?;
        let evicted = self
            .upsert_manifest_id_unlocked(
                &self.abstract_embedding_shard_path(node.session_id.as_ref()),
                node.id,
            )
            .await?;
        for id in evicted {
            self.remove_file_if_exists(&self.abstract_embedding_path(&id))
                .await?;
        }
        self.write_json(
            &self.graph_path(&node.id),
            &StoredGraphEdges {
                node_id: node.id,
                session_id: node.session_id,
                edges: ObjectGraphRepository::edges_for_abstract(node),
            },
        )
        .await?;
        self.touch_metadata_unlocked().await
    }

    async fn apply_raw_lifecycle_unlocked(
        &self,
        ids: &[RawNodeId],
        patch: &RawLifecyclePatch,
    ) -> Result<()> {
        let mut changed = false;
        for id in ids {
            if let Some(mut node) = self.try_read_json::<RawNode>(&self.raw_path(id)).await? {
                if let Some(distillation_state) = &patch.distillation_state {
                    node.distillation_state = distillation_state.clone();
                }
                if let Some(overflow) = &patch.overflow {
                    node.overflow = overflow.clone();
                }
                self.write_json(&self.raw_path(id), &node).await?;
                self.sync_raw_indexes_unlocked(&node).await?;
                changed = true;
            }
        }
        if changed {
            self.touch_metadata_unlocked().await?;
        }
        Ok(())
    }

    async fn sync_directory(&self, directory: &Path) -> Result<()> {
        let directory = directory.to_path_buf();
        tokio::task::spawn_blocking(move || File::open(&directory)?.sync_all())
            .await
            .map_err(|err| EngineError::Storage(format!("directory sync task failed: {err}")))?
            .map_err(|err| EngineError::Storage(format!("failed to sync directory: {err}")))
    }

    pub(super) async fn read_raw_by_ids_unlocked(&self, ids: &[RawNodeId]) -> Result<Vec<RawNode>> {
        let mut nodes = Vec::new();
        for id in ids {
            if let Some(node) = self.try_read_json::<RawNode>(&self.raw_path(id)).await? {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }
}

/// Process-wide registry mapping a canonical store root to the single gate
/// shared by every `FileObjectStore` opened on that root. This is what makes
/// concurrent runs on one (space, installation) directory mutually exclude even
/// though each run constructs its own store handle. Entries are never removed
/// (one tiny entry per installation ever opened in this process).
fn store_gate_registry() -> &'static std::sync::Mutex<HashMap<PathBuf, Arc<Mutex<()>>>> {
    static REGISTRY: std::sync::OnceLock<std::sync::Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> =
        std::sync::OnceLock::new();
    REGISTRY.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Fetch (or create) the shared gate for `root`. The key is the canonicalized
/// path so two callers that name the same directory differently (relative vs
/// absolute, symlinks) still share one gate; if canonicalization fails (e.g.
/// the directory does not exist yet) we fall back to the literal path, which is
/// still stable for the common case where every run passes the same string.
async fn shared_gate_for_root(root: &Path) -> Arc<Mutex<()>> {
    // Best-effort: ensure the directory exists so canonicalization succeeds and
    // produces a stable key. Ignore errors here — `ensure_layout` surfaces any
    // real failure with a precise message right after.
    let _ = fs::create_dir_all(root).await;
    let key = fs::canonicalize(root)
        .await
        .unwrap_or_else(|_| root.to_path_buf());
    let mut registry = store_gate_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    registry
        .entry(key)
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

/// Bridges synchronous `FileObjectStore::open` callers to the tokio-fs based
/// async constructor. The work runs on a dedicated thread with its own
/// current-thread tokio runtime so we never need to assume the caller is on
/// the multi-threaded scheduler (`block_in_place` panics on current-thread
/// runtimes) and never re-enter the caller's runtime.
fn tokio_block_on<F>(future: F) -> F::Output
where
    F: std::future::Future + Send + 'static,
    F::Output: Send + 'static,
{
    std::thread::scope(|scope| {
        scope
            .spawn(|| {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("tokio current-thread runtime should build for object store open");
                runtime.block_on(future)
            })
            .join()
            .expect("object store open thread should not panic")
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct StoredId<Id> {
    pub(super) id: Id,
    #[serde(default)]
    pub(super) operation_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum MutationJournal {
    CommitRaw {
        node: RawNode,
        embedding: Embedding,
    },
    CommitAbstract {
        node: AbstractNode,
        embedding: Embedding,
    },
    UpdateRawLifecycle {
        ids: Vec<RawNodeId>,
        patch: RawLifecyclePatch,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoreMetadata {
    format_version: u32,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    last_index_rebuild_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IdManifest<Id> {
    ids: Vec<Id>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct StoredEmbedding<Id> {
    pub(super) id: Id,
    pub(super) embedding: Embedding,
    #[serde(default = "default_indexed_at")]
    pub(super) indexed_at: DateTime<Utc>,
    /// Session id this embedding was indexed for. `None` represents either
    /// (a) legacy entries written before the session-aware index existed, or
    /// (b) intentionally session-less indexing. The search filter treats
    /// `None` entries as legacy: they are only returned when the search
    /// itself passes `session_id = None`, never when a specific session is
    /// requested. The field is `#[serde(default)]` so existing on-disk JSON
    /// without the field deserializes cleanly into the legacy bucket.
    #[serde(default)]
    pub(super) session_id: Option<SessionId>,
}

fn default_indexed_at() -> DateTime<Utc> {
    DateTime::<Utc>::UNIX_EPOCH
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(super) struct RawIndexEntry {
    pub(super) id: RawNodeId,
    pub(super) timestamp: DateTime<Utc>,
}

impl From<&RawNode> for RawIndexEntry {
    fn from(node: &RawNode) -> Self {
        Self {
            id: node.id,
            timestamp: node.timestamp,
        }
    }
}

/// Returns `true` when the path is either absent or parses as JSON. This is a
/// cheap sanity probe used by `ensure_indexes` to detect a corrupted
/// materialized index without holding the index version marker hostage.
async fn path_index_is_parseable_or_missing(path: &Path) -> bool {
    match fs::read(path).await {
        Ok(payload) => serde_json::from_slice::<serde_json::Value>(&payload).is_ok(),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => true,
        Err(_) => false,
    }
}

/// Day bucket (`YYYYMMDD`, UTC) used to shard the global timeline. Lexical
/// ordering of these strings matches chronological ordering, so newest-first
/// shard iteration is just a reverse sort of the bucket names.
fn timeline_bucket(timestamp: DateTime<Utc>) -> String {
    timestamp.format("%Y%m%d").to_string()
}

/// Bucket name for an embedding shard. A session id renders as its UUID string
/// (filesystem-safe); session-less / legacy entries use the reserved `none`
/// bucket (UUIDs never render as the literal "none", so there is no collision).
fn embedding_shard_bucket(session_id: Option<&SessionId>) -> String {
    session_id.map_or_else(|| "none".to_string(), ToString::to_string)
}

fn sort_raw_index_entries(entries: &mut [RawIndexEntry]) {
    entries.sort_by(|left, right| {
        left.timestamp
            .cmp(&right.timestamp)
            .then_with(|| left.id.cmp(&right.id))
    });
}

fn encode_operation_key(source: &str) -> String {
    uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, source.as_bytes()).to_string()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use super::{FileObjectStore, MutationJournal};
    use crate::domain::{DistillationState, RawNode, RawNodeKind};
    use crate::model::Embedding;
    use crate::storage::object_store::ObjectNodeRepository;
    use crate::storage::{NodeRepository, RawLifecyclePatch, VectorIndex};

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "takos-agent-engine-object-store-{name}-{}",
            uuid::Uuid::new_v4()
        ))
    }

    #[test]
    fn object_store_writes_metadata_file_on_open() -> crate::Result<()> {
        let root = temp_root("metadata");
        let store = FileObjectStore::open(&root)?;
        let payload = std::fs::read_to_string(store.root().join("store.json")).map_err(|err| {
            crate::EngineError::Storage(format!("failed to read object store metadata file: {err}"))
        })?;
        let metadata: Value = serde_json::from_str(&payload).map_err(|err| {
            crate::EngineError::Storage(format!(
                "failed to parse object store metadata file: {err}"
            ))
        })?;

        assert_eq!(metadata["format_version"], 1);
        assert!(metadata["created_at"].is_string());
        assert!(metadata["updated_at"].is_string());
        assert!(metadata["last_index_rebuild_at"].is_string());

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    // S1 regression: two `FileObjectStore` handles opened on the SAME root must
    // share one gate, so concurrent inserts serialize instead of racing the
    // shared store.json / index files. Without the shared-root lock these
    // interleaved read-modify-writes drop timeline entries and/or fail a run
    // with a rename ENOENT.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_stores_on_one_root_serialize() -> crate::Result<()> {
        let root = temp_root("concurrent");
        let session_id = crate::SessionId::new();
        let loop_id = crate::LoopId::new();

        // Two independent handles on the same directory, exactly like two runs.
        let repo_a = ObjectNodeRepository::new(FileObjectStore::open_async(&root).await?);
        let repo_b = ObjectNodeRepository::new(FileObjectStore::open_async(&root).await?);

        let total = 24usize;
        let mut handles = Vec::new();
        for index in 0..total {
            let repo = if index % 2 == 0 {
                repo_a.clone()
            } else {
                repo_b.clone()
            };
            handles.push(tokio::spawn(async move {
                let node = RawNode::text(
                    RawNodeKind::UserUtterance,
                    Some(session_id),
                    Some(loop_id),
                    "user",
                    format!("message {index}"),
                    0.5,
                    Vec::new(),
                );
                repo.insert_raw(node).await
            }));
        }
        for handle in handles {
            handle.await.expect("insert task panicked")?;
        }

        let reopened = ObjectNodeRepository::new(FileObjectStore::open_async(&root).await?);
        let timeline = reopened.timeline_raw(None, None, None, 1000).await?;
        assert_eq!(
            timeline.len(),
            total,
            "every concurrent insert must be present in the global timeline"
        );
        let session = reopened.session_raw(&session_id).await?;
        assert_eq!(session.len(), total);

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn recovery_journal_replays_raw_commit_before_index_validation() -> crate::Result<()> {
        let root = temp_root("journal-raw");
        let store = FileObjectStore::open_async(&root).await?;
        let session_id = crate::SessionId::new();
        let node = RawNode::text(
            RawNodeKind::Note,
            Some(session_id),
            Some(crate::LoopId::new()),
            "test",
            "journaled",
            0.5,
            Vec::new(),
        )
        .with_operation_key("journal-raw-operation");
        store
            .write_json(
                &store.journal_path(),
                &MutationJournal::CommitRaw {
                    node: node.clone(),
                    embedding: Embedding(vec![1.0, 0.0]),
                },
            )
            .await?;
        drop(store);

        let reopened = FileObjectStore::open_async(&root).await?;
        let repository = ObjectNodeRepository::new(reopened.clone());
        assert_eq!(
            repository
                .get_raw_by_operation_key("journal-raw-operation")
                .await?
                .map(|value| value.id),
            Some(node.id)
        );
        let vector = super::super::ObjectVectorIndex::new(reopened);
        assert_eq!(
            vector
                .search_raw(&Embedding(vec![1.0, 0.0]), 1, Some(&session_id))
                .await?
                .first()
                .map(|hit| hit.id),
            Some(node.id)
        );
        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn recovery_journal_finishes_partial_lifecycle_update() -> crate::Result<()> {
        let root = temp_root("journal-lifecycle");
        let store = FileObjectStore::open_async(&root).await?;
        let repository = ObjectNodeRepository::new(store.clone());
        let session_id = crate::SessionId::new();
        let loop_id = crate::LoopId::new();
        let first = RawNode::text(
            RawNodeKind::Note,
            Some(session_id),
            Some(loop_id),
            "test",
            "first",
            0.5,
            Vec::new(),
        );
        let second = RawNode::text(
            RawNodeKind::Note,
            Some(session_id),
            Some(loop_id),
            "test",
            "second",
            0.5,
            Vec::new(),
        );
        repository.insert_raw(first.clone()).await?;
        repository.insert_raw(second.clone()).await?;
        let patch = RawLifecyclePatch {
            distillation_state: Some(DistillationState::Distilled),
            overflow: None,
        };
        store
            .write_json(
                &store.journal_path(),
                &MutationJournal::UpdateRawLifecycle {
                    ids: vec![first.id, second.id],
                    patch: patch.clone(),
                },
            )
            .await?;
        let mut partially_updated = first.clone();
        partially_updated.distillation_state = DistillationState::Distilled;
        store
            .write_json(&store.raw_path(&first.id), &partially_updated)
            .await?;
        drop(repository);
        drop(store);

        let reopened = FileObjectStore::open_async(&root).await?;
        let repository = ObjectNodeRepository::new(reopened);
        for id in [first.id, second.id] {
            assert_eq!(
                repository
                    .get_raw(&id)
                    .await?
                    .expect("replayed raw")
                    .distillation_state,
                DistillationState::Distilled
            );
        }
        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn corrupt_body_is_quarantined_and_open_fails_closed() -> crate::Result<()> {
        let root = temp_root("quarantine");
        let store = FileObjectStore::open_async(&root).await?;
        let corrupt_path = store.raw_path(&crate::RawNodeId::new());
        std::fs::write(&corrupt_path, b"{not-json").map_err(|err| {
            crate::EngineError::Storage(format!("failed to stage corrupt body: {err}"))
        })?;
        drop(store);

        assert!(FileObjectStore::open_async(&root).await.is_err());
        let quarantined = std::fs::read_dir(root.join("quarantine"))
            .map_err(|err| crate::EngineError::Storage(err.to_string()))?
            .count();
        assert_eq!(quarantined, 1);
        FileObjectStore::open_async(&root).await?;
        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn corrupt_recovery_journal_remains_blocking() -> crate::Result<()> {
        let root = temp_root("corrupt-journal");
        let store = FileObjectStore::open_async(&root).await?;
        std::fs::write(store.journal_path(), b"{not-json").map_err(|err| {
            crate::EngineError::Storage(format!("failed to stage corrupt journal: {err}"))
        })?;
        drop(store);
        assert!(FileObjectStore::open_async(&root).await.is_err());
        assert!(root.join("journal/current.json").exists());
        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn open_reaps_abandoned_temporary_files() -> crate::Result<()> {
        let root = temp_root("temp-reaper");
        let store = FileObjectStore::open_async(&root).await?;
        let abandoned = root.join("raw/abandoned.tmp");
        std::fs::write(&abandoned, b"partial").map_err(|err| {
            crate::EngineError::Storage(format!("failed to stage temporary file: {err}"))
        })?;
        drop(store);
        FileObjectStore::open_async(&root).await?;
        assert!(!abandoned.exists());
        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn symlink_root_is_rejected() -> crate::Result<()> {
        let target = temp_root("symlink-target");
        let link = temp_root("symlink-link");
        std::fs::create_dir_all(&target)
            .map_err(|err| crate::EngineError::Storage(err.to_string()))?;
        std::os::unix::fs::symlink(&target, &link)
            .map_err(|err| crate::EngineError::Storage(err.to_string()))?;
        assert!(FileObjectStore::open(&link).is_err());
        let _ = std::fs::remove_file(link);
        let _ = std::fs::remove_dir_all(target);
        Ok(())
    }
}
