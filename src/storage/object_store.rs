use std::cmp::Ordering;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use tokio::fs;
use tokio::sync::{Mutex, MutexGuard};

use crate::domain::{AbstractNode, DistillationState, LoopState, RawNode};
use crate::error::{EngineError, Result};
use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};
use crate::model::embedding::{cosine_similarity, Embedding};

use super::traits::{
    GraphRepository, GraphTraversalHit, LoopStateRepository, NodeRepository, RawLifecyclePatch,
    ScoredAbstractRef, ScoredRawRef, VectorIndex,
};

const STORE_FORMAT_VERSION: u32 = 1;
/// Bumped whenever the on-disk index layout written by
/// `rebuild_indexes_unlocked` changes. `open()` compares this with the value
/// stored in `<root>/indexes/.index-version` and skips the (expensive) full
/// rebuild whenever they match.
const INDEX_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct FileObjectStore {
    root: PathBuf,
    gate: Arc<Mutex<()>>,
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
        let store = Self {
            root: root.as_ref().to_path_buf(),
            gate: Arc::new(Mutex::new(())),
        };
        store.ensure_layout().await?;
        store.ensure_indexes().await?;
        Ok(store)
    }

    #[must_use]
    pub fn root(&self) -> &Path {
        &self.root
    }

    async fn lock(&self) -> MutexGuard<'_, ()> {
        self.gate.lock().await
    }

    async fn ensure_layout(&self) -> Result<()> {
        for directory in [
            self.raw_dir(),
            self.abstract_dir(),
            self.raw_embedding_dir(),
            self.abstract_embedding_dir(),
            self.graph_dir(),
            self.checkpoint_dir(),
        ] {
            fs::create_dir_all(&directory).await.map_err(|err| {
                EngineError::Storage(format!(
                    "failed to create object store directory {}: {err}",
                    directory.display()
                ))
            })?;
        }
        self.ensure_index_layout().await
    }

    async fn ensure_index_layout(&self) -> Result<()> {
        for directory in [
            self.raw_operation_dir(),
            self.abstract_operation_dir(),
            self.session_index_dir(),
            self.loop_index_dir(),
            self.timeline_index_dir(),
            self.backlog_index_dir(),
            self.vector_index_dir(),
        ] {
            fs::create_dir_all(&directory).await.map_err(|err| {
                EngineError::Storage(format!(
                    "failed to create object store index directory {}: {err}",
                    directory.display()
                ))
            })?;
        }
        Ok(())
    }

    async fn ensure_indexes(&self) -> Result<()> {
        let _guard = self.lock().await;
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
        let timeline = self
            .read_raw_index_entries_unlocked(&self.raw_timeline_path())
            .await?;
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
    /// upserting the id into the vector manifest
    /// (`indexes/vector/{raw,abstract}_embeddings.json`). The body is the source
    /// of truth and `rebuild_indexes_unlocked` regenerates both manifests from a
    /// body scan (`read_embedding_ids_scan_unlocked` ->
    /// `write_manifest_unlocked`). If the process is SIGKILLed/OOM-killed in that
    /// window the body exists but is absent from the manifest, so `search_raw` /
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
    /// fast path, preserving the cheap-repeated-open property for the common
    /// (consistent) case.
    async fn embedding_indexes_are_complete(&self) -> Result<bool> {
        if !self
            .embedding_dir_covered_by_manifest::<RawNodeId>(
                &self.raw_embedding_dir(),
                &self.raw_embedding_manifest_path(),
            )
            .await?
        {
            return Ok(false);
        }
        self.embedding_dir_covered_by_manifest::<AbstractNodeId>(
            &self.abstract_embedding_dir(),
            &self.abstract_embedding_manifest_path(),
        )
        .await
    }

    /// Returns `true` when every `*.json` body in `embedding_dir` has its id
    /// (the file stem) present in the manifest at `manifest_path`. Empty body
    /// directories are vacuously complete. An unparseable body stem is treated
    /// as a mismatch so a rebuild re-derives a clean manifest from the bodies.
    async fn embedding_dir_covered_by_manifest<Id>(
        &self,
        embedding_dir: &Path,
        manifest_path: &Path,
    ) -> Result<bool>
    where
        Id: DeserializeOwned + FromStr + Eq + std::hash::Hash,
    {
        let body_paths = self.list_paths(embedding_dir).await?;
        if body_paths.is_empty() {
            // No embedding bodies on disk: nothing can be orphaned from the
            // manifest.
            return Ok(true);
        }
        let manifest_ids: HashSet<Id> = self
            .read_manifest_unlocked::<Id>(manifest_path)
            .await?
            .into_iter()
            .collect();
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
                Some(id) if manifest_ids.contains(&id) => {}
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
            self.raw_timeline_path(),
            self.undistilled_index_path(),
            self.pushed_undistilled_index_path(),
            self.raw_embedding_manifest_path(),
            self.abstract_embedding_manifest_path(),
        ] {
            if !path_index_is_parseable_or_missing(&path).await {
                return false;
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

        let mut global_timeline = Vec::new();
        let mut session_timelines: HashMap<SessionId, Vec<RawIndexEntry>> = HashMap::new();
        let mut loop_timelines: HashMap<LoopId, Vec<RawIndexEntry>> = HashMap::new();
        let mut undistilled = Vec::new();
        let mut pushed_undistilled = Vec::new();

        for node in raw_nodes {
            let entry = RawIndexEntry::from(&node);
            global_timeline.push(entry.clone());
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
                    &StoredId { id: node.id },
                )
                .await?;
            }
        }

        sort_raw_index_entries(&mut global_timeline);
        self.write_json(&self.raw_timeline_path(), &global_timeline)
            .await?;
        sort_raw_index_entries(&mut undistilled);
        self.write_json(&self.undistilled_index_path(), &undistilled)
            .await?;
        sort_raw_index_entries(&mut pushed_undistilled);
        self.write_json(&self.pushed_undistilled_index_path(), &pushed_undistilled)
            .await?;

        for (session_id, mut entries) in session_timelines {
            sort_raw_index_entries(&mut entries);
            self.write_json(&self.session_index_path(&session_id), &entries)
                .await?;
        }
        for (loop_id, mut entries) in loop_timelines {
            sort_raw_index_entries(&mut entries);
            self.write_json(&self.loop_index_path(&loop_id), &entries)
                .await?;
        }

        for node in abstract_nodes {
            if let Some(operation_key) = &node.operation_key {
                self.write_json(
                    &self.abstract_operation_path(operation_key),
                    &StoredId { id: node.id },
                )
                .await?;
            }
        }

        let raw_embedding_ids = self
            .read_embedding_ids_scan_unlocked::<RawNodeId>(&self.raw_embedding_dir())
            .await?;
        self.write_manifest_unlocked(&self.raw_embedding_manifest_path(), &raw_embedding_ids)
            .await?;
        let abstract_embedding_ids = self
            .read_embedding_ids_scan_unlocked::<AbstractNodeId>(&self.abstract_embedding_dir())
            .await?;
        self.write_manifest_unlocked(
            &self.abstract_embedding_manifest_path(),
            &abstract_embedding_ids,
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

    fn raw_operation_dir(&self) -> PathBuf {
        self.index_root().join("raw_operation")
    }

    fn abstract_operation_dir(&self) -> PathBuf {
        self.index_root().join("abstract_operation")
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

    fn raw_path(&self, id: &RawNodeId) -> PathBuf {
        self.raw_dir().join(format!("{id}.json"))
    }

    fn abstract_path(&self, id: &AbstractNodeId) -> PathBuf {
        self.abstract_dir().join(format!("{id}.json"))
    }

    fn raw_embedding_path(&self, id: &RawNodeId) -> PathBuf {
        self.raw_embedding_dir().join(format!("{id}.json"))
    }

    fn abstract_embedding_path(&self, id: &AbstractNodeId) -> PathBuf {
        self.abstract_embedding_dir().join(format!("{id}.json"))
    }

    fn graph_path(&self, id: &AbstractNodeId) -> PathBuf {
        self.graph_dir().join(format!("{id}.json"))
    }

    fn checkpoint_path(&self, session_id: &SessionId, loop_id: &LoopId) -> PathBuf {
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

    async fn touch_metadata_unlocked(&self) -> Result<()> {
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

    fn raw_operation_path(&self, operation_key: &str) -> PathBuf {
        self.raw_operation_dir()
            .join(format!("{}.json", encode_operation_key(operation_key)))
    }

    fn abstract_operation_path(&self, operation_key: &str) -> PathBuf {
        self.abstract_operation_dir()
            .join(format!("{}.json", encode_operation_key(operation_key)))
    }

    fn session_index_path(&self, session_id: &SessionId) -> PathBuf {
        self.session_index_dir().join(format!("{session_id}.json"))
    }

    fn loop_index_path(&self, loop_id: &LoopId) -> PathBuf {
        self.loop_index_dir().join(format!("{loop_id}.json"))
    }

    fn raw_timeline_path(&self) -> PathBuf {
        self.timeline_index_dir().join("raw.json")
    }

    fn undistilled_index_path(&self) -> PathBuf {
        self.backlog_index_dir().join("undistilled_raw.json")
    }

    fn pushed_undistilled_index_path(&self) -> PathBuf {
        self.backlog_index_dir().join("pushed_undistilled_raw.json")
    }

    fn raw_embedding_manifest_path(&self) -> PathBuf {
        self.vector_index_dir().join("raw_embeddings.json")
    }

    fn abstract_embedding_manifest_path(&self) -> PathBuf {
        self.vector_index_dir().join("abstract_embeddings.json")
    }

    #[allow(clippy::unused_self)] // grouped with FileObjectStore for cohesion
    async fn write_json<T: Serialize>(&self, path: &Path, value: &T) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await.map_err(|err| {
                EngineError::Storage(format!(
                    "failed to create parent directory {}: {err}",
                    parent.display()
                ))
            })?;
        }
        let payload = serde_json::to_vec_pretty(value).map_err(|err| {
            EngineError::Storage(format!(
                "failed to serialize object store payload {}: {err}",
                path.display()
            ))
        })?;
        let temporary = path.with_extension("tmp");
        fs::write(&temporary, payload).await.map_err(|err| {
            EngineError::Storage(format!(
                "failed to write temporary object {}: {err}",
                temporary.display()
            ))
        })?;
        fs::rename(&temporary, path).await.map_err(|err| {
            EngineError::Storage(format!(
                "failed to move object {} into place: {err}",
                path.display()
            ))
        })?;
        Ok(())
    }

    #[allow(clippy::unused_self)] // grouped with FileObjectStore for cohesion
    async fn read_json<T: DeserializeOwned>(&self, path: &Path) -> Result<T> {
        let payload = fs::read(path).await.map_err(|err| {
            EngineError::Storage(format!("failed to read object {}: {err}", path.display()))
        })?;
        serde_json::from_slice(&payload).map_err(|err| {
            EngineError::Storage(format!(
                "failed to deserialize object {}: {err}",
                path.display()
            ))
        })
    }

    async fn try_read_json<T: DeserializeOwned>(&self, path: &Path) -> Result<Option<T>> {
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
    async fn remove_file_if_exists(&self, path: &Path) -> Result<()> {
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

    async fn read_raw_index_entries_unlocked(&self, path: &Path) -> Result<Vec<RawIndexEntry>> {
        Ok(self
            .try_read_json::<Vec<RawIndexEntry>>(path)
            .await?
            .unwrap_or_default())
    }

    async fn write_manifest_unlocked<Id>(&self, path: &Path, ids: &[Id]) -> Result<()>
    where
        Id: Serialize + Clone + Ord,
    {
        let mut ids = ids.to_vec();
        ids.sort();
        ids.dedup();
        self.write_json(path, &IdManifest { ids }).await
    }

    async fn read_manifest_unlocked<Id>(&self, path: &Path) -> Result<Vec<Id>>
    where
        Id: DeserializeOwned,
    {
        Ok(self
            .try_read_json::<IdManifest<Id>>(path)
            .await?
            .map(|manifest| manifest.ids)
            .unwrap_or_default())
    }

    async fn upsert_manifest_id_unlocked<Id>(&self, path: &Path, id: Id) -> Result<()>
    where
        Id: Serialize + DeserializeOwned + Copy + Ord,
    {
        let mut ids = self.read_manifest_unlocked::<Id>(path).await?;
        if !ids.contains(&id) {
            ids.push(id);
        }
        self.write_manifest_unlocked(path, &ids).await
    }

    async fn read_embedding_ids_scan_unlocked<Id>(&self, directory: &Path) -> Result<Vec<Id>>
    where
        Id: DeserializeOwned + Ord + Copy,
    {
        let mut ids = Vec::new();
        for path in self.list_paths(directory).await? {
            let record: StoredEmbedding<Id> = self.read_json(&path).await?;
            ids.push(record.id);
        }
        ids.sort();
        ids.dedup();
        Ok(ids)
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

    async fn remove_raw_index_entry_unlocked(&self, path: &Path, id: RawNodeId) -> Result<()> {
        let mut entries = self.read_raw_index_entries_unlocked(path).await?;
        let before = entries.len();
        entries.retain(|entry| entry.id != id);
        if before != entries.len() {
            self.write_json(path, &entries).await?;
        }
        Ok(())
    }

    async fn sync_raw_indexes_unlocked(&self, node: &RawNode) -> Result<()> {
        let entry = RawIndexEntry::from(node);
        self.upsert_raw_index_entry_unlocked(&self.raw_timeline_path(), entry.clone())
            .await?;
        if let Some(session_id) = node.session_id {
            self.upsert_raw_index_entry_unlocked(
                &self.session_index_path(&session_id),
                entry.clone(),
            )
            .await?;
        }
        if let Some(loop_id) = node.loop_id {
            self.upsert_raw_index_entry_unlocked(&self.loop_index_path(&loop_id), entry.clone())
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
            self.upsert_raw_index_entry_unlocked(&self.pushed_undistilled_index_path(), entry)
                .await?;
        } else {
            self.remove_raw_index_entry_unlocked(&self.pushed_undistilled_index_path(), node.id)
                .await?;
        }

        Ok(())
    }

    async fn read_raw_by_ids_unlocked(&self, ids: &[RawNodeId]) -> Result<Vec<RawNode>> {
        let mut nodes = Vec::new();
        for id in ids {
            if let Some(node) = self.try_read_json::<RawNode>(&self.raw_path(id)).await? {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }
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
struct StoredId<Id> {
    id: Id,
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
struct StoredEmbedding<Id> {
    id: Id,
    embedding: Embedding,
    /// Session id this embedding was indexed for. `None` represents either
    /// (a) legacy entries written before the session-aware index existed, or
    /// (b) intentionally session-less indexing. The search filter treats
    /// `None` entries as legacy: they are only returned when the search
    /// itself passes `session_id = None`, never when a specific session is
    /// requested. The field is `#[serde(default)]` so existing on-disk JSON
    /// without the field deserializes cleanly into the legacy bucket.
    #[serde(default)]
    session_id: Option<SessionId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct RawIndexEntry {
    id: RawNodeId,
    timestamp: DateTime<Utc>,
}

impl From<&RawNode> for RawIndexEntry {
    fn from(node: &RawNode) -> Self {
        Self {
            id: node.id,
            timestamp: node.timestamp,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
struct GraphEdge {
    to: AbstractNodeId,
    predicate: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredGraphEdges {
    node_id: AbstractNodeId,
    edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone)]
pub struct ObjectNodeRepository {
    store: FileObjectStore,
}

impl ObjectNodeRepository {
    #[must_use]
    pub const fn new(store: FileObjectStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl NodeRepository for ObjectNodeRepository {
    async fn insert_raw(&self, node: RawNode) -> Result<()> {
        let _guard = self.store.lock().await;
        // Cross-file write ordering invariant: the durable raw node body MUST be
        // written (tmp + atomic rename) BEFORE the retrieval indexes are synced.
        // The body is the source of truth; `FileObjectStore::indexes_are_complete`
        // reconciles index coverage against the bodies on open and forces a
        // rebuild if any body is missing from the timeline. If this order were
        // reversed (index first) a mid-insert crash would instead leave an index
        // entry pointing at a non-existent body — a worse, self-healing-free
        // state. Do not reorder these writes.
        self.store
            .write_json(&self.store.raw_path(&node.id), &node)
            .await?;
        if let Some(operation_key) = &node.operation_key {
            self.store
                .write_json(
                    &self.store.raw_operation_path(operation_key),
                    &StoredId { id: node.id },
                )
                .await?;
        }
        self.store.sync_raw_indexes_unlocked(&node).await?;
        self.store.touch_metadata_unlocked().await
    }

    async fn insert_abstract(&self, node: AbstractNode) -> Result<()> {
        let _guard = self.store.lock().await;
        self.store
            .write_json(&self.store.abstract_path(&node.id), &node)
            .await?;
        if let Some(operation_key) = &node.operation_key {
            self.store
                .write_json(
                    &self.store.abstract_operation_path(operation_key),
                    &StoredId { id: node.id },
                )
                .await?;
        }
        self.store.touch_metadata_unlocked().await
    }

    async fn get_raw(&self, id: &RawNodeId) -> Result<Option<RawNode>> {
        let _guard = self.store.lock().await;
        self.store.try_read_json(&self.store.raw_path(id)).await
    }

    async fn get_abstract(&self, id: &AbstractNodeId) -> Result<Option<AbstractNode>> {
        let _guard = self.store.lock().await;
        self.store
            .try_read_json(&self.store.abstract_path(id))
            .await
    }

    async fn get_raw_by_operation_key(&self, operation_key: &str) -> Result<Option<RawNode>> {
        let _guard = self.store.lock().await;
        match self
            .store
            .try_read_json::<StoredId<RawNodeId>>(&self.store.raw_operation_path(operation_key))
            .await?
        {
            Some(record) => {
                self.store
                    .try_read_json(&self.store.raw_path(&record.id))
                    .await
            }
            None => Ok(None),
        }
    }

    async fn get_abstract_by_operation_key(
        &self,
        operation_key: &str,
    ) -> Result<Option<AbstractNode>> {
        let _guard = self.store.lock().await;
        match self
            .store
            .try_read_json::<StoredId<AbstractNodeId>>(
                &self.store.abstract_operation_path(operation_key),
            )
            .await?
        {
            Some(record) => {
                self.store
                    .try_read_json(&self.store.abstract_path(&record.id))
                    .await
            }
            None => Ok(None),
        }
    }

    async fn list_raw(&self, ids: &[RawNodeId]) -> Result<Vec<RawNode>> {
        let _guard = self.store.lock().await;
        self.store.read_raw_by_ids_unlocked(ids).await
    }

    async fn list_abstract(&self, ids: &[AbstractNodeId]) -> Result<Vec<AbstractNode>> {
        let _guard = self.store.lock().await;
        let mut nodes = Vec::new();
        for id in ids {
            if let Some(node) = self
                .store
                .try_read_json::<AbstractNode>(&self.store.abstract_path(id))
                .await?
            {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }

    async fn recent_session_raw(
        &self,
        session_id: &SessionId,
        limit: usize,
    ) -> Result<Vec<RawNode>> {
        let _guard = self.store.lock().await;
        let mut entries = self
            .store
            .read_raw_index_entries_unlocked(&self.store.session_index_path(session_id))
            .await?;
        entries.reverse();
        entries.truncate(limit);
        entries.reverse();
        let ids: Vec<_> = entries.into_iter().map(|entry| entry.id).collect();
        self.store.read_raw_by_ids_unlocked(&ids).await
    }

    async fn session_raw(&self, session_id: &SessionId) -> Result<Vec<RawNode>> {
        let _guard = self.store.lock().await;
        let entries = self
            .store
            .read_raw_index_entries_unlocked(&self.store.session_index_path(session_id))
            .await?;
        let ids: Vec<_> = entries.into_iter().map(|entry| entry.id).collect();
        self.store.read_raw_by_ids_unlocked(&ids).await
    }

    async fn raw_for_loop(&self, loop_id: &LoopId) -> Result<Vec<RawNode>> {
        let _guard = self.store.lock().await;
        let entries = self
            .store
            .read_raw_index_entries_unlocked(&self.store.loop_index_path(loop_id))
            .await?;
        let ids: Vec<_> = entries.into_iter().map(|entry| entry.id).collect();
        self.store.read_raw_by_ids_unlocked(&ids).await
    }

    async fn timeline_raw(
        &self,
        session_id: Option<&SessionId>,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<RawNode>> {
        let _guard = self.store.lock().await;
        let index_path = session_id.map_or_else(
            || self.store.raw_timeline_path(),
            |value| self.store.session_index_path(value),
        );
        let mut entries = self
            .store
            .read_raw_index_entries_unlocked(&index_path)
            .await?;
        entries.retain(|entry| from.is_none_or(|value| entry.timestamp >= value));
        entries.retain(|entry| to.is_none_or(|value| entry.timestamp <= value));
        entries.sort_by(|left, right| {
            right
                .timestamp
                .cmp(&left.timestamp)
                .then_with(|| left.id.cmp(&right.id))
        });
        entries.truncate(limit);
        let ids: Vec<_> = entries.into_iter().map(|entry| entry.id).collect();
        self.store.read_raw_by_ids_unlocked(&ids).await
    }

    async fn update_raw_lifecycle(
        &self,
        ids: &[RawNodeId],
        patch: &RawLifecyclePatch,
    ) -> Result<()> {
        let _guard = self.store.lock().await;
        let mut changed = false;
        for id in ids {
            if let Some(mut node) = self
                .store
                .try_read_json::<RawNode>(&self.store.raw_path(id))
                .await?
            {
                if let Some(distillation_state) = &patch.distillation_state {
                    node.distillation_state = distillation_state.clone();
                }
                if let Some(overflow) = &patch.overflow {
                    node.overflow = overflow.clone();
                }
                // Same body-before-indexes ordering invariant as `insert_raw`:
                // persist the updated body first so a crash before the index
                // sync is reconciled (rebuilt from the body) on the next open.
                self.store
                    .write_json(&self.store.raw_path(id), &node)
                    .await?;
                self.store.sync_raw_indexes_unlocked(&node).await?;
                changed = true;
            }
        }
        if changed {
            self.store.touch_metadata_unlocked().await?;
        }
        Ok(())
    }

    async fn undistilled_raw(&self, limit: usize, only_pushed_out: bool) -> Result<Vec<RawNode>> {
        let _guard = self.store.lock().await;
        let index_path = if only_pushed_out {
            self.store.pushed_undistilled_index_path()
        } else {
            self.store.undistilled_index_path()
        };
        let mut entries = self
            .store
            .read_raw_index_entries_unlocked(&index_path)
            .await?;
        entries.truncate(limit);
        let ids: Vec<_> = entries.into_iter().map(|entry| entry.id).collect();
        self.store.read_raw_by_ids_unlocked(&ids).await
    }
}

#[derive(Debug, Clone)]
pub struct ObjectVectorIndex {
    store: FileObjectStore,
}

impl ObjectVectorIndex {
    #[must_use]
    pub const fn new(store: FileObjectStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl VectorIndex for ObjectVectorIndex {
    async fn index_raw(&self, id: RawNodeId, embedding: Embedding) -> Result<()> {
        self.index_raw_with_session(id, embedding, None).await
    }

    async fn index_abstract(&self, id: AbstractNodeId, embedding: Embedding) -> Result<()> {
        self.index_abstract_with_session(id, embedding, None).await
    }

    async fn index_raw_with_session(
        &self,
        id: RawNodeId,
        embedding: Embedding,
        session_id: Option<SessionId>,
    ) -> Result<()> {
        let _guard = self.store.lock().await;
        self.store
            .write_json(
                &self.store.raw_embedding_path(&id),
                &StoredEmbedding {
                    id,
                    embedding,
                    session_id,
                },
            )
            .await?;
        self.store
            .upsert_manifest_id_unlocked(&self.store.raw_embedding_manifest_path(), id)
            .await?;
        self.store.touch_metadata_unlocked().await
    }

    async fn index_abstract_with_session(
        &self,
        id: AbstractNodeId,
        embedding: Embedding,
        session_id: Option<SessionId>,
    ) -> Result<()> {
        let _guard = self.store.lock().await;
        self.store
            .write_json(
                &self.store.abstract_embedding_path(&id),
                &StoredEmbedding {
                    id,
                    embedding,
                    session_id,
                },
            )
            .await?;
        self.store
            .upsert_manifest_id_unlocked(&self.store.abstract_embedding_manifest_path(), id)
            .await?;
        self.store.touch_metadata_unlocked().await
    }

    async fn search_raw(
        &self,
        query: &Embedding,
        top_k: usize,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<ScoredRawRef>> {
        let _guard = self.store.lock().await;
        let mut scored = Vec::new();
        for id in self
            .store
            .read_manifest_unlocked::<RawNodeId>(&self.store.raw_embedding_manifest_path())
            .await?
        {
            if let Some(record) = self
                .store
                .try_read_json::<StoredEmbedding<RawNodeId>>(&self.store.raw_embedding_path(&id))
                .await?
            {
                if !object_store_session_matches(record.session_id.as_ref(), session_id) {
                    continue;
                }
                scored.push(ScoredRawRef {
                    id: record.id,
                    score: cosine_similarity(query, &record.embedding),
                });
            }
        }
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.id.cmp(&right.id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn search_abstract(
        &self,
        query: &Embedding,
        top_k: usize,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<ScoredAbstractRef>> {
        let _guard = self.store.lock().await;
        let mut scored = Vec::new();
        for id in self
            .store
            .read_manifest_unlocked::<AbstractNodeId>(
                &self.store.abstract_embedding_manifest_path(),
            )
            .await?
        {
            if let Some(record) = self
                .store
                .try_read_json::<StoredEmbedding<AbstractNodeId>>(
                    &self.store.abstract_embedding_path(&id),
                )
                .await?
            {
                if !object_store_session_matches(record.session_id.as_ref(), session_id) {
                    continue;
                }
                scored.push(ScoredAbstractRef {
                    id: record.id,
                    score: cosine_similarity(query, &record.embedding),
                });
            }
        }
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.id.cmp(&right.id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

/// Filter helper for the file-backed vector index.
///
/// Mirrors `entry_matches_session_filter` in `vector.rs`:
/// - `Some(stored) == Some(requested)` matches.
/// - Both `None` matches (= legacy entry returned for legacy / session-less
///   searches).
/// - Anything else does not match (no cross-session leakage, no legacy entry
///   returned when a specific session is requested).
fn object_store_session_matches(stored: Option<&SessionId>, requested: Option<&SessionId>) -> bool {
    match (stored, requested) {
        (None, None) => true,
        (Some(stored), Some(requested)) => stored == requested,
        _ => false,
    }
}

#[derive(Debug, Clone)]
pub struct ObjectGraphRepository {
    store: FileObjectStore,
}

impl ObjectGraphRepository {
    #[must_use]
    pub const fn new(store: FileObjectStore) -> Self {
        Self { store }
    }

    fn edges_for_abstract(node: &AbstractNode) -> Vec<GraphEdge> {
        let mut edges = HashSet::new();
        for abstract_id in &node.references.abstract_node_ids {
            edges.insert(GraphEdge {
                to: *abstract_id,
                predicate: "reference".to_string(),
            });
        }
        for relation in &node.graph.relations {
            if let Ok(abstract_id) = AbstractNodeId::from_str(&relation.object) {
                edges.insert(GraphEdge {
                    to: abstract_id,
                    predicate: relation.predicate.clone(),
                });
            }
        }
        let mut edges: Vec<_> = edges.into_iter().collect();
        edges.sort_by(|left, right| {
            left.predicate
                .cmp(&right.predicate)
                .then_with(|| left.to.cmp(&right.to))
        });
        edges
    }
}

#[async_trait]
impl GraphRepository for ObjectGraphRepository {
    async fn index_abstract(&self, node: &AbstractNode) -> Result<()> {
        let _guard = self.store.lock().await;
        self.store
            .write_json(
                &self.store.graph_path(&node.id),
                &StoredGraphEdges {
                    node_id: node.id,
                    edges: Self::edges_for_abstract(node),
                },
            )
            .await?;
        self.store.touch_metadata_unlocked().await
    }

    async fn traverse(
        &self,
        start: &AbstractNodeId,
        max_depth: usize,
        relation_types: Option<&[String]>,
    ) -> Result<Vec<GraphTraversalHit>> {
        let _guard = self.store.lock().await;
        let filters = relation_types.map(|values| values.iter().cloned().collect::<HashSet<_>>());
        let mut visited = HashSet::new();
        let mut queue = VecDeque::from([(*start, 0usize, None::<String>)]);
        let mut output = Vec::new();

        while let Some((current, depth, via_predicate)) = queue.pop_front() {
            if depth > max_depth || !visited.insert(current) {
                continue;
            }
            output.push(GraphTraversalHit {
                node_id: current,
                depth,
                via_predicate: via_predicate.clone(),
            });
            if let Some(record) = self
                .store
                .try_read_json::<StoredGraphEdges>(&self.store.graph_path(&current))
                .await?
            {
                for edge in record.edges {
                    if let Some(filters) = &filters {
                        if !filters.contains(&edge.predicate) {
                            continue;
                        }
                    }
                    queue.push_back((edge.to, depth + 1, Some(edge.predicate)));
                }
            }
        }

        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct ObjectLoopStateRepository {
    store: FileObjectStore,
}

impl ObjectLoopStateRepository {
    #[must_use]
    pub const fn new(store: FileObjectStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl LoopStateRepository for ObjectLoopStateRepository {
    async fn save_checkpoint(&self, state: LoopState) -> Result<()> {
        let _guard = self.store.lock().await;
        self.store
            .write_json(
                &self
                    .store
                    .checkpoint_path(&state.session_id, &state.loop_id),
                &state,
            )
            .await?;
        self.store.touch_metadata_unlocked().await
    }

    async fn load_checkpoint(
        &self,
        session_id: &SessionId,
        loop_id: &LoopId,
    ) -> Result<Option<LoopState>> {
        let _guard = self.store.lock().await;
        self.store
            .try_read_json(&self.store.checkpoint_path(session_id, loop_id))
            .await
    }

    async fn clear_checkpoint(&self, session_id: &SessionId, loop_id: &LoopId) -> Result<()> {
        let _guard = self.store.lock().await;
        self.store
            .remove_file_if_exists(&self.store.checkpoint_path(session_id, loop_id))
            .await?;
        self.store.touch_metadata_unlocked().await
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

fn sort_raw_index_entries(entries: &mut [RawIndexEntry]) {
    entries.sort_by(|left, right| {
        left.timestamp
            .cmp(&right.timestamp)
            .then_with(|| left.id.cmp(&right.id))
    });
}

fn encode_operation_key(source: &str) -> String {
    let mut encoded = String::with_capacity(source.len() * 2);
    for byte in source.bytes() {
        encoded.push(hex_char(byte >> 4));
        encoded.push(hex_char(byte & 0x0f));
    }
    encoded
}

fn hex_char(value: u8) -> char {
    match value {
        0..=9 => (b'0' + value) as char,
        10..=15 => (b'a' + (value - 10)) as char,
        _ => unreachable!("hex nibble must be between 0 and 15"),
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use chrono::{Duration as ChronoDuration, Utc};
    use serde_json::Value;

    use crate::domain::{OverflowPolicy, RawNode, RawNodeKind};
    use crate::model::embedding::Embedding;

    use super::{FileObjectStore, ObjectNodeRepository, ObjectVectorIndex, RawLifecyclePatch};
    use crate::storage::traits::{NodeRepository, VectorIndex};

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

    #[tokio::test]
    async fn object_store_rebuilds_materialized_indexes_on_open() -> crate::Result<()> {
        let root = temp_root("rebuild");
        let session_id = crate::SessionId::new();
        let loop_id = crate::LoopId::new();

        let store = FileObjectStore::open(&root)?;
        let repository = ObjectNodeRepository::new(store.clone());
        let vector_index = ObjectVectorIndex::new(store.clone());

        let mut older = RawNode::text(
            RawNodeKind::UserUtterance,
            Some(session_id),
            Some(loop_id),
            "user",
            "older",
            0.5,
            Vec::new(),
        );
        older.timestamp = Utc::now() - ChronoDuration::minutes(1);
        repository.insert_raw(older.clone()).await?;
        vector_index
            .index_raw(older.id, Embedding(vec![1.0, 0.0]))
            .await?;

        let newer = RawNode::text(
            RawNodeKind::AssistantUtterance,
            Some(session_id),
            Some(loop_id),
            "assistant",
            "newer",
            0.6,
            Vec::new(),
        );
        repository.insert_raw(newer.clone()).await?;
        vector_index
            .index_raw(newer.id, Embedding(vec![0.9, 0.1]))
            .await?;

        std::fs::remove_dir_all(root.join("indexes")).map_err(|err| {
            crate::EngineError::Storage(format!(
                "failed to remove object store indexes for rebuild test: {err}"
            ))
        })?;

        let reopened = FileObjectStore::open(&root)?;
        let repository = ObjectNodeRepository::new(reopened.clone());
        let vector_index = ObjectVectorIndex::new(reopened);

        let session_nodes = repository.session_raw(&session_id).await?;
        assert_eq!(session_nodes.len(), 2);
        assert_eq!(session_nodes[0].id, older.id);
        assert_eq!(session_nodes[1].id, newer.id);

        let recent = repository.timeline_raw(None, None, None, 1).await?;
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].id, newer.id);

        let scored = vector_index
            .search_raw(&Embedding(vec![1.0, 0.0]), 2, None)
            .await?;
        assert_eq!(scored.len(), 2);
        assert_eq!(scored[0].id, older.id);

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn object_store_recovers_from_corrupted_index_payloads_on_open() -> crate::Result<()> {
        let root = temp_root("corrupt-index");
        let session_id = crate::SessionId::new();
        let loop_id = crate::LoopId::new();

        let store = FileObjectStore::open(&root)?;
        let repository = ObjectNodeRepository::new(store.clone());
        let vector_index = ObjectVectorIndex::new(store.clone());

        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            Some(session_id),
            Some(loop_id),
            "user",
            "recover me",
            0.5,
            Vec::new(),
        );
        repository.insert_raw(node.clone()).await?;
        vector_index
            .index_raw(node.id, Embedding(vec![1.0, 0.0]))
            .await?;

        std::fs::write(
            root.join("indexes").join("timeline").join("raw.json"),
            b"{ broken json",
        )
        .map_err(|err| {
            crate::EngineError::Storage(format!(
                "failed to corrupt object store index for test: {err}"
            ))
        })?;

        let reopened = FileObjectStore::open(&root)?;
        let repository = ObjectNodeRepository::new(reopened.clone());
        let vector_index = ObjectVectorIndex::new(reopened);

        let recovered = repository.timeline_raw(None, None, None, 8).await?;
        assert_eq!(recovered.len(), 1);
        assert_eq!(recovered[0].id, node.id);

        let scored = vector_index
            .search_raw(&Embedding(vec![1.0, 0.0]), 1, None)
            .await?;
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].id, node.id);

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn object_store_vector_index_filters_by_session_id() -> crate::Result<()> {
        let root = temp_root("vector-session-filter");
        let session_a = crate::SessionId::new();
        let session_b = crate::SessionId::new();
        let store = FileObjectStore::open(&root)?;
        let vector_index = ObjectVectorIndex::new(store);

        let raw_a = crate::ids::RawNodeId::new();
        let raw_b = crate::ids::RawNodeId::new();
        let raw_legacy = crate::ids::RawNodeId::new();

        vector_index
            .index_raw_with_session(raw_a, Embedding(vec![1.0, 0.0]), Some(session_a))
            .await?;
        vector_index
            .index_raw_with_session(raw_b, Embedding(vec![1.0, 0.0]), Some(session_b))
            .await?;
        // Legacy entry indexed via the no-session path.
        vector_index
            .index_raw(raw_legacy, Embedding(vec![1.0, 0.0]))
            .await?;

        let scoped_a = vector_index
            .search_raw(&Embedding(vec![1.0, 0.0]), 8, Some(&session_a))
            .await?;
        assert_eq!(scoped_a.len(), 1);
        assert_eq!(scoped_a[0].id, raw_a);

        let legacy_only = vector_index
            .search_raw(&Embedding(vec![1.0, 0.0]), 8, None)
            .await?;
        assert_eq!(legacy_only.len(), 1);
        assert_eq!(legacy_only[0].id, raw_legacy);

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn object_store_vector_index_backfills_legacy_payloads_without_session_field(
    ) -> crate::Result<()> {
        // A `StoredEmbedding` JSON payload that pre-dates the session_id
        // field must still load cleanly and be returned by legacy (= None)
        // searches, but never by session-scoped searches.
        let root = temp_root("vector-legacy-backfill");
        let store = FileObjectStore::open(&root)?;
        let vector_index = ObjectVectorIndex::new(store.clone());

        let raw_legacy = crate::ids::RawNodeId::new();
        let legacy_path = store.root().join("embeddings").join("raw");
        std::fs::create_dir_all(&legacy_path).map_err(|err| {
            crate::EngineError::Storage(format!("failed to prepare legacy embedding dir: {err}"))
        })?;
        let legacy_file = legacy_path.join(format!("{raw_legacy}.json"));
        let legacy_payload = format!(
            r#"{{"id":"{raw_legacy}","embedding":[1.0,0.0]}}"#,
            raw_legacy = raw_legacy
        );
        std::fs::write(&legacy_file, legacy_payload.as_bytes()).map_err(|err| {
            crate::EngineError::Storage(format!("failed to write legacy embedding file: {err}"))
        })?;

        // Manifest must list the id so the search loop visits it.
        let manifest_path = store
            .root()
            .join("indexes")
            .join("vector")
            .join("raw_embeddings.json");
        std::fs::write(
            &manifest_path,
            format!(r#"{{"ids":["{raw_legacy}"]}}"#).as_bytes(),
        )
        .map_err(|err| {
            crate::EngineError::Storage(format!("failed to seed manifest with legacy id: {err}"))
        })?;

        let scoped = vector_index
            .search_raw(
                &Embedding(vec![1.0, 0.0]),
                4,
                Some(&crate::SessionId::new()),
            )
            .await?;
        assert!(
            scoped.is_empty(),
            "legacy entry must not surface for a session-scoped search"
        );

        let legacy = vector_index
            .search_raw(&Embedding(vec![1.0, 0.0]), 4, None)
            .await?;
        assert_eq!(legacy.len(), 1);
        assert_eq!(legacy[0].id, raw_legacy);

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn object_store_backlog_indexes_follow_lifecycle_updates() -> crate::Result<()> {
        let root = temp_root("backlog");
        let session_id = crate::SessionId::new();
        let loop_id = crate::LoopId::new();
        let store = FileObjectStore::open(&root)?;
        let repository = ObjectNodeRepository::new(store);

        let node = RawNode::text(
            RawNodeKind::Note,
            Some(session_id),
            Some(loop_id),
            "system",
            "backlog candidate",
            0.5,
            Vec::new(),
        );
        repository.insert_raw(node.clone()).await?;

        assert_eq!(repository.undistilled_raw(10, false).await?.len(), 1);
        assert_eq!(repository.undistilled_raw(10, true).await?.len(), 0);

        repository
            .update_raw_lifecycle(
                &[node.id],
                &RawLifecyclePatch {
                    distillation_state: None,
                    overflow: Some(OverflowPolicy {
                        was_pushed_out_of_session: true,
                        relax_retrieval_until: None,
                    }),
                },
            )
            .await?;

        assert_eq!(repository.undistilled_raw(10, false).await?.len(), 1);
        assert_eq!(repository.undistilled_raw(10, true).await?.len(), 1);

        repository
            .update_raw_lifecycle(
                &[node.id],
                &RawLifecyclePatch {
                    distillation_state: Some(crate::domain::DistillationState::Distilled),
                    overflow: Some(OverflowPolicy {
                        was_pushed_out_of_session: false,
                        relax_retrieval_until: None,
                    }),
                },
            )
            .await?;

        assert_eq!(repository.undistilled_raw(10, false).await?.len(), 0);
        assert_eq!(repository.undistilled_raw(10, true).await?.len(), 0);

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn object_store_reindexes_node_missing_from_timeline_on_open() -> crate::Result<()> {
        // Simulate a mid-insert crash: the raw node body and the still-valid
        // `.index-version` marker survive, but the node never made it into the
        // timeline index (its index sync was interrupted). On reopen the
        // completeness reconciliation must detect the orphaned body and rebuild,
        // so the node is reachable again through every retrieval path. The two
        // existing recovery tests only cover an unparseable index / a wiped
        // index dir — neither covers a structurally-valid-but-incomplete index.
        let root = temp_root("incomplete-timeline");
        let session_id = crate::SessionId::new();
        let loop_id = crate::LoopId::new();

        let store = FileObjectStore::open(&root)?;
        let repository = ObjectNodeRepository::new(store.clone());

        let kept = RawNode::text(
            RawNodeKind::UserUtterance,
            Some(session_id),
            Some(loop_id),
            "user",
            "kept node",
            0.5,
            Vec::new(),
        );
        let orphan = RawNode::text(
            RawNodeKind::AssistantUtterance,
            Some(session_id),
            Some(loop_id),
            "assistant",
            "orphaned node",
            0.6,
            Vec::new(),
        );
        repository.insert_raw(kept.clone()).await?;
        repository.insert_raw(orphan.clone()).await?;

        // Rewrite the timeline to drop only the orphan's entry, leaving the raw
        // body, the `.index-version` marker, and the JSON shape all intact — so
        // the index-version check and the parse sanity check both still pass and
        // the *only* trigger for a rebuild is the completeness probe.
        let timeline_path = root.join("indexes").join("timeline").join("raw.json");
        let timeline_payload = std::fs::read_to_string(&timeline_path).map_err(|err| {
            crate::EngineError::Storage(format!("failed to read timeline index for test: {err}"))
        })?;
        let mut timeline: Value = serde_json::from_str(&timeline_payload).map_err(|err| {
            crate::EngineError::Storage(format!("failed to parse timeline index for test: {err}"))
        })?;
        let orphan_id = orphan.id.to_string();
        if let Some(entries) = timeline.as_array_mut() {
            entries.retain(|entry| entry["id"].as_str() != Some(orphan_id.as_str()));
        }
        std::fs::write(
            &timeline_path,
            serde_json::to_vec(&timeline).map_err(|err| {
                crate::EngineError::Storage(format!(
                    "failed to serialize trimmed timeline for test: {err}"
                ))
            })?,
        )
        .map_err(|err| {
            crate::EngineError::Storage(format!("failed to write trimmed timeline for test: {err}"))
        })?;

        // Confirm the orphan really is invisible through the stale index we
        // trimmed. We removed it from the global timeline index, so probe via
        // `timeline_raw` (which reads `raw_timeline_path`); `session_raw` reads
        // the session index, which we deliberately left intact and would still
        // surface the orphan.
        let stale = ObjectNodeRepository::new(store);
        let stale_timeline = stale.timeline_raw(None, None, None, 16).await?;
        assert_eq!(
            stale_timeline.len(),
            1,
            "before reconciliation the orphaned node must be invisible in the trimmed timeline"
        );

        // Reopen: the completeness probe must force a rebuild from the bodies.
        let reopened = FileObjectStore::open(&root)?;
        let repository = ObjectNodeRepository::new(reopened);

        let session_nodes = repository.session_raw(&session_id).await?;
        assert_eq!(
            session_nodes.len(),
            2,
            "reopen must reindex the orphaned node from its surviving body"
        );
        let timeline_nodes = repository.timeline_raw(None, None, None, 16).await?;
        let ids: Vec<_> = timeline_nodes.iter().map(|node| node.id).collect();
        assert!(ids.contains(&kept.id));
        assert!(ids.contains(&orphan.id));

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[tokio::test]
    async fn object_store_reindexes_embedding_missing_from_manifest_on_open() -> crate::Result<()> {
        // Simulate the embedding mid-insert crash: the embedding body and the
        // still-valid `.index-version` marker survive, but the embedding id
        // never made it into the vector manifest (its manifest upsert was
        // interrupted between the body write and the upsert). On reopen the
        // embedding completeness probe must detect the orphaned body and
        // rebuild, so the embedding is reachable again through vector search.
        // This mirrors the raw-timeline incompleteness test but for the
        // structurally-identical embedding-body/manifest window.
        let root = temp_root("incomplete-embedding-manifest");
        let store = FileObjectStore::open(&root)?;
        let vector_index = ObjectVectorIndex::new(store.clone());

        let kept_raw = crate::ids::RawNodeId::new();
        let orphan_raw = crate::ids::RawNodeId::new();
        let orphan_abstract = crate::ids::AbstractNodeId::new();

        vector_index
            .index_raw(kept_raw, Embedding(vec![0.0, 1.0]))
            .await?;
        vector_index
            .index_raw(orphan_raw, Embedding(vec![1.0, 0.0]))
            .await?;
        vector_index
            .index_abstract(orphan_abstract, Embedding(vec![1.0, 0.0]))
            .await?;

        // Rewrite each manifest to drop only the orphan id, leaving the
        // embedding body, the `.index-version` marker, and the JSON shape all
        // intact — so the index-version check and the parse sanity check both
        // still pass and the *only* trigger for a rebuild is the embedding
        // completeness probe.
        let drop_id_from_manifest = |relative: &str, orphan: &str| -> crate::Result<()> {
            let manifest_path = root.join("indexes").join("vector").join(relative);
            let payload = std::fs::read_to_string(&manifest_path).map_err(|err| {
                crate::EngineError::Storage(format!(
                    "failed to read embedding manifest for test: {err}"
                ))
            })?;
            let mut manifest: Value = serde_json::from_str(&payload).map_err(|err| {
                crate::EngineError::Storage(format!(
                    "failed to parse embedding manifest for test: {err}"
                ))
            })?;
            if let Some(ids) = manifest.get_mut("ids").and_then(Value::as_array_mut) {
                ids.retain(|id| id.as_str() != Some(orphan));
            }
            std::fs::write(
                &manifest_path,
                serde_json::to_vec(&manifest).map_err(|err| {
                    crate::EngineError::Storage(format!(
                        "failed to serialize trimmed embedding manifest for test: {err}"
                    ))
                })?,
            )
            .map_err(|err| {
                crate::EngineError::Storage(format!(
                    "failed to write trimmed embedding manifest for test: {err}"
                ))
            })
        };
        drop_id_from_manifest("raw_embeddings.json", &orphan_raw.to_string())?;
        drop_id_from_manifest("abstract_embeddings.json", &orphan_abstract.to_string())?;

        // Confirm the orphaned embeddings really are invisible through the
        // stale manifests (search iterates the manifest, not the directory).
        let stale_index = ObjectVectorIndex::new(store);
        let stale_raw = stale_index
            .search_raw(&Embedding(vec![1.0, 0.0]), 8, None)
            .await?;
        assert!(
            stale_raw.iter().all(|hit| hit.id != orphan_raw),
            "before reconciliation the orphaned raw embedding must be invisible"
        );
        let stale_abstract = stale_index
            .search_abstract(&Embedding(vec![1.0, 0.0]), 8, None)
            .await?;
        assert!(
            stale_abstract.is_empty(),
            "before reconciliation the orphaned abstract embedding must be invisible"
        );

        // Reopen: the embedding completeness probe must force a rebuild from the
        // bodies, regenerating both manifests.
        let reopened = FileObjectStore::open(&root)?;
        let vector_index = ObjectVectorIndex::new(reopened);

        let recovered_raw = vector_index
            .search_raw(&Embedding(vec![1.0, 0.0]), 8, None)
            .await?;
        let recovered_raw_ids: Vec<_> = recovered_raw.iter().map(|hit| hit.id).collect();
        assert!(
            recovered_raw_ids.contains(&orphan_raw),
            "reopen must reindex the orphaned raw embedding from its surviving body"
        );
        assert!(
            recovered_raw_ids.contains(&kept_raw),
            "the already-indexed embedding must remain searchable after rebuild"
        );

        let recovered_abstract = vector_index
            .search_abstract(&Embedding(vec![1.0, 0.0]), 8, None)
            .await?;
        assert_eq!(
            recovered_abstract.len(),
            1,
            "reopen must reindex the orphaned abstract embedding from its surviving body"
        );
        assert_eq!(recovered_abstract[0].id, orphan_abstract);

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }
}
