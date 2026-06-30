use async_trait::async_trait;
use chrono::{DateTime, Utc};

use crate::domain::{AbstractNode, RawNode};
use crate::error::Result;
use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};

use crate::storage::traits::{NodeRepository, RawLifecyclePatch};

use super::store::{FileObjectStore, StoredId};

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
        let entries = match session_id {
            // Session-scoped: read the (naturally bounded) per-session index.
            Some(value) => {
                let mut entries = self
                    .store
                    .read_raw_index_entries_unlocked(&self.store.session_index_path(value))
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
                entries
            }
            // Global: merge the per-day timeline shards newest-first with a
            // bounded read (see `read_global_timeline_entries_unlocked`).
            None => {
                self.store
                    .read_global_timeline_entries_unlocked(from, to, limit)
                    .await?
            }
        };
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use chrono::{Duration as ChronoDuration, Utc};
    use serde_json::Value;

    use crate::domain::{OverflowPolicy, RawNode, RawNodeKind};
    use crate::model::embedding::Embedding;

    use super::super::store::FileObjectStore;
    use super::super::{ObjectNodeRepository, ObjectVectorIndex};
    use crate::storage::traits::{NodeRepository, RawLifecyclePatch, VectorIndex};

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "takos-agent-engine-object-store-{name}-{}",
            uuid::Uuid::new_v4()
        ))
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

        // Corrupt the (single) per-day timeline shard so the parse sanity check
        // forces a rebuild on reopen.
        let shard_dir = root.join("indexes").join("timeline").join("raw");
        let shard = std::fs::read_dir(&shard_dir)
            .map_err(|err| {
                crate::EngineError::Storage(format!("failed to read timeline shard dir: {err}"))
            })?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .find(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
            .expect("a timeline shard should exist after an insert");
        std::fs::write(&shard, b"{ broken json").map_err(|err| {
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

        // Rewrite the timeline shards to drop only the orphan's entry, leaving
        // the raw body, the `.index-version` marker, and the JSON shape all
        // intact — so the index-version check and the parse sanity check both
        // still pass and the *only* trigger for a rebuild is the completeness
        // probe. Both nodes were inserted ~now, so they share a day shard, but
        // iterate every shard to be robust across a midnight boundary.
        let shard_dir = root.join("indexes").join("timeline").join("raw");
        let orphan_id = orphan.id.to_string();
        let shard_paths: Vec<_> = std::fs::read_dir(&shard_dir)
            .map_err(|err| {
                crate::EngineError::Storage(format!("failed to read timeline shard dir: {err}"))
            })?
            .filter_map(std::result::Result::ok)
            .map(|entry| entry.path())
            .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
            .collect();
        for shard_path in shard_paths {
            let payload = std::fs::read_to_string(&shard_path).map_err(|err| {
                crate::EngineError::Storage(format!("failed to read timeline shard for test: {err}"))
            })?;
            let mut shard: Value = serde_json::from_str(&payload).map_err(|err| {
                crate::EngineError::Storage(format!(
                    "failed to parse timeline shard for test: {err}"
                ))
            })?;
            if let Some(entries) = shard.as_array_mut() {
                entries.retain(|entry| entry["id"].as_str() != Some(orphan_id.as_str()));
            }
            std::fs::write(
                &shard_path,
                serde_json::to_vec(&shard).map_err(|err| {
                    crate::EngineError::Storage(format!(
                        "failed to serialize trimmed timeline shard for test: {err}"
                    ))
                })?,
            )
            .map_err(|err| {
                crate::EngineError::Storage(format!(
                    "failed to write trimmed timeline shard for test: {err}"
                ))
            })?;
        }

        // Confirm the orphan really is invisible through the stale index we
        // trimmed. We removed it from the global timeline shards, so probe via
        // `timeline_raw(None, ...)` (which merges the shards); `session_raw`
        // reads the session index, which we deliberately left intact and would
        // still surface the orphan.
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
    async fn timeline_insert_touches_only_its_day_shard() -> crate::Result<()> {
        // C2 regression: an insert must rewrite only the bounded per-day shard
        // for the node's timestamp, not the whole never-pruned timeline. We
        // prove this by inserting into one day, snapshotting that shard's bytes,
        // inserting into a *different* day, and asserting the first shard is
        // byte-for-byte unchanged — and that the merged read still returns
        // correctly ordered + limited results.
        let root = temp_root("timeline-shard-bounded");
        let session_id = crate::SessionId::new();
        let loop_id = crate::LoopId::new();
        let store = FileObjectStore::open(&root)?;
        let repository = ObjectNodeRepository::new(store);

        let mut old = RawNode::text(
            RawNodeKind::UserUtterance,
            Some(session_id),
            Some(loop_id),
            "user",
            "old day",
            0.5,
            Vec::new(),
        );
        old.timestamp = Utc::now() - ChronoDuration::days(2);
        repository.insert_raw(old.clone()).await?;

        let shard_dir = root.join("indexes").join("timeline").join("raw");
        let list_shards = || -> Vec<PathBuf> {
            std::fs::read_dir(&shard_dir)
                .into_iter()
                .flatten()
                .filter_map(std::result::Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("json"))
                .collect()
        };
        let old_shards = list_shards();
        assert_eq!(old_shards.len(), 1, "one day inserted -> one shard");
        let old_shard_path = old_shards[0].clone();
        let old_shard_bytes = std::fs::read(&old_shard_path).map_err(|err| {
            crate::EngineError::Storage(format!("failed to snapshot timeline shard: {err}"))
        })?;

        // Two inserts on a different (current) day.
        for label in ["today a", "today b"] {
            let mut node = RawNode::text(
                RawNodeKind::AssistantUtterance,
                Some(session_id),
                Some(loop_id),
                "assistant",
                label,
                0.6,
                Vec::new(),
            );
            node.timestamp = Utc::now();
            repository.insert_raw(node).await?;
        }

        // The earlier day's shard must be untouched by the later inserts.
        let after_bytes = std::fs::read(&old_shard_path).map_err(|err| {
            crate::EngineError::Storage(format!("failed to re-read timeline shard: {err}"))
        })?;
        assert_eq!(
            after_bytes, old_shard_bytes,
            "inserting into another day must not rewrite an existing day's shard"
        );
        assert_eq!(list_shards().len(), 2, "second day -> a second shard");

        // Read semantics preserved: newest-first, limited, and full merge.
        let recent = repository.timeline_raw(None, None, None, 2).await?;
        assert_eq!(recent.len(), 2);
        assert!(
            recent.iter().all(|node| node.id != old.id),
            "the two newest entries are from today, not the old day"
        );
        let all = repository.timeline_raw(None, None, None, 16).await?;
        assert_eq!(all.len(), 3);
        for window in all.windows(2) {
            assert!(window[0].timestamp >= window[1].timestamp);
        }

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }
}
