use async_trait::async_trait;

use crate::error::Result;
use crate::ids::{AbstractNodeId, RawNodeId, SessionId};
use crate::model::embedding::{cmp_score_desc, cosine_similarity, Embedding};

use crate::storage::traits::{ScoredAbstractRef, ScoredRawRef, VectorIndex};

use super::store::{FileObjectStore, StoredEmbedding};

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
        // Index into the per-session shard so a session-scoped search only scans
        // its own session's embeddings (the `none` shard for legacy/session-less
        // entries). [C3]
        self.store
            .upsert_manifest_id_unlocked(
                &self.store.raw_embedding_shard_path(session_id.as_ref()),
                id,
            )
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
            .upsert_manifest_id_unlocked(
                &self.store.abstract_embedding_shard_path(session_id.as_ref()),
                id,
            )
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
        // Read only the requested session's shard (the `none` shard for a
        // session-less search). Every id in that shard belongs to the session,
        // so no per-entry session filter is needed. [C3]
        for id in self
            .store
            .read_manifest_unlocked::<RawNodeId>(&self.store.raw_embedding_shard_path(session_id))
            .await?
        {
            if let Some(record) = self
                .store
                .try_read_json::<StoredEmbedding<RawNodeId>>(&self.store.raw_embedding_path(&id))
                .await?
            {
                scored.push(ScoredRawRef {
                    id: record.id,
                    score: cosine_similarity(query, &record.embedding),
                });
            }
        }
        scored.sort_by(|left, right| {
            cmp_score_desc(left.score, right.score).then_with(|| left.id.cmp(&right.id))
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
                &self.store.abstract_embedding_shard_path(session_id),
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
                scored.push(ScoredAbstractRef {
                    id: record.id,
                    score: cosine_similarity(query, &record.embedding),
                });
            }
        }
        scored.sort_by(|left, right| {
            cmp_score_desc(left.score, right.score).then_with(|| left.id.cmp(&right.id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use crate::model::embedding::Embedding;

    use super::super::store::FileObjectStore;
    use super::super::ObjectVectorIndex;
    use crate::storage::traits::VectorIndex;

    fn temp_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "takos-agent-engine-object-store-{name}-{}",
            uuid::Uuid::new_v4()
        ))
    }

    #[tokio::test]
    async fn object_store_rebuilds_from_wrong_shape_embedding_manifest_on_open() -> crate::Result<()>
    {
        // A valid-JSON but wrong-shape embedding manifest (e.g. a bare array
        // instead of `{"ids": [...]}`) must force a rebuild on open rather
        // than surfacing a Storage error.
        let root = temp_root("wrong-shape-manifest");
        let store = FileObjectStore::open(&root)?;
        let vector_index = ObjectVectorIndex::new(store.clone());

        let raw = crate::ids::RawNodeId::new();
        vector_index
            .index_raw(raw, Embedding(vec![1.0, 0.0]))
            .await?;

        // Embedding body stays on disk; clobber the (session-less) shard with
        // valid JSON of the wrong shape.
        let manifest_path = store
            .root()
            .join("indexes")
            .join("vector")
            .join("raw")
            .join("none.json");
        std::fs::write(&manifest_path, b"[1,2,3]").map_err(|err| {
            crate::EngineError::Storage(format!(
                "failed to write wrong-shape embedding manifest: {err}"
            ))
        })?;

        // Reopen: rebuild must succeed (not error) and re-derive the manifest
        // from the surviving body so the embedding is searchable again.
        let reopened = FileObjectStore::open(&root)?;
        let vector_index = ObjectVectorIndex::new(reopened);

        let scored = vector_index
            .search_raw(&Embedding(vec![1.0, 0.0]), 1, None)
            .await?;
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].id, raw);

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

    // C3 regression: embeddings are sharded per session, so a session-scoped
    // search only reads its own session's shard (bounded by that session, not
    // the whole installation). Assert the shard files are separate and the
    // session-A shard holds only session-A ids.
    #[tokio::test]
    async fn vector_search_uses_per_session_shards() -> crate::Result<()> {
        let root = temp_root("vector-per-session-shard");
        let session_a = crate::SessionId::new();
        let session_b = crate::SessionId::new();
        let store = FileObjectStore::open(&root)?;
        let vector_index = ObjectVectorIndex::new(store);

        let a1 = crate::ids::RawNodeId::new();
        let a2 = crate::ids::RawNodeId::new();
        let b1 = crate::ids::RawNodeId::new();
        for (id, session) in [(a1, session_a), (a2, session_a), (b1, session_b)] {
            vector_index
                .index_raw_with_session(id, Embedding(vec![1.0, 0.0]), Some(session))
                .await?;
        }

        let shard_dir = root.join("indexes").join("vector").join("raw");
        let shard_a = shard_dir.join(format!("{session_a}.json"));
        let shard_b = shard_dir.join(format!("{session_b}.json"));
        assert!(shard_a.exists(), "session A must have its own shard file");
        assert!(shard_b.exists(), "session B must have its own shard file");

        let payload = std::fs::read_to_string(&shard_a).map_err(|err| {
            crate::EngineError::Storage(format!("failed to read session-A shard: {err}"))
        })?;
        let manifest: serde_json::Value = serde_json::from_str(&payload).map_err(|err| {
            crate::EngineError::Storage(format!("failed to parse session-A shard: {err}"))
        })?;
        let ids: Vec<String> = manifest["ids"]
            .as_array()
            .expect("shard has ids array")
            .iter()
            .map(|value| value.as_str().unwrap().to_string())
            .collect();
        assert_eq!(ids.len(), 2, "session-A shard holds only session-A entries");
        assert!(!ids.contains(&b1.to_string()), "no cross-session bleed");

        let scoped_a = vector_index
            .search_raw(&Embedding(vec![1.0, 0.0]), 8, Some(&session_a))
            .await?;
        assert_eq!(scoped_a.len(), 2);

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

        // The session-less shard must list the id so the search loop visits it.
        let manifest_path = store
            .root()
            .join("indexes")
            .join("vector")
            .join("raw")
            .join("none.json");
        std::fs::create_dir_all(manifest_path.parent().unwrap()).map_err(|err| {
            crate::EngineError::Storage(format!("failed to create shard dir: {err}"))
        })?;
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
            let mut manifest: serde_json::Value =
                serde_json::from_str(&payload).map_err(|err| {
                    crate::EngineError::Storage(format!(
                        "failed to parse embedding manifest for test: {err}"
                    ))
                })?;
            if let Some(ids) = manifest
                .get_mut("ids")
                .and_then(serde_json::Value::as_array_mut)
            {
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
        drop_id_from_manifest("raw/none.json", &orphan_raw.to_string())?;
        drop_id_from_manifest("abstract/none.json", &orphan_abstract.to_string())?;

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
