use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::ids::{AbstractNodeId, RawNodeId, SessionId};
use crate::model::embedding::{cmp_score_desc, cosine_similarity, Embedding};
use crate::storage::traits::{ScoredAbstractRef, ScoredRawRef, VectorIndex};

/// Returns true when an entry's stored session matches the search filter.
///
/// - If the caller requests a specific `query` session (`Some(_)`), only
///   entries indexed with the same `entry` session are eligible.
/// - If the caller passes `None`, only legacy entries that were indexed
///   without a session id stay eligible. This keeps backfill-free data
///   reachable without leaking it across sessions when a session-scoped
///   search is requested.
#[cfg(test)]
fn entry_matches_session_filter(entry: Option<&SessionId>, query: Option<&SessionId>) -> bool {
    match (entry, query) {
        (None, None) => true,
        (Some(stored), Some(requested)) => stored == requested,
        _ => false,
    }
}

#[derive(Debug)]
#[cfg(test)]
struct InMemoryEmbeddingEntry {
    embedding: Embedding,
    session_id: Option<SessionId>,
}

#[derive(Debug, Default)]
#[cfg(test)]
pub struct InMemoryVectorIndex {
    raw_embeddings: RwLock<HashMap<RawNodeId, InMemoryEmbeddingEntry>>,
    abstract_embeddings: RwLock<HashMap<AbstractNodeId, InMemoryEmbeddingEntry>>,
}

#[async_trait]
#[cfg(test)]
impl VectorIndex for InMemoryVectorIndex {
    async fn index_raw(&self, id: RawNodeId, embedding: Embedding) -> Result<()> {
        self.raw_embeddings.write().await.insert(
            id,
            InMemoryEmbeddingEntry {
                embedding,
                session_id: None,
            },
        );
        Ok(())
    }

    async fn index_abstract(&self, id: AbstractNodeId, embedding: Embedding) -> Result<()> {
        self.abstract_embeddings.write().await.insert(
            id,
            InMemoryEmbeddingEntry {
                embedding,
                session_id: None,
            },
        );
        Ok(())
    }

    async fn index_raw_with_session(
        &self,
        id: RawNodeId,
        embedding: Embedding,
        session_id: Option<SessionId>,
    ) -> Result<()> {
        self.raw_embeddings.write().await.insert(
            id,
            InMemoryEmbeddingEntry {
                embedding,
                session_id,
            },
        );
        Ok(())
    }

    async fn index_abstract_with_session(
        &self,
        id: AbstractNodeId,
        embedding: Embedding,
        session_id: Option<SessionId>,
    ) -> Result<()> {
        self.abstract_embeddings.write().await.insert(
            id,
            InMemoryEmbeddingEntry {
                embedding,
                session_id,
            },
        );
        Ok(())
    }

    async fn search_raw(
        &self,
        query: &Embedding,
        top_k: usize,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<ScoredRawRef>> {
        let guard = self.raw_embeddings.read().await;
        let mut scored: Vec<_> = guard
            .iter()
            .filter(|(_, entry)| {
                entry_matches_session_filter(entry.session_id.as_ref(), session_id)
            })
            .map(|(id, entry)| ScoredRawRef {
                id: *id,
                score: cosine_similarity(query, &entry.embedding),
            })
            .collect();
        scored.sort_by(|left, right| {
            // Match the production object store's native-id tiebreak (not a
            // stringified comparison) so the doubles cannot diverge. [Q4]
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
        let guard = self.abstract_embeddings.read().await;
        let mut scored: Vec<_> = guard
            .iter()
            .filter(|(_, entry)| {
                entry_matches_session_filter(entry.session_id.as_ref(), session_id)
            })
            .map(|(id, entry)| ScoredAbstractRef {
                id: *id,
                score: cosine_similarity(query, &entry.embedding),
            })
            .collect();
        scored.sort_by(|left, right| {
            // Match the production object store's native-id tiebreak (not a
            // stringified comparison) so the doubles cannot diverge. [Q4]
            cmp_score_desc(left.score, right.score).then_with(|| left.id.cmp(&right.id))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}
