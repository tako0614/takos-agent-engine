use std::str::FromStr;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::config::ToolsConfig;
use crate::domain::{AbstractNode, RawNode};
use crate::error::{EngineError, Result};
use crate::ids::{AbstractNodeId, SessionId};
use crate::model::{Embedder, Embedding};
use crate::storage::{GraphRepository, NodeRepository, VectorIndex};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MemorySearchTarget {
    Raw,
    Abstract,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchParams {
    pub query: String,
    #[serde(default = "default_memory_search_target")]
    pub target: MemorySearchTarget,
    #[serde(default = "default_memory_search_top_k")]
    pub top_k: usize,
    pub threshold: Option<f32>,
    /// Session scope for the search. The engine forces this to the running
    /// session in `prepare_tool_call_for_config`, overriding any model-supplied
    /// value, so a tool call cannot read another session's memory and so the
    /// search actually matches the engine's session-tagged embeddings (passing
    /// `None` matches only legacy/None-tagged entries and returns nothing in
    /// production). `None` = session-less, used only off the engine tool path
    /// (e.g. unit tests). [S2, C8]
    #[serde(default)]
    pub session_id: Option<String>,
}

const fn default_memory_search_target() -> MemorySearchTarget {
    MemorySearchTarget::Both
}

const fn default_memory_search_top_k() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchParams {
    pub start_node_id: String,
    #[serde(default = "default_graph_depth")]
    pub max_depth: usize,
    pub relation_types: Option<Vec<String>>,
}

const fn default_graph_depth() -> usize {
    2
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceLookupParams {
    pub abstract_node_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineSearchParams {
    pub session_id: Option<String>,
    pub from: Option<DateTime<Utc>>,
    pub to: Option<DateTime<Utc>>,
    #[serde(default = "default_timeline_limit")]
    pub limit: usize,
}

const fn default_timeline_limit() -> usize {
    10
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredRawHit {
    pub node: RawNode,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredAbstractHit {
    pub node: AbstractNode,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySearchResult {
    pub raw_hits: Vec<ScoredRawHit>,
    pub abstract_hits: Vec<ScoredAbstractHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchHit {
    pub node: AbstractNode,
    pub depth: usize,
    pub via_predicate: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphSearchResult {
    pub hits: Vec<GraphSearchHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceLookupResult {
    pub abstract_node: AbstractNode,
    pub raw_nodes: Vec<RawNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineSearchResult {
    pub raw_nodes: Vec<RawNode>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryToolBounds {
    pub max_memory_search_top_k: usize,
    pub max_graph_search_depth: usize,
    pub max_timeline_search_limit: usize,
}

impl Default for MemoryToolBounds {
    fn default() -> Self {
        Self::from(&ToolsConfig::default())
    }
}

impl From<&ToolsConfig> for MemoryToolBounds {
    fn from(config: &ToolsConfig) -> Self {
        Self {
            max_memory_search_top_k: config.max_memory_search_top_k,
            max_graph_search_depth: config.max_graph_search_depth,
            max_timeline_search_limit: config.max_timeline_search_limit,
        }
    }
}

impl MemoryToolBounds {
    #[must_use]
    pub fn clamp_memory_search_top_k(&self, requested: usize) -> usize {
        requested.max(1).min(self.max_memory_search_top_k.max(1))
    }

    #[must_use]
    pub fn clamp_graph_search_depth(&self, requested: usize) -> usize {
        requested.min(self.max_graph_search_depth)
    }

    #[must_use]
    pub fn clamp_timeline_search_limit(&self, requested: usize) -> usize {
        requested.max(1).min(self.max_timeline_search_limit.max(1))
    }
}

pub struct MemoryTools {
    repository: Arc<dyn NodeRepository>,
    vector_index: Arc<dyn VectorIndex>,
    graph_repository: Arc<dyn GraphRepository>,
    embedder: Arc<dyn Embedder>,
}

impl MemoryTools {
    pub fn new(
        repository: Arc<dyn NodeRepository>,
        vector_index: Arc<dyn VectorIndex>,
        graph_repository: Arc<dyn GraphRepository>,
        embedder: Arc<dyn Embedder>,
    ) -> Self {
        // Argument bounds are enforced once, at the engine's
        // `prepare_tool_call_for_config` chokepoint using the LIVE config; the
        // tool methods below intentionally do not re-clamp (a second clamp from a
        // separate source would silently override raised operator config). [Q2]
        Self {
            repository,
            vector_index,
            graph_repository,
            embedder,
        }
    }

    /// # Errors
    ///
    /// Returns an [`EngineError`] when the embedder fails to embed the query
    /// or the underlying vector / node repositories return an error.
    pub async fn semantic_search(&self, params: MemorySearchParams) -> Result<MemorySearchResult> {
        let top_k = params.top_k.max(1);
        let session_id = params
            .session_id
            .as_deref()
            .map(SessionId::from_str)
            .transpose()
            .map_err(|err| EngineError::Tool(format!("invalid session id: {err}")))?;
        let query_embedding = self.embedder.embed_text(&params.query).await?;
        let threshold = params.threshold.unwrap_or(0.0);

        let raw_hits = if matches!(
            params.target,
            MemorySearchTarget::Raw | MemorySearchTarget::Both
        ) {
            self.search_raw(&query_embedding, top_k, threshold, session_id.as_ref())
                .await?
        } else {
            Vec::new()
        };

        let abstract_hits = if matches!(
            params.target,
            MemorySearchTarget::Abstract | MemorySearchTarget::Both
        ) {
            self.search_abstract(&query_embedding, top_k, threshold, session_id.as_ref())
                .await?
        } else {
            Vec::new()
        };

        Ok(MemorySearchResult {
            raw_hits,
            abstract_hits,
        })
    }

    /// # Errors
    ///
    /// Returns an [`EngineError`] when the start node id cannot be parsed or
    /// the graph / node repositories return an error.
    pub async fn graph_search(&self, params: GraphSearchParams) -> Result<GraphSearchResult> {
        let start = AbstractNodeId::from_str(&params.start_node_id)
            .map_err(|err| EngineError::Tool(format!("invalid abstract node id: {err}")))?;
        let max_depth = params.max_depth;
        let hits = self
            .graph_repository
            .traverse(&start, max_depth, params.relation_types.as_deref())
            .await?;

        let ids: Vec<_> = hits.iter().map(|hit| hit.node_id).collect();
        let mut nodes_by_id = self
            .repository
            .list_abstract(&ids)
            .await?
            .into_iter()
            .map(|node| (node.id, node))
            .collect::<std::collections::HashMap<_, _>>();

        let mut ordered_hits = Vec::new();
        for hit in hits {
            if let Some(node) = nodes_by_id.remove(&hit.node_id) {
                ordered_hits.push(GraphSearchHit {
                    node,
                    depth: hit.depth,
                    via_predicate: hit.via_predicate,
                });
            }
        }

        Ok(GraphSearchResult { hits: ordered_hits })
    }

    /// # Errors
    ///
    /// Returns an [`EngineError`] when the abstract node id cannot be parsed,
    /// the referenced node does not exist, or the repository returns an error.
    pub async fn provenance_lookup(
        &self,
        params: ProvenanceLookupParams,
    ) -> Result<ProvenanceLookupResult> {
        let node_id = AbstractNodeId::from_str(&params.abstract_node_id)
            .map_err(|err| EngineError::Tool(format!("invalid abstract node id: {err}")))?;
        let abstract_node = self
            .repository
            .get_abstract(&node_id)
            .await?
            .ok_or_else(|| EngineError::Tool("abstract node not found".to_string()))?;
        let raw_nodes = self
            .repository
            .list_raw(&abstract_node.references.raw_node_ids)
            .await?;
        Ok(ProvenanceLookupResult {
            abstract_node,
            raw_nodes,
        })
    }

    /// # Errors
    ///
    /// Returns an [`EngineError`] when the optional session id cannot be
    /// parsed or the node repository returns an error.
    pub async fn timeline_search(
        &self,
        params: TimelineSearchParams,
    ) -> Result<TimelineSearchResult> {
        let session_id = params
            .session_id
            .as_deref()
            .map(SessionId::from_str)
            .transpose()
            .map_err(|err| EngineError::Tool(format!("invalid session id: {err}")))?;
        let raw_nodes = self
            .repository
            .timeline_raw(
                session_id.as_ref(),
                params.from,
                params.to,
                params.limit.max(1),
            )
            .await?;
        Ok(TimelineSearchResult { raw_nodes })
    }

    async fn search_raw(
        &self,
        query_embedding: &Embedding,
        top_k: usize,
        threshold: f32,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<ScoredRawHit>> {
        // Scope the search to the requested session. The engine forces this to
        // the running session (so it matches the engine's session-tagged
        // embeddings and never leaks across sessions); `None` matches only
        // legacy/None-tagged entries (off-engine callers / tests).
        let refs = self
            .vector_index
            .search_raw(query_embedding, top_k, session_id)
            .await?;
        let mut hits = Vec::new();
        for candidate in refs {
            if candidate.score < threshold {
                continue;
            }
            if let Some(node) = self.repository.get_raw(&candidate.id).await? {
                hits.push(ScoredRawHit {
                    node,
                    score: candidate.score,
                });
            }
        }
        Ok(hits)
    }

    async fn search_abstract(
        &self,
        query_embedding: &Embedding,
        top_k: usize,
        threshold: f32,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<ScoredAbstractHit>> {
        let refs = self
            .vector_index
            .search_abstract(query_embedding, top_k, session_id)
            .await?;
        let mut hits = Vec::new();
        for candidate in refs {
            if candidate.score < threshold {
                continue;
            }
            if let Some(node) = self.repository.get_abstract(&candidate.id).await? {
                hits.push(ScoredAbstractHit {
                    node,
                    score: candidate.score,
                });
            }
        }
        Ok(hits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        AbstractNode, AbstractNodeMetadata, GraphFragment, RawNode, RawNodeKind, References,
    };
    use crate::model::Embedder;
    use crate::storage::{InMemoryGraphRepository, InMemoryNodeRepository, InMemoryVectorIndex};
    use crate::test_support::TestHashEmbedder;

    struct TestHarness {
        tools: MemoryTools,
        repo: Arc<InMemoryNodeRepository>,
        vector: Arc<InMemoryVectorIndex>,
        embedder: TestHashEmbedder,
    }

    fn setup() -> TestHarness {
        let repo = Arc::new(InMemoryNodeRepository::default());
        let vector = Arc::new(InMemoryVectorIndex::default());
        let graph = Arc::new(InMemoryGraphRepository::default());
        let embedder = TestHashEmbedder::default();

        let tools = MemoryTools::new(
            repo.clone() as Arc<dyn NodeRepository>,
            vector.clone() as Arc<dyn VectorIndex>,
            graph.clone() as Arc<dyn GraphRepository>,
            Arc::new(embedder.clone()) as Arc<dyn Embedder>,
        );
        TestHarness {
            tools,
            repo,
            vector,
            embedder,
        }
    }

    #[tokio::test]
    async fn semantic_search_empty_returns_no_hits() {
        let h = setup();
        let params = MemorySearchParams {
            query: "hello".to_string(),
            target: MemorySearchTarget::Both,
            top_k: 5,
            threshold: None,
            session_id: None,
        };
        let result = h.tools.semantic_search(params).await.unwrap();
        assert!(result.raw_hits.is_empty());
        assert!(result.abstract_hits.is_empty());
    }

    #[tokio::test]
    async fn semantic_search_returns_raw_hits() {
        let h = setup();

        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "hello world",
            0.5,
            Vec::new(),
        );
        let node_id = node.id;
        let emb = h.embedder.embed_text("hello world").await.unwrap();
        h.repo.insert_raw(node).await.unwrap();
        h.vector.index_raw(node_id, emb).await.unwrap();

        let params = MemorySearchParams {
            query: "hello world".to_string(),
            target: MemorySearchTarget::Raw,
            top_k: 5,
            threshold: None,
            session_id: None,
        };
        let result = h.tools.semantic_search(params).await.unwrap();
        assert!(!result.raw_hits.is_empty());
        assert_eq!(result.raw_hits[0].node.id, node_id);
        assert!(result.abstract_hits.is_empty());
    }

    // C8 regression: session-tagged embeddings (what the engine actually
    // persists) must be returned by a session-scoped semantic search, and never
    // leak to a different session.
    #[tokio::test]
    async fn semantic_search_returns_session_tagged_hits() {
        let h = setup();
        let session = SessionId::new();
        let other = SessionId::new();

        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            Some(session),
            None,
            "user",
            "hello world",
            0.5,
            Vec::new(),
        );
        let node_id = node.id;
        let emb = h.embedder.embed_text("hello world").await.unwrap();
        h.repo.insert_raw(node).await.unwrap();
        h.vector
            .index_raw_with_session(node_id, emb, Some(session))
            .await
            .unwrap();

        let scoped = h
            .tools
            .semantic_search(MemorySearchParams {
                query: "hello world".to_string(),
                target: MemorySearchTarget::Raw,
                top_k: 5,
                threshold: None,
                session_id: Some(session.to_string()),
            })
            .await
            .unwrap();
        assert_eq!(scoped.raw_hits.len(), 1);
        assert_eq!(scoped.raw_hits[0].node.id, node_id);

        let foreign = h
            .tools
            .semantic_search(MemorySearchParams {
                query: "hello world".to_string(),
                target: MemorySearchTarget::Raw,
                top_k: 5,
                threshold: None,
                session_id: Some(other.to_string()),
            })
            .await
            .unwrap();
        assert!(
            foreign.raw_hits.is_empty(),
            "a session-scoped search must not return another session's memory"
        );
    }

    #[tokio::test]
    async fn semantic_search_returns_abstract_hits() {
        let h = setup();

        let node = AbstractNode::new(
            "hello concept",
            "a summary about hello",
            References::default(),
            GraphFragment::default(),
            AbstractNodeMetadata::default(),
        );
        let node_id = node.id;
        let emb = h
            .embedder
            .embed_text("hello concept: a summary about hello")
            .await
            .unwrap();
        h.repo.insert_abstract(node).await.unwrap();
        h.vector.index_abstract(node_id, emb).await.unwrap();

        let params = MemorySearchParams {
            query: "hello concept: a summary about hello".to_string(),
            target: MemorySearchTarget::Abstract,
            top_k: 5,
            threshold: None,
            session_id: None,
        };
        let result = h.tools.semantic_search(params).await.unwrap();
        assert!(result.raw_hits.is_empty());
        assert!(!result.abstract_hits.is_empty());
        assert_eq!(result.abstract_hits[0].node.id, node_id);
    }

    #[tokio::test]
    async fn semantic_search_both_targets() {
        let h = setup();

        let raw = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "shared topic here",
            0.5,
            Vec::new(),
        );
        let raw_id = raw.id;
        let raw_emb = h.embedder.embed_text("shared topic here").await.unwrap();
        h.repo.insert_raw(raw).await.unwrap();
        h.vector.index_raw(raw_id, raw_emb).await.unwrap();

        let abs = AbstractNode::new(
            "shared topic",
            "summary about shared topic",
            References::default(),
            GraphFragment::default(),
            AbstractNodeMetadata::default(),
        );
        let abs_id = abs.id;
        let abs_emb = h
            .embedder
            .embed_text("shared topic: summary about shared topic")
            .await
            .unwrap();
        h.repo.insert_abstract(abs).await.unwrap();
        h.vector.index_abstract(abs_id, abs_emb).await.unwrap();

        let params = MemorySearchParams {
            query: "shared topic".to_string(),
            target: MemorySearchTarget::Both,
            top_k: 5,
            threshold: None,
            session_id: None,
        };
        let result = h.tools.semantic_search(params).await.unwrap();
        assert!(!result.raw_hits.is_empty());
        assert!(!result.abstract_hits.is_empty());
    }

    #[tokio::test]
    async fn semantic_search_respects_threshold() {
        let h = setup();

        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "apples and oranges",
            0.5,
            Vec::new(),
        );
        let node_id = node.id;
        let emb = h.embedder.embed_text("apples and oranges").await.unwrap();
        h.repo.insert_raw(node).await.unwrap();
        h.vector.index_raw(node_id, emb).await.unwrap();

        let params = MemorySearchParams {
            query: "completely unrelated rockets in space".to_string(),
            target: MemorySearchTarget::Raw,
            top_k: 5,
            threshold: Some(0.99),
            session_id: None,
        };
        let result = h.tools.semantic_search(params).await.unwrap();
        assert!(result.raw_hits.is_empty());
    }

    // Argument-bound clamping is enforced once at the engine chokepoint
    // (`prepare_tool_call_for_config`, tested in session_engine), so MemoryTools
    // no longer re-clamps and the former per-tool clamp tests were removed. [Q2]

    #[tokio::test]
    async fn timeline_search_empty() {
        let h = setup();
        let params = TimelineSearchParams {
            session_id: None,
            from: None,
            to: None,
            limit: 10,
        };
        let result = h.tools.timeline_search(params).await.unwrap();
        assert!(result.raw_nodes.is_empty());
    }

    #[tokio::test]
    async fn timeline_search_returns_recent_nodes() {
        let h = setup();

        for i in 0..5 {
            let node = RawNode::text(
                RawNodeKind::UserUtterance,
                None,
                None,
                "user",
                format!("message {i}"),
                0.5,
                Vec::new(),
            );
            h.repo.insert_raw(node).await.unwrap();
        }

        let params = TimelineSearchParams {
            session_id: None,
            from: None,
            to: None,
            limit: 3,
        };
        let result = h.tools.timeline_search(params).await.unwrap();
        assert_eq!(result.raw_nodes.len(), 3);
    }

    #[tokio::test]
    async fn timeline_search_with_session_filter() {
        let h = setup();
        let sid = SessionId::new();

        let in_session = RawNode::text(
            RawNodeKind::UserUtterance,
            Some(sid),
            None,
            "user",
            "in session",
            0.5,
            Vec::new(),
        );
        h.repo.insert_raw(in_session).await.unwrap();

        let outside = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "outside session",
            0.5,
            Vec::new(),
        );
        h.repo.insert_raw(outside).await.unwrap();

        let params = TimelineSearchParams {
            session_id: Some(sid.to_string()),
            from: None,
            to: None,
            limit: 10,
        };
        let result = h.tools.timeline_search(params).await.unwrap();
        assert_eq!(result.raw_nodes.len(), 1);
        assert_eq!(result.raw_nodes[0].content_text(), "in session");
    }

    #[tokio::test]
    async fn timeline_search_invalid_session_id_returns_error() {
        let h = setup();
        let params = TimelineSearchParams {
            session_id: Some("not-a-uuid".to_string()),
            from: None,
            to: None,
            limit: 10,
        };
        let result = h.tools.timeline_search(params).await;
        assert!(result.is_err());
    }
}
