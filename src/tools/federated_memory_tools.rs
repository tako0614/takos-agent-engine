use std::cmp::Ordering;
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
use crate::tools::memory_tools::{
    GraphSearchParams, MemorySearchTarget, MemoryToolBounds, TimelineSearchParams,
};

#[derive(Clone)]
pub struct MemorySource {
    pub source_id: String,
    pub repository: Arc<dyn NodeRepository>,
    pub vector_index: Arc<dyn VectorIndex>,
    pub graph_repository: Option<Arc<dyn GraphRepository>>,
}

impl MemorySource {
    pub fn new(
        source_id: impl Into<String>,
        repository: Arc<dyn NodeRepository>,
        vector_index: Arc<dyn VectorIndex>,
    ) -> Self {
        Self {
            source_id: source_id.into(),
            repository,
            vector_index,
            graph_repository: None,
        }
    }

    #[must_use]
    pub fn with_graph_repository(mut self, graph_repository: Arc<dyn GraphRepository>) -> Self {
        self.graph_repository = Some(graph_repository);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedMemorySearchParams {
    pub query: String,
    #[serde(default = "default_federated_memory_search_target")]
    pub target: MemorySearchTarget,
    #[serde(default = "default_federated_memory_search_top_k")]
    pub top_k: usize,
    pub threshold: Option<f32>,
}

fn default_federated_memory_search_target() -> MemorySearchTarget {
    MemorySearchTarget::Both
}

fn default_federated_memory_search_top_k() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedMemorySearchResult {
    pub hits: Vec<FederatedMemoryHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedMemoryHit {
    pub source_id: String,
    pub score: f32,
    pub node: FederatedMemoryNode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", content = "node", rename_all = "snake_case")]
pub enum FederatedMemoryNode {
    Raw(RawNode),
    Abstract(AbstractNode),
}

impl FederatedMemoryNode {
    fn id_string(&self) -> String {
        match self {
            Self::Raw(node) => node.id.to_string(),
            Self::Abstract(node) => node.id.to_string(),
        }
    }

    fn kind_rank(&self) -> u8 {
        match self {
            Self::Raw(_) => 0,
            Self::Abstract(_) => 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedGraphSearchResult {
    pub hits: Vec<FederatedGraphSearchHit>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedGraphSearchHit {
    pub source_id: String,
    pub node: AbstractNode,
    pub depth: usize,
    pub via_predicate: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedTimelineSearchResult {
    pub raw_nodes: Vec<FederatedTimelineRawNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedTimelineRawNode {
    pub source_id: String,
    pub node: RawNode,
}

pub struct FederatedMemoryTools {
    sources: Vec<MemorySource>,
    embedder: Arc<dyn Embedder>,
    bounds: MemoryToolBounds,
}

impl FederatedMemoryTools {
    pub fn new(sources: Vec<MemorySource>, embedder: Arc<dyn Embedder>) -> Self {
        Self::new_with_bounds(sources, embedder, MemoryToolBounds::default())
    }

    pub fn new_with_config(
        sources: Vec<MemorySource>,
        embedder: Arc<dyn Embedder>,
        config: &ToolsConfig,
    ) -> Self {
        Self::new_with_bounds(sources, embedder, MemoryToolBounds::from(config))
    }

    pub fn new_with_bounds(
        sources: Vec<MemorySource>,
        embedder: Arc<dyn Embedder>,
        bounds: MemoryToolBounds,
    ) -> Self {
        Self {
            sources,
            embedder,
            bounds,
        }
    }

    #[must_use]
    pub fn sources(&self) -> &[MemorySource] {
        &self.sources
    }

    /// # Errors
    ///
    /// Returns an [`EngineError`] when the embedder fails or any federated
    /// source repository returns an error during the per-source search.
    pub async fn semantic_search(
        &self,
        params: FederatedMemorySearchParams,
    ) -> Result<FederatedMemorySearchResult> {
        let top_k = self.bounds.clamp_memory_search_top_k(params.top_k);
        let threshold = params.threshold.unwrap_or(0.0);
        let query_embedding = self.embedder.embed_text(&params.query).await?;
        let mut hits = Vec::new();

        for source in &self.sources {
            if matches!(
                params.target,
                MemorySearchTarget::Raw | MemorySearchTarget::Both
            ) {
                self.search_raw_source(source, &query_embedding, top_k, threshold, &mut hits)
                    .await?;
            }

            if matches!(
                params.target,
                MemorySearchTarget::Abstract | MemorySearchTarget::Both
            ) {
                self.search_abstract_source(source, &query_embedding, top_k, threshold, &mut hits)
                    .await?;
            }
        }

        sort_federated_hits(&mut hits);
        hits.truncate(top_k);
        Ok(FederatedMemorySearchResult { hits })
    }

    /// # Errors
    ///
    /// Returns an [`EngineError`] when the start node id cannot be parsed or
    /// any federated graph / node repository returns an error during traversal.
    pub async fn graph_search(
        &self,
        params: GraphSearchParams,
    ) -> Result<FederatedGraphSearchResult> {
        let start = AbstractNodeId::from_str(&params.start_node_id)
            .map_err(|err| EngineError::Tool(format!("invalid abstract node id: {err}")))?;
        let max_depth = self.bounds.clamp_graph_search_depth(params.max_depth);
        let mut hits = Vec::new();

        for source in &self.sources {
            let Some(graph_repository) = &source.graph_repository else {
                continue;
            };
            for hit in graph_repository
                .traverse(&start, max_depth, params.relation_types.as_deref())
                .await?
            {
                if let Some(node) = source.repository.get_abstract(&hit.node_id).await? {
                    hits.push(FederatedGraphSearchHit {
                        source_id: source.source_id.clone(),
                        node,
                        depth: hit.depth,
                        via_predicate: hit.via_predicate,
                    });
                }
            }
        }

        sort_federated_graph_hits(&mut hits);
        Ok(FederatedGraphSearchResult { hits })
    }

    /// # Errors
    ///
    /// Returns an [`EngineError`] when the optional session id cannot be
    /// parsed or any federated node repository returns an error.
    pub async fn timeline_search(
        &self,
        params: TimelineSearchParams,
    ) -> Result<FederatedTimelineSearchResult> {
        let session_id = params
            .session_id
            .as_deref()
            .map(SessionId::from_str)
            .transpose()
            .map_err(|err| EngineError::Tool(format!("invalid session id: {err}")))?;
        let limit = self.bounds.clamp_timeline_search_limit(params.limit);
        let mut raw_nodes = Vec::new();

        for source in &self.sources {
            for node in source
                .repository
                .timeline_raw(session_id.as_ref(), params.from, params.to, limit)
                .await?
            {
                raw_nodes.push(FederatedTimelineRawNode {
                    source_id: source.source_id.clone(),
                    node,
                });
            }
        }

        sort_federated_timeline_nodes(&mut raw_nodes);
        raw_nodes.truncate(limit);
        Ok(FederatedTimelineSearchResult { raw_nodes })
    }

    async fn search_raw_source(
        &self,
        source: &MemorySource,
        query_embedding: &Embedding,
        top_k: usize,
        threshold: f32,
        hits: &mut Vec<FederatedMemoryHit>,
    ) -> Result<()> {
        for candidate in source
            .vector_index
            .search_raw(query_embedding, top_k)
            .await?
        {
            if candidate.score < threshold {
                continue;
            }
            if let Some(node) = source.repository.get_raw(&candidate.id).await? {
                hits.push(FederatedMemoryHit {
                    source_id: source.source_id.clone(),
                    score: candidate.score,
                    node: FederatedMemoryNode::Raw(node),
                });
            }
        }
        Ok(())
    }

    async fn search_abstract_source(
        &self,
        source: &MemorySource,
        query_embedding: &Embedding,
        top_k: usize,
        threshold: f32,
        hits: &mut Vec<FederatedMemoryHit>,
    ) -> Result<()> {
        for candidate in source
            .vector_index
            .search_abstract(query_embedding, top_k)
            .await?
        {
            if candidate.score < threshold {
                continue;
            }
            if let Some(node) = source.repository.get_abstract(&candidate.id).await? {
                hits.push(FederatedMemoryHit {
                    source_id: source.source_id.clone(),
                    score: candidate.score,
                    node: FederatedMemoryNode::Abstract(node),
                });
            }
        }
        Ok(())
    }
}

fn sort_federated_hits(hits: &mut [FederatedMemoryHit]) {
    hits.sort_by(compare_federated_hits);
}

fn compare_federated_hits(left: &FederatedMemoryHit, right: &FederatedMemoryHit) -> Ordering {
    right
        .score
        .partial_cmp(&left.score)
        .unwrap_or(Ordering::Equal)
        .then_with(|| left.source_id.cmp(&right.source_id))
        .then_with(|| left.node.id_string().cmp(&right.node.id_string()))
        .then_with(|| left.node.kind_rank().cmp(&right.node.kind_rank()))
}

fn sort_federated_graph_hits(hits: &mut [FederatedGraphSearchHit]) {
    hits.sort_by(|left, right| {
        left.depth
            .cmp(&right.depth)
            .then_with(|| left.source_id.cmp(&right.source_id))
            .then_with(|| left.node.id.to_string().cmp(&right.node.id.to_string()))
            .then_with(|| left.via_predicate.cmp(&right.via_predicate))
    });
}

fn sort_federated_timeline_nodes(nodes: &mut [FederatedTimelineRawNode]) {
    nodes.sort_by(compare_timeline_position);
}

fn compare_timeline_position(
    left: &FederatedTimelineRawNode,
    right: &FederatedTimelineRawNode,
) -> Ordering {
    compare_timestamp_desc(left.node.timestamp, right.node.timestamp)
        .then_with(|| left.source_id.cmp(&right.source_id))
        .then_with(|| left.node.id.to_string().cmp(&right.node.id.to_string()))
}

fn compare_timestamp_desc(left: DateTime<Utc>, right: DateTime<Utc>) -> Ordering {
    right.cmp(&left)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    use crate::domain::{AbstractNodeMetadata, GraphFragment, RawNodeKind, References, Relation};
    use crate::model::Embedding;
    use crate::storage::{InMemoryGraphRepository, InMemoryNodeRepository, InMemoryVectorIndex};

    #[derive(Debug, Clone)]
    struct FixedEmbedder;

    #[async_trait]
    impl Embedder for FixedEmbedder {
        async fn embed_text(&self, _text: &str) -> Result<Embedding> {
            Ok(Embedding(vec![1.0, 0.0]))
        }
    }

    struct TestSource {
        repo: Arc<InMemoryNodeRepository>,
        vector: Arc<InMemoryVectorIndex>,
        source: MemorySource,
    }

    fn test_source(source_id: &str) -> TestSource {
        let repo = Arc::new(InMemoryNodeRepository::default());
        let vector = Arc::new(InMemoryVectorIndex::default());
        let source = MemorySource::new(
            source_id,
            repo.clone() as Arc<dyn NodeRepository>,
            vector.clone() as Arc<dyn VectorIndex>,
        );
        TestSource {
            repo,
            vector,
            source,
        }
    }

    fn test_source_with_graph(source_id: &str) -> (TestSource, Arc<InMemoryGraphRepository>) {
        let graph = Arc::new(InMemoryGraphRepository::default());
        let mut source = test_source(source_id);
        source.source = source
            .source
            .with_graph_repository(graph.clone() as Arc<dyn GraphRepository>);
        (source, graph)
    }

    #[tokio::test]
    async fn semantic_search_merges_sources_with_stable_order() {
        let source_b = test_source("source-b");
        let raw_b = RawNode::text(
            RawNodeKind::Note,
            None,
            None,
            "test",
            "same score from b",
            0.5,
            Vec::new(),
        );
        let raw_b_id = raw_b.id;
        source_b.repo.insert_raw(raw_b).await.unwrap();
        source_b
            .vector
            .index_raw(raw_b_id, Embedding(vec![1.0, 0.0]))
            .await
            .unwrap();

        let source_a = test_source("source-a");
        let raw_a = RawNode::text(
            RawNodeKind::Note,
            None,
            None,
            "test",
            "same score from a",
            0.5,
            Vec::new(),
        );
        let raw_a_id = raw_a.id;
        source_a.repo.insert_raw(raw_a).await.unwrap();
        source_a
            .vector
            .index_raw(raw_a_id, Embedding(vec![1.0, 0.0]))
            .await
            .unwrap();

        let tools = FederatedMemoryTools::new(
            vec![source_b.source, source_a.source],
            Arc::new(FixedEmbedder),
        );
        let result = tools
            .semantic_search(FederatedMemorySearchParams {
                query: "query".to_string(),
                target: MemorySearchTarget::Raw,
                top_k: 10,
                threshold: None,
            })
            .await
            .unwrap();

        assert_eq!(result.hits.len(), 2);
        assert_eq!(result.hits[0].source_id, "source-a");
        assert_eq!(result.hits[1].source_id, "source-b");
    }

    #[tokio::test]
    async fn semantic_search_respects_threshold_top_k_and_target() {
        let source = test_source("local");

        let raw = RawNode::text(
            RawNodeKind::Note,
            None,
            None,
            "test",
            "raw should not be searched",
            0.5,
            Vec::new(),
        );
        let raw_id = raw.id;
        source.repo.insert_raw(raw).await.unwrap();
        source
            .vector
            .index_raw(raw_id, Embedding(vec![1.0, 0.0]))
            .await
            .unwrap();

        let high = AbstractNode::new(
            "high",
            "high score",
            References::default(),
            GraphFragment::default(),
            AbstractNodeMetadata::default(),
        );
        let high_id = high.id;
        source.repo.insert_abstract(high).await.unwrap();
        source
            .vector
            .index_abstract(high_id, Embedding(vec![1.0, 0.0]))
            .await
            .unwrap();

        let low = AbstractNode::new(
            "low",
            "low score",
            References::default(),
            GraphFragment::default(),
            AbstractNodeMetadata::default(),
        );
        let low_id = low.id;
        source.repo.insert_abstract(low).await.unwrap();
        source
            .vector
            .index_abstract(low_id, Embedding(vec![0.5, 1.0]))
            .await
            .unwrap();

        let tools = FederatedMemoryTools::new(vec![source.source], Arc::new(FixedEmbedder));
        let result = tools
            .semantic_search(FederatedMemorySearchParams {
                query: "query".to_string(),
                target: MemorySearchTarget::Abstract,
                top_k: 1,
                threshold: Some(0.9),
            })
            .await
            .unwrap();

        assert_eq!(result.hits.len(), 1);
        assert_eq!(result.hits[0].source_id, "local");
        assert!(matches!(
            result.hits[0].node,
            FederatedMemoryNode::Abstract(_)
        ));
    }

    #[tokio::test]
    async fn semantic_search_ties_by_node_id_within_source() {
        let source = test_source("local");
        let mut ids = Vec::new();
        for text in ["first", "second"] {
            let raw = RawNode::text(RawNodeKind::Note, None, None, "test", text, 0.5, Vec::new());
            let id = raw.id;
            ids.push(id.to_string());
            source.repo.insert_raw(raw).await.unwrap();
            source
                .vector
                .index_raw(id, Embedding(vec![1.0, 0.0]))
                .await
                .unwrap();
        }
        ids.sort();

        let tools = FederatedMemoryTools::new(vec![source.source], Arc::new(FixedEmbedder));
        let result = tools
            .semantic_search(FederatedMemorySearchParams {
                query: "query".to_string(),
                target: MemorySearchTarget::Raw,
                top_k: 10,
                threshold: None,
            })
            .await
            .unwrap();
        let result_ids: Vec<_> = result.hits.iter().map(|hit| hit.node.id_string()).collect();

        assert_eq!(result_ids, ids);
    }

    #[tokio::test]
    async fn graph_search_merges_sources_with_source_ids() {
        let (source_b, graph_b) = test_source_with_graph("source-b");
        let (source_a, graph_a) = test_source_with_graph("source-a");
        let leaf = AbstractNode::new(
            "leaf",
            "shared leaf",
            References::default(),
            GraphFragment::default(),
            AbstractNodeMetadata::default(),
        );
        let root = AbstractNode::new(
            "root",
            "shared root",
            References::default(),
            GraphFragment {
                entities: Vec::new(),
                relations: vec![Relation {
                    subject: "root".to_string(),
                    predicate: "links".to_string(),
                    object: leaf.id.to_string(),
                    weight: 1.0,
                    provenance_raw_node_ids: Vec::new(),
                }],
            },
            AbstractNodeMetadata::default(),
        );

        for (source, graph) in [(&source_b, &graph_b), (&source_a, &graph_a)] {
            for node in [&root, &leaf] {
                source.repo.insert_abstract(node.clone()).await.unwrap();
                graph.index_abstract(node).await.unwrap();
            }
        }

        let tools = FederatedMemoryTools::new(
            vec![source_b.source, source_a.source],
            Arc::new(FixedEmbedder),
        );
        let result = tools
            .graph_search(GraphSearchParams {
                start_node_id: root.id.to_string(),
                max_depth: 1,
                relation_types: Some(vec!["links".to_string()]),
            })
            .await
            .unwrap();

        assert_eq!(result.hits.len(), 4);
        assert_eq!(result.hits[0].source_id, "source-a");
        assert_eq!(result.hits[0].depth, 0);
        assert_eq!(result.hits[1].source_id, "source-b");
        assert_eq!(result.hits[1].depth, 0);
        assert_eq!(result.hits[2].source_id, "source-a");
        assert_eq!(result.hits[2].depth, 1);
        assert_eq!(result.hits[3].source_id, "source-b");
        assert_eq!(result.hits[3].depth, 1);
    }

    #[tokio::test]
    async fn timeline_search_merges_sources_with_global_limit() {
        let source_b = test_source("source-b");
        let source_a = test_source("source-a");

        let old = RawNode::text(
            RawNodeKind::Note,
            None,
            None,
            "test",
            "old from a",
            0.5,
            Vec::new(),
        );
        let old_time = old.timestamp;
        source_a.repo.insert_raw(old).await.unwrap();

        let mut newest = RawNode::text(
            RawNodeKind::Note,
            None,
            None,
            "test",
            "newest from b",
            0.5,
            Vec::new(),
        );
        newest.timestamp = old_time + chrono::Duration::seconds(1);
        source_b.repo.insert_raw(newest).await.unwrap();

        let tools = FederatedMemoryTools::new(
            vec![source_a.source, source_b.source],
            Arc::new(FixedEmbedder),
        );
        let result = tools
            .timeline_search(TimelineSearchParams {
                session_id: None,
                from: None,
                to: None,
                limit: 1,
            })
            .await
            .unwrap();

        assert_eq!(result.raw_nodes.len(), 1);
        assert_eq!(result.raw_nodes[0].source_id, "source-b");
    }
}
