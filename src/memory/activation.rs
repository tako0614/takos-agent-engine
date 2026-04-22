use std::cmp::Ordering;
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::config::EngineConfig;
use crate::domain::{AbstractNode, DistillationState, RawNode};
use crate::error::Result;
use crate::model::embedding::Embedding;
use crate::storage::{NodeRepository, VectorIndex};

use super::scoring::ScoringPolicy;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedRawNode {
    pub node: RawNode,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RankedAbstractNode {
    pub node: AbstractNode,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct ActivatedMemory {
    pub raw_nodes: Vec<RankedRawNode>,
    pub abstract_nodes: Vec<RankedAbstractNode>,
}

pub struct ActivationService {
    repository: Arc<dyn NodeRepository>,
    vector_index: Arc<dyn VectorIndex>,
    scoring_policy: Arc<dyn ScoringPolicy>,
}

impl ActivationService {
    pub fn new(
        repository: Arc<dyn NodeRepository>,
        vector_index: Arc<dyn VectorIndex>,
        scoring_policy: Arc<dyn ScoringPolicy>,
    ) -> Self {
        Self {
            repository,
            vector_index,
            scoring_policy,
        }
    }

    pub async fn activate(
        &self,
        config: &EngineConfig,
        query_embedding: &Embedding,
        now: DateTime<Utc>,
    ) -> Result<ActivatedMemory> {
        let raw_ratio = config.memory.activation.target_ratio.raw.max(1);
        let abstract_ratio = config.memory.activation.target_ratio.abstract_nodes.max(1);
        let total_ratio = raw_ratio + abstract_ratio;
        let top_k_total = config.memory.activation.top_k_total.max(2);
        let raw_budget = ((top_k_total * raw_ratio) / total_ratio).max(1);
        let abstract_budget = top_k_total.saturating_sub(raw_budget).max(1);
        let search_window = top_k_total * 2;

        let raw_candidates = self
            .vector_index
            .search_raw(query_embedding, search_window)
            .await?;
        let mut raw_nodes = Vec::new();
        for candidate in raw_candidates {
            if let Some(node) = self.repository.get_raw(&candidate.id).await? {
                let threshold = if node.overflow.was_pushed_out_of_session
                    && node.distillation_state != DistillationState::Distilled
                {
                    config.memory.retrieval.relaxed_threshold_for_pushed_raw
                } else {
                    config.memory.retrieval.similarity_threshold.raw
                };
                if candidate.score < threshold {
                    continue;
                }
                raw_nodes.push(RankedRawNode {
                    score: self.scoring_policy.score_raw(candidate.score, &node, now),
                    node,
                });
            }
        }
        raw_nodes.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        raw_nodes.truncate(raw_budget);

        let abstract_candidates = self
            .vector_index
            .search_abstract(query_embedding, search_window)
            .await?;
        let mut abstract_nodes = Vec::new();
        for candidate in abstract_candidates {
            if let Some(node) = self.repository.get_abstract(&candidate.id).await? {
                if candidate.score < config.memory.retrieval.similarity_threshold.abstract_nodes {
                    continue;
                }
                abstract_nodes.push(RankedAbstractNode {
                    score: self
                        .scoring_policy
                        .score_abstract(candidate.score, &node, now),
                    node,
                });
            }
        }
        abstract_nodes.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
        });
        abstract_nodes.truncate(abstract_budget);

        Ok(ActivatedMemory {
            raw_nodes,
            abstract_nodes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineConfig;
    use crate::domain::{
        AbstractNode, AbstractNodeMetadata, GraphFragment, RawNode, RawNodeKind, References,
    };
    use crate::memory::scoring::DefaultScoringPolicy;
    use crate::model::Embedder;
    use crate::storage::{InMemoryNodeRepository, InMemoryVectorIndex};
    use crate::test_support::TestHashEmbedder;

    struct TestHarness {
        service: ActivationService,
        repo: Arc<InMemoryNodeRepository>,
        vector: Arc<InMemoryVectorIndex>,
        embedder: TestHashEmbedder,
    }

    fn setup() -> TestHarness {
        let repo = Arc::new(InMemoryNodeRepository::default());
        let vector = Arc::new(InMemoryVectorIndex::default());
        let scoring = Arc::new(DefaultScoringPolicy::default());
        let embedder = TestHashEmbedder::default();
        let service = ActivationService::new(
            repo.clone() as Arc<dyn NodeRepository>,
            vector.clone() as Arc<dyn VectorIndex>,
            scoring as Arc<dyn ScoringPolicy>,
        );
        TestHarness {
            service,
            repo,
            vector,
            embedder,
        }
    }

    #[tokio::test]
    async fn activate_with_no_nodes_returns_empty() {
        let h = setup();
        let config = EngineConfig::default();
        let query_emb = h.embedder.embed_text("hello").await.unwrap();
        let result = h
            .service
            .activate(&config, &query_emb, Utc::now())
            .await
            .unwrap();
        assert!(result.raw_nodes.is_empty());
        assert!(result.abstract_nodes.is_empty());
    }

    #[tokio::test]
    async fn activate_returns_matching_raw_nodes() {
        let h = setup();
        let mut config = EngineConfig::default();
        config.memory.retrieval.similarity_threshold.raw = 0.0;
        config.memory.retrieval.similarity_threshold.abstract_nodes = 0.0;

        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "hello world",
            0.8,
            Vec::new(),
        );
        let node_id = node.id;
        let emb = h.embedder.embed_text("hello world").await.unwrap();
        h.repo.insert_raw(node).await.unwrap();
        h.vector.index_raw(node_id, emb).await.unwrap();

        let query_emb = h.embedder.embed_text("hello world").await.unwrap();
        let result = h
            .service
            .activate(&config, &query_emb, Utc::now())
            .await
            .unwrap();
        assert!(!result.raw_nodes.is_empty());
        assert_eq!(result.raw_nodes[0].node.id, node_id);
        assert!(result.raw_nodes[0].score > 0.5);
    }

    #[tokio::test]
    async fn activate_returns_matching_abstract_nodes() {
        let h = setup();
        let mut config = EngineConfig::default();
        config.memory.retrieval.similarity_threshold.raw = 0.0;
        config.memory.retrieval.similarity_threshold.abstract_nodes = 0.0;

        let node = AbstractNode::new(
            "test topic",
            "a summary about testing",
            References::default(),
            GraphFragment::default(),
            AbstractNodeMetadata::default(),
        );
        let node_id = node.id;
        let emb = h
            .embedder
            .embed_text("test topic: a summary about testing")
            .await
            .unwrap();
        h.repo.insert_abstract(node).await.unwrap();
        h.vector.index_abstract(node_id, emb).await.unwrap();

        let query_emb = h
            .embedder
            .embed_text("test topic: a summary about testing")
            .await
            .unwrap();
        let result = h
            .service
            .activate(&config, &query_emb, Utc::now())
            .await
            .unwrap();
        assert!(!result.abstract_nodes.is_empty());
        assert_eq!(result.abstract_nodes[0].node.id, node_id);
    }

    #[tokio::test]
    async fn activate_respects_similarity_threshold() {
        let h = setup();
        let mut config = EngineConfig::default();
        config.memory.retrieval.similarity_threshold.raw = 0.99;
        config.memory.retrieval.similarity_threshold.abstract_nodes = 0.99;

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

        let query_emb = h
            .embedder
            .embed_text("completely unrelated query about rockets")
            .await
            .unwrap();
        let result = h
            .service
            .activate(&config, &query_emb, Utc::now())
            .await
            .unwrap();
        assert!(result.raw_nodes.is_empty());
    }

    #[tokio::test]
    async fn activate_respects_budget() {
        let h = setup();
        let mut config = EngineConfig::default();
        config.memory.retrieval.similarity_threshold.raw = 0.0;
        config.memory.activation.top_k_total = 2;
        config.memory.activation.target_ratio.raw = 1;
        config.memory.activation.target_ratio.abstract_nodes = 1;

        for i in 0..10 {
            let text = format!("node number {i} with some content");
            let node = RawNode::text(
                RawNodeKind::UserUtterance,
                None,
                None,
                "user",
                &text,
                0.5,
                Vec::new(),
            );
            let node_id = node.id;
            let emb = h.embedder.embed_text(&text).await.unwrap();
            h.repo.insert_raw(node).await.unwrap();
            h.vector.index_raw(node_id, emb).await.unwrap();
        }

        let query_emb = h
            .embedder
            .embed_text("node number 5 with some content")
            .await
            .unwrap();
        let result = h
            .service
            .activate(&config, &query_emb, Utc::now())
            .await
            .unwrap();
        assert!(result.raw_nodes.len() <= 2);
    }

    #[tokio::test]
    async fn activate_results_sorted_by_score_descending() {
        let h = setup();
        let mut config = EngineConfig::default();
        config.memory.retrieval.similarity_threshold.raw = 0.0;
        config.memory.activation.top_k_total = 20;

        let texts = ["hello world", "goodbye world", "hello there friend"];
        for text in &texts {
            let node = RawNode::text(
                RawNodeKind::UserUtterance,
                None,
                None,
                "user",
                *text,
                0.5,
                Vec::new(),
            );
            let node_id = node.id;
            let emb = h.embedder.embed_text(text).await.unwrap();
            h.repo.insert_raw(node).await.unwrap();
            h.vector.index_raw(node_id, emb).await.unwrap();
        }

        let query_emb = h.embedder.embed_text("hello world").await.unwrap();
        let result = h
            .service
            .activate(&config, &query_emb, Utc::now())
            .await
            .unwrap();
        for window in result.raw_nodes.windows(2) {
            assert!(window[0].score >= window[1].score);
        }
    }
}
