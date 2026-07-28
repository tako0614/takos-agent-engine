use std::time::Duration;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::domain::{AbstractNode, DistillationState, LoopState, OverflowPolicy, RawNode};
use crate::error::Result;
use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};
use crate::model::embedding::Embedding;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredRawRef {
    pub id: RawNodeId,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ScoredAbstractRef {
    pub id: AbstractNodeId,
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphTraversalHit {
    pub node_id: AbstractNodeId,
    pub depth: usize,
    pub via_predicate: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawLifecyclePatch {
    pub distillation_state: Option<DistillationState>,
    pub overflow: Option<OverflowPolicy>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NodeCommit<T> {
    pub node: T,
    pub inserted: bool,
    /// `true` when node, embedding and (for abstract nodes) graph projection
    /// were committed by one durable mutation boundary.
    pub projections_committed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DistillationClaim {
    pub session_id: SessionId,
    pub loop_id: LoopId,
    pub input_digest: String,
    pub lease_id: String,
    pub expires_at: DateTime<Utc>,
}

#[async_trait]
pub trait NodeRepository: Send + Sync {
    async fn insert_raw(&self, node: RawNode) -> Result<()>;
    async fn insert_abstract(&self, node: AbstractNode) -> Result<()>;
    async fn insert_raw_once(&self, node: RawNode) -> Result<NodeCommit<RawNode>> {
        if let Some(operation_key) = &node.operation_key {
            if let Some(existing) = self.get_raw_by_operation_key(operation_key).await? {
                return Ok(NodeCommit {
                    node: existing,
                    inserted: false,
                    projections_committed: false,
                });
            }
        }
        self.insert_raw(node.clone()).await?;
        Ok(NodeCommit {
            node,
            inserted: true,
            projections_committed: false,
        })
    }
    async fn insert_abstract_once(&self, node: AbstractNode) -> Result<NodeCommit<AbstractNode>> {
        if let Some(operation_key) = &node.operation_key {
            if let Some(existing) = self.get_abstract_by_operation_key(operation_key).await? {
                return Ok(NodeCommit {
                    node: existing,
                    inserted: false,
                    projections_committed: false,
                });
            }
        }
        self.insert_abstract(node.clone()).await?;
        Ok(NodeCommit {
            node,
            inserted: true,
            projections_committed: false,
        })
    }
    async fn commit_raw(
        &self,
        node: RawNode,
        _embedding: Embedding,
    ) -> Result<NodeCommit<RawNode>> {
        self.insert_raw_once(node).await
    }
    async fn commit_abstract(
        &self,
        node: AbstractNode,
        _embedding: Embedding,
    ) -> Result<NodeCommit<AbstractNode>> {
        self.insert_abstract_once(node).await
    }
    async fn get_raw(&self, id: &RawNodeId) -> Result<Option<RawNode>>;
    async fn get_abstract(&self, id: &AbstractNodeId) -> Result<Option<AbstractNode>>;
    async fn get_raw_by_operation_key(&self, operation_key: &str) -> Result<Option<RawNode>>;
    async fn get_abstract_by_operation_key(
        &self,
        operation_key: &str,
    ) -> Result<Option<AbstractNode>>;
    async fn list_raw(&self, ids: &[RawNodeId]) -> Result<Vec<RawNode>>;
    async fn list_abstract(&self, ids: &[AbstractNodeId]) -> Result<Vec<AbstractNode>>;
    async fn recent_session_raw(
        &self,
        session_id: &SessionId,
        limit: usize,
    ) -> Result<Vec<RawNode>>;
    async fn session_raw(&self, session_id: &SessionId) -> Result<Vec<RawNode>>;
    async fn raw_for_loop(&self, loop_id: &LoopId) -> Result<Vec<RawNode>>;
    async fn raw_for_loop_bounded(&self, loop_id: &LoopId, limit: usize) -> Result<Vec<RawNode>> {
        let mut nodes = self.raw_for_loop(loop_id).await?;
        nodes.truncate(limit);
        Ok(nodes)
    }
    async fn timeline_raw(
        &self,
        session_id: Option<&SessionId>,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<RawNode>>;
    async fn update_raw_lifecycle(
        &self,
        ids: &[RawNodeId],
        patch: &RawLifecyclePatch,
    ) -> Result<()>;
    async fn undistilled_raw(&self, limit: usize, only_pushed_out: bool) -> Result<Vec<RawNode>>;
    async fn try_claim_distillation(
        &self,
        session_id: SessionId,
        loop_id: LoopId,
        input_digest: String,
        lease_for: Duration,
    ) -> Result<Option<DistillationClaim>>;
    async fn distillation_claim_is_current(&self, claim: &DistillationClaim) -> Result<bool>;
    async fn release_distillation_claim(&self, claim: &DistillationClaim) -> Result<()>;
}

#[async_trait]
pub trait VectorIndex: Send + Sync {
    async fn index_raw(&self, id: RawNodeId, embedding: Embedding) -> Result<()>;
    async fn index_abstract(&self, id: AbstractNodeId, embedding: Embedding) -> Result<()>;

    /// Index a raw embedding together with the session that produced it.
    ///
    /// The default implementation defers to [`VectorIndex::index_raw`] so
    /// existing legacy callers keep working. Production implementations
    /// override this to persist the `session_id` alongside the embedding so
    /// later searches can apply a per-session filter.
    async fn index_raw_with_session(
        &self,
        id: RawNodeId,
        embedding: Embedding,
        _session_id: Option<SessionId>,
    ) -> Result<()> {
        self.index_raw(id, embedding).await
    }

    /// Same as [`VectorIndex::index_raw_with_session`] but for abstract nodes.
    async fn index_abstract_with_session(
        &self,
        id: AbstractNodeId,
        embedding: Embedding,
        _session_id: Option<SessionId>,
    ) -> Result<()> {
        self.index_abstract(id, embedding).await
    }

    /// Search raw embeddings.
    ///
    /// When `session_id` is `Some`, only entries whose stored session_id
    /// matches the requested session are returned. When `session_id` is `None`,
    /// only entries without a stored session_id (= legacy entries that
    /// pre-date the session-aware index) are returned. This convention keeps
    /// existing legacy data reachable without leaking across sessions when
    /// session-scoped retrieval is requested.
    async fn search_raw(
        &self,
        query: &Embedding,
        top_k: usize,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<ScoredRawRef>>;

    async fn search_abstract(
        &self,
        query: &Embedding,
        top_k: usize,
        session_id: Option<&SessionId>,
    ) -> Result<Vec<ScoredAbstractRef>>;
}

#[async_trait]
pub trait GraphRepository: Send + Sync {
    async fn index_abstract(&self, node: &AbstractNode) -> Result<()>;
    async fn traverse(
        &self,
        start: &AbstractNodeId,
        max_depth: usize,
        relation_types: Option<&[String]>,
        session_id: Option<&SessionId>,
        max_hits: usize,
    ) -> Result<Vec<GraphTraversalHit>>;
}

#[async_trait]
pub trait LoopStateRepository: Send + Sync {
    async fn save_checkpoint(&self, state: LoopState) -> Result<()>;
    async fn load_checkpoint(
        &self,
        session_id: &SessionId,
        loop_id: &LoopId,
    ) -> Result<Option<LoopState>>;
    async fn clear_checkpoint(&self, session_id: &SessionId, loop_id: &LoopId) -> Result<()>;
}
