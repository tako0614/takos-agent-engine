use std::collections::HashMap;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use tokio::sync::RwLock;

use crate::domain::{AbstractNode, DistillationState, LoopState, RawNode};
use crate::error::Result;
use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};

use super::traits::{LoopStateRepository, NodeRepository, RawLifecyclePatch};

#[derive(Debug, Default)]
pub struct InMemoryNodeRepository {
    raw_nodes: RwLock<HashMap<RawNodeId, RawNode>>,
    abstract_nodes: RwLock<HashMap<AbstractNodeId, AbstractNode>>,
    session_index: RwLock<HashMap<SessionId, Vec<RawNodeId>>>,
    loop_index: RwLock<HashMap<LoopId, Vec<RawNodeId>>>,
    raw_operation_index: RwLock<HashMap<String, RawNodeId>>,
    abstract_operation_index: RwLock<HashMap<String, AbstractNodeId>>,
}

#[async_trait]
impl NodeRepository for InMemoryNodeRepository {
    async fn insert_raw(&self, node: RawNode) -> Result<()> {
        if let Some(session_id) = node.session_id {
            let mut session_index = self.session_index.write().await;
            session_index.entry(session_id).or_default().push(node.id);
        }
        if let Some(loop_id) = node.loop_id {
            let mut loop_index = self.loop_index.write().await;
            loop_index.entry(loop_id).or_default().push(node.id);
        }
        if let Some(operation_key) = &node.operation_key {
            self.raw_operation_index
                .write()
                .await
                .insert(operation_key.clone(), node.id);
        }
        self.raw_nodes.write().await.insert(node.id, node);
        Ok(())
    }

    async fn insert_abstract(&self, node: AbstractNode) -> Result<()> {
        if let Some(operation_key) = &node.operation_key {
            self.abstract_operation_index
                .write()
                .await
                .insert(operation_key.clone(), node.id);
        }
        self.abstract_nodes.write().await.insert(node.id, node);
        Ok(())
    }

    async fn get_raw(&self, id: &RawNodeId) -> Result<Option<RawNode>> {
        Ok(self.raw_nodes.read().await.get(id).cloned())
    }

    async fn get_abstract(&self, id: &AbstractNodeId) -> Result<Option<AbstractNode>> {
        Ok(self.abstract_nodes.read().await.get(id).cloned())
    }

    async fn get_raw_by_operation_key(&self, operation_key: &str) -> Result<Option<RawNode>> {
        let raw_id = self
            .raw_operation_index
            .read()
            .await
            .get(operation_key)
            .copied();
        match raw_id {
            Some(raw_id) => self.get_raw(&raw_id).await,
            None => Ok(None),
        }
    }

    async fn get_abstract_by_operation_key(
        &self,
        operation_key: &str,
    ) -> Result<Option<AbstractNode>> {
        let abstract_id = self
            .abstract_operation_index
            .read()
            .await
            .get(operation_key)
            .copied();
        match abstract_id {
            Some(abstract_id) => self.get_abstract(&abstract_id).await,
            None => Ok(None),
        }
    }

    async fn list_raw(&self, ids: &[RawNodeId]) -> Result<Vec<RawNode>> {
        let guard = self.raw_nodes.read().await;
        Ok(ids.iter().filter_map(|id| guard.get(id).cloned()).collect())
    }

    async fn list_abstract(&self, ids: &[AbstractNodeId]) -> Result<Vec<AbstractNode>> {
        let guard = self.abstract_nodes.read().await;
        Ok(ids.iter().filter_map(|id| guard.get(id).cloned()).collect())
    }

    async fn recent_session_raw(
        &self,
        session_id: &SessionId,
        limit: usize,
    ) -> Result<Vec<RawNode>> {
        let mut nodes = self.session_raw(session_id).await?;
        nodes.reverse();
        nodes.truncate(limit);
        nodes.reverse();
        Ok(nodes)
    }

    async fn session_raw(&self, session_id: &SessionId) -> Result<Vec<RawNode>> {
        let ids = {
            let index = self.session_index.read().await;
            index.get(session_id).cloned().unwrap_or_default()
        };
        let guard = self.raw_nodes.read().await;
        let mut nodes: Vec<_> = ids.iter().filter_map(|id| guard.get(id).cloned()).collect();
        nodes.sort_by(|left, right| left.timestamp.cmp(&right.timestamp));
        Ok(nodes)
    }

    async fn raw_for_loop(&self, loop_id: &LoopId) -> Result<Vec<RawNode>> {
        let ids = {
            let index = self.loop_index.read().await;
            index.get(loop_id).cloned().unwrap_or_default()
        };
        let guard = self.raw_nodes.read().await;
        let mut nodes: Vec<_> = ids.iter().filter_map(|id| guard.get(id).cloned()).collect();
        nodes.sort_by(|left, right| left.timestamp.cmp(&right.timestamp));
        Ok(nodes)
    }

    async fn timeline_raw(
        &self,
        session_id: Option<&SessionId>,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<RawNode>> {
        let mut nodes: Vec<_> = if let Some(session_id) = session_id {
            self.session_raw(session_id).await?
        } else {
            self.raw_nodes.read().await.values().cloned().collect()
        };
        nodes.retain(|node| {
            let lower_ok = from.is_none_or(|value| node.timestamp >= value);
            let upper_ok = to.is_none_or(|value| node.timestamp <= value);
            lower_ok && upper_ok
        });
        nodes.sort_by(|left, right| right.timestamp.cmp(&left.timestamp));
        nodes.truncate(limit);
        Ok(nodes)
    }

    async fn update_raw_lifecycle(
        &self,
        ids: &[RawNodeId],
        patch: &RawLifecyclePatch,
    ) -> Result<()> {
        let mut guard = self.raw_nodes.write().await;
        for id in ids {
            if let Some(node) = guard.get_mut(id) {
                if let Some(distillation_state) = &patch.distillation_state {
                    node.distillation_state = distillation_state.clone();
                }
                if let Some(overflow) = &patch.overflow {
                    node.overflow = overflow.clone();
                }
            }
        }
        Ok(())
    }

    async fn undistilled_raw(&self, limit: usize, only_pushed_out: bool) -> Result<Vec<RawNode>> {
        let mut nodes: Vec<_> = self
            .raw_nodes
            .read()
            .await
            .values()
            .filter(|node| node.distillation_state != DistillationState::Distilled)
            .filter(|node| !only_pushed_out || node.overflow.was_pushed_out_of_session)
            .cloned()
            .collect();
        nodes.sort_by(|left, right| left.timestamp.cmp(&right.timestamp));
        nodes.truncate(limit);
        Ok(nodes)
    }
}

#[derive(Debug, Default)]
pub struct InMemoryLoopStateRepository {
    checkpoints: RwLock<HashMap<(SessionId, LoopId), LoopState>>,
}

#[async_trait]
impl LoopStateRepository for InMemoryLoopStateRepository {
    async fn save_checkpoint(&self, state: LoopState) -> Result<()> {
        self.checkpoints
            .write()
            .await
            .insert((state.session_id, state.loop_id), state);
        Ok(())
    }

    async fn load_checkpoint(
        &self,
        session_id: &SessionId,
        loop_id: &LoopId,
    ) -> Result<Option<LoopState>> {
        Ok(self
            .checkpoints
            .read()
            .await
            .get(&(*session_id, *loop_id))
            .cloned())
    }

    async fn clear_checkpoint(&self, session_id: &SessionId, loop_id: &LoopId) -> Result<()> {
        self.checkpoints
            .write()
            .await
            .remove(&(*session_id, *loop_id));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{
        AbstractNode, AbstractNodeMetadata, GraphFragment, OverflowPolicy, RawNode, RawNodeKind,
        References,
    };

    fn make_raw_node(
        session_id: Option<SessionId>,
        loop_id: Option<LoopId>,
        text: &str,
    ) -> RawNode {
        RawNode::text(
            RawNodeKind::UserUtterance,
            session_id,
            loop_id,
            "test",
            text,
            0.5,
            Vec::new(),
        )
    }

    fn make_abstract_node(title: &str) -> AbstractNode {
        AbstractNode::new(
            title,
            "summary",
            References::default(),
            GraphFragment::default(),
            AbstractNodeMetadata::default(),
        )
    }

    // --- NodeRepository: raw nodes ---

    #[tokio::test]
    async fn insert_and_get_raw() {
        let repo = InMemoryNodeRepository::default();
        let node = make_raw_node(None, None, "hello");
        let id = node.id;
        repo.insert_raw(node.clone()).await.unwrap();
        let fetched = repo.get_raw(&id).await.unwrap();
        assert_eq!(fetched.unwrap().id, id);
    }

    #[tokio::test]
    async fn get_raw_missing_returns_none() {
        let repo = InMemoryNodeRepository::default();
        let id = RawNodeId::new();
        let result = repo.get_raw(&id).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn list_raw_returns_matching() {
        let repo = InMemoryNodeRepository::default();
        let a = make_raw_node(None, None, "a");
        let b = make_raw_node(None, None, "b");
        let c = make_raw_node(None, None, "c");
        let id_a = a.id;
        let id_c = c.id;
        repo.insert_raw(a).await.unwrap();
        repo.insert_raw(b).await.unwrap();
        repo.insert_raw(c).await.unwrap();

        let result = repo.list_raw(&[id_a, id_c]).await.unwrap();
        assert_eq!(result.len(), 2);
        let ids: Vec<_> = result.iter().map(|n| n.id).collect();
        assert!(ids.contains(&id_a));
        assert!(ids.contains(&id_c));
    }

    #[tokio::test]
    async fn list_raw_skips_missing() {
        let repo = InMemoryNodeRepository::default();
        let node = make_raw_node(None, None, "present");
        let id = node.id;
        repo.insert_raw(node).await.unwrap();

        let missing_id = RawNodeId::new();
        let result = repo.list_raw(&[id, missing_id]).await.unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn session_raw_returns_sorted_by_timestamp() {
        let repo = InMemoryNodeRepository::default();
        let sid = SessionId::new();
        let a = make_raw_node(Some(sid), None, "first");
        let b = make_raw_node(Some(sid), None, "second");
        repo.insert_raw(a.clone()).await.unwrap();
        repo.insert_raw(b.clone()).await.unwrap();

        let result = repo.session_raw(&sid).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].timestamp <= result[1].timestamp);
    }

    #[tokio::test]
    async fn session_raw_empty_for_unknown_session() {
        let repo = InMemoryNodeRepository::default();
        let result = repo.session_raw(&SessionId::new()).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn raw_for_loop_returns_sorted() {
        let repo = InMemoryNodeRepository::default();
        let lid = LoopId::new();
        let a = make_raw_node(None, Some(lid), "first");
        let b = make_raw_node(None, Some(lid), "second");
        repo.insert_raw(a).await.unwrap();
        repo.insert_raw(b).await.unwrap();

        let result = repo.raw_for_loop(&lid).await.unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].timestamp <= result[1].timestamp);
    }

    #[tokio::test]
    async fn raw_for_loop_empty_for_unknown() {
        let repo = InMemoryNodeRepository::default();
        let result = repo.raw_for_loop(&LoopId::new()).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn update_raw_lifecycle_distillation_state() {
        let repo = InMemoryNodeRepository::default();
        let node = make_raw_node(None, None, "test");
        let id = node.id;
        repo.insert_raw(node).await.unwrap();

        let patch = RawLifecyclePatch {
            distillation_state: Some(DistillationState::Distilled),
            overflow: None,
        };
        repo.update_raw_lifecycle(&[id], &patch).await.unwrap();

        let updated = repo.get_raw(&id).await.unwrap().unwrap();
        assert_eq!(updated.distillation_state, DistillationState::Distilled);
    }

    #[tokio::test]
    async fn update_raw_lifecycle_overflow() {
        let repo = InMemoryNodeRepository::default();
        let node = make_raw_node(None, None, "test");
        let id = node.id;
        repo.insert_raw(node).await.unwrap();

        let patch = RawLifecyclePatch {
            distillation_state: None,
            overflow: Some(OverflowPolicy {
                was_pushed_out_of_session: true,
                relax_retrieval_until: None,
            }),
        };
        repo.update_raw_lifecycle(&[id], &patch).await.unwrap();

        let updated = repo.get_raw(&id).await.unwrap().unwrap();
        assert!(updated.overflow.was_pushed_out_of_session);
    }

    #[tokio::test]
    async fn update_raw_lifecycle_ignores_missing_ids() {
        let repo = InMemoryNodeRepository::default();
        let missing_id = RawNodeId::new();
        let patch = RawLifecyclePatch {
            distillation_state: Some(DistillationState::Distilled),
            overflow: None,
        };
        // Should not error
        repo.update_raw_lifecycle(&[missing_id], &patch)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn recent_session_raw_limits_results() {
        let repo = InMemoryNodeRepository::default();
        let sid = SessionId::new();
        for i in 0..5 {
            let node = make_raw_node(Some(sid), None, &format!("msg {i}"));
            repo.insert_raw(node).await.unwrap();
        }

        let result = repo.recent_session_raw(&sid, 2).await.unwrap();
        assert_eq!(result.len(), 2);
        // Should be the most recent 2
        assert!(result[0].timestamp <= result[1].timestamp);
    }

    #[tokio::test]
    async fn timeline_raw_no_filters() {
        let repo = InMemoryNodeRepository::default();
        let sid = SessionId::new();
        for i in 0..3 {
            let node = make_raw_node(Some(sid), None, &format!("msg {i}"));
            repo.insert_raw(node).await.unwrap();
        }

        let result = repo.timeline_raw(None, None, None, 10).await.unwrap();
        assert_eq!(result.len(), 3);
        // timeline_raw returns newest first
        assert!(result[0].timestamp >= result[1].timestamp);
    }

    #[tokio::test]
    async fn timeline_raw_with_limit() {
        let repo = InMemoryNodeRepository::default();
        for i in 0..5 {
            let node = make_raw_node(None, None, &format!("msg {i}"));
            repo.insert_raw(node).await.unwrap();
        }

        let result = repo.timeline_raw(None, None, None, 2).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn undistilled_raw_returns_undistilled() {
        let repo = InMemoryNodeRepository::default();
        let a = make_raw_node(None, None, "undistilled");
        let id_a = a.id;
        repo.insert_raw(a).await.unwrap();

        let mut b = make_raw_node(None, None, "distilled");
        b.distillation_state = DistillationState::Distilled;
        repo.insert_raw(b).await.unwrap();

        let result = repo.undistilled_raw(10, false).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, id_a);
    }

    #[tokio::test]
    async fn undistilled_raw_only_pushed_out() {
        let repo = InMemoryNodeRepository::default();
        let a = make_raw_node(None, None, "not pushed out");
        repo.insert_raw(a).await.unwrap();

        let mut b = make_raw_node(None, None, "pushed out");
        b.overflow.was_pushed_out_of_session = true;
        let id_b = b.id;
        repo.insert_raw(b).await.unwrap();

        let result = repo.undistilled_raw(10, true).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, id_b);
    }

    // --- NodeRepository: abstract nodes ---

    #[tokio::test]
    async fn insert_and_get_abstract() {
        let repo = InMemoryNodeRepository::default();
        let node = make_abstract_node("test title");
        let id = node.id;
        repo.insert_abstract(node).await.unwrap();
        let fetched = repo.get_abstract(&id).await.unwrap();
        assert_eq!(fetched.unwrap().id, id);
    }

    #[tokio::test]
    async fn get_abstract_missing_returns_none() {
        let repo = InMemoryNodeRepository::default();
        let result = repo.get_abstract(&AbstractNodeId::new()).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn list_abstract_returns_matching() {
        let repo = InMemoryNodeRepository::default();
        let a = make_abstract_node("a");
        let b = make_abstract_node("b");
        let id_a = a.id;
        let id_b = b.id;
        repo.insert_abstract(a).await.unwrap();
        repo.insert_abstract(b).await.unwrap();

        let result = repo.list_abstract(&[id_a, id_b]).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    // --- NodeRepository: operation key lookups ---

    #[tokio::test]
    async fn get_raw_by_operation_key() {
        let repo = InMemoryNodeRepository::default();
        let node = make_raw_node(None, None, "keyed").with_operation_key("op-1");
        let id = node.id;
        repo.insert_raw(node).await.unwrap();

        let found = repo.get_raw_by_operation_key("op-1").await.unwrap();
        assert_eq!(found.unwrap().id, id);

        let missing = repo.get_raw_by_operation_key("op-2").await.unwrap();
        assert!(missing.is_none());
    }

    #[tokio::test]
    async fn get_abstract_by_operation_key() {
        let repo = InMemoryNodeRepository::default();
        let node = make_abstract_node("keyed").with_operation_key("abs-op-1");
        let id = node.id;
        repo.insert_abstract(node).await.unwrap();

        let found = repo
            .get_abstract_by_operation_key("abs-op-1")
            .await
            .unwrap();
        assert_eq!(found.unwrap().id, id);

        let missing = repo
            .get_abstract_by_operation_key("abs-op-2")
            .await
            .unwrap();
        assert!(missing.is_none());
    }

    // --- LoopStateRepository ---

    #[tokio::test]
    async fn save_and_load_checkpoint() {
        let repo = InMemoryLoopStateRepository::default();
        let sid = SessionId::new();
        let lid = LoopId::new();
        let state = LoopState::new_for_test(sid, lid, "test goal");

        repo.save_checkpoint(state.clone()).await.unwrap();
        let loaded = repo.load_checkpoint(&sid, &lid).await.unwrap();
        assert_eq!(loaded.unwrap(), state);
    }

    #[tokio::test]
    async fn load_checkpoint_missing_returns_none() {
        let repo = InMemoryLoopStateRepository::default();
        let result = repo
            .load_checkpoint(&SessionId::new(), &LoopId::new())
            .await
            .unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn clear_checkpoint() {
        let repo = InMemoryLoopStateRepository::default();
        let sid = SessionId::new();
        let lid = LoopId::new();
        let state = LoopState::new_for_test(sid, lid, "test");

        repo.save_checkpoint(state).await.unwrap();
        assert!(repo.load_checkpoint(&sid, &lid).await.unwrap().is_some());

        repo.clear_checkpoint(&sid, &lid).await.unwrap();
        assert!(repo.load_checkpoint(&sid, &lid).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn clear_checkpoint_noop_for_missing() {
        let repo = InMemoryLoopStateRepository::default();
        // Should not error
        repo.clear_checkpoint(&SessionId::new(), &LoopId::new())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn save_checkpoint_overwrites() {
        let repo = InMemoryLoopStateRepository::default();
        let sid = SessionId::new();
        let lid = LoopId::new();

        let mut state1 = LoopState::new_for_test(sid, lid, "goal v1");
        state1.iteration = 1;
        repo.save_checkpoint(state1).await.unwrap();

        let mut state2 = LoopState::new_for_test(sid, lid, "goal v2");
        state2.iteration = 5;
        repo.save_checkpoint(state2).await.unwrap();

        let loaded = repo.load_checkpoint(&sid, &lid).await.unwrap().unwrap();
        assert_eq!(loaded.iteration, 5);
        assert_eq!(loaded.user_goal, "goal v2");
    }
}
