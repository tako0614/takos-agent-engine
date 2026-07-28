use std::collections::{HashSet, VecDeque};
use std::str::FromStr;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::AbstractNode;
use crate::error::Result;
use crate::ids::{AbstractNodeId, SessionId};

use crate::storage::traits::{GraphRepository, GraphTraversalHit};

use super::store::FileObjectStore;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub(super) struct GraphEdge {
    pub(super) to: AbstractNodeId,
    pub(super) predicate: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct StoredGraphEdges {
    pub(super) node_id: AbstractNodeId,
    #[serde(default)]
    pub(super) session_id: Option<SessionId>,
    pub(super) edges: Vec<GraphEdge>,
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

    pub(super) fn edges_for_abstract(node: &AbstractNode) -> Vec<GraphEdge> {
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
        let _guard = self.store.lock().await?;
        self.store
            .write_json(
                &self.store.graph_path(&node.id),
                &StoredGraphEdges {
                    node_id: node.id,
                    session_id: node.session_id,
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
        session_id: Option<&SessionId>,
        max_hits: usize,
    ) -> Result<Vec<GraphTraversalHit>> {
        let _guard = self.store.lock().await?;
        let filters = relation_types.map(|values| values.iter().cloned().collect::<HashSet<_>>());
        let mut visited = HashSet::new();
        let mut queue = VecDeque::from([(*start, 0usize, None::<String>)]);
        let mut output = Vec::new();

        while let Some((current, depth, via_predicate)) = queue.pop_front() {
            if depth > max_depth || !visited.insert(current) {
                continue;
            }
            let Some(record) = self
                .store
                .try_read_json::<StoredGraphEdges>(&self.store.graph_path(&current))
                .await?
            else {
                continue;
            };
            if session_id.is_some_and(|scope| record.session_id.as_ref() != Some(scope)) {
                continue;
            }
            output.push(GraphTraversalHit {
                node_id: current,
                depth,
                via_predicate: via_predicate.clone(),
            });
            if output.len() >= max_hits.max(1) {
                break;
            }
            for edge in record.edges {
                if let Some(filters) = &filters {
                    if !filters.contains(&edge.predicate) {
                        continue;
                    }
                }
                if output.len().saturating_add(queue.len()) < max_hits.max(1) {
                    queue.push_back((edge.to, depth + 1, Some(edge.predicate)));
                }
            }
        }

        Ok(output)
    }
}
