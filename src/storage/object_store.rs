use std::cmp::Ordering;
use std::collections::{HashSet, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use crate::domain::{AbstractNode, DistillationState, LoopState, RawNode};
use crate::error::{EngineError, Result};
use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};
use crate::model::embedding::{cosine_similarity, Embedding};

use super::traits::{
    GraphRepository, GraphTraversalHit, LoopStateRepository, NodeRepository, RawLifecyclePatch,
    ScoredAbstractRef, ScoredRawRef, VectorIndex,
};

#[derive(Debug, Clone)]
pub struct FileObjectStore {
    root: PathBuf,
}

impl FileObjectStore {
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        let store = Self {
            root: root.as_ref().to_path_buf(),
        };
        for directory in [
            store.raw_dir(),
            store.abstract_dir(),
            store.raw_embedding_dir(),
            store.abstract_embedding_dir(),
            store.graph_dir(),
            store.checkpoint_dir(),
            store.raw_operation_dir(),
            store.abstract_operation_dir(),
        ] {
            fs::create_dir_all(&directory).map_err(|err| {
                EngineError::Storage(format!(
                    "failed to create object store directory {}: {err}",
                    directory.display()
                ))
            })?;
        }
        Ok(store)
    }

    pub fn root(&self) -> &Path {
        &self.root
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
        self.root.join("indexes").join("raw_operation")
    }

    fn abstract_operation_dir(&self) -> PathBuf {
        self.root.join("indexes").join("abstract_operation")
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

    fn raw_operation_path(&self, operation_key: &str) -> PathBuf {
        self.raw_operation_dir()
            .join(format!("{}.json", encode_operation_key(operation_key)))
    }

    fn abstract_operation_path(&self, operation_key: &str) -> PathBuf {
        self.abstract_operation_dir()
            .join(format!("{}.json", encode_operation_key(operation_key)))
    }

    fn write_json<T: Serialize>(&self, path: &Path, value: &T) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| {
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
        fs::write(&temporary, payload).map_err(|err| {
            EngineError::Storage(format!(
                "failed to write temporary object {}: {err}",
                temporary.display()
            ))
        })?;
        fs::rename(&temporary, path).map_err(|err| {
            EngineError::Storage(format!(
                "failed to move object {} into place: {err}",
                path.display()
            ))
        })?;
        Ok(())
    }

    fn read_json<T: DeserializeOwned>(&self, path: &Path) -> Result<T> {
        let payload = fs::read(path).map_err(|err| {
            EngineError::Storage(format!("failed to read object {}: {err}", path.display()))
        })?;
        serde_json::from_slice(&payload).map_err(|err| {
            EngineError::Storage(format!(
                "failed to deserialize object {}: {err}",
                path.display()
            ))
        })
    }

    fn try_read_json<T: DeserializeOwned>(&self, path: &Path) -> Result<Option<T>> {
        if !path.exists() {
            return Ok(None);
        }
        self.read_json(path).map(Some)
    }

    fn remove_file_if_exists(&self, path: &Path) -> Result<()> {
        if !path.exists() {
            return Ok(());
        }
        fs::remove_file(path).map_err(|err| {
            EngineError::Storage(format!("failed to remove object {}: {err}", path.display()))
        })?;
        Ok(())
    }

    fn list_paths(&self, directory: &Path) -> Result<Vec<PathBuf>> {
        if !directory.exists() {
            return Ok(Vec::new());
        }
        let mut paths = Vec::new();
        let entries = fs::read_dir(directory).map_err(|err| {
            EngineError::Storage(format!(
                "failed to read object store directory {}: {err}",
                directory.display()
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(|err| {
                EngineError::Storage(format!(
                    "failed to enumerate object store directory {}: {err}",
                    directory.display()
                ))
            })?;
            let path = entry.path();
            if path.extension().and_then(|value| value.to_str()) == Some("json") {
                paths.push(path);
            }
        }
        paths.sort();
        Ok(paths)
    }

    fn list_raw_nodes(&self) -> Result<Vec<RawNode>> {
        let mut nodes = Vec::new();
        for path in self.list_paths(&self.raw_dir())? {
            nodes.push(self.read_json(&path)?);
        }
        nodes.sort_by(|left: &RawNode, right: &RawNode| {
            left.timestamp
                .cmp(&right.timestamp)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        Ok(nodes)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredId<Id> {
    id: Id,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredEmbedding<Id> {
    id: Id,
    embedding: Embedding,
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
    pub fn new(store: FileObjectStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl NodeRepository for ObjectNodeRepository {
    async fn insert_raw(&self, node: RawNode) -> Result<()> {
        self.store
            .write_json(&self.store.raw_path(&node.id), &node)?;
        if let Some(operation_key) = &node.operation_key {
            self.store.write_json(
                &self.store.raw_operation_path(operation_key),
                &StoredId { id: node.id },
            )?;
        }
        Ok(())
    }

    async fn insert_abstract(&self, node: AbstractNode) -> Result<()> {
        self.store
            .write_json(&self.store.abstract_path(&node.id), &node)?;
        if let Some(operation_key) = &node.operation_key {
            self.store.write_json(
                &self.store.abstract_operation_path(operation_key),
                &StoredId { id: node.id },
            )?;
        }
        Ok(())
    }

    async fn get_raw(&self, id: &RawNodeId) -> Result<Option<RawNode>> {
        self.store.try_read_json(&self.store.raw_path(id))
    }

    async fn get_abstract(&self, id: &AbstractNodeId) -> Result<Option<AbstractNode>> {
        self.store.try_read_json(&self.store.abstract_path(id))
    }

    async fn get_raw_by_operation_key(&self, operation_key: &str) -> Result<Option<RawNode>> {
        match self
            .store
            .try_read_json::<StoredId<RawNodeId>>(&self.store.raw_operation_path(operation_key))?
        {
            Some(record) => self.get_raw(&record.id).await,
            None => Ok(None),
        }
    }

    async fn get_abstract_by_operation_key(
        &self,
        operation_key: &str,
    ) -> Result<Option<AbstractNode>> {
        match self.store.try_read_json::<StoredId<AbstractNodeId>>(
            &self.store.abstract_operation_path(operation_key),
        )? {
            Some(record) => self.get_abstract(&record.id).await,
            None => Ok(None),
        }
    }

    async fn list_raw(&self, ids: &[RawNodeId]) -> Result<Vec<RawNode>> {
        let mut nodes = Vec::new();
        for id in ids {
            if let Some(node) = self.get_raw(id).await? {
                nodes.push(node);
            }
        }
        Ok(nodes)
    }

    async fn list_abstract(&self, ids: &[AbstractNodeId]) -> Result<Vec<AbstractNode>> {
        let mut nodes = Vec::new();
        for id in ids {
            if let Some(node) = self.get_abstract(id).await? {
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
        let mut nodes = self.session_raw(session_id).await?;
        nodes.reverse();
        nodes.truncate(limit);
        nodes.reverse();
        Ok(nodes)
    }

    async fn session_raw(&self, session_id: &SessionId) -> Result<Vec<RawNode>> {
        let mut nodes: Vec<_> = self
            .store
            .list_raw_nodes()?
            .into_iter()
            .filter(|node| node.session_id == Some(*session_id))
            .collect();
        nodes.sort_by(|left, right| {
            left.timestamp
                .cmp(&right.timestamp)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        Ok(nodes)
    }

    async fn raw_for_loop(&self, loop_id: &LoopId) -> Result<Vec<RawNode>> {
        let mut nodes: Vec<_> = self
            .store
            .list_raw_nodes()?
            .into_iter()
            .filter(|node| node.loop_id == Some(*loop_id))
            .collect();
        nodes.sort_by(|left, right| {
            left.timestamp
                .cmp(&right.timestamp)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        Ok(nodes)
    }

    async fn timeline_raw(
        &self,
        session_id: Option<&SessionId>,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<RawNode>> {
        let mut nodes: Vec<_> = self
            .store
            .list_raw_nodes()?
            .into_iter()
            .filter(|node| {
                session_id
                    .map(|value| node.session_id == Some(*value))
                    .unwrap_or(true)
            })
            .filter(|node| from.map(|value| node.timestamp >= value).unwrap_or(true))
            .filter(|node| to.map(|value| node.timestamp <= value).unwrap_or(true))
            .collect();
        nodes.sort_by(|left, right| {
            right
                .timestamp
                .cmp(&left.timestamp)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        nodes.truncate(limit);
        Ok(nodes)
    }

    async fn update_raw_lifecycle(
        &self,
        ids: &[RawNodeId],
        patch: &RawLifecyclePatch,
    ) -> Result<()> {
        for id in ids {
            if let Some(mut node) = self.get_raw(id).await? {
                if let Some(distillation_state) = &patch.distillation_state {
                    node.distillation_state = distillation_state.clone();
                }
                if let Some(overflow) = &patch.overflow {
                    node.overflow = overflow.clone();
                }
                self.store.write_json(&self.store.raw_path(id), &node)?;
            }
        }
        Ok(())
    }

    async fn undistilled_raw(&self, limit: usize, only_pushed_out: bool) -> Result<Vec<RawNode>> {
        let mut nodes: Vec<_> = self
            .store
            .list_raw_nodes()?
            .into_iter()
            .filter(|node| node.distillation_state != DistillationState::Distilled)
            .filter(|node| !only_pushed_out || node.overflow.was_pushed_out_of_session)
            .collect();
        nodes.sort_by(|left, right| {
            left.timestamp
                .cmp(&right.timestamp)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        nodes.truncate(limit);
        Ok(nodes)
    }
}

#[derive(Debug, Clone)]
pub struct ObjectVectorIndex {
    store: FileObjectStore,
}

impl ObjectVectorIndex {
    pub fn new(store: FileObjectStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl VectorIndex for ObjectVectorIndex {
    async fn index_raw(&self, id: RawNodeId, embedding: Embedding) -> Result<()> {
        self.store.write_json(
            &self.store.raw_embedding_path(&id),
            &StoredEmbedding { id, embedding },
        )
    }

    async fn index_abstract(&self, id: AbstractNodeId, embedding: Embedding) -> Result<()> {
        self.store.write_json(
            &self.store.abstract_embedding_path(&id),
            &StoredEmbedding { id, embedding },
        )
    }

    async fn search_raw(&self, query: &Embedding, top_k: usize) -> Result<Vec<ScoredRawRef>> {
        let mut scored = Vec::new();
        for path in self.store.list_paths(&self.store.raw_embedding_dir())? {
            let record: StoredEmbedding<RawNodeId> = self.store.read_json(&path)?;
            scored.push(ScoredRawRef {
                id: record.id,
                score: cosine_similarity(query, &record.embedding),
            });
        }
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn search_abstract(
        &self,
        query: &Embedding,
        top_k: usize,
    ) -> Result<Vec<ScoredAbstractRef>> {
        let mut scored = Vec::new();
        for path in self
            .store
            .list_paths(&self.store.abstract_embedding_dir())?
        {
            let record: StoredEmbedding<AbstractNodeId> = self.store.read_json(&path)?;
            scored.push(ScoredAbstractRef {
                id: record.id,
                score: cosine_similarity(query, &record.embedding),
            });
        }
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

#[derive(Debug, Clone)]
pub struct ObjectGraphRepository {
    store: FileObjectStore,
}

impl ObjectGraphRepository {
    pub fn new(store: FileObjectStore) -> Self {
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
                .then_with(|| left.to.to_string().cmp(&right.to.to_string()))
        });
        edges
    }
}

#[async_trait]
impl GraphRepository for ObjectGraphRepository {
    async fn index_abstract(&self, node: &AbstractNode) -> Result<()> {
        self.store.write_json(
            &self.store.graph_path(&node.id),
            &StoredGraphEdges {
                node_id: node.id,
                edges: Self::edges_for_abstract(node),
            },
        )
    }

    async fn traverse(
        &self,
        start: &AbstractNodeId,
        max_depth: usize,
        relation_types: Option<&[String]>,
    ) -> Result<Vec<GraphTraversalHit>> {
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
                .try_read_json::<StoredGraphEdges>(&self.store.graph_path(&current))?
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
    pub fn new(store: FileObjectStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl LoopStateRepository for ObjectLoopStateRepository {
    async fn save_checkpoint(&self, state: LoopState) -> Result<()> {
        self.store.write_json(
            &self
                .store
                .checkpoint_path(&state.session_id, &state.loop_id),
            &state,
        )
    }

    async fn load_checkpoint(
        &self,
        session_id: &SessionId,
        loop_id: &LoopId,
    ) -> Result<Option<LoopState>> {
        self.store
            .try_read_json(&self.store.checkpoint_path(session_id, loop_id))
    }

    async fn clear_checkpoint(&self, session_id: &SessionId, loop_id: &LoopId) -> Result<()> {
        self.store
            .remove_file_if_exists(&self.store.checkpoint_path(session_id, loop_id))
    }
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
