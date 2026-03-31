use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::str::FromStr;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};

use crate::domain::{AbstractNode, DistillationState, LoopState, RawNode};
use crate::error::{EngineError, Result};
use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};
use crate::model::embedding::{cosine_similarity, Embedding};

use super::traits::{
    GraphRepository, GraphTraversalHit, LoopStateRepository, NodeRepository, RawLifecyclePatch,
    ScoredAbstractRef, ScoredRawRef, VectorIndex,
};

#[derive(Clone)]
pub struct SqliteDatabase {
    connection: Arc<Mutex<Connection>>,
}

impl SqliteDatabase {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let connection = Connection::open(path).map_err(storage_error)?;
        let database = Self {
            connection: Arc::new(Mutex::new(connection)),
        };
        database.init_schema()?;
        Ok(database)
    }

    pub fn open_in_memory() -> Result<Self> {
        let connection = Connection::open_in_memory().map_err(storage_error)?;
        let database = Self {
            connection: Arc::new(Mutex::new(connection)),
        };
        database.init_schema()?;
        Ok(database)
    }

    fn init_schema(&self) -> Result<()> {
        self.with_connection(|connection| {
            connection.execute_batch(
                "
                CREATE TABLE IF NOT EXISTS raw_nodes (
                    id TEXT PRIMARY KEY,
                    operation_key TEXT UNIQUE,
                    session_id TEXT,
                    loop_id TEXT,
                    timestamp TEXT NOT NULL,
                    distillation_state TEXT NOT NULL,
                    was_pushed_out INTEGER NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_raw_nodes_session_timestamp
                    ON raw_nodes(session_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_raw_nodes_loop_timestamp
                    ON raw_nodes(loop_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_raw_nodes_pushed_out_undistilled
                    ON raw_nodes(was_pushed_out, distillation_state, timestamp);

                CREATE TABLE IF NOT EXISTS abstract_nodes (
                    id TEXT PRIMARY KEY,
                    operation_key TEXT UNIQUE,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_abstract_nodes_timestamp
                    ON abstract_nodes(timestamp);

                CREATE TABLE IF NOT EXISTS embeddings (
                    node_kind TEXT NOT NULL,
                    node_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (node_kind, node_id)
                );

                CREATE TABLE IF NOT EXISTS abstract_edges (
                    from_id TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    predicate TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_abstract_edges_from
                    ON abstract_edges(from_id);
                CREATE INDEX IF NOT EXISTS idx_abstract_edges_predicate
                    ON abstract_edges(from_id, predicate, to_id);

                CREATE TABLE IF NOT EXISTS loop_states (
                    session_id TEXT NOT NULL,
                    loop_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    current_node TEXT NOT NULL,
                    iteration INTEGER NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (session_id, loop_id)
                );
                ",
            )
        })?;
        self.ensure_column("raw_nodes", "operation_key", "TEXT")?;
        self.ensure_column("abstract_nodes", "operation_key", "TEXT")?;
        self.ensure_column(
            "loop_states",
            "updated_at",
            "TEXT NOT NULL DEFAULT '1970-01-01T00:00:00+00:00'",
        )?;
        self.with_connection(|connection| {
            connection.execute_batch(
                "
                CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_nodes_operation_key
                    ON raw_nodes(operation_key)
                    WHERE operation_key IS NOT NULL;
                CREATE UNIQUE INDEX IF NOT EXISTS idx_abstract_nodes_operation_key
                    ON abstract_nodes(operation_key)
                    WHERE operation_key IS NOT NULL;
                ",
            )
        })?;
        Ok(())
    }

    fn with_connection<T>(
        &self,
        operation: impl FnOnce(&Connection) -> rusqlite::Result<T>,
    ) -> Result<T> {
        let guard = self
            .connection
            .lock()
            .map_err(|err| EngineError::Storage(format!("sqlite mutex poisoned: {err}")))?;
        operation(&guard).map_err(storage_error)
    }

    fn ensure_column(&self, table: &str, column: &str, spec: &str) -> Result<()> {
        let exists = self.with_connection(|connection| {
            let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
            let rows = statement.query_map(params![], |row| row.get::<_, String>(1))?;
            let mut columns = Vec::new();
            for row in rows {
                columns.push(row?);
            }
            Ok(columns.into_iter().any(|name| name == column))
        })?;
        if exists {
            return Ok(());
        }

        self.with_connection(|connection| {
            connection.execute(
                &format!("ALTER TABLE {table} ADD COLUMN {column} {spec}"),
                params![],
            )?;
            Ok(())
        })
    }
}

#[derive(Clone)]
pub struct SqliteNodeRepository {
    database: SqliteDatabase,
}

impl SqliteNodeRepository {
    pub fn new(database: SqliteDatabase) -> Self {
        Self { database }
    }

    fn serialize<T: serde::Serialize>(value: &T) -> Result<String> {
        serde_json::to_string(value).map_err(|err| {
            EngineError::Storage(format!("failed to serialize sqlite payload: {err}"))
        })
    }

    fn deserialize<T: serde::de::DeserializeOwned>(payload: String) -> Result<T> {
        serde_json::from_str(&payload).map_err(|err| {
            EngineError::Storage(format!("failed to deserialize sqlite payload: {err}"))
        })
    }

    fn upsert_raw(&self, node: &RawNode) -> Result<()> {
        let payload = Self::serialize(node)?;
        let operation_key = node.operation_key.clone();
        let session_id = node.session_id.map(|value| value.to_string());
        let loop_id = node.loop_id.map(|value| value.to_string());
        let distillation_state = serde_json::to_string(&node.distillation_state)
            .map_err(|err| EngineError::Storage(format!("failed to encode raw state: {err}")))?;
        self.database.with_connection(|connection| {
            connection.execute(
                "
                INSERT OR REPLACE INTO raw_nodes
                    (id, operation_key, session_id, loop_id, timestamp, distillation_state, was_pushed_out, payload)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
                ",
                params![
                    node.id.to_string(),
                    operation_key,
                    session_id,
                    loop_id,
                    node.timestamp.to_rfc3339(),
                    distillation_state,
                    if node.overflow.was_pushed_out_of_session { 1 } else { 0 },
                    payload,
                ],
            )?;
            Ok(())
        })
    }

    fn upsert_abstract(&self, node: &AbstractNode) -> Result<()> {
        let payload = Self::serialize(node)?;
        let operation_key = node.operation_key.clone();
        self.database.with_connection(|connection| {
            connection.execute(
                "
                INSERT OR REPLACE INTO abstract_nodes (id, operation_key, timestamp, payload)
                VALUES (?1, ?2, ?3, ?4)
                ",
                params![
                    node.id.to_string(),
                    operation_key,
                    node.timestamp.to_rfc3339(),
                    payload
                ],
            )?;
            Ok(())
        })
    }

    fn load_raw_by_sql(&self, sql: &str, params: impl rusqlite::Params) -> Result<Vec<RawNode>> {
        self.database
            .with_connection(|connection| {
                let mut statement = connection.prepare(sql)?;
                let rows = statement.query_map(params, |row| row.get::<_, String>(0))?;
                let mut nodes = Vec::new();
                for row in rows {
                    nodes.push(row?);
                }
                Ok(nodes)
            })?
            .into_iter()
            .map(Self::deserialize)
            .collect()
    }

    fn load_abstract_by_sql(
        &self,
        sql: &str,
        params: impl rusqlite::Params,
    ) -> Result<Vec<AbstractNode>> {
        self.database
            .with_connection(|connection| {
                let mut statement = connection.prepare(sql)?;
                let rows = statement.query_map(params, |row| row.get::<_, String>(0))?;
                let mut nodes = Vec::new();
                for row in rows {
                    nodes.push(row?);
                }
                Ok(nodes)
            })?
            .into_iter()
            .map(Self::deserialize)
            .collect()
    }
}

#[async_trait]
impl NodeRepository for SqliteNodeRepository {
    async fn insert_raw(&self, node: RawNode) -> Result<()> {
        self.upsert_raw(&node)
    }

    async fn insert_abstract(&self, node: AbstractNode) -> Result<()> {
        self.upsert_abstract(&node)
    }

    async fn get_raw(&self, id: &RawNodeId) -> Result<Option<RawNode>> {
        let mut nodes = self.load_raw_by_sql(
            "SELECT payload FROM raw_nodes WHERE id = ?1 LIMIT 1",
            params![id.to_string()],
        )?;
        Ok(nodes.pop())
    }

    async fn get_abstract(&self, id: &AbstractNodeId) -> Result<Option<AbstractNode>> {
        let mut nodes = self.load_abstract_by_sql(
            "SELECT payload FROM abstract_nodes WHERE id = ?1 LIMIT 1",
            params![id.to_string()],
        )?;
        Ok(nodes.pop())
    }

    async fn get_raw_by_operation_key(&self, operation_key: &str) -> Result<Option<RawNode>> {
        let mut nodes = self.load_raw_by_sql(
            "SELECT payload FROM raw_nodes WHERE operation_key = ?1 LIMIT 1",
            params![operation_key],
        )?;
        Ok(nodes.pop())
    }

    async fn get_abstract_by_operation_key(
        &self,
        operation_key: &str,
    ) -> Result<Option<AbstractNode>> {
        let mut nodes = self.load_abstract_by_sql(
            "SELECT payload FROM abstract_nodes WHERE operation_key = ?1 LIMIT 1",
            params![operation_key],
        )?;
        Ok(nodes.pop())
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
        let mut nodes = self.load_raw_by_sql(
            "
            SELECT payload
            FROM raw_nodes
            WHERE session_id = ?1
            ORDER BY timestamp DESC
            LIMIT ?2
            ",
            params![session_id.to_string(), limit as i64],
        )?;
        nodes.reverse();
        Ok(nodes)
    }

    async fn session_raw(&self, session_id: &SessionId) -> Result<Vec<RawNode>> {
        self.load_raw_by_sql(
            "
            SELECT payload
            FROM raw_nodes
            WHERE session_id = ?1
            ORDER BY timestamp ASC
            ",
            params![session_id.to_string()],
        )
    }

    async fn raw_for_loop(&self, loop_id: &LoopId) -> Result<Vec<RawNode>> {
        self.load_raw_by_sql(
            "
            SELECT payload
            FROM raw_nodes
            WHERE loop_id = ?1
            ORDER BY timestamp ASC
            ",
            params![loop_id.to_string()],
        )
    }

    async fn timeline_raw(
        &self,
        session_id: Option<&SessionId>,
        from: Option<DateTime<Utc>>,
        to: Option<DateTime<Utc>>,
        limit: usize,
    ) -> Result<Vec<RawNode>> {
        let mut nodes = if let Some(session_id) = session_id {
            self.session_raw(session_id).await?
        } else {
            self.load_raw_by_sql(
                "SELECT payload FROM raw_nodes ORDER BY timestamp DESC",
                params![],
            )?
        };
        nodes.retain(|node| {
            let lower_ok = from.map(|value| node.timestamp >= value).unwrap_or(true);
            let upper_ok = to.map(|value| node.timestamp <= value).unwrap_or(true);
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
        for id in ids {
            if let Some(mut node) = self.get_raw(id).await? {
                if let Some(distillation_state) = &patch.distillation_state {
                    node.distillation_state = distillation_state.clone();
                }
                if let Some(overflow) = &patch.overflow {
                    node.overflow = overflow.clone();
                }
                self.upsert_raw(&node)?;
            }
        }
        Ok(())
    }

    async fn undistilled_raw(&self, limit: usize, only_pushed_out: bool) -> Result<Vec<RawNode>> {
        let mut nodes = self.load_raw_by_sql(
            "SELECT payload FROM raw_nodes ORDER BY timestamp ASC",
            params![],
        )?;
        nodes.retain(|node| node.distillation_state != DistillationState::Distilled);
        if only_pushed_out {
            nodes.retain(|node| node.overflow.was_pushed_out_of_session);
        }
        nodes.truncate(limit);
        Ok(nodes)
    }
}

#[derive(Clone)]
pub struct SqliteVectorIndex {
    database: SqliteDatabase,
}

impl SqliteVectorIndex {
    pub fn new(database: SqliteDatabase) -> Self {
        Self { database }
    }

    fn serialize_embedding(embedding: &Embedding) -> Result<String> {
        serde_json::to_string(embedding)
            .map_err(|err| EngineError::Storage(format!("failed to serialize embedding: {err}")))
    }

    fn deserialize_embedding(payload: String) -> Result<Embedding> {
        serde_json::from_str(&payload)
            .map_err(|err| EngineError::Storage(format!("failed to deserialize embedding: {err}")))
    }

    fn load_embeddings(&self, kind: &str) -> Result<Vec<(String, Embedding)>> {
        let rows = self.database.with_connection(|connection| {
            let mut statement = connection
                .prepare("SELECT node_id, payload FROM embeddings WHERE node_kind = ?1")?;
            let rows = statement.query_map(params![kind], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut values = Vec::new();
            for row in rows {
                values.push(row?);
            }
            Ok(values)
        })?;

        rows.into_iter()
            .map(|(id, payload)| Ok((id, Self::deserialize_embedding(payload)?)))
            .collect()
    }
}

#[async_trait]
impl VectorIndex for SqliteVectorIndex {
    async fn index_raw(&self, id: RawNodeId, embedding: Embedding) -> Result<()> {
        let payload = Self::serialize_embedding(&embedding)?;
        self.database.with_connection(|connection| {
            connection.execute(
                "
                INSERT OR REPLACE INTO embeddings (node_kind, node_id, payload)
                VALUES ('raw', ?1, ?2)
                ",
                params![id.to_string(), payload],
            )?;
            Ok(())
        })
    }

    async fn index_abstract(&self, id: AbstractNodeId, embedding: Embedding) -> Result<()> {
        let payload = Self::serialize_embedding(&embedding)?;
        self.database.with_connection(|connection| {
            connection.execute(
                "
                INSERT OR REPLACE INTO embeddings (node_kind, node_id, payload)
                VALUES ('abstract', ?1, ?2)
                ",
                params![id.to_string(), payload],
            )?;
            Ok(())
        })
    }

    async fn search_raw(&self, query: &Embedding, top_k: usize) -> Result<Vec<ScoredRawRef>> {
        let mut scored = Vec::new();
        for (id, embedding) in self.load_embeddings("raw")? {
            let raw_id = RawNodeId::from_str(&id).map_err(|err| {
                EngineError::Storage(format!("invalid raw id stored in sqlite: {err}"))
            })?;
            scored.push(ScoredRawRef {
                id: raw_id,
                score: cosine_similarity(query, &embedding),
            });
        }
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
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
        for (id, embedding) in self.load_embeddings("abstract")? {
            let abstract_id = AbstractNodeId::from_str(&id).map_err(|err| {
                EngineError::Storage(format!("invalid abstract id stored in sqlite: {err}"))
            })?;
            scored.push(ScoredAbstractRef {
                id: abstract_id,
                score: cosine_similarity(query, &embedding),
            });
        }
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

#[derive(Clone)]
pub struct SqliteGraphRepository {
    database: SqliteDatabase,
}

impl SqliteGraphRepository {
    pub fn new(database: SqliteDatabase) -> Self {
        Self { database }
    }

    fn edge_rows(node: &AbstractNode) -> Vec<(AbstractNodeId, AbstractNodeId, String)> {
        let mut edges = HashSet::new();
        for reference in &node.references.abstract_node_ids {
            edges.insert((node.id, *reference, "reference".to_string()));
        }
        for relation in &node.graph.relations {
            if let Ok(target) = AbstractNodeId::from_str(&relation.object) {
                edges.insert((node.id, target, relation.predicate.clone()));
            }
        }
        edges.into_iter().collect()
    }
}

#[async_trait]
impl GraphRepository for SqliteGraphRepository {
    async fn index_abstract(&self, node: &AbstractNode) -> Result<()> {
        let edges = Self::edge_rows(node);
        self.database.with_connection(|connection| {
            connection.execute(
                "DELETE FROM abstract_edges WHERE from_id = ?1",
                params![node.id.to_string()],
            )?;
            for (from_id, to_id, predicate) in &edges {
                connection.execute(
                    "
                    INSERT INTO abstract_edges (from_id, to_id, predicate)
                    VALUES (?1, ?2, ?3)
                    ",
                    params![from_id.to_string(), to_id.to_string(), predicate],
                )?;
            }
            Ok(())
        })
    }

    async fn traverse(
        &self,
        start: &AbstractNodeId,
        max_depth: usize,
        relation_types: Option<&[String]>,
    ) -> Result<Vec<GraphTraversalHit>> {
        let edges = self.database.with_connection(|connection| {
            let mut statement =
                connection.prepare("SELECT from_id, to_id, predicate FROM abstract_edges")?;
            let rows = statement.query_map(params![], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?;
            let mut values = Vec::new();
            for row in rows {
                values.push(row?);
            }
            Ok(values)
        })?;

        let mut adjacency: HashMap<AbstractNodeId, Vec<(AbstractNodeId, String)>> = HashMap::new();
        for (from_id, to_id, predicate) in edges {
            let from = AbstractNodeId::from_str(&from_id).map_err(|err| {
                EngineError::Storage(format!("invalid graph from_id stored in sqlite: {err}"))
            })?;
            let to = AbstractNodeId::from_str(&to_id).map_err(|err| {
                EngineError::Storage(format!("invalid graph to_id stored in sqlite: {err}"))
            })?;
            adjacency.entry(from).or_default().push((to, predicate));
        }
        for neighbors in adjacency.values_mut() {
            neighbors.sort_by(|left, right| {
                left.1
                    .cmp(&right.1)
                    .then_with(|| left.0.to_string().cmp(&right.0.to_string()))
            });
        }

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
            if let Some(neighbors) = adjacency.get(&current) {
                for (neighbor, predicate) in neighbors {
                    if let Some(filters) = &filters {
                        if !filters.contains(predicate) {
                            continue;
                        }
                    }
                    queue.push_back((*neighbor, depth + 1, Some(predicate.clone())));
                }
            }
        }

        Ok(output)
    }
}

#[derive(Clone)]
pub struct SqliteLoopStateRepository {
    database: SqliteDatabase,
}

impl SqliteLoopStateRepository {
    pub fn new(database: SqliteDatabase) -> Self {
        Self { database }
    }
}

#[async_trait]
impl LoopStateRepository for SqliteLoopStateRepository {
    async fn save_checkpoint(&self, state: LoopState) -> Result<()> {
        let status = serde_json::to_string(&state.status)
            .map_err(|err| EngineError::Storage(format!("failed to encode loop status: {err}")))?;
        let payload = serde_json::to_string(&state).map_err(|err| {
            EngineError::Storage(format!("failed to encode loop checkpoint: {err}"))
        })?;
        self.database.with_connection(|connection| {
            connection.execute(
                "
                INSERT OR REPLACE INTO loop_states
                    (session_id, loop_id, status, current_node, iteration, updated_at, payload)
                VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                ",
                params![
                    state.session_id.to_string(),
                    state.loop_id.to_string(),
                    status,
                    state.current_node,
                    state.iteration as i64,
                    chrono::Utc::now().to_rfc3339(),
                    payload
                ],
            )?;
            Ok(())
        })
    }

    async fn load_checkpoint(
        &self,
        session_id: &SessionId,
        loop_id: &LoopId,
    ) -> Result<Option<LoopState>> {
        let payload = self.database.with_connection(|connection| {
            connection
                .query_row(
                    "
                    SELECT payload
                    FROM loop_states
                    WHERE session_id = ?1 AND loop_id = ?2
                    LIMIT 1
                    ",
                    params![session_id.to_string(), loop_id.to_string()],
                    |row| row.get::<_, String>(0),
                )
                .optional()
        })?;

        payload
            .map(|payload| {
                serde_json::from_str(&payload).map_err(|err| {
                    EngineError::Storage(format!("failed to decode loop checkpoint: {err}"))
                })
            })
            .transpose()
    }

    async fn clear_checkpoint(&self, session_id: &SessionId, loop_id: &LoopId) -> Result<()> {
        self.database.with_connection(|connection| {
            connection.execute(
                "
                DELETE FROM loop_states
                WHERE session_id = ?1 AND loop_id = ?2
                ",
                params![session_id.to_string(), loop_id.to_string()],
            )?;
            Ok(())
        })
    }
}

fn storage_error(error: impl std::fmt::Display) -> EngineError {
    EngineError::Storage(format!("sqlite error: {error}"))
}
