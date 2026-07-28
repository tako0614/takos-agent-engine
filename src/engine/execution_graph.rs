use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::config::EngineConfig;
use crate::domain::{LoopState, LoopStatus, RawNode};
use crate::error::{EngineError, Result};
use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};
use crate::memory::{ActivatedMemory, ActivationQuery};
use crate::model::{ConversationMessage, Embedding, ModelOutput, ToolCallRequest};
use crate::tools::executor::ToolCallResult;

use super::context_assembler::AssembledContext;
use super::session_engine::{EngineDeps, SessionRequest};

pub const DEFAULT_EDGE: &str = "__default__";

/// Selects which execution topology and local-state behaviour the engine uses.
///
/// [`Self::MemoryAware`] preserves the original engine behaviour: the graph
/// ingests the turn into the injected repositories, activates memory, manages
/// the session window, and distills the completed loop. [`Self::ExternalContext`]
/// is for product wrappers that already own durable history, memory selection,
/// and the current-turn transcript. It runs only the bounded model/tool loop and
/// never writes turn data into the engine's node/vector/graph repositories.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProfile {
    #[default]
    MemoryAware,
    ExternalContext,
}

impl ExecutionProfile {
    #[must_use]
    pub const fn uses_local_memory(self) -> bool {
        matches!(self, Self::MemoryAware)
    }

    #[must_use]
    pub const fn graph_id(self) -> &'static str {
        match self {
            Self::MemoryAware => "memory-aware-v1",
            Self::ExternalContext => "external-context-v1",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    pub session_id: SessionId,
    pub loop_id: LoopId,
    pub user_message: String,
    pub plan: Option<String>,
    #[serde(default)]
    pub conversation_history: Vec<ConversationMessage>,
    #[serde(default)]
    pub turn_messages: Vec<ConversationMessage>,
    pub recent_session: Vec<RawNode>,
    pub activation_query: Option<ActivationQuery>,
    pub query_embedding: Option<Embedding>,
    pub activated_memory: ActivatedMemory,
    pub assembled_context: Option<AssembledContext>,
    pub latest_model_output: Option<ModelOutput>,
    pub pending_tool_calls: Vec<ToolCallRequest>,
    pub tool_results: Vec<ToolCallResult>,
    pub persisted_raw_ids: Vec<RawNodeId>,
    pub tool_result_ids: Vec<RawNodeId>,
    pub new_abstract_ids: Vec<AbstractNodeId>,
    pub session_window_ids: Vec<RawNodeId>,
    pub pushed_out_raw_ids: Vec<RawNodeId>,
    pub assistant_message: Option<String>,
    pub iteration: u32,
    pub tool_rounds_completed: u32,
    pub model_invocations: u32,
    pub last_completed_node: Option<String>,
    /// Stored in checkpoints so a paused loop cannot accidentally resume under
    /// a graph with different local-memory semantics. Older checkpoints omit
    /// this field and therefore retain the memory-aware default.
    #[serde(default)]
    pub execution_profile: ExecutionProfile,
}

#[cfg(test)]
mod checkpoint_compatibility_tests {
    use super::*;

    #[test]
    fn pre_external_context_checkpoint_defaults_new_transcript_fields() -> Result<()> {
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let state = ExecutionState::from_request(
            SessionRequest {
                session_id: Some(session_id),
                user_message: "legacy turn".to_string(),
                plan: None,
            },
            session_id,
            loop_id,
        );
        let mut checkpoint = state.checkpoint("run_model".to_string(), LoopStatus::Paused)?;
        let object = checkpoint
            .state_json
            .as_object_mut()
            .expect("execution state checkpoint must be an object");
        object.remove("conversation_history");
        object.remove("turn_messages");
        object.remove("execution_profile");

        let (restored, current_node, status) = ExecutionState::from_checkpoint(checkpoint)?;

        assert!(restored.conversation_history.is_empty());
        assert!(restored.turn_messages.is_empty());
        assert_eq!(restored.execution_profile, ExecutionProfile::MemoryAware);
        assert_eq!(current_node, "run_model");
        assert_eq!(status, LoopStatus::Paused);
        Ok(())
    }
}

impl ExecutionState {
    #[must_use]
    pub fn from_request(request: SessionRequest, session_id: SessionId, loop_id: LoopId) -> Self {
        Self {
            session_id,
            loop_id,
            user_message: request.user_message,
            plan: request.plan,
            conversation_history: Vec::new(),
            turn_messages: Vec::new(),
            recent_session: Vec::new(),
            activation_query: None,
            query_embedding: None,
            activated_memory: ActivatedMemory::default(),
            assembled_context: None,
            latest_model_output: None,
            pending_tool_calls: Vec::new(),
            tool_results: Vec::new(),
            persisted_raw_ids: Vec::new(),
            tool_result_ids: Vec::new(),
            new_abstract_ids: Vec::new(),
            session_window_ids: Vec::new(),
            pushed_out_raw_ids: Vec::new(),
            assistant_message: None,
            iteration: 0,
            tool_rounds_completed: 0,
            model_invocations: 0,
            last_completed_node: None,
            execution_profile: ExecutionProfile::default(),
        }
    }

    /// # Errors
    ///
    /// Returns an [`EngineError::Storage`] when the execution state cannot be
    /// serialized into JSON for inclusion in the checkpoint.
    pub fn checkpoint(&self, current_node: String, status: LoopStatus) -> Result<LoopState> {
        Ok(LoopState {
            checkpoint_version: 1,
            graph_id: self.execution_profile.graph_id().to_string(),
            session_id: self.session_id,
            loop_id: self.loop_id,
            current_node,
            status,
            state_json: serde_json::to_value(self).map_err(|err| {
                EngineError::Storage(format!("failed to serialize loop checkpoint state: {err}"))
            })?,
        })
    }

    /// # Errors
    ///
    /// Returns an [`EngineError::Storage`] when the checkpoint's `state_json`
    /// cannot be deserialized back into an `ExecutionState`.
    pub fn from_checkpoint(checkpoint: LoopState) -> Result<(Self, String, LoopStatus)> {
        if checkpoint.checkpoint_version != 1 {
            return Err(EngineError::RecoveryUnsafe(format!(
                "unsupported checkpoint schema version {}",
                checkpoint.checkpoint_version
            )));
        }
        let state: Self = serde_json::from_value(checkpoint.state_json).map_err(|err| {
            EngineError::Storage(format!(
                "failed to deserialize loop checkpoint state: {err}"
            ))
        })?;
        if state.session_id != checkpoint.session_id || state.loop_id != checkpoint.loop_id {
            return Err(EngineError::RecoveryUnsafe(
                "checkpoint envelope identity does not match serialized execution state"
                    .to_string(),
            ));
        }
        if checkpoint.graph_id != state.execution_profile.graph_id() {
            return Err(EngineError::RecoveryUnsafe(format!(
                "checkpoint graph {} does not match execution profile graph {}",
                checkpoint.graph_id,
                state.execution_profile.graph_id()
            )));
        }
        Ok((state, checkpoint.current_node, checkpoint.status))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeOutcome {
    Continue,
    Branch(String),
    Finish,
    Pause,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeRuntimeClass {
    Standard,
    /// Model-inference nodes. These await the LLM, whose own HTTP client is
    /// budgeted far higher than a `Standard` node, so they get a dedicated
    /// `model_timeout` instead of the small `node_timeout` (otherwise any
    /// completion taking longer than `node_timeout` is spuriously aborted).
    Model,
    ToolExecution,
    Distillation,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraphRunResult {
    pub status: LoopStatus,
    pub completed_steps: u32,
    pub tool_rounds_completed: u32,
    pub last_node: Option<String>,
}

#[derive(Clone, Default)]
pub struct RunOptions {
    /// Caller-stable loop identity for durable recovery. Ephemeral callers may
    /// leave it unset and receive a generated id in `SessionResponse`.
    pub loop_id: Option<LoopId>,
    pub max_graph_steps: Option<u32>,
    pub max_tool_rounds: Option<u32>,
    pub node_timeout: Option<Duration>,
    pub model_timeout: Option<Duration>,
    pub tool_timeout: Option<Duration>,
    pub distillation_timeout: Option<Duration>,
    pub maintenance_batch_size: Option<usize>,
    pub cancellation_token: Option<CancellationToken>,
    /// Product-owned durable history transported into the model context. It
    /// is deliberately an execution option so the stable `SessionRequest`
    /// surface remains a single-turn request and existing embedders can opt in.
    pub conversation_history: Vec<ConversationMessage>,
    /// The memory-aware graph remains the library default. Product runtimes
    /// that supply canonical external context must opt into the lean profile.
    pub execution_profile: ExecutionProfile,
}

#[derive(Clone)]
pub struct ResolvedRunOptions {
    pub max_graph_steps: u32,
    pub max_tool_rounds: u32,
    pub node_timeout: Duration,
    pub model_timeout: Duration,
    pub tool_timeout: Duration,
    pub distillation_timeout: Duration,
    pub maintenance_batch_size: usize,
    pub cancellation_token: Option<CancellationToken>,
    pub execution_profile: ExecutionProfile,
}

impl ResolvedRunOptions {
    #[must_use]
    pub fn from_config(config: &EngineConfig, options: RunOptions) -> Self {
        Self {
            max_graph_steps: options
                .max_graph_steps
                .unwrap_or(config.runtime.max_graph_steps)
                .min(config.runtime.max_graph_steps),
            max_tool_rounds: options
                .max_tool_rounds
                .unwrap_or(config.runtime.max_tool_rounds)
                .min(config.runtime.max_tool_rounds),
            node_timeout: options
                .node_timeout
                .unwrap_or_else(|| config.runtime.node_timeout())
                .min(config.runtime.node_timeout()),
            model_timeout: options
                .model_timeout
                .unwrap_or_else(|| config.runtime.model_timeout())
                .min(config.runtime.model_timeout()),
            tool_timeout: options
                .tool_timeout
                .unwrap_or_else(|| config.runtime.tool_timeout())
                .min(config.runtime.tool_timeout()),
            distillation_timeout: options
                .distillation_timeout
                .unwrap_or_else(|| config.runtime.distillation_timeout())
                .min(config.runtime.distillation_timeout()),
            maintenance_batch_size: options
                .maintenance_batch_size
                .unwrap_or(config.runtime.maintenance_batch_size)
                .min(config.runtime.maintenance_batch_size),
            cancellation_token: options.cancellation_token,
            execution_profile: options.execution_profile,
        }
    }

    #[must_use]
    pub const fn timeout_for_class(&self, class: NodeRuntimeClass) -> Duration {
        match class {
            NodeRuntimeClass::Standard => self.node_timeout,
            NodeRuntimeClass::Model => self.model_timeout,
            NodeRuntimeClass::ToolExecution => self.tool_timeout,
            NodeRuntimeClass::Distillation => self.distillation_timeout,
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancellation_token
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    }

    pub fn validate(&self) -> Result<()> {
        if self.max_graph_steps == 0
            || self.max_tool_rounds == 0
            || self.maintenance_batch_size == 0
        {
            return Err(EngineError::Configuration(
                "run option budgets must be greater than zero".to_string(),
            ));
        }
        if [
            self.node_timeout,
            self.model_timeout,
            self.tool_timeout,
            self.distillation_timeout,
        ]
        .into_iter()
        .any(|timeout| timeout.is_zero())
        {
            return Err(EngineError::Configuration(
                "run option timeouts must be greater than zero".to_string(),
            ));
        }
        Ok(())
    }
}

#[async_trait]
pub trait GraphNode: Send + Sync {
    fn id(&self) -> &'static str;

    fn runtime_class(&self) -> NodeRuntimeClass {
        NodeRuntimeClass::Standard
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        config: &EngineConfig,
        deps: &EngineDeps,
        options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome>;
}

#[derive(Default)]
pub struct ExecutionGraph {
    start: String,
    nodes: HashMap<String, Arc<dyn GraphNode>>,
    edges: HashMap<(String, String), String>,
}

impl ExecutionGraph {
    pub fn new(start: impl Into<String>) -> Self {
        Self {
            start: start.into(),
            nodes: HashMap::new(),
            edges: HashMap::new(),
        }
    }

    pub fn add_node(&mut self, node: Arc<dyn GraphNode>) {
        self.nodes.insert(node.id().to_string(), node);
    }

    pub fn add_edge(
        &mut self,
        from: impl Into<String>,
        branch: impl Into<String>,
        to: impl Into<String>,
    ) {
        self.edges.insert((from.into(), branch.into()), to.into());
    }

    #[must_use]
    pub fn start_node(&self) -> &str {
        &self.start
    }

    /// Number of registered nodes. Used by tests asserting graph topology.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of registered edges. Used by tests asserting graph topology.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    fn resolve_next(&self, current_node: &str, outcome: &NodeOutcome) -> Result<Option<String>> {
        match outcome {
            NodeOutcome::Finish | NodeOutcome::Pause => Ok(None),
            NodeOutcome::Continue => self
                .edges
                .get(&(current_node.to_string(), DEFAULT_EDGE.to_string()))
                .cloned()
                .map(Some)
                .ok_or_else(|| {
                    EngineError::Configuration(format!(
                        "missing default edge from execution node {current_node}"
                    ))
                }),
            NodeOutcome::Branch(branch) => self
                .edges
                .get(&(current_node.to_string(), branch.clone()))
                .cloned()
                .map(Some)
                .ok_or_else(|| {
                    EngineError::Configuration(format!(
                        "missing branch edge {branch} from execution node {current_node}"
                    ))
                }),
        }
    }

    fn node(&self, id: &str) -> Result<&Arc<dyn GraphNode>> {
        self.nodes.get(id).ok_or_else(|| {
            EngineError::Configuration(format!("execution graph node {id} is not registered"))
        })
    }
}

pub struct GraphRunner {
    graph: Arc<ExecutionGraph>,
}

impl GraphRunner {
    #[must_use]
    pub const fn new(graph: Arc<ExecutionGraph>) -> Self {
        Self { graph }
    }

    /// # Errors
    ///
    /// Returns the first [`EngineError`] surfaced by a graph node — node
    /// timeouts, repository / model / tool failures, cancellation, or
    /// checkpoint persistence errors.
    pub async fn run(
        &self,
        state: &mut ExecutionState,
        config: &EngineConfig,
        deps: &EngineDeps,
        options: &ResolvedRunOptions,
    ) -> Result<GraphRunResult> {
        options.validate()?;
        // `GraphRunner` is public and callers are not required to construct
        // state through `run_turn_with_options`. Persist the selected profile
        // before the first checkpoint so a direct external-context run can be
        // resumed under the same graph semantics.
        state.execution_profile = options.execution_profile;
        self.run_from_node(
            self.graph.start_node().to_string(),
            state,
            config,
            deps,
            options,
        )
        .await
    }

    /// # Errors
    ///
    /// Returns [`EngineError::LoopTerminated`] when the checkpoint is not
    /// paused, plus any [`EngineError`] raised by the resumed execution per
    /// [`Self::run`].
    pub async fn resume(
        &self,
        checkpoint: LoopState,
        config: &EngineConfig,
        deps: &EngineDeps,
        options: &ResolvedRunOptions,
    ) -> Result<(ExecutionState, GraphRunResult)> {
        options.validate()?;
        if checkpoint.status != LoopStatus::Paused {
            return Err(EngineError::LoopTerminated(checkpoint.status));
        }
        let (mut state, current_node, _status) = ExecutionState::from_checkpoint(checkpoint)?;
        if state.execution_profile != options.execution_profile {
            return Err(EngineError::Configuration(format!(
                "checkpoint execution profile {:?} does not match requested profile {:?}",
                state.execution_profile, options.execution_profile
            )));
        }
        let result = self
            .run_from_node(current_node, &mut state, config, deps, options)
            .await?;
        Ok((state, result))
    }

    /// Recover a checkpoint left in `Running` state by an interrupted process.
    /// The checkpoint is written immediately before its `current_node`, so the
    /// node is intentionally executed again. Callers must only use this path
    /// when that node's external effects are idempotent or otherwise fenced.
    ///
    /// # Errors
    ///
    /// Returns [`EngineError::LoopTerminated`] unless the checkpoint is
    /// `Running`, plus the same profile/configuration/runtime errors as
    /// [`Self::resume`].
    pub async fn recover_running(
        &self,
        checkpoint: LoopState,
        config: &EngineConfig,
        deps: &EngineDeps,
        options: &ResolvedRunOptions,
    ) -> Result<(ExecutionState, GraphRunResult)> {
        options.validate()?;
        if checkpoint.status != LoopStatus::Running {
            return Err(EngineError::LoopTerminated(checkpoint.status));
        }
        let (mut state, current_node, _status) = ExecutionState::from_checkpoint(checkpoint)?;
        if state.execution_profile != options.execution_profile {
            return Err(EngineError::Configuration(format!(
                "checkpoint execution profile {:?} does not match requested profile {:?}",
                state.execution_profile, options.execution_profile
            )));
        }
        let result = self
            .run_from_node(current_node, &mut state, config, deps, options)
            .await?;
        Ok((state, result))
    }

    async fn save_cancelled_checkpoint(
        &self,
        current_node: &str,
        state: &ExecutionState,
        deps: &EngineDeps,
    ) -> Result<GraphRunResult> {
        warn!(
            session_id = %state.session_id,
            loop_id = %state.loop_id,
            node = current_node,
            "graph execution cancelled"
        );
        let cancelled = state.checkpoint(current_node.to_string(), LoopStatus::Cancelled)?;
        deps.loop_state_repository
            .save_checkpoint(cancelled)
            .await?;
        Ok(GraphRunResult {
            status: LoopStatus::Cancelled,
            completed_steps: state.iteration,
            tool_rounds_completed: state.tool_rounds_completed,
            last_node: Some(current_node.to_string()),
        })
    }

    // Each iteration handles one node — splitting the match arms into helpers
    // would just spread the lifecycle (cancellation / timeout / error / outcome
    // / branching / completion) across functions that all need the same locals.
    #[allow(clippy::too_many_lines)]
    async fn run_from_node(
        &self,
        mut current_node: String,
        state: &mut ExecutionState,
        config: &EngineConfig,
        deps: &EngineDeps,
        options: &ResolvedRunOptions,
    ) -> Result<GraphRunResult> {
        enum NodeExecutionResult {
            Completed(Result<NodeOutcome>),
            TimedOut,
            Cancelled,
        }

        loop {
            if options.is_cancelled() {
                return self
                    .save_cancelled_checkpoint(&current_node, state, deps)
                    .await;
            }

            if state.iteration >= options.max_graph_steps {
                warn!(session_id = %state.session_id, loop_id = %state.loop_id, node = %current_node, "graph execution hit max step budget");
                let timed_out = state.checkpoint(current_node.clone(), LoopStatus::TimedOut)?;
                deps.loop_state_repository
                    .save_checkpoint(timed_out)
                    .await?;
                return Ok(GraphRunResult {
                    status: LoopStatus::TimedOut,
                    completed_steps: state.iteration,
                    tool_rounds_completed: state.tool_rounds_completed,
                    last_node: Some(current_node),
                });
            }

            let node = self.graph.node(&current_node)?.clone();
            state.iteration = state.iteration.saturating_add(1);
            let running = state.checkpoint(current_node.clone(), LoopStatus::Running)?;
            deps.loop_state_repository.save_checkpoint(running).await?;
            if options.is_cancelled() {
                return self
                    .save_cancelled_checkpoint(&current_node, state, deps)
                    .await;
            }

            debug!(
                session_id = %state.session_id,
                loop_id = %state.loop_id,
                node = %current_node,
                step = state.iteration,
                "running execution node"
            );

            let runtime_class = node.runtime_class();
            let node_future = node.run(state, config, deps, options);
            let execution = if runtime_class == NodeRuntimeClass::ToolExecution {
                // ExecuteToolsNode applies `tool_timeout` to every individual
                // call. A second outer timeout with the same duration cuts a
                // serial side-effect phase off part-way through and loses the
                // completed results. The per-call timeouts, fan-out cap, graph
                // step budget, and cancellation token keep this node bounded.
                // ExecuteToolsNode owns cancellation after dispatch so it can
                // distinguish a clean pre-dispatch cancel from an ambiguous
                // side-effecting outcome.
                NodeExecutionResult::Completed(node_future.await)
            } else {
                let run_node = timeout(options.timeout_for_class(runtime_class), node_future);
                if let Some(token) = &options.cancellation_token {
                    tokio::select! {
                        () = token.cancelled() => NodeExecutionResult::Cancelled,
                        result = run_node => match result {
                            Ok(outcome) => NodeExecutionResult::Completed(outcome),
                            Err(_) => NodeExecutionResult::TimedOut,
                        },
                    }
                } else {
                    match run_node.await {
                        Ok(outcome) => NodeExecutionResult::Completed(outcome),
                        Err(_) => NodeExecutionResult::TimedOut,
                    }
                }
            };

            let outcome = match execution {
                NodeExecutionResult::Cancelled
                | NodeExecutionResult::Completed(Err(EngineError::Cancelled)) => {
                    return self
                        .save_cancelled_checkpoint(&current_node, state, deps)
                        .await;
                }
                NodeExecutionResult::Completed(Ok(outcome)) => outcome,
                NodeExecutionResult::Completed(Err(error)) => {
                    let failed = state.checkpoint(current_node.clone(), LoopStatus::Failed)?;
                    deps.loop_state_repository.save_checkpoint(failed).await?;
                    return Err(error);
                }
                NodeExecutionResult::TimedOut => {
                    warn!(
                        session_id = %state.session_id,
                        loop_id = %state.loop_id,
                        node = %current_node,
                        "execution node timed out"
                    );
                    let timed_out = state.checkpoint(current_node.clone(), LoopStatus::TimedOut)?;
                    deps.loop_state_repository
                        .save_checkpoint(timed_out)
                        .await?;
                    return Ok(GraphRunResult {
                        status: LoopStatus::TimedOut,
                        completed_steps: state.iteration,
                        tool_rounds_completed: state.tool_rounds_completed,
                        last_node: Some(current_node),
                    });
                }
            };

            if options.is_cancelled() && runtime_class != NodeRuntimeClass::ToolExecution {
                return self
                    .save_cancelled_checkpoint(&current_node, state, deps)
                    .await;
            }

            state.last_completed_node = Some(current_node.clone());
            match outcome {
                NodeOutcome::Finish => {
                    info!(
                        session_id = %state.session_id,
                        loop_id = %state.loop_id,
                        steps = state.iteration,
                        tool_rounds = state.tool_rounds_completed,
                        "graph execution finished"
                    );
                    deps.loop_state_repository
                        .clear_checkpoint(&state.session_id, &state.loop_id)
                        .await?;
                    return Ok(GraphRunResult {
                        status: LoopStatus::Finished,
                        completed_steps: state.iteration,
                        tool_rounds_completed: state.tool_rounds_completed,
                        last_node: state.last_completed_node.clone(),
                    });
                }
                NodeOutcome::Pause => {
                    let paused = state.checkpoint(current_node.clone(), LoopStatus::Paused)?;
                    deps.loop_state_repository.save_checkpoint(paused).await?;
                    return Ok(GraphRunResult {
                        status: LoopStatus::Paused,
                        completed_steps: state.iteration,
                        tool_rounds_completed: state.tool_rounds_completed,
                        last_node: Some(current_node),
                    });
                }
                NodeOutcome::Continue | NodeOutcome::Branch(_) => {
                    let next_node = self
                        .graph
                        .resolve_next(&current_node, &outcome)?
                        .ok_or_else(|| {
                            EngineError::Configuration(format!(
                                "execution graph node {current_node} did not resolve to a next node"
                            ))
                        })?;
                    current_node = next_node;
                }
            }
        }
    }
}

#[cfg(test)]
mod checkpoint_identity_tests {
    use super::*;
    use crate::engine::session_engine::SessionRequest;

    fn checkpoint() -> LoopState {
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let state = ExecutionState::from_request(
            SessionRequest {
                session_id: Some(session_id),
                user_message: "checkpoint".to_string(),
                plan: None,
            },
            session_id,
            loop_id,
        );
        state
            .checkpoint("load_session_view".to_string(), LoopStatus::Running)
            .expect("checkpoint should serialize")
    }

    #[test]
    fn checkpoint_rejects_outer_inner_identity_mismatch() {
        let mut checkpoint = checkpoint();
        checkpoint.session_id = SessionId::new();
        assert!(matches!(
            ExecutionState::from_checkpoint(checkpoint),
            Err(EngineError::RecoveryUnsafe(_))
        ));
    }

    #[test]
    fn checkpoint_rejects_graph_and_schema_mismatch() {
        let mut graph_mismatch = checkpoint();
        graph_mismatch.graph_id = "different-graph".to_string();
        assert!(matches!(
            ExecutionState::from_checkpoint(graph_mismatch),
            Err(EngineError::RecoveryUnsafe(_))
        ));

        let mut version_mismatch = checkpoint();
        version_mismatch.checkpoint_version = 99;
        assert!(matches!(
            ExecutionState::from_checkpoint(version_mismatch),
            Err(EngineError::RecoveryUnsafe(_))
        ));
    }

    #[test]
    fn run_options_cannot_disable_or_expand_configured_guards() {
        let config = EngineConfig::default();
        let zero = ResolvedRunOptions::from_config(
            &config,
            RunOptions {
                max_graph_steps: Some(0),
                ..RunOptions::default()
            },
        );
        assert!(zero.validate().is_err());

        let expanded = ResolvedRunOptions::from_config(
            &config,
            RunOptions {
                max_graph_steps: Some(u32::MAX),
                tool_timeout: Some(Duration::MAX),
                maintenance_batch_size: Some(usize::MAX),
                ..RunOptions::default()
            },
        );
        assert_eq!(expanded.max_graph_steps, config.runtime.max_graph_steps);
        assert_eq!(expanded.tool_timeout, config.runtime.tool_timeout());
        assert_eq!(
            expanded.maintenance_batch_size,
            config.runtime.maintenance_batch_size
        );
    }
}
