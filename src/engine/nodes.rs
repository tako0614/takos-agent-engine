//! Execution-graph node implementations and the shared helpers they call.
//!
//! The engine's [`ExecutionGraph`](crate::engine::execution_graph::ExecutionGraph)
//! is wired from these [`GraphNode`] structs by
//! [`graph_spec`](crate::engine::graph_spec). The lifecycle entry points in
//! [`session_engine`](crate::engine::session_engine) drive the runner; the node
//! impls and the persistence / model-input helpers that back them live here so
//! `session_engine.rs` stays focused on the public lifecycle surface.

use async_trait::async_trait;
use chrono::{Duration as ChronoDuration, Utc};

use crate::config::{EngineConfig, ToolsConfig};
use crate::domain::{
    AbstractNode, DistillationState, OverflowPolicy, RawContent, RawNode, RawNodeKind,
};
use crate::engine::execution_graph::{
    ExecutionState, GraphNode, GraphRunResult, NodeOutcome, NodeRuntimeClass, ResolvedRunOptions,
};
use crate::engine::session_engine::{EngineDeps, SessionResponse};
use crate::error::{EngineError, Result};
use crate::ids::{LoopId, SessionId};
use crate::memory::{ActivationQuery, DistillationInput};
use crate::model::{ModelInput, ToolCallRequest};
use crate::storage::RawLifecyclePatch;
use crate::tools::executor::ToolCallResult;
use crate::tools::memory_tools::{
    GraphSearchParams, MemorySearchParams, MemoryToolBounds, TimelineSearchParams,
};

// Node ids that a fixed-id node returns from `GraphNode::id` (or that tests
// reference), so they must stay named. The query/activate/assemble/model nodes
// carry their id as a `&'static str` field instead of one constant each,
// because the graph instantiates two of each: the opening pass and the
// post-tool re-activation pass.
pub(crate) const INGEST_USER_INPUT_NODE: &str = "ingest_user_input";
pub(crate) const LOAD_SESSION_VIEW_NODE: &str = "load_session_view";
pub(crate) const EXECUTE_TOOLS_NODE: &str = "execute_tools";
pub(crate) const PERSIST_ASSISTANT_OUTPUT_NODE: &str = "persist_assistant_output";
pub(crate) const MARK_SESSION_OVERFLOW_NODE: &str = "mark_session_overflow";
pub(crate) const DISTILL_CURRENT_LOOP_NODE: &str = "distill_current_loop";

// Per-kind importance priors assigned to raw nodes at creation time.
//
// These seed `RawNodeMetadata.importance`, which `DefaultScoringPolicy`
// multiplies by `importance_weight` and adds to the similarity score when
// ranking memories for activation (see `memory::scoring`). They are a coarse
// prior, not a learned value: user utterances carry the goal/intent and are
// weighted highest; assistant utterances are the committed response and rank
// just below; tool results are supporting evidence that is often verbose and
// duplicative, so they rank lowest. The relative ordering
// (user > assistant > tool) is what matters; absolute values are arbitrary
// within (0.0, 1.0]. Tuning the retrieval prior means editing these constants.
const USER_UTTERANCE_IMPORTANCE: f32 = 0.85;
const ASSISTANT_UTTERANCE_IMPORTANCE: f32 = 0.78;
const TOOL_RESULT_IMPORTANCE: f32 = 0.72;

pub(crate) struct IngestUserInputNode;

#[async_trait]
impl GraphNode for IngestUserInputNode {
    fn id(&self) -> &'static str {
        INGEST_USER_INPUT_NODE
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        _config: &EngineConfig,
        deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            Some(state.session_id),
            Some(state.loop_id),
            "user",
            state.user_message.clone(),
            USER_UTTERANCE_IMPORTANCE,
            vec!["input".to_string()],
        )
        .with_operation_key(user_input_operation_key(state.loop_id));
        let node = persist_raw_node(deps, node).await?;
        push_raw_node_into_state(state, node);
        Ok(NodeOutcome::Continue)
    }
}

pub(crate) struct LoadSessionViewNode;

#[async_trait]
impl GraphNode for LoadSessionViewNode {
    fn id(&self) -> &'static str {
        LOAD_SESSION_VIEW_NODE
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        _config: &EngineConfig,
        deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        state.recent_session = deps.repository.session_raw(&state.session_id).await?;
        Ok(NodeOutcome::Continue)
    }
}

pub(crate) struct BuildActivationQueryNode {
    pub(crate) id: &'static str,
}

#[async_trait]
impl GraphNode for BuildActivationQueryNode {
    fn id(&self) -> &'static str {
        self.id
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        _config: &EngineConfig,
        _deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        let recent_context = state
            .recent_session
            .iter()
            .rev()
            .take(12)
            .map(crate::domain::raw_node::RawNode::context_text)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        state.activation_query = Some(ActivationQuery::new(
            state.user_message.clone(),
            recent_context,
            state.plan.clone(),
            state
                .tool_results
                .iter()
                .map(|result| result.summary.clone())
                .collect(),
        ));
        Ok(NodeOutcome::Continue)
    }
}

pub(crate) struct ActivateMemoryNode {
    pub(crate) id: &'static str,
}

#[async_trait]
impl GraphNode for ActivateMemoryNode {
    fn id(&self) -> &'static str {
        self.id
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        config: &EngineConfig,
        deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        let activation_query = state.activation_query.as_ref().ok_or_else(|| {
            EngineError::Configuration(
                "activation query must exist before memory activation".into(),
            )
        })?;
        let query_embedding = deps
            .embedder
            .embed_text(&activation_query.as_embedding_input())
            .await?;
        state.query_embedding = Some(query_embedding.clone());
        state.activated_memory = deps
            .activation_service()
            .activate(
                config,
                &query_embedding,
                Utc::now(),
                Some(&state.session_id),
            )
            .await?;
        Ok(NodeOutcome::Continue)
    }
}

pub(crate) struct AssembleContextNode {
    pub(crate) id: &'static str,
    pub(crate) reload_session: bool,
}

#[async_trait]
impl GraphNode for AssembleContextNode {
    fn id(&self) -> &'static str {
        self.id
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        config: &EngineConfig,
        deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        if self.reload_session {
            state.recent_session = deps.repository.session_raw(&state.session_id).await?;
        }
        let context = deps.context_assembler().assemble(
            &config.context_budget,
            &config.system_prompt,
            &state.recent_session,
            &state.activated_memory,
            &state.tool_results,
        );
        state
            .session_window_ids
            .clone_from(&context.session_window.included_raw_ids);
        state
            .pushed_out_raw_ids
            .clone_from(&context.session_window.pushed_out_raw_ids);
        state.assembled_context = Some(context);
        Ok(NodeOutcome::Continue)
    }
}

pub(crate) struct ModelNode {
    pub(crate) id: &'static str,
}

#[async_trait]
impl GraphNode for ModelNode {
    fn id(&self) -> &'static str {
        self.id
    }

    fn runtime_class(&self) -> NodeRuntimeClass {
        // Model nodes await the LLM; budget them with `model_timeout` (default
        // 60s, >= the model HTTP client timeout) rather than the small
        // `node_timeout`, which would force-abort any completion >10s.
        NodeRuntimeClass::Model
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        _config: &EngineConfig,
        deps: &EngineDeps,
        options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        let context = state.assembled_context.as_ref().ok_or_else(|| {
            EngineError::Configuration("assembled context must exist before model execution".into())
        })?;
        let output = deps
            .model_runner
            .run(to_model_input(
                state.session_id,
                state.loop_id,
                &state.user_message,
                state.plan.clone(),
                context,
            ))
            .await?;
        state.model_invocations = state.model_invocations.saturating_add(1);
        state.pending_tool_calls.clone_from(&output.tool_calls);
        if let Some(message) = &output.assistant_message {
            state.assistant_message = Some(message.clone());
        }
        state.latest_model_output = Some(output);

        if !state.pending_tool_calls.is_empty()
            && state.tool_rounds_completed < options.max_tool_rounds
        {
            return Ok(NodeOutcome::Branch("needs_tools".to_string()));
        }

        if !state.pending_tool_calls.is_empty() && state.assistant_message.is_none() {
            state.assistant_message = Some(format!(
                "Tool round limit reached after {} rounds without a final assistant message.",
                options.max_tool_rounds
            ));
        }
        Ok(NodeOutcome::Continue)
    }
}

pub(crate) struct ExecuteToolsNode;

#[async_trait]
impl GraphNode for ExecuteToolsNode {
    fn id(&self) -> &'static str {
        EXECUTE_TOOLS_NODE
    }

    fn runtime_class(&self) -> NodeRuntimeClass {
        NodeRuntimeClass::ToolExecution
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        config: &EngineConfig,
        deps: &EngineDeps,
        options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        let calls = std::mem::take(&mut state.pending_tool_calls);
        let round = state.tool_rounds_completed.saturating_add(1);

        // Prepare every pending call up front (validation, arg clamping,
        // operation_key derivation). This stays sequential because it is
        // cheap, deterministic, and lets us bail early on a configuration
        // error before spawning any tasks. We also probe the dedup cache
        // here so already-persisted tool results skip the tool invocation
        // entirely and parallel execution only fans out the real work.
        let mut prepared: Vec<PreparedToolCall> = Vec::with_capacity(calls.len());
        for (index, call) in calls.into_iter().enumerate() {
            if options.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            let call = prepare_tool_call_for_config(call, &config.tools)?;
            let operation_key = tool_result_operation_key(state.loop_id, round, index, &call.name);
            let cached = deps
                .repository
                .get_raw_by_operation_key(&operation_key)
                .await?;
            prepared.push(PreparedToolCall {
                index,
                call,
                operation_key,
                cached,
            });
        }

        // Fan out: every prepared call that does not have a cached raw node
        // is spawned with its own `tool_timeout` so a single slow tool can
        // not consume the entire node budget. `tokio::spawn` ensures the
        // tasks make progress in parallel on the multi-thread scheduler.
        let mut handles: Vec<(usize, tokio::task::JoinHandle<TimedToolOutcome>)> = Vec::new();
        for prep in &prepared {
            if prep.cached.is_some() {
                continue;
            }
            let executor = deps.tool_executor.clone();
            let cancellation = options.cancellation_token.clone();
            let timeout = options.tool_timeout;
            let call = prep.call.clone();
            let tool_name = call.name.clone();
            let handle = tokio::spawn(async move {
                // Per-task cancellation race: if the run-wide token fires we
                // surface `Cancelled` immediately instead of waiting for the
                // tool to honor cancellation.
                let exec_future = executor.execute(call);
                let cancel_aware: std::pin::Pin<
                    Box<dyn std::future::Future<Output = Result<ToolCallResult>> + Send>,
                > = if let Some(token) = cancellation {
                    Box::pin(async move {
                        tokio::select! {
                            biased;
                            () = token.cancelled() => Err(EngineError::Cancelled),
                            result = exec_future => result,
                        }
                    })
                } else {
                    Box::pin(exec_future)
                };
                match tokio::time::timeout(timeout, cancel_aware).await {
                    Ok(result) => TimedToolOutcome::Completed(result),
                    Err(_) => TimedToolOutcome::TimedOut { tool_name },
                }
            });
            handles.push((prep.index, handle));
        }

        // Collect results into a sparse vector keyed by the prepared index so
        // we can recombine deterministically with the cached entries below.
        let mut completed: std::collections::HashMap<usize, Result<ToolCallResult>> =
            std::collections::HashMap::new();
        for (index, handle) in handles {
            let outcome = match handle.await {
                Ok(value) => value,
                Err(join_err) => {
                    // A panic / cancellation of the spawn task itself is a
                    // hard tool error rather than a silent skip.
                    return Err(EngineError::Tool(format!(
                        "tool task join failed: {join_err}"
                    )));
                }
            };
            match outcome {
                TimedToolOutcome::Completed(result) => {
                    completed.insert(index, result);
                }
                TimedToolOutcome::TimedOut { tool_name } => {
                    completed.insert(
                        index,
                        Err(EngineError::Tool(format!(
                            "tool {tool_name} exceeded the configured tool_timeout of {:?}",
                            options.tool_timeout
                        ))),
                    );
                }
            }
        }

        if options.is_cancelled() {
            return Err(EngineError::Cancelled);
        }

        // Reassemble results in the original tool-call order so downstream
        // model rounds see them deterministically. Persistence still happens
        // sequentially because the raw-node insert path is not designed for
        // concurrent writers (operation_key uniqueness, sync_raw_indexes).
        for prep in prepared {
            let raw_node = if let Some(existing) = prep.cached {
                existing
            } else {
                let result_value = completed.remove(&prep.index).ok_or_else(|| {
                    EngineError::Tool(format!(
                        "missing tool result for index {} (tool {})",
                        prep.index, prep.call.name
                    ))
                })?;
                let result = result_value?;
                let raw = RawNode::json(
                    RawNodeKind::ToolResult,
                    Some(state.session_id),
                    Some(state.loop_id),
                    format!("tool:{}", result.name),
                    serde_json::to_value(&result).map_err(|err| {
                        EngineError::Tool(format!("failed to encode tool result payload: {err}"))
                    })?,
                    TOOL_RESULT_IMPORTANCE,
                    vec!["tool".to_string()],
                )
                .with_operation_key(prep.operation_key.clone());
                let raw = persist_raw_node(deps, raw).await?;
                state.last_effect_key = Some(prep.operation_key);
                raw
            };

            let tool_result = decode_tool_result_from_raw(&raw_node)?;
            push_raw_node_into_state(state, raw_node);
            state.tool_results.push(tool_result);
        }

        state.tool_rounds_completed = round;
        Ok(NodeOutcome::Continue)
    }
}

/// Holds a fully-prepared tool call ready for either dedup-replay or
/// parallel execution.
struct PreparedToolCall {
    /// Position in the original `pending_tool_calls` vector. Used to keep
    /// downstream ordering deterministic regardless of which task finishes
    /// first.
    index: usize,
    call: ToolCallRequest,
    operation_key: String,
    /// `Some(existing)` when an earlier persisted raw node already covers
    /// this operation_key. In that case we do not spawn the tool at all.
    cached: Option<RawNode>,
}

/// Result of a single parallel tool task. We distinguish a per-tool timeout
/// from a generic engine error so the surfaced message clearly attributes
/// the failure mode.
enum TimedToolOutcome {
    Completed(Result<ToolCallResult>),
    TimedOut { tool_name: String },
}

pub(crate) struct PersistAssistantOutputNode;

#[async_trait]
impl GraphNode for PersistAssistantOutputNode {
    fn id(&self) -> &'static str {
        PERSIST_ASSISTANT_OUTPUT_NODE
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        _config: &EngineConfig,
        deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        let assistant_message = state
            .assistant_message
            .clone()
            .or_else(|| {
                state
                    .latest_model_output
                    .as_ref()
                    .and_then(|output| output.assistant_message.clone())
            })
            .unwrap_or_else(|| "No assistant message generated.".to_string());
        let node = RawNode::text(
            RawNodeKind::AssistantUtterance,
            Some(state.session_id),
            Some(state.loop_id),
            "assistant",
            assistant_message.clone(),
            ASSISTANT_UTTERANCE_IMPORTANCE,
            vec!["output".to_string()],
        )
        .with_operation_key(assistant_output_operation_key(state.loop_id));
        let node = persist_raw_node(deps, node).await?;
        state.assistant_message = Some(assistant_message);
        push_raw_node_into_state(state, node);
        Ok(NodeOutcome::Continue)
    }
}

pub(crate) struct MarkSessionOverflowNode;

#[async_trait]
impl GraphNode for MarkSessionOverflowNode {
    fn id(&self) -> &'static str {
        MARK_SESSION_OVERFLOW_NODE
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        _config: &EngineConfig,
        deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        let clear_ids: Vec<_> = state
            .recent_session
            .iter()
            .filter(|node| {
                node.distillation_state != DistillationState::Distilled
                    && state.session_window_ids.contains(&node.id)
            })
            .map(|node| node.id)
            .collect();
        if !clear_ids.is_empty() {
            deps.repository
                .update_raw_lifecycle(
                    &clear_ids,
                    &RawLifecyclePatch {
                        distillation_state: None,
                        overflow: Some(OverflowPolicy {
                            was_pushed_out_of_session: false,
                            relax_retrieval_until: None,
                        }),
                    },
                )
                .await?;
        }

        let pushed_ids: Vec<_> = state
            .recent_session
            .iter()
            .filter(|node| {
                node.distillation_state != DistillationState::Distilled
                    && state.pushed_out_raw_ids.contains(&node.id)
            })
            .map(|node| node.id)
            .collect();
        if !pushed_ids.is_empty() {
            deps.repository
                .update_raw_lifecycle(
                    &pushed_ids,
                    &RawLifecyclePatch {
                        distillation_state: None,
                        overflow: Some(OverflowPolicy {
                            was_pushed_out_of_session: true,
                            relax_retrieval_until: Some(Utc::now() + ChronoDuration::hours(24)),
                        }),
                    },
                )
                .await?;
        }

        state.recent_session = deps.repository.session_raw(&state.session_id).await?;
        Ok(NodeOutcome::Continue)
    }
}

pub(crate) struct DistillCurrentLoopNode;

#[async_trait]
impl GraphNode for DistillCurrentLoopNode {
    fn id(&self) -> &'static str {
        DISTILL_CURRENT_LOOP_NODE
    }

    fn runtime_class(&self) -> NodeRuntimeClass {
        NodeRuntimeClass::Distillation
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        _config: &EngineConfig,
        deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        let raw_nodes = deps.repository.raw_for_loop(&state.loop_id).await?;
        let distilled = deps
            .distiller
            .distill(DistillationInput {
                session_id: state.session_id,
                loop_id: state.loop_id,
                raw_nodes,
                activated_abstract_ids: state
                    .activated_memory
                    .abstract_nodes
                    .iter()
                    .map(|entry| entry.node.id)
                    .collect(),
            })
            .await?;

        for node in distilled.new_nodes {
            if let Some(node) = persist_abstract_node(deps, node, Some(state.session_id)).await? {
                state.new_abstract_ids.push(node.id);
                if let Some(operation_key) = &node.operation_key {
                    state.last_effect_key = Some(operation_key.clone());
                }
            }
        }
        for update in distilled.raw_updates {
            deps.repository
                .update_raw_lifecycle(&[update.raw_node_id], &update.patch)
                .await?;
        }
        Ok(NodeOutcome::Finish)
    }
}

pub(crate) fn build_response(state: ExecutionState, result: GraphRunResult) -> SessionResponse {
    SessionResponse {
        session_id: state.session_id,
        loop_id: state.loop_id,
        status: result.status,
        assistant_message: state.assistant_message,
        activated_raw_count: state.activated_memory.raw_nodes.len(),
        activated_abstract_count: state.activated_memory.abstract_nodes.len(),
        tool_results_count: state.tool_results.len(),
        completed_steps: result.completed_steps,
        tool_rounds_completed: result.tool_rounds_completed,
    }
}

fn to_model_input(
    session_id: SessionId,
    loop_id: LoopId,
    user_message: &str,
    plan: Option<String>,
    context: &crate::engine::context_assembler::AssembledContext,
) -> ModelInput {
    ModelInput {
        session_id,
        loop_id,
        system_prompt: context.system_prompt.clone(),
        session_context: context.session_context.clone(),
        memory_context: context.memory_context.clone(),
        tool_context: context.tool_context.clone(),
        user_message: user_message.to_string(),
        plan,
    }
}

async fn persist_raw_node(deps: &EngineDeps, node: RawNode) -> Result<RawNode> {
    if let Some(operation_key) = &node.operation_key {
        if let Some(existing) = deps
            .repository
            .get_raw_by_operation_key(operation_key)
            .await?
        {
            return Ok(existing);
        }
    }
    let embedding = deps.embedder.embed_text(&node.content_text()).await?;
    let session_id = node.session_id;
    deps.repository.insert_raw(node.clone()).await?;
    // Use the session-aware indexing path so per-session retrieval stays
    // isolated. The raw node's own `session_id` is the source of truth here.
    deps.vector_index
        .index_raw_with_session(node.id, embedding, session_id)
        .await?;
    Ok(node)
}

/// Persist a distilled abstract node, deduplicating on its `operation_key`.
///
/// Returns `Ok(Some(node))` when a NEW abstract was inserted, and `Ok(None)`
/// when an abstract already existed for the `operation_key` (a dedup hit, no
/// write). Callers use this distinction to count only fresh creations.
pub(crate) async fn persist_abstract_node(
    deps: &EngineDeps,
    node: AbstractNode,
    session_id: Option<SessionId>,
) -> Result<Option<AbstractNode>> {
    if let Some(operation_key) = &node.operation_key {
        if deps
            .repository
            .get_abstract_by_operation_key(operation_key)
            .await?
            .is_some()
        {
            // Already persisted under this operation_key. Return `None` (not
            // `Some(existing)`) so callers can distinguish a NEW abstract from a
            // dedup hit — otherwise "created" is always true, miscounting
            // maintenance stats and the engine's new_abstract_ids. [C6]
            return Ok(None);
        }
    }
    let embedding = deps
        .embedder
        .embed_text(&format!("{} {}", node.title, node.summary))
        .await?;
    deps.repository.insert_abstract(node.clone()).await?;
    // Persist the producing session id alongside the abstract embedding so
    // distilled / summary nodes also respect the per-session retrieval guard.
    deps.vector_index
        .index_abstract_with_session(node.id, embedding, session_id)
        .await?;
    deps.graph_repository.index_abstract(&node).await?;
    Ok(Some(node))
}

fn push_raw_node_into_state(state: &mut ExecutionState, node: RawNode) {
    if !state.persisted_raw_ids.contains(&node.id) {
        state.persisted_raw_ids.push(node.id);
    }
    if matches!(node.kind, RawNodeKind::ToolResult) && !state.tool_result_ids.contains(&node.id) {
        state.tool_result_ids.push(node.id);
    }
    if !state.recent_session.iter().any(|entry| entry.id == node.id) {
        state.recent_session.push(node);
        state
            .recent_session
            .sort_by(|left, right| left.timestamp.cmp(&right.timestamp));
    }
}

fn decode_tool_result_from_raw(node: &RawNode) -> Result<ToolCallResult> {
    match &node.content {
        RawContent::Json(value) => serde_json::from_value(value.clone()).map_err(|err| {
            EngineError::Tool(format!("failed to decode tool result from raw node: {err}"))
        }),
        RawContent::Text(_) => Err(EngineError::Tool(
            "tool result raw node must store a JSON payload".to_string(),
        )),
    }
}

pub(crate) fn prepare_tool_call_for_config(
    mut call: ToolCallRequest,
    tools: &ToolsConfig,
) -> Result<ToolCallRequest> {
    let bounds = MemoryToolBounds::from(tools);
    match call.name.as_str() {
        "semantic_search_memory" => {
            ensure_tool_enabled(tools.memory_search, &call.name)?;
            let mut params: MemorySearchParams =
                serde_json::from_value(std::mem::take(&mut call.arguments)).map_err(|err| {
                    EngineError::Tool(format!("invalid semantic search args: {err}"))
                })?;
            params.top_k = bounds.clamp_memory_search_top_k(params.top_k);
            call.arguments = serde_json::to_value(params).map_err(|err| {
                EngineError::Tool(format!("failed to encode semantic search args: {err}"))
            })?;
            Ok(call)
        }
        "graph_search_memory" => {
            ensure_tool_enabled(tools.graph_search, &call.name)?;
            let mut params: GraphSearchParams =
                serde_json::from_value(std::mem::take(&mut call.arguments)).map_err(|err| {
                    EngineError::Tool(format!("invalid graph search args: {err}"))
                })?;
            params.max_depth = bounds.clamp_graph_search_depth(params.max_depth);
            call.arguments = serde_json::to_value(params).map_err(|err| {
                EngineError::Tool(format!("failed to encode graph search args: {err}"))
            })?;
            Ok(call)
        }
        "provenance_lookup" => {
            ensure_tool_enabled(tools.provenance_lookup, &call.name)?;
            Ok(call)
        }
        "timeline_search" => {
            ensure_tool_enabled(tools.timeline_search, &call.name)?;
            let mut params: TimelineSearchParams =
                serde_json::from_value(std::mem::take(&mut call.arguments))
                    .map_err(|err| EngineError::Tool(format!("invalid timeline args: {err}")))?;
            params.limit = bounds.clamp_timeline_search_limit(params.limit);
            call.arguments = serde_json::to_value(params).map_err(|err| {
                EngineError::Tool(format!("failed to encode timeline args: {err}"))
            })?;
            Ok(call)
        }
        _ => Ok(call),
    }
}

fn ensure_tool_enabled(enabled: bool, tool_name: &str) -> Result<()> {
    if enabled {
        Ok(())
    } else {
        Err(EngineError::Tool(format!(
            "tool {tool_name} is disabled by config"
        )))
    }
}

pub(crate) fn user_input_operation_key(loop_id: LoopId) -> String {
    format!("loop:{loop_id}:user_input")
}

pub(crate) fn assistant_output_operation_key(loop_id: LoopId) -> String {
    format!("loop:{loop_id}:assistant_output")
}

pub(crate) fn tool_result_operation_key(
    loop_id: LoopId,
    round: u32,
    index: usize,
    name: &str,
) -> String {
    format!("loop:{loop_id}:tool:{round}:{index}:{name}")
}
