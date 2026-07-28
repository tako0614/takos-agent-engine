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
use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};
use crate::memory::{ActivationQuery, DistillationInput};
use crate::model::{ConversationMessage, ConversationRole, ModelInput, ToolCallRequest};
use crate::storage::RawLifecyclePatch;
use crate::tools::executor::{ToolCallResult, ToolExecutionContext, ToolExecutionKind};
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
pub(crate) const ASSEMBLE_EXTERNAL_CONTEXT_NODE: &str = "assemble_external_context";
pub(crate) const FINALIZE_EXTERNAL_RESPONSE_NODE: &str = "finalize_external_response";

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
        config: &EngineConfig,
        deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        state.recent_session = deps
            .repository
            .recent_session_raw(&state.session_id, config.runtime.max_session_nodes)
            .await?;
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
}

/// Builds the deliberately empty engine-owned context used by wrappers that
/// transport canonical durable history themselves. The model still receives
/// the configured system prompt, product-owned `conversation_history`, current
/// `user_message`, and the exact structured `turn_messages`; none of those are
/// copied through the legacy session/memory string buckets.
pub(crate) struct AssembleExternalContextNode;

#[async_trait]
impl GraphNode for AssembleExternalContextNode {
    fn id(&self) -> &'static str {
        ASSEMBLE_EXTERNAL_CONTEXT_NODE
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        config: &EngineConfig,
        deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        state.conversation_history = trim_conversation_history(
            &state.conversation_history,
            config,
            deps.token_estimator.as_ref(),
            &state.user_message,
            state.plan.as_deref(),
        )?;
        state.assembled_context = Some(crate::engine::context_assembler::AssembledContext {
            system_prompt: config.system_prompt.clone(),
            ..crate::engine::context_assembler::AssembledContext::default()
        });
        Ok(NodeOutcome::Continue)
    }
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
        // No session reload here. `recent_session` is loaded once by
        // `LoadSessionViewNode` and kept current in memory by
        // `push_raw_node_into_state` as the user/tool/assistant nodes are
        // persisted, so the post-tool re-assembly already sees them. Reloading
        // the whole session from disk on every pass was redundant N+ reads. [C4]
        // Current-loop tool exchanges are represented by native structured
        // `turn_messages`. Do not flatten the same result through the legacy
        // raw/session and summary buckets a second time.
        let current_session_view = state
            .recent_session
            .iter()
            .filter(|node| {
                state.turn_messages.is_empty()
                    || node.loop_id != Some(state.loop_id)
                    || node.kind != RawNodeKind::ToolResult
            })
            .cloned()
            .collect::<Vec<_>>();
        let model_tool_results = if state.turn_messages.is_empty() {
            state.tool_results.as_slice()
        } else {
            &[]
        };
        let context = deps.context_assembler().assemble(
            &config.context_budget,
            &config.system_prompt,
            &current_session_view,
            &state.activated_memory,
            model_tool_results,
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
        // Model nodes await the LLM; budget them with the embedder-configured
        // `model_timeout` rather than the much smaller standard-node timeout.
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
                &state.conversation_history,
                &state.turn_messages,
            ))
            .await?;
        if output.tool_calls.len() > _config.runtime.max_tool_calls_per_round {
            return Err(EngineError::Tool(format!(
                "model returned {} tool calls, exceeding max_tool_calls_per_round={}",
                output.tool_calls.len(),
                _config.runtime.max_tool_calls_per_round
            )));
        }
        for call in &output.tool_calls {
            validate_tool_call_payload(call, &_config.tools)?;
        }
        state.model_invocations = state.model_invocations.saturating_add(1);
        state.pending_tool_calls.clone_from(&output.tool_calls);
        state.latest_model_output = Some(output);

        if !state.pending_tool_calls.is_empty()
            && state.tool_rounds_completed < options.max_tool_rounds
        {
            return Ok(NodeOutcome::Branch("needs_tools".to_string()));
        }

        if !state.pending_tool_calls.is_empty() {
            state.assistant_message = Some(format!(
                "Tool round limit reached after {} rounds without a final assistant message.",
                options.max_tool_rounds
            ));
        } else {
            state.assistant_message = state
                .latest_model_output
                .as_ref()
                .and_then(|output| output.assistant_message.clone());
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
        // Keep the checkpoint replay input intact until every result has been
        // durably handled. A timeout/cancellation after dispatch must retain
        // the original calls so recovery derives the identical operation keys.
        let calls = state.pending_tool_calls.clone();
        let round = state.tool_rounds_completed.saturating_add(1);

        // The model boundary normally rejects this before the checkpoint is
        // written. Re-check recovered/legacy checkpoints and fail closed rather
        // than silently dropping calls that may be semantically required.
        let max_calls = config.runtime.max_tool_calls_per_round.max(1);
        if calls.len() > max_calls {
            return Err(EngineError::Tool(format!(
                "checkpoint contains {} tool calls, exceeding max_tool_calls_per_round={max_calls}",
                calls.len()
            )));
        }

        // Prepare every pending call up front (validation, arg clamping,
        // operation_key derivation). This stays sequential because it is
        // cheap, deterministic, and lets us bail early on a configuration
        // error before spawning any tasks. We also probe the dedup cache
        // here so already-persisted tool results skip the tool invocation
        // entirely and parallel execution only fans out the real work.
        let mut prepared: Vec<PreparedToolCall> = Vec::with_capacity(calls.len());
        let mut call_ids = state
            .turn_messages
            .iter()
            .flat_map(|message| message.tool_calls.iter())
            .filter_map(|call| call.id.clone())
            .collect::<std::collections::HashSet<_>>();
        for (index, mut call) in calls.into_iter().enumerate() {
            if options.is_cancelled() {
                return Err(EngineError::Cancelled);
            }
            if call.id.is_none() {
                call.id = Some(format!("call-{}-{round}-{index}", state.loop_id));
            }
            // External-context wrappers own the remote tool catalog and its
            // schemas. Never reinterpret an external MCP tool merely because
            // its name collides with an engine-local memory primitive.
            let call = if options.execution_profile.uses_local_memory() {
                prepare_tool_call_for_config(call, &config.tools, state.session_id)?
            } else {
                call
            };
            let call_id = call.id.as_deref().ok_or_else(|| {
                EngineError::Tool(format!("tool call {} has no correlation id", call.name))
            })?;
            if call_id.trim().is_empty() {
                return Err(EngineError::Tool(format!(
                    "tool call {} has an empty correlation id",
                    call.name
                )));
            }
            if !call_ids.insert(call_id.to_string()) {
                return Err(EngineError::Tool(format!(
                    "tool call {} reused correlation id {call_id}",
                    call.name
                )));
            }
            if call.name.trim().is_empty() {
                return Err(EngineError::Tool(
                    "tool call name must not be empty".to_string(),
                ));
            }
            let operation_key = tool_result_operation_key(state.loop_id, round, index, &call.name);
            let cached = if options.execution_profile.uses_local_memory() {
                deps.repository
                    .get_raw_by_operation_key(&operation_key)
                    .await?
            } else {
                None
            };
            prepared.push(PreparedToolCall {
                index,
                execution_kind: deps.tool_executor.execution_kind(&call),
                call,
                operation_key,
                cached,
            });
        }

        // Reserve a coherent native transcript before invoking any tool. If
        // the configured context cannot hold the call/result envelopes, a
        // side-effecting tool must not run and leave an unreportable result.
        let estimator = deps.token_estimator.as_ref();
        let round_calls = prepared
            .iter()
            .map(|prep| prep.call.clone())
            .collect::<Vec<_>>();
        let used_turn_tokens = state
            .turn_messages
            .iter()
            .map(|message| estimate_conversation_message(estimator, message))
            .sum::<usize>();
        let available_round_tokens = config
            .context_budget
            .reserve_tools
            .checked_sub(used_turn_tokens)
            .ok_or_else(|| {
                EngineError::Configuration(
                    "current tool transcript already exceeds context_budget.reserve_tools"
                        .to_string(),
                )
            })?;
        let minimum_result_tokens = round_calls
            .iter()
            .map(|call| minimum_tool_result_message_tokens(call, estimator))
            .collect::<Result<Vec<_>>>()?;
        let minimum_results_total = minimum_result_tokens.iter().sum::<usize>();
        let minimum_assistant_turn = ConversationMessage {
            role: ConversationRole::Assistant,
            content: String::new(),
            tool_call_id: None,
            tool_calls: round_calls
                .iter()
                .cloned()
                .map(|mut call| {
                    call.arguments = serde_json::json!({});
                    call
                })
                .collect(),
        };
        let minimum_assistant_tokens =
            estimate_conversation_message(estimator, &minimum_assistant_turn);
        let minimum_round_tokens = minimum_assistant_tokens.saturating_add(minimum_results_total);
        if minimum_round_tokens > available_round_tokens {
            return Err(EngineError::Configuration(format!(
                "tool transcript needs at least {minimum_round_tokens} tokens but only \
                 {available_round_tokens} remain in context_budget.reserve_tools"
            )));
        }

        let variable_round_tokens = available_round_tokens - minimum_round_tokens;
        let assistant_variable_tokens = variable_round_tokens / 3;
        let assistant_call_budget =
            minimum_assistant_tokens.saturating_add(assistant_variable_tokens / 2);
        let model_round_calls =
            bound_tool_calls_for_model(&round_calls, estimator, assistant_call_budget)?;
        let mut assistant_turn = ConversationMessage {
            role: ConversationRole::Assistant,
            content: String::new(),
            tool_call_id: None,
            tool_calls: model_round_calls,
        };
        let assistant_content_budget = minimum_assistant_tokens
            .saturating_add(assistant_variable_tokens)
            .saturating_sub(estimate_conversation_message(estimator, &assistant_turn));
        assistant_turn.content = truncate_text_to_token_budget(
            state
                .latest_model_output
                .as_ref()
                .and_then(|output| output.assistant_message.as_deref())
                .unwrap_or_default(),
            estimator,
            assistant_content_budget,
        );
        let assistant_turn_tokens = estimate_conversation_message(estimator, &assistant_turn);
        let available_result_tokens = available_round_tokens.saturating_sub(assistant_turn_tokens);
        if available_result_tokens < minimum_results_total {
            return Err(EngineError::Configuration(
                "bounded assistant tool call left insufficient room for tool results".to_string(),
            ));
        }

        // Adjacent read-only calls may overlap. Side-effecting/destructive
        // calls are barriers and run one at a time in the provider's order, so
        // writes cannot race each other or an adjacent read.
        let mut phases: Vec<Vec<&PreparedToolCall>> = Vec::new();
        let mut read_only_phase = Vec::new();
        for prep in &prepared {
            if prep.cached.is_some() {
                continue;
            }
            if prep.execution_kind == ToolExecutionKind::ReadOnly {
                read_only_phase.push(prep);
            } else {
                if !read_only_phase.is_empty() {
                    phases.push(std::mem::take(&mut read_only_phase));
                }
                phases.push(vec![prep]);
            }
        }
        if !read_only_phase.is_empty() {
            phases.push(read_only_phase);
        }

        // Collect results into a sparse map keyed by the prepared index so we
        // can recombine deterministically with cached entries below.
        let mut completed: std::collections::HashMap<usize, Result<ToolCallResult>> =
            std::collections::HashMap::new();
        let mut terminal_error = None;
        for phase in phases {
            let phase_results = execute_tool_phase(
                &phase,
                deps,
                options,
                state.session_id,
                state.loop_id,
                config.tools.max_tool_result_bytes,
            )
            .await?;
            terminal_error = phase_results.values().find_map(|result| match result {
                Err(EngineError::Cancelled) => Some(EngineError::Cancelled),
                Err(EngineError::ToolOutcomeIndeterminate {
                    idempotency_key,
                    reason,
                }) => Some(EngineError::ToolOutcomeIndeterminate {
                    idempotency_key: idempotency_key.clone(),
                    reason: reason.clone(),
                }),
                _ => None,
            });
            completed.extend(phase_results);
            if terminal_error.is_some() {
                break;
            }
        }

        // Reassemble results in the original tool-call order so downstream
        // model rounds see them deterministically. Persistence still happens
        // sequentially because the raw-node insert path is not designed for
        // concurrent writers (operation_key uniqueness, sync_raw_indexes).
        let mut round_results = Vec::with_capacity(prepared.len());
        let mut result_extra_tokens = available_result_tokens - minimum_results_total;
        let prepared_count = prepared.len();
        for (position, prep) in prepared.into_iter().enumerate() {
            let (mut full_tool_result, existing_raw) = if let Some(existing) = prep.cached {
                let tool_result = decode_tool_result_from_raw(&existing)?;
                (tool_result, Some(existing))
            } else {
                let Some(result_value) = completed.remove(&prep.index) else {
                    if let Some(error) = terminal_error.take() {
                        return Err(error);
                    }
                    return Err(EngineError::Tool(format!(
                        "missing tool result for index {} (tool {})",
                        prep.index, prep.call.name
                    )));
                };
                let result = match result_value {
                    Ok(result) => result,
                    Err(EngineError::Cancelled) => return Err(EngineError::Cancelled),
                    Err(error @ EngineError::ToolOutcomeIndeterminate { .. }) => {
                        return Err(error);
                    }
                    Err(error) => ToolCallResult {
                        tool_call_id: prep.call.id.clone(),
                        name: prep.call.name.clone(),
                        content: serde_json::json!({ "error": error.to_string() }),
                        summary: format!("{} error={error}", prep.call.name),
                    },
                };
                (result, None)
            };
            // The request id/name are the engine's correlation authority. A
            // third-party executor or a legacy cached payload must not inject
            // a missing or mismatched id into the provider transcript.
            full_tool_result.tool_call_id.clone_from(&prep.call.id);
            full_tool_result.name.clone_from(&prep.call.name);
            bound_tool_result_for_storage(
                &mut full_tool_result,
                config.tools.max_tool_result_bytes,
            )?;
            let remaining_results = prepared_count.saturating_sub(position).max(1);
            let extra_allowance = result_extra_tokens / remaining_results;
            let result_metadata_tokens =
                minimum_result_tokens[position].saturating_sub(estimator.estimate_text("{}"));
            let mut result_allowance = estimator
                .estimate_text("{}")
                .saturating_add(extra_allowance);
            let bounded_tool_result = clamp_tool_result_for_model(
                full_tool_result.clone(),
                estimator,
                &mut result_allowance,
            );
            let bounded_content_tokens =
                estimator.estimate_text(&bounded_tool_result.content.to_string());
            let used_extra = result_metadata_tokens
                .saturating_add(bounded_content_tokens)
                .saturating_sub(minimum_result_tokens[position]);
            result_extra_tokens = result_extra_tokens.saturating_sub(used_extra);
            if let Some(existing) = existing_raw {
                // A pre-upgrade checkpoint may contain a larger cached raw
                // result. Keep the persisted evidence intact but only expose
                // the bounded representation to the model/transcript.
                let mut state_raw = existing;
                state_raw.content = RawContent::Json(
                    serde_json::to_value(&bounded_tool_result).map_err(|err| {
                        EngineError::Tool(format!(
                            "failed to encode bounded cached tool result: {err}"
                        ))
                    })?,
                );
                push_raw_node_into_state(state, state_raw);
            } else if options.execution_profile.uses_local_memory() {
                let raw = RawNode::json(
                    RawNodeKind::ToolResult,
                    Some(state.session_id),
                    Some(state.loop_id),
                    format!("tool:{}", full_tool_result.name),
                    serde_json::to_value(&full_tool_result).map_err(|err| {
                        EngineError::Tool(format!("failed to encode tool result payload: {err}"))
                    })?,
                    TOOL_RESULT_IMPORTANCE,
                    vec!["tool".to_string()],
                )
                .with_operation_key(prep.operation_key.clone());
                let raw = persist_raw_node(deps, raw).await?;
                // The repository retains the complete result for retrieval and
                // distillation. Only the in-flight model view is bounded.
                let mut state_raw = raw;
                state_raw.content = RawContent::Json(
                    serde_json::to_value(&bounded_tool_result).map_err(|err| {
                        EngineError::Tool(format!("failed to encode bounded tool result: {err}"))
                    })?,
                );
                push_raw_node_into_state(state, state_raw);
            }
            round_results.push(bounded_tool_result.clone());
            state.tool_results.push(bounded_tool_result);
        }

        state.turn_messages.push(assistant_turn);
        state
            .turn_messages
            .extend(round_results.into_iter().map(|result| ConversationMessage {
                role: ConversationRole::Tool,
                content: result.content.to_string(),
                tool_call_id: result.tool_call_id,
                tool_calls: Vec::new(),
            }));

        let transcript_tokens = state
            .turn_messages
            .iter()
            .map(|message| estimate_conversation_message(estimator, message))
            .sum::<usize>();
        if transcript_tokens > config.context_budget.reserve_tools {
            return Err(EngineError::Configuration(format!(
                "bounded tool transcript used {transcript_tokens} tokens, exceeding reserve_tools={} ",
                config.context_budget.reserve_tools
            )));
        }

        state.tool_rounds_completed = round;
        state.pending_tool_calls.clear();
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
    execution_kind: ToolExecutionKind,
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
    Cancelled { tool_name: String },
}

async fn execute_tool_phase(
    phase: &[&PreparedToolCall],
    deps: &EngineDeps,
    options: &ResolvedRunOptions,
    session_id: SessionId,
    loop_id: LoopId,
    max_result_bytes: usize,
) -> Result<std::collections::HashMap<usize, Result<ToolCallResult>>> {
    // A JoinSet is scoped to one read-only phase (or one side-effecting call).
    // Dropping the node future aborts every still-running task in this phase.
    let mut join_set: tokio::task::JoinSet<(usize, TimedToolOutcome)> = tokio::task::JoinSet::new();
    for prep in phase {
        let index = prep.index;
        let executor = deps.tool_executor.clone();
        let cancellation = options.cancellation_token.clone();
        let timeout = options.tool_timeout;
        let call = prep.call.clone();
        let tool_name = call.name.clone();
        let context = ToolExecutionContext {
            session_id,
            loop_id,
            idempotency_key: prep.operation_key.clone(),
            timeout,
            max_result_bytes,
            cancellation_token: cancellation.clone(),
        };
        join_set.spawn(async move {
            let exec_future = executor.execute_with_context(context, call);
            let outcome = if let Some(token) = cancellation {
                tokio::select! {
                    biased;
                    () = token.cancelled() => TimedToolOutcome::Cancelled { tool_name },
                    result = tokio::time::timeout(timeout, exec_future) => match result {
                        Ok(result) => TimedToolOutcome::Completed(result),
                        Err(_) => TimedToolOutcome::TimedOut { tool_name },
                    },
                }
            } else {
                match tokio::time::timeout(timeout, exec_future).await {
                    Ok(result) => TimedToolOutcome::Completed(result),
                    Err(_) => TimedToolOutcome::TimedOut { tool_name },
                }
            };
            (index, outcome)
        });
    }

    let mut completed = std::collections::HashMap::new();
    while let Some(joined) = join_set.join_next().await {
        let (index, outcome) = joined
            .map_err(|join_err| EngineError::Tool(format!("tool task join failed: {join_err}")))?;
        let result = match outcome {
            TimedToolOutcome::Completed(result) => result,
            TimedToolOutcome::TimedOut { tool_name } => {
                let prep = phase
                    .iter()
                    .find(|prep| prep.index == index)
                    .expect("joined tool index must belong to the phase");
                if prep.execution_kind == ToolExecutionKind::SideEffecting {
                    Err(EngineError::ToolOutcomeIndeterminate {
                        idempotency_key: prep.operation_key.clone(),
                        reason: format!(
                            "tool {tool_name} exceeded the configured timeout of {:?}",
                            options.tool_timeout
                        ),
                    })
                } else {
                    Err(EngineError::Tool(format!(
                        "tool {tool_name} exceeded the configured tool_timeout of {:?}",
                        options.tool_timeout
                    )))
                }
            }
            TimedToolOutcome::Cancelled { tool_name } => {
                let prep = phase
                    .iter()
                    .find(|prep| prep.index == index)
                    .expect("joined tool index must belong to the phase");
                if prep.execution_kind == ToolExecutionKind::SideEffecting {
                    Err(EngineError::ToolOutcomeIndeterminate {
                        idempotency_key: prep.operation_key.clone(),
                        reason: format!("tool {tool_name} was cancelled after dispatch"),
                    })
                } else {
                    Err(EngineError::Cancelled)
                }
            }
        };
        completed.insert(index, result);
    }
    Ok(completed)
}

pub(crate) struct PersistAssistantOutputNode;

/// Terminates an external-context run without persisting an engine-local copy
/// of the assistant response. The product wrapper commits `assistant_message`
/// and `turn_messages` atomically to its durable Thread authority.
pub(crate) struct FinalizeExternalResponseNode;

#[async_trait]
impl GraphNode for FinalizeExternalResponseNode {
    fn id(&self) -> &'static str {
        FINALIZE_EXTERNAL_RESPONSE_NODE
    }

    async fn run(
        &self,
        state: &mut ExecutionState,
        _config: &EngineConfig,
        _deps: &EngineDeps,
        _options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        if state.assistant_message.is_none() {
            state.assistant_message = state
                .latest_model_output
                .as_ref()
                .and_then(|output| output.assistant_message.clone())
                .or_else(|| Some("No assistant message generated.".to_string()));
        }
        Ok(NodeOutcome::Finish)
    }
}

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
        // Clear the overflow mark on nodes that are back inside the window —
        // but only those that actually carry a mark, so we don't rewrite
        // never-overflowed nodes every turn.
        let session_window_ids = state
            .session_window_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let pushed_out_raw_ids = state
            .pushed_out_raw_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let clear_ids: Vec<_> = state
            .recent_session
            .iter()
            .filter(|node| {
                node.distillation_state != DistillationState::Distilled
                    && node.overflow.was_pushed_out_of_session
                    && session_window_ids.contains(&node.id)
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

        // Stamp the relaxation deadline ONCE, only on newly pushed-out nodes
        // (those without an existing deadline). Re-stamping every turn would
        // keep extending the window so it never elapses, defeating the 24h
        // expiry enforced in activation/scoring. [C7]
        let pushed_ids: Vec<_> = state
            .recent_session
            .iter()
            .filter(|node| {
                node.distillation_state != DistillationState::Distilled
                    && pushed_out_raw_ids.contains(&node.id)
                    && node.overflow.relax_retrieval_until.is_none()
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

        // No session reload: nothing after this node reads `recent_session`
        // (DistillCurrentLoopNode reads `raw_for_loop`, build_response does not
        // touch it), so the end-of-turn full reload was pure overhead. [C4]
        Ok(NodeOutcome::Continue)
    }
}

pub(crate) struct DistillCurrentLoopNode;

#[derive(Debug, Default)]
pub(crate) struct DistillationBatchResult {
    pub(crate) new_abstract_ids: Vec<AbstractNodeId>,
    pub(crate) updated_raw_nodes: usize,
}

pub(crate) async fn distill_claimed_raw_nodes(
    config: &EngineConfig,
    deps: &EngineDeps,
    options: &ResolvedRunOptions,
    session_id: SessionId,
    loop_id: LoopId,
    raw_nodes: Vec<RawNode>,
    activated_abstract_ids: Vec<AbstractNodeId>,
) -> Result<Option<DistillationBatchResult>> {
    if raw_nodes.is_empty() {
        return Ok(Some(DistillationBatchResult::default()));
    }
    if raw_nodes.len() > config.runtime.max_loop_nodes {
        return Err(EngineError::Configuration(format!(
            "distillation batch contains {} raw nodes, exceeding max_loop_nodes={}",
            raw_nodes.len(),
            config.runtime.max_loop_nodes
        )));
    }
    if raw_nodes
        .iter()
        .any(|node| node.session_id != Some(session_id) || node.loop_id != Some(loop_id))
    {
        return Err(EngineError::Storage(
            "distillation batch contains a cross-session or cross-loop raw node".to_string(),
        ));
    }

    let mut input_ids = raw_nodes.iter().map(|node| node.id).collect::<Vec<_>>();
    input_ids.sort();
    input_ids.dedup();
    if input_ids.len() != raw_nodes.len() {
        return Err(EngineError::Storage(
            "distillation batch contains duplicate raw node ids".to_string(),
        ));
    }
    let digest_source = input_ids
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(":");
    let input_digest =
        uuid::Uuid::new_v5(&uuid::Uuid::NAMESPACE_OID, digest_source.as_bytes()).to_string();
    let lease_for = options
        .distillation_timeout
        .checked_mul(2)
        .and_then(|duration| duration.checked_add(std::time::Duration::from_secs(5)))
        .ok_or_else(|| {
            EngineError::Configuration("distillation timeout is too large".to_string())
        })?;
    let Some(claim) = deps
        .repository
        .try_claim_distillation(session_id, loop_id, input_digest.clone(), lease_for)
        .await?
    else {
        return Ok(None);
    };

    let work = async {
        let distilled = tokio::time::timeout(
            options.distillation_timeout,
            deps.distiller.distill(DistillationInput {
                session_id,
                loop_id,
                raw_nodes,
                activated_abstract_ids,
            }),
        )
        .await
        .map_err(|_| {
            EngineError::Tool(format!(
                "distillation exceeded the configured timeout of {:?}",
                options.distillation_timeout
            ))
        })??;

        let input_set = input_ids
            .iter()
            .copied()
            .collect::<std::collections::HashSet<_>>();
        let mut updated_ids = std::collections::HashSet::<RawNodeId>::new();
        for update in &distilled.raw_updates {
            if !input_set.contains(&update.raw_node_id) || !updated_ids.insert(update.raw_node_id) {
                return Err(EngineError::Storage(
                    "distiller returned an out-of-batch or duplicate raw update".to_string(),
                ));
            }
        }
        if !deps
            .repository
            .distillation_claim_is_current(&claim)
            .await?
        {
            return Err(EngineError::RecoveryUnsafe(
                "distillation claim expired before commit".to_string(),
            ));
        }

        let mut result = DistillationBatchResult::default();
        for (index, mut node) in distilled.new_nodes.into_iter().enumerate() {
            if !deps
                .repository
                .distillation_claim_is_current(&claim)
                .await?
            {
                return Err(EngineError::RecoveryUnsafe(
                    "distillation claim was replaced during commit".to_string(),
                ));
            }
            node.operation_key = Some(format!(
                "distill:{session_id}:{loop_id}:{input_digest}:{index}"
            ));
            if let Some(node) = persist_abstract_node(deps, node, Some(session_id)).await? {
                result.new_abstract_ids.push(node.id);
            }
        }
        for update in distilled.raw_updates {
            if !deps
                .repository
                .distillation_claim_is_current(&claim)
                .await?
            {
                return Err(EngineError::RecoveryUnsafe(
                    "distillation claim was replaced during lifecycle commit".to_string(),
                ));
            }
            deps.repository
                .update_raw_lifecycle(&[update.raw_node_id], &update.patch)
                .await?;
            result.updated_raw_nodes += 1;
        }
        Ok(result)
    }
    .await;

    let release_result = deps.repository.release_distillation_claim(&claim).await;
    match (work, release_result) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(result), Ok(())) => Ok(Some(result)),
    }
}

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
        config: &EngineConfig,
        deps: &EngineDeps,
        options: &ResolvedRunOptions,
    ) -> Result<NodeOutcome> {
        let raw_nodes = deps
            .repository
            .raw_for_loop_bounded(
                &state.loop_id,
                config.runtime.max_loop_nodes.saturating_add(1),
            )
            .await?;
        if raw_nodes.len() > config.runtime.max_loop_nodes {
            return Err(EngineError::Configuration(format!(
                "loop {} exceeds max_loop_nodes={}; split the turn before distillation",
                state.loop_id, config.runtime.max_loop_nodes
            )));
        }
        if let Some(result) = distill_claimed_raw_nodes(
            config,
            deps,
            options,
            state.session_id,
            state.loop_id,
            raw_nodes,
            state
                .activated_memory
                .abstract_nodes
                .iter()
                .map(|entry| entry.node.id)
                .collect(),
        )
        .await?
        {
            state.new_abstract_ids.extend(result.new_abstract_ids);
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
        turn_messages: state.turn_messages,
        activated_raw_count: state.activated_memory.raw_nodes.len(),
        activated_abstract_count: state.activated_memory.abstract_nodes.len(),
        tool_results_count: state.tool_results.len(),
        completed_steps: result.completed_steps,
        tool_rounds_completed: result.tool_rounds_completed,
    }
}

pub(crate) fn trim_conversation_history(
    history: &[ConversationMessage],
    config: &EngineConfig,
    estimator: &dyn crate::engine::context_assembler::TokenEstimator,
    user_message: &str,
    plan: Option<&str>,
) -> Result<Vec<ConversationMessage>> {
    let system_tokens = estimator.estimate_text(&config.system_prompt);
    let working_tokens = estimator
        .estimate_text(user_message)
        .saturating_add(plan.map_or(0, |value| estimator.estimate_text(value)));
    let fixed_session_tokens = working_tokens
        .saturating_sub(config.context_budget.reserve_working)
        .saturating_add(system_tokens.saturating_sub(config.context_budget.reserve_system));
    let available = config.context_budget.remaining_tokens();
    if fixed_session_tokens > available {
        return Err(EngineError::Configuration(
            "external context fixed prompt exceeds the configured context budget".to_string(),
        ));
    }

    let mut remaining = available.saturating_sub(fixed_session_tokens);
    let groups = coherent_conversation_groups(history);
    let mut selected = Vec::new();
    for group in groups.into_iter().rev() {
        let tokens = group
            .iter()
            .map(|message| estimate_conversation_message(estimator, message))
            .sum::<usize>();
        if tokens > remaining {
            break;
        }
        remaining = remaining.saturating_sub(tokens);
        selected.push(group);
    }
    selected.reverse();
    Ok(selected.into_iter().flatten().collect())
}

fn validate_tool_call_payload(call: &ToolCallRequest, tools: &ToolsConfig) -> Result<()> {
    const MAX_TOOL_IDENTIFIER_BYTES: usize = 256;
    if call.name.len() > MAX_TOOL_IDENTIFIER_BYTES
        || call
            .id
            .as_ref()
            .is_some_and(|value| value.len() > MAX_TOOL_IDENTIFIER_BYTES)
    {
        return Err(EngineError::Tool(
            "tool call id/name exceeds the 256-byte boundary".to_string(),
        ));
    }
    let argument_bytes = serde_json::to_vec(&call.arguments)
        .map_err(|err| EngineError::Tool(format!("failed to encode tool arguments: {err}")))?
        .len();
    if argument_bytes > tools.max_tool_argument_bytes {
        return Err(EngineError::Tool(format!(
            "tool {} arguments use {argument_bytes} bytes, exceeding max_tool_argument_bytes={}",
            call.name, tools.max_tool_argument_bytes
        )));
    }
    Ok(())
}

fn coherent_conversation_groups(history: &[ConversationMessage]) -> Vec<Vec<ConversationMessage>> {
    let mut groups = Vec::new();
    let mut index = 0;
    while index < history.len() {
        let message = &history[index];
        if message.role == ConversationRole::Tool {
            index += 1;
            continue;
        }
        if message.role != ConversationRole::Assistant || message.tool_calls.is_empty() {
            groups.push(vec![message.clone()]);
            index += 1;
            continue;
        }

        let expected = message
            .tool_calls
            .iter()
            .filter_map(|call| call.id.as_deref())
            .filter(|id| !id.trim().is_empty())
            .collect::<std::collections::HashSet<_>>();
        let mut results = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut next = index + 1;
        while next < history.len() && history[next].role == ConversationRole::Tool {
            if let Some(id) = history[next].tool_call_id.as_deref() {
                if expected.contains(id) && seen.insert(id) {
                    results.push(history[next].clone());
                }
            }
            next += 1;
        }
        let complete = expected.len() == message.tool_calls.len() && seen.len() == expected.len();
        if complete {
            let mut group = Vec::with_capacity(1 + results.len());
            group.push(message.clone());
            group.extend(results);
            groups.push(group);
        } else if !message.content.trim().is_empty() {
            let mut plain = message.clone();
            plain.tool_calls.clear();
            groups.push(vec![plain]);
        }
        index = next;
    }
    groups
}

fn estimate_conversation_message(
    estimator: &dyn crate::engine::context_assembler::TokenEstimator,
    message: &ConversationMessage,
) -> usize {
    let mut tokens = estimator.estimate_text(&message.content);
    if let Some(tool_call_id) = &message.tool_call_id {
        tokens = tokens.saturating_add(estimator.estimate_text(tool_call_id));
    }
    for call in &message.tool_calls {
        if let Some(id) = &call.id {
            tokens = tokens.saturating_add(estimator.estimate_text(id));
        }
        tokens = tokens
            .saturating_add(estimator.estimate_text(&call.name))
            .saturating_add(estimator.estimate_text(&call.arguments.to_string()));
    }
    tokens.saturating_add(1)
}

fn bound_tool_calls_for_model(
    calls: &[ToolCallRequest],
    estimator: &dyn crate::engine::context_assembler::TokenEstimator,
    budget: usize,
) -> Result<Vec<ToolCallRequest>> {
    let empty_arguments = serde_json::json!({});
    let empty_argument_tokens = estimator.estimate_text(&empty_arguments.to_string());
    let minimum_calls = calls
        .iter()
        .cloned()
        .map(|mut call| {
            call.arguments = empty_arguments.clone();
            call
        })
        .collect::<Vec<_>>();
    let minimum_message = ConversationMessage {
        role: ConversationRole::Assistant,
        content: String::new(),
        tool_call_id: None,
        tool_calls: minimum_calls,
    };
    let minimum_tokens = estimate_conversation_message(estimator, &minimum_message);
    if minimum_tokens > budget {
        return Err(EngineError::Configuration(format!(
            "tool-call transcript needs at least {minimum_tokens} tokens but only {budget} were allocated"
        )));
    }

    let mut extra_tokens = budget - minimum_tokens;
    let mut bounded = Vec::with_capacity(calls.len());
    for (index, mut call) in calls.iter().cloned().enumerate() {
        let remaining_calls = calls.len().saturating_sub(index).max(1);
        let extra_allowance = extra_tokens / remaining_calls;
        let original_tokens = estimator.estimate_text(&call.arguments.to_string());
        if original_tokens.saturating_sub(empty_argument_tokens) > extra_allowance {
            let marker = serde_json::json!({ "truncated": true });
            let marker_tokens = estimator.estimate_text(&marker.to_string());
            call.arguments =
                if marker_tokens.saturating_sub(empty_argument_tokens) <= extra_allowance {
                    marker
                } else {
                    empty_arguments.clone()
                };
        }
        let used_extra = estimator
            .estimate_text(&call.arguments.to_string())
            .saturating_sub(empty_argument_tokens);
        extra_tokens = extra_tokens.saturating_sub(used_extra);
        bounded.push(call);
    }
    Ok(bounded)
}

fn minimum_tool_result_message_tokens(
    call: &ToolCallRequest,
    estimator: &dyn crate::engine::context_assembler::TokenEstimator,
) -> Result<usize> {
    let call_id = call.id.clone().ok_or_else(|| {
        EngineError::Tool(format!("tool call {} has no correlation id", call.name))
    })?;
    Ok(estimate_conversation_message(
        estimator,
        &ConversationMessage {
            role: ConversationRole::Tool,
            content: "{}".to_string(),
            tool_call_id: Some(call_id),
            tool_calls: Vec::new(),
        },
    ))
}

fn truncate_text_to_token_budget(
    text: &str,
    estimator: &dyn crate::engine::context_assembler::TokenEstimator,
    budget: usize,
) -> String {
    if text.is_empty() || budget == 0 {
        return String::new();
    }
    if estimator.estimate_text(text) <= budget {
        return text.to_string();
    }
    let boundaries = text
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(text.len()))
        .collect::<Vec<_>>();
    let mut low = 0;
    let mut high = boundaries.len().saturating_sub(1);
    let mut best = String::new();
    while low <= high {
        let middle = low + (high - low) / 2;
        let candidate = &text[..boundaries[middle]];
        if estimator.estimate_text(candidate) <= budget {
            best = candidate.to_string();
            low = middle.saturating_add(1);
        } else if middle == 0 {
            break;
        } else {
            high = middle - 1;
        }
    }
    best
}

fn to_model_input(
    session_id: SessionId,
    loop_id: LoopId,
    user_message: &str,
    plan: Option<String>,
    context: &crate::engine::context_assembler::AssembledContext,
    conversation_history: &[ConversationMessage],
    turn_messages: &[ConversationMessage],
) -> ModelInput {
    ModelInput {
        session_id,
        loop_id,
        system_prompt: context.system_prompt.clone(),
        session_context: context.session_context.clone(),
        memory_context: context.memory_context.clone(),
        tool_context: context.tool_context.clone(),
        conversation_history: conversation_history.to_vec(),
        turn_messages: turn_messages.to_vec(),
        user_message: user_message.to_string(),
        plan,
    }
}

async fn persist_raw_node(deps: &EngineDeps, node: RawNode) -> Result<RawNode> {
    let embedding = deps.embedder.embed_text(&node.content_text()).await?;
    let commit = deps.repository.commit_raw(node, embedding.clone()).await?;
    if !commit.projections_committed {
        let embedding = if commit.inserted {
            embedding
        } else {
            deps.embedder
                .embed_text(&commit.node.content_text())
                .await?
        };
        deps.vector_index
            .index_raw_with_session(commit.node.id, embedding, commit.node.session_id)
            .await?;
    }
    Ok(commit.node)
}

/// Persist a distilled abstract node, deduplicating on its `operation_key`.
///
/// Returns `Ok(Some(node))` when a NEW abstract was inserted, and `Ok(None)`
/// when an abstract already existed for the `operation_key` (a dedup hit, no
/// write). Callers use this distinction to count only fresh creations.
pub(crate) async fn persist_abstract_node(
    deps: &EngineDeps,
    mut node: AbstractNode,
    session_id: Option<SessionId>,
) -> Result<Option<AbstractNode>> {
    node.session_id = session_id;
    if let Some(session_id) = session_id {
        let mut raw_reference_ids = node.references.raw_node_ids.clone();
        raw_reference_ids.extend(
            node.graph
                .relations
                .iter()
                .flat_map(|relation| relation.provenance_raw_node_ids.iter().copied()),
        );
        raw_reference_ids.sort();
        raw_reference_ids.dedup();
        let raw_references = deps.repository.list_raw(&raw_reference_ids).await?;
        if raw_references.len() != raw_reference_ids.len()
            || raw_references
                .iter()
                .any(|reference| reference.session_id != Some(session_id))
        {
            return Err(EngineError::Storage(
                "abstract node contains missing or cross-session raw provenance".to_string(),
            ));
        }

        let mut abstract_reference_ids = node.references.abstract_node_ids.clone();
        abstract_reference_ids.sort();
        abstract_reference_ids.dedup();
        let abstract_references = deps
            .repository
            .list_abstract(&abstract_reference_ids)
            .await?;
        if abstract_references.len() != abstract_reference_ids.len()
            || abstract_references
                .iter()
                .any(|reference| reference.session_id != Some(session_id))
        {
            return Err(EngineError::Storage(
                "abstract node contains missing or cross-session abstract provenance".to_string(),
            ));
        }
    }
    let embedding = deps
        .embedder
        .embed_text(&format!("{} {}", node.title, node.summary))
        .await?;
    let commit = deps
        .repository
        .commit_abstract(node, embedding.clone())
        .await?;
    if !commit.projections_committed {
        let embedding = if commit.inserted {
            embedding
        } else {
            deps.embedder
                .embed_text(&format!("{} {}", commit.node.title, commit.node.summary))
                .await?
        };
        deps.vector_index
            .index_abstract_with_session(commit.node.id, embedding, commit.node.session_id)
            .await?;
        deps.graph_repository.index_abstract(&commit.node).await?;
    }
    Ok(commit.inserted.then_some(commit.node))
}

fn push_raw_node_into_state(state: &mut ExecutionState, node: RawNode) {
    if !state.persisted_raw_ids.contains(&node.id) {
        state.persisted_raw_ids.push(node.id);
    }
    if matches!(node.kind, RawNodeKind::ToolResult) && !state.tool_result_ids.contains(&node.id) {
        state.tool_result_ids.push(node.id);
    }
    if !state.recent_session.iter().any(|entry| entry.id == node.id) {
        // `recent_session` is maintained sorted by timestamp; insert at the
        // correct position instead of re-sorting the whole (growing) vector on
        // every push. Using `<=` keeps equal-timestamp ties in insertion order,
        // matching the previous stable full sort. [C4]
        let position = state
            .recent_session
            .partition_point(|entry| entry.timestamp <= node.timestamp);
        state.recent_session.insert(position, node);
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

fn clamp_tool_result_for_model(
    mut result: ToolCallResult,
    estimator: &dyn crate::engine::context_assembler::TokenEstimator,
    remaining_tokens: &mut usize,
) -> ToolCallResult {
    let serialized = result.content.to_string();
    let full_estimate = estimator.estimate_text(&serialized);
    if full_estimate <= *remaining_tokens {
        *remaining_tokens = remaining_tokens.saturating_sub(full_estimate);
        return result;
    }

    let minimum_payload = serde_json::json!({});
    let minimum_payload_estimate = estimator.estimate_text(&minimum_payload.to_string());
    if minimum_payload_estimate > *remaining_tokens {
        // Callers reserve the minimum protocol envelope before executing the
        // tool. Reaching this branch indicates an internal budgeting bug, but
        // keep the provider transcript syntactically valid rather than
        // emitting an empty non-JSON message.
        result.content = minimum_payload;
        *remaining_tokens = 0;
        return result;
    }

    let empty_marker = serde_json::json!({
        "truncated": true,
        "preview": "",
    });
    let empty_marker_estimate = estimator.estimate_text(&empty_marker.to_string());
    if empty_marker_estimate > *remaining_tokens {
        result.content = minimum_payload;
        *remaining_tokens = remaining_tokens.saturating_sub(minimum_payload_estimate);
        return result;
    }

    let boundaries = serialized
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(serialized.len()))
        .collect::<Vec<_>>();
    let mut low = 0usize;
    let mut high = boundaries.len().saturating_sub(1);
    let mut best = empty_marker;
    let mut best_estimate = empty_marker_estimate;
    while low <= high {
        let middle = low + (high - low) / 2;
        let candidate = serde_json::json!({
            "truncated": true,
            "preview": &serialized[..boundaries[middle]],
        });
        let estimate = estimator.estimate_text(&candidate.to_string());
        if estimate <= *remaining_tokens {
            best = candidate;
            best_estimate = estimate;
            low = middle.saturating_add(1);
        } else if middle == 0 {
            break;
        } else {
            high = middle - 1;
        }
    }
    result.content = best;
    *remaining_tokens = remaining_tokens.saturating_sub(best_estimate);
    result
}

fn bound_tool_result_for_storage(result: &mut ToolCallResult, max_bytes: usize) -> Result<()> {
    fn encoded_len(result: &ToolCallResult) -> Result<usize> {
        serde_json::to_vec(result)
            .map(|payload| payload.len())
            .map_err(|err| EngineError::Tool(format!("failed to encode tool result: {err}")))
    }

    fn truncate_utf8(value: &mut String, max_bytes: usize) {
        if value.len() <= max_bytes {
            return;
        }
        let mut boundary = max_bytes;
        while boundary > 0 && !value.is_char_boundary(boundary) {
            boundary -= 1;
        }
        value.truncate(boundary);
    }

    if encoded_len(result)? <= max_bytes {
        return Ok(());
    }

    // Bound metadata first so the content preview has a deterministic share of
    // the envelope. Correlation id/name are engine-validated before dispatch.
    truncate_utf8(&mut result.summary, (max_bytes / 8).min(4_096));
    let serialized = serde_json::to_string(&result.content)
        .map_err(|err| EngineError::Tool(format!("failed to encode tool result: {err}")))?;

    let empty_marker = serde_json::json!({ "truncated": true, "preview": "" });
    result.content = empty_marker.clone();
    if encoded_len(result)? > max_bytes {
        result.summary.clear();
    }
    if encoded_len(result)? > max_bytes {
        return Err(EngineError::Configuration(format!(
            "max_tool_result_bytes={max_bytes} is too small for the validated tool result envelope"
        )));
    }

    // Binary-search a UTF-8-safe preview without allocating a boundary vector
    // proportional to the (potentially large) result.
    let mut low = 0usize;
    let mut high = serialized.len();
    let mut best = empty_marker;
    while low <= high {
        let middle = low + (high - low) / 2;
        let mut boundary = middle;
        while boundary > 0 && !serialized.is_char_boundary(boundary) {
            boundary -= 1;
        }
        let candidate = serde_json::json!({
            "truncated": true,
            "preview": &serialized[..boundary],
        });
        result.content = candidate.clone();
        if encoded_len(result)? <= max_bytes {
            best = candidate;
            if middle == serialized.len() {
                break;
            }
            low = middle.saturating_add(1);
        } else if middle == 0 {
            break;
        } else {
            high = middle - 1;
        }
    }
    result.content = best;
    debug_assert!(encoded_len(result)? <= max_bytes);
    Ok(())
}

pub(crate) fn prepare_tool_call_for_config(
    mut call: ToolCallRequest,
    tools: &ToolsConfig,
    session_id: SessionId,
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
            // Bind the search to THIS run's session. This both prevents a
            // model-supplied scope from reading another session's memory and
            // makes the search match the engine's session-tagged embeddings
            // (a `None` scope would match only legacy entries -> empty). [S2, C8]
            params.session_id = Some(session_id.to_string());
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
            params.max_hits = bounds.clamp_graph_search_hits(params.max_hits);
            params.session_id = Some(session_id.to_string());
            call.arguments = serde_json::to_value(params).map_err(|err| {
                EngineError::Tool(format!("failed to encode graph search args: {err}"))
            })?;
            Ok(call)
        }
        "provenance_lookup" => {
            ensure_tool_enabled(tools.provenance_lookup, &call.name)?;
            let mut params: crate::tools::memory_tools::ProvenanceLookupParams =
                serde_json::from_value(std::mem::take(&mut call.arguments))
                    .map_err(|err| EngineError::Tool(format!("invalid provenance args: {err}")))?;
            params.session_id = Some(session_id.to_string());
            params.limit = bounds.clamp_provenance_raw_nodes(params.limit);
            call.arguments = serde_json::to_value(params).map_err(|err| {
                EngineError::Tool(format!("failed to encode provenance args: {err}"))
            })?;
            Ok(call)
        }
        "timeline_search" => {
            ensure_tool_enabled(tools.timeline_search, &call.name)?;
            let mut params: TimelineSearchParams =
                serde_json::from_value(std::mem::take(&mut call.arguments))
                    .map_err(|err| EngineError::Tool(format!("invalid timeline args: {err}")))?;
            params.limit = bounds.clamp_timeline_search_limit(params.limit);
            // Force the run's session: a `None`/foreign session_id otherwise
            // reads the global cross-session timeline. [S2]
            params.session_id = Some(session_id.to_string());
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
