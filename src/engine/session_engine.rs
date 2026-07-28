//! Session lifecycle entry points for the agent engine.
//!
//! This module owns the public lifecycle surface — [`run_turn`],
//! [`run_turn_with_options`], [`resume_loop`], [`run_maintenance_pass`] and the
//! request/response/config types they exchange. The execution-graph topology is
//! built by [`graph_spec`](crate::engine::graph_spec) and the per-node behaviour
//! lives in [`nodes`](crate::engine::nodes); both are re-exported here so the
//! crate-facing API paths stay stable.

use std::collections::BTreeMap;
use std::sync::Arc;

use tracing::{info_span, instrument};

use crate::config::EngineConfig;
use crate::domain::{LoopState, LoopStatus};
use crate::engine::context_assembler::{ContextAssembler, TokenEstimator};
use crate::engine::execution_graph::{
    ExecutionProfile, ExecutionState, GraphRunner, ResolvedRunOptions, RunOptions,
};
use crate::engine::nodes::{
    build_response, distill_claimed_raw_nodes, trim_conversation_history, EXECUTE_TOOLS_NODE,
};
use crate::error::{EngineError, Result};
use crate::ids::{LoopId, SessionId};
use crate::memory::{ActivationService, Distiller, ScoringPolicy};
use crate::model::{Embedder, ModelRunner};
use crate::storage::{GraphRepository, LoopStateRepository, NodeRepository, VectorIndex};
use crate::tools::executor::ToolExecutor;

// Re-export the graph builder so the crate-facing API keeps serving it from
// `crate::engine::session_engine::*` (lib.rs and engine/mod.rs re-export it).
pub use crate::engine::graph_spec::{
    build_default_execution_graph, build_external_context_execution_graph,
};

fn graph_for_profile(profile: ExecutionProfile) -> crate::engine::execution_graph::ExecutionGraph {
    match profile {
        ExecutionProfile::MemoryAware => build_default_execution_graph(),
        ExecutionProfile::ExternalContext => build_external_context_execution_graph(),
    }
}

#[derive(Clone)]
pub struct EngineDeps {
    pub repository: Arc<dyn NodeRepository>,
    pub vector_index: Arc<dyn VectorIndex>,
    pub graph_repository: Arc<dyn GraphRepository>,
    pub loop_state_repository: Arc<dyn LoopStateRepository>,
    pub embedder: Arc<dyn Embedder>,
    pub model_runner: Arc<dyn ModelRunner>,
    pub tool_executor: Arc<dyn ToolExecutor>,
    pub distiller: Arc<dyn Distiller>,
    pub scoring_policy: Arc<dyn ScoringPolicy>,
    pub token_estimator: Arc<dyn TokenEstimator>,
}

impl EngineDeps {
    pub(crate) fn activation_service(&self) -> ActivationService {
        ActivationService::new(
            self.repository.clone(),
            self.vector_index.clone(),
            self.scoring_policy.clone(),
        )
    }

    pub(crate) fn context_assembler(&self) -> ContextAssembler {
        ContextAssembler::new(self.token_estimator.clone())
    }
}

#[derive(Debug, Clone)]
pub struct SessionRequest {
    pub session_id: Option<SessionId>,
    pub user_message: String,
    pub plan: Option<String>,
}

#[derive(Debug, Clone)]
pub struct SessionResponse {
    pub session_id: SessionId,
    pub loop_id: LoopId,
    pub status: LoopStatus,
    pub assistant_message: Option<String>,
    /// Structured assistant tool-call/tool-result transcript produced during
    /// this turn. Product wrappers can commit it with the terminal outcome;
    /// the engine itself does not claim durable conversation authority.
    pub turn_messages: Vec<crate::model::ConversationMessage>,
    pub activated_raw_count: usize,
    pub activated_abstract_count: usize,
    pub tool_results_count: usize,
    pub completed_steps: u32,
    pub tool_rounds_completed: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MaintenanceReport {
    pub processed_loops: usize,
    pub skipped_loops: usize,
    pub new_abstract_nodes: usize,
    pub updated_raw_nodes: usize,
}

/// # Errors
///
/// Surfaces the same [`EngineError`] variants as
/// [`run_turn_with_options`], to which this is a thin wrapper.
#[instrument(skip(config, deps), fields(session_id, loop_id))]
pub async fn run_turn(
    config: &EngineConfig,
    deps: &EngineDeps,
    request: SessionRequest,
) -> Result<SessionResponse> {
    run_turn_with_options(config, deps, request, RunOptions::default()).await
}

/// # Errors
///
/// Returns [`EngineError::Configuration`] when the config does not validate,
/// plus any [`EngineError`] raised by the underlying [`GraphRunner::run`].
#[instrument(skip(config, deps, options), fields(session_id, loop_id))]
pub async fn run_turn_with_options(
    config: &EngineConfig,
    deps: &EngineDeps,
    request: SessionRequest,
    mut options: RunOptions,
) -> Result<SessionResponse> {
    config.validate()?;

    let session_id = request.session_id.unwrap_or_default();
    let loop_id = options.loop_id.unwrap_or_default();
    tracing::Span::current().record("session_id", tracing::field::display(session_id));
    tracing::Span::current().record("loop_id", tracing::field::display(loop_id));

    let conversation_history = trim_conversation_history(
        &std::mem::take(&mut options.conversation_history),
        config,
        deps.token_estimator.as_ref(),
        &request.user_message,
        request.plan.as_deref(),
    )?;
    let resolved_options = ResolvedRunOptions::from_config(config, options);
    let graph = Arc::new(graph_for_profile(resolved_options.execution_profile));
    let runner = GraphRunner::new(graph);
    let mut state = ExecutionState::from_request(request, session_id, loop_id);
    state.conversation_history = conversation_history;
    state.execution_profile = resolved_options.execution_profile;
    let result = runner
        .run(&mut state, config, deps, &resolved_options)
        .await?;

    Ok(build_response(state, result))
}

/// # Errors
///
/// Returns [`EngineError::Configuration`] when the config does not validate,
/// [`EngineError::CheckpointNotFound`] when no checkpoint exists for the
/// `(session_id, loop_id)` pair, plus any [`EngineError`] raised by
/// [`GraphRunner::resume`].
#[instrument(skip(config, deps, options), fields(session_id, loop_id))]
pub async fn resume_loop(
    config: &EngineConfig,
    deps: &EngineDeps,
    session_id: SessionId,
    loop_id: LoopId,
    options: RunOptions,
) -> Result<SessionResponse> {
    config.validate()?;
    tracing::Span::current().record("session_id", tracing::field::display(session_id));
    tracing::Span::current().record("loop_id", tracing::field::display(loop_id));

    let checkpoint = deps
        .loop_state_repository
        .load_checkpoint(&session_id, &loop_id)
        .await?
        .ok_or_else(|| EngineError::CheckpointNotFound {
            session_id: session_id.to_string(),
            loop_id: loop_id.to_string(),
        })?;

    let resolved_options = ResolvedRunOptions::from_config(config, options);
    let graph = Arc::new(graph_for_profile(resolved_options.execution_profile));
    let runner = GraphRunner::new(graph);
    let (state, result) = runner
        .resume(checkpoint, config, deps, &resolved_options)
        .await?;

    Ok(build_response(state, result))
}

/// Recover a `Running` checkpoint left by an interrupted process.
///
/// Model requests do not have a provider-neutral idempotency contract. A
/// checkpoint written immediately before a model node therefore has an
/// ambiguous billing/outcome boundary and is rejected instead of silently
/// issuing a second completion. Tool nodes may be recovered when the embedding
/// product supplies idempotent/fenced tool execution through `ToolExecutor`.
///
/// # Errors
///
/// Returns [`EngineError::RecoveryUnsafe`] for an interrupted model node,
/// [`EngineError::LoopTerminated`] for a non-running checkpoint, and otherwise
/// the same errors as [`GraphRunner::recover_running`].
pub async fn recover_interrupted_loop_with_options(
    config: &EngineConfig,
    deps: &EngineDeps,
    checkpoint: LoopState,
    options: RunOptions,
) -> Result<SessionResponse> {
    config.validate()?;
    if checkpoint.current_node.starts_with("run_model") {
        return Err(EngineError::RecoveryUnsafe(format!(
            "model node {} may already have produced a billable completion",
            checkpoint.current_node
        )));
    }
    if checkpoint.current_node == EXECUTE_TOOLS_NODE {
        let (state, _, _) = ExecutionState::from_checkpoint(checkpoint.clone())?;
        if let Some(call) = state
            .pending_tool_calls
            .iter()
            .find(|call| !deps.tool_executor.recovery_is_idempotent(call))
        {
            return Err(EngineError::RecoveryUnsafe(format!(
                "side-effecting tool {} does not guarantee recovery with the engine idempotency key",
                call.name
            )));
        }
    }
    let resolved_options = ResolvedRunOptions::from_config(config, options);
    let graph = Arc::new(graph_for_profile(resolved_options.execution_profile));
    let runner = GraphRunner::new(graph);
    let (state, result) = runner
        .recover_running(checkpoint, config, deps, &resolved_options)
        .await?;
    Ok(build_response(state, result))
}

/// # Errors
///
/// Returns [`EngineError::Configuration`] when the config does not validate,
/// plus any [`EngineError`] raised by the repository or distillation tools
/// during the maintenance sweep.
pub async fn run_maintenance_pass(
    config: &EngineConfig,
    deps: &EngineDeps,
    limit: usize,
) -> Result<MaintenanceReport> {
    config.validate()?;
    let resolved_options = ResolvedRunOptions::from_config(config, RunOptions::default());
    let backlog_limit = limit.max(1).min(resolved_options.maintenance_batch_size);
    let backlog = deps.repository.undistilled_raw(backlog_limit, true).await?;
    // The backlog query is the bounded work queue. Group only the materialized
    // batch rather than reloading an unbounded loop; stable input-digest keys
    // let later passes safely distill the next chunk.
    let mut loop_batches: BTreeMap<(SessionId, LoopId), Vec<_>> = BTreeMap::new();
    for node in backlog {
        if let (Some(session_id), Some(loop_id)) = (node.session_id, node.loop_id) {
            loop_batches
                .entry((session_id, loop_id))
                .or_default()
                .push(node);
        }
    }

    let mut report = MaintenanceReport::default();
    for ((session_id, loop_id), mut raw_nodes) in loop_batches {
        raw_nodes.truncate(config.runtime.max_loop_nodes);
        if raw_nodes.is_empty() {
            continue;
        }
        let span = info_span!(
            "maintenance_distillation",
            session_id = %session_id,
            loop_id = %loop_id,
            raw_nodes = raw_nodes.len()
        );
        let _guard = span.enter();
        match distill_claimed_raw_nodes(
            config,
            deps,
            &resolved_options,
            session_id,
            loop_id,
            raw_nodes,
            Vec::new(),
        )
        .await?
        {
            Some(result) => {
                report.new_abstract_nodes += result.new_abstract_ids.len();
                report.updated_raw_nodes += result.updated_raw_nodes;
                if result.new_abstract_ids.is_empty() {
                    report.skipped_loops += 1;
                } else {
                    report.processed_loops += 1;
                }
            }
            None => report.skipped_loops += 1,
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::sync::{Barrier, Notify};
    use tokio::time::sleep;
    use tokio_util::sync::CancellationToken;

    use crate::config::{ContextBudgetConfig, EngineConfig, ToolsConfig};
    use crate::domain::{DistillationState, LoopStatus, RawNode, RawNodeKind};
    use crate::engine::execution_graph::{
        ExecutionGraph, ExecutionProfile, ExecutionState, GraphNode, GraphRunner, NodeOutcome,
        ResolvedRunOptions, RunOptions, DEFAULT_EDGE,
    };
    use crate::engine::nodes::{
        assistant_output_operation_key, prepare_tool_call_for_config, tool_result_operation_key,
        user_input_operation_key,
    };
    use crate::engine::session_engine::{
        build_default_execution_graph, build_external_context_execution_graph,
        recover_interrupted_loop_with_options, run_maintenance_pass, run_turn,
        run_turn_with_options, EngineDeps, SessionRequest,
    };
    use crate::error::{EngineError, Result};
    use crate::ids::{AbstractNodeId, LoopId, SessionId};
    use crate::memory::scoring::DefaultScoringPolicy;
    use crate::memory::{DistillationInput, DistillationOutput, Distiller};
    use crate::model::{
        ConversationMessage, ConversationRole, Embedder, Embedding, ModelInput, ModelOutput,
        ModelRunner, ToolCallRequest,
    };
    use crate::storage::object_store::{
        FileObjectStore, ObjectGraphRepository, ObjectLoopStateRepository, ObjectNodeRepository,
        ObjectVectorIndex,
    };
    use crate::storage::{
        InMemoryGraphRepository, InMemoryLoopStateRepository, InMemoryNodeRepository,
        InMemoryVectorIndex,
    };
    use crate::test_support::{
        TestHashEmbedder, TestRuleBasedModelRunner, TestSimpleDistiller,
        TestWhitespaceTokenEstimator,
    };
    use crate::tools::executor::{
        DefaultToolExecutor, ToolCallResult, ToolExecutionContext, ToolExecutionKind, ToolExecutor,
    };
    use crate::tools::memory_tools::{
        GraphSearchParams, MemorySearchParams, MemoryTools, TimelineSearchParams,
    };

    fn build_demo_deps() -> EngineDeps {
        let repository = Arc::new(InMemoryNodeRepository::default());
        let vector_index = Arc::new(InMemoryVectorIndex::default());
        let graph_repository = Arc::new(InMemoryGraphRepository::default());
        let loop_state_repository = Arc::new(InMemoryLoopStateRepository::default());
        let embedder = Arc::new(TestHashEmbedder::default());
        let scoring_policy = Arc::new(DefaultScoringPolicy::default());
        let token_estimator = Arc::new(TestWhitespaceTokenEstimator);
        let model_runner = Arc::new(TestRuleBasedModelRunner);
        let distiller = Arc::new(TestSimpleDistiller);
        let memory_tools = MemoryTools::new(
            repository.clone(),
            vector_index.clone(),
            graph_repository.clone(),
            embedder.clone(),
        );
        let tool_executor = Arc::new(DefaultToolExecutor::new(memory_tools));

        EngineDeps {
            repository,
            vector_index,
            graph_repository,
            loop_state_repository,
            embedder,
            model_runner,
            tool_executor,
            distiller,
            scoring_policy,
            token_estimator,
        }
    }

    #[tokio::test]
    async fn session_engine_runs_end_to_end() -> Result<()> {
        let deps = build_demo_deps();
        let response = run_turn(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "memory: session and memory".to_string(),
                plan: Some("Demonstrate memory retrieval.".to_string()),
            },
        )
        .await?;

        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(response.tool_results_count, 1);
        assert!(response.assistant_message.is_some());
        Ok(())
    }

    #[tokio::test]
    async fn caller_supplied_loop_id_is_preserved() -> Result<()> {
        let deps = build_demo_deps();
        let loop_id = LoopId::new();
        let response = run_turn_with_options(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "stable recovery identity".to_string(),
                plan: None,
            },
            RunOptions {
                loop_id: Some(loop_id),
                ..RunOptions::default()
            },
        )
        .await?;
        assert_eq!(response.loop_id, loop_id);
        Ok(())
    }

    struct PauseNode;
    struct FinishNode;
    struct LoopNode;
    struct SlowNode;
    #[derive(Debug)]
    struct PendingModelRunner {
        started: Arc<Notify>,
    }

    #[derive(Debug)]
    struct PendingToolExecutor {
        started: Arc<Notify>,
    }

    #[async_trait]
    impl GraphNode for PauseNode {
        fn id(&self) -> &'static str {
            "pause"
        }

        async fn run(
            &self,
            state: &mut ExecutionState,
            _config: &EngineConfig,
            _deps: &EngineDeps,
            _options: &ResolvedRunOptions,
        ) -> Result<NodeOutcome> {
            if state.assistant_message.is_none() {
                state.assistant_message = Some("paused".to_string());
                Ok(NodeOutcome::Pause)
            } else {
                Ok(NodeOutcome::Continue)
            }
        }
    }

    #[async_trait]
    impl GraphNode for FinishNode {
        fn id(&self) -> &'static str {
            "finish"
        }

        async fn run(
            &self,
            state: &mut ExecutionState,
            _config: &EngineConfig,
            _deps: &EngineDeps,
            _options: &ResolvedRunOptions,
        ) -> Result<NodeOutcome> {
            state.assistant_message = Some("finished".to_string());
            Ok(NodeOutcome::Finish)
        }
    }

    #[async_trait]
    impl GraphNode for LoopNode {
        fn id(&self) -> &'static str {
            "loop"
        }

        async fn run(
            &self,
            _state: &mut ExecutionState,
            _config: &EngineConfig,
            _deps: &EngineDeps,
            _options: &ResolvedRunOptions,
        ) -> Result<NodeOutcome> {
            Ok(NodeOutcome::Continue)
        }
    }

    #[async_trait]
    impl GraphNode for SlowNode {
        fn id(&self) -> &'static str {
            "slow"
        }

        async fn run(
            &self,
            _state: &mut ExecutionState,
            _config: &EngineConfig,
            _deps: &EngineDeps,
            _options: &ResolvedRunOptions,
        ) -> Result<NodeOutcome> {
            sleep(Duration::from_millis(50)).await;
            Ok(NodeOutcome::Finish)
        }
    }

    #[async_trait]
    impl ModelRunner for PendingModelRunner {
        async fn run(&self, _input: ModelInput) -> Result<ModelOutput> {
            self.started.notify_waiters();
            std::future::pending::<()>().await;
            unreachable!("pending model runner should be cancelled before returning")
        }
    }

    #[async_trait]
    impl ToolExecutor for PendingToolExecutor {
        fn execution_kind(&self, _call: &ToolCallRequest) -> ToolExecutionKind {
            ToolExecutionKind::ReadOnly
        }

        async fn execute(&self, _call: ToolCallRequest) -> Result<ToolCallResult> {
            self.started.notify_waiters();
            std::future::pending::<()>().await;
            unreachable!("pending tool executor should be cancelled before returning")
        }
    }

    #[tokio::test]
    async fn graph_runner_can_pause_and_resume() -> Result<()> {
        let deps = build_demo_deps();
        let mut graph = ExecutionGraph::new("pause");
        graph.add_node(Arc::new(PauseNode));
        graph.add_node(Arc::new(FinishNode));
        graph.add_edge("pause", DEFAULT_EDGE, "finish");
        let runner = GraphRunner::new(Arc::new(graph));
        let resolved_options =
            ResolvedRunOptions::from_config(&EngineConfig::default(), RunOptions::default());

        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let request = SessionRequest {
            session_id: Some(session_id),
            user_message: "pause".to_string(),
            plan: None,
        };
        let mut state = ExecutionState::from_request(request, session_id, loop_id);
        let first = runner
            .run(
                &mut state,
                &EngineConfig::default(),
                &deps,
                &resolved_options,
            )
            .await?;
        assert_eq!(first.status, LoopStatus::Paused);

        let checkpoint = deps
            .loop_state_repository
            .load_checkpoint(&session_id, &loop_id)
            .await?
            .expect("checkpoint");
        let (resumed_state, resumed) = runner
            .resume(
                checkpoint,
                &EngineConfig::default(),
                &deps,
                &resolved_options,
            )
            .await?;
        assert_eq!(resumed.status, LoopStatus::Finished);
        assert_eq!(resumed_state.assistant_message.as_deref(), Some("finished"));
        Ok(())
    }

    #[tokio::test]
    async fn graph_runner_stamps_external_profile_before_first_checkpoint() -> Result<()> {
        let deps = build_demo_deps();
        let mut graph = ExecutionGraph::new("pause");
        graph.add_node(Arc::new(PauseNode));
        let runner = GraphRunner::new(Arc::new(graph));
        let config = EngineConfig::default();
        let resolved_options = ResolvedRunOptions::from_config(
            &config,
            RunOptions {
                execution_profile: ExecutionProfile::ExternalContext,
                ..RunOptions::default()
            },
        );

        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let request = SessionRequest {
            session_id: Some(session_id),
            user_message: "pause external".to_string(),
            plan: None,
        };
        let mut state = ExecutionState::from_request(request, session_id, loop_id);
        assert_eq!(state.execution_profile, ExecutionProfile::MemoryAware);

        let result = runner
            .run(&mut state, &config, &deps, &resolved_options)
            .await?;
        assert_eq!(result.status, LoopStatus::Paused);
        let checkpoint = deps
            .loop_state_repository
            .load_checkpoint(&session_id, &loop_id)
            .await?
            .expect("external checkpoint");
        let checkpoint_state: ExecutionState =
            serde_json::from_value(checkpoint.state_json).expect("checkpoint execution state");
        assert_eq!(
            checkpoint_state.execution_profile,
            ExecutionProfile::ExternalContext
        );
        Ok(())
    }

    #[tokio::test]
    async fn graph_runner_rejects_resume_when_checkpoint_is_not_paused() -> Result<()> {
        let deps = build_demo_deps();
        let graph = ExecutionGraph::new("finish");
        let runner = GraphRunner::new(Arc::new(graph));
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let request = SessionRequest {
            session_id: Some(session_id),
            user_message: "running".to_string(),
            plan: None,
        };
        let state = ExecutionState::from_request(request, session_id, loop_id);
        let checkpoint = state.checkpoint("finish".to_string(), LoopStatus::Running)?;

        let err = runner
            .resume(
                checkpoint,
                &EngineConfig::default(),
                &deps,
                &ResolvedRunOptions::from_config(&EngineConfig::default(), RunOptions::default()),
            )
            .await
            .expect_err("running checkpoints must not resume");

        assert!(matches!(
            err,
            EngineError::LoopTerminated(LoopStatus::Running)
        ));
        Ok(())
    }

    #[tokio::test]
    async fn graph_runner_rejects_resume_under_a_different_execution_profile() -> Result<()> {
        let deps = build_demo_deps();
        let graph = ExecutionGraph::new("pause");
        let runner = GraphRunner::new(Arc::new(graph));
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let request = SessionRequest {
            session_id: Some(session_id),
            user_message: "paused external run".to_string(),
            plan: None,
        };
        let mut state = ExecutionState::from_request(request, session_id, loop_id);
        state.execution_profile = ExecutionProfile::ExternalContext;
        let checkpoint = state.checkpoint("pause".to_string(), LoopStatus::Paused)?;

        let err = runner
            .resume(
                checkpoint,
                &EngineConfig::default(),
                &deps,
                &ResolvedRunOptions::from_config(&EngineConfig::default(), RunOptions::default()),
            )
            .await
            .expect_err("a checkpoint cannot switch execution profiles");

        assert!(
            matches!(err, EngineError::Configuration(message) if message.contains("execution profile"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn graph_runner_respects_cancellation() -> Result<()> {
        let deps = build_demo_deps();
        let token = CancellationToken::new();
        token.cancel();
        let response = run_turn_with_options(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "cancel me".to_string(),
                plan: None,
            },
            RunOptions {
                cancellation_token: Some(token),
                ..RunOptions::default()
            },
        )
        .await?;
        assert_eq!(response.status, LoopStatus::Cancelled);
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_interrupts_model_execution() -> Result<()> {
        let mut deps = build_demo_deps();
        let started = Arc::new(Notify::new());
        deps.model_runner = Arc::new(PendingModelRunner {
            started: started.clone(),
        });
        let token = CancellationToken::new();
        let run_token = token.clone();

        let handle = tokio::spawn(async move {
            run_turn_with_options(
                &EngineConfig::default(),
                &deps,
                SessionRequest {
                    session_id: None,
                    user_message: "wait in model".to_string(),
                    plan: None,
                },
                RunOptions {
                    node_timeout: Some(Duration::from_secs(5)),
                    cancellation_token: Some(run_token),
                    ..RunOptions::default()
                },
            )
            .await
        });

        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("model runner did not start");
        token.cancel();
        let response = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("cancelled model run timed out")
            .expect("model task panicked")?;

        assert_eq!(response.status, LoopStatus::Cancelled);
        Ok(())
    }

    #[tokio::test]
    async fn cancellation_interrupts_tool_execution() -> Result<()> {
        let mut deps = build_demo_deps();
        let started = Arc::new(Notify::new());
        deps.tool_executor = Arc::new(PendingToolExecutor {
            started: started.clone(),
        });
        let token = CancellationToken::new();
        let run_token = token.clone();

        let handle = tokio::spawn(async move {
            run_turn_with_options(
                &EngineConfig::default(),
                &deps,
                SessionRequest {
                    session_id: None,
                    user_message: "timeline: recent".to_string(),
                    plan: None,
                },
                RunOptions {
                    tool_timeout: Some(Duration::from_secs(5)),
                    cancellation_token: Some(run_token),
                    ..RunOptions::default()
                },
            )
            .await
        });

        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("tool executor did not start");
        token.cancel();
        let response = tokio::time::timeout(Duration::from_secs(1), handle)
            .await
            .expect("cancelled tool run timed out")
            .expect("tool task panicked")?;

        assert_eq!(response.status, LoopStatus::Cancelled);
        Ok(())
    }

    #[tokio::test]
    async fn disabled_memory_search_tool_is_rejected() -> Result<()> {
        let deps = build_demo_deps();
        let mut config = EngineConfig::default();
        config.tools.memory_search = false;

        let err = run_turn(
            &config,
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "memory: blocked".to_string(),
                plan: None,
            },
        )
        .await
        .expect_err("disabled memory search should fail");

        assert!(matches!(err, EngineError::Tool(message) if message.contains("disabled")));
        Ok(())
    }

    #[test]
    fn prepare_tool_call_for_config_clamps_memory_tool_args() -> Result<()> {
        let tools = ToolsConfig {
            max_memory_search_top_k: 2,
            max_graph_search_depth: 1,
            max_timeline_search_limit: 3,
            ..ToolsConfig::default()
        };
        let session_id = SessionId::new();

        let semantic = prepare_tool_call_for_config(
            ToolCallRequest {
                id: None,
                name: "semantic_search_memory".to_string(),
                arguments: serde_json::json!({
                    "query": "topic",
                    "target": "both",
                    "top_k": 100
                }),
            },
            &tools,
            session_id,
        )?;
        let semantic_params: MemorySearchParams =
            serde_json::from_value(semantic.arguments).expect("semantic args should deserialize");
        assert_eq!(semantic_params.top_k, 2);
        assert_eq!(
            semantic_params.session_id.as_deref(),
            Some(session_id.to_string().as_str()),
            "semantic search must be bound to the run's session"
        );

        let graph = prepare_tool_call_for_config(
            ToolCallRequest {
                id: None,
                name: "graph_search_memory".to_string(),
                arguments: serde_json::json!({
                    "start_node_id": AbstractNodeId::new().to_string(),
                    "max_depth": 100
                }),
            },
            &tools,
            session_id,
        )?;
        let graph_params: GraphSearchParams =
            serde_json::from_value(graph.arguments).expect("graph args should deserialize");
        assert_eq!(graph_params.max_depth, 1);

        let timeline = prepare_tool_call_for_config(
            ToolCallRequest {
                id: None,
                name: "timeline_search".to_string(),
                arguments: serde_json::json!({
                    "limit": 100
                }),
            },
            &tools,
            session_id,
        )?;
        let timeline_params: TimelineSearchParams =
            serde_json::from_value(timeline.arguments).expect("timeline args should deserialize");
        assert_eq!(timeline_params.limit, 3);
        assert_eq!(
            timeline_params.session_id.as_deref(),
            Some(session_id.to_string().as_str()),
            "timeline search must be bound to the run's session"
        );

        Ok(())
    }

    // S2 regression: a model-supplied session_id (here a foreign one) must be
    // overridden with the run's session, so a tool call cannot read another
    // session's timeline.
    #[test]
    fn prepare_tool_call_overrides_model_supplied_session() -> Result<()> {
        let tools = ToolsConfig::default();
        let run_session = SessionId::new();
        let attacker_session = SessionId::new();

        let timeline = prepare_tool_call_for_config(
            ToolCallRequest {
                id: None,
                name: "timeline_search".to_string(),
                arguments: serde_json::json!({
                    "session_id": attacker_session.to_string(),
                    "limit": 10
                }),
            },
            &tools,
            run_session,
        )?;
        let params: TimelineSearchParams =
            serde_json::from_value(timeline.arguments).expect("timeline args should deserialize");
        assert_eq!(
            params.session_id.as_deref(),
            Some(run_session.to_string().as_str())
        );
        assert_ne!(
            params.session_id.as_deref(),
            Some(attacker_session.to_string().as_str())
        );
        Ok(())
    }

    #[tokio::test]
    async fn graph_runner_respects_step_budget() -> Result<()> {
        let deps = build_demo_deps();
        let mut graph = ExecutionGraph::new("loop");
        graph.add_node(Arc::new(LoopNode));
        graph.add_edge("loop", DEFAULT_EDGE, "loop");
        let runner = GraphRunner::new(Arc::new(graph));
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let request = SessionRequest {
            session_id: Some(session_id),
            user_message: "loop".to_string(),
            plan: None,
        };
        let mut state = ExecutionState::from_request(request, session_id, loop_id);
        let result = runner
            .run(
                &mut state,
                &EngineConfig::default(),
                &deps,
                &ResolvedRunOptions::from_config(
                    &EngineConfig::default(),
                    RunOptions {
                        max_graph_steps: Some(3),
                        ..RunOptions::default()
                    },
                ),
            )
            .await?;
        assert_eq!(result.status, LoopStatus::TimedOut);
        assert_eq!(result.completed_steps, 3);
        Ok(())
    }

    #[tokio::test]
    async fn graph_runner_respects_node_timeout() -> Result<()> {
        let deps = build_demo_deps();
        let mut graph = ExecutionGraph::new("slow");
        graph.add_node(Arc::new(SlowNode));
        let runner = GraphRunner::new(Arc::new(graph));
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let request = SessionRequest {
            session_id: Some(session_id),
            user_message: "slow".to_string(),
            plan: None,
        };
        let mut state = ExecutionState::from_request(request, session_id, loop_id);
        let result = runner
            .run(
                &mut state,
                &EngineConfig::default(),
                &deps,
                &ResolvedRunOptions::from_config(
                    &EngineConfig::default(),
                    RunOptions {
                        node_timeout: Some(Duration::from_millis(10)),
                        ..RunOptions::default()
                    },
                ),
            )
            .await?;
        assert_eq!(result.status, LoopStatus::TimedOut);
        Ok(())
    }

    #[tokio::test]
    async fn maintenance_pass_distills_pushed_out_raw_without_duplication() -> Result<()> {
        let deps = build_demo_deps();
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let mut node = RawNode::text(
            RawNodeKind::Note,
            Some(session_id),
            Some(loop_id),
            "system",
            "backlog item",
            0.5,
            Vec::new(),
        );
        node.overflow.was_pushed_out_of_session = true;
        node.distillation_state = DistillationState::Undistilled;
        deps.repository.insert_raw(node.clone()).await?;
        deps.vector_index
            .index_raw(
                node.id,
                deps.embedder.embed_text(&node.content_text()).await?,
            )
            .await?;

        let first = run_maintenance_pass(&EngineConfig::default(), &deps, 10).await?;
        let second = run_maintenance_pass(&EngineConfig::default(), &deps, 10).await?;
        assert_eq!(first.processed_loops, 1);
        assert_eq!(first.new_abstract_nodes, 1);
        assert_eq!(second.processed_loops, 0);
        Ok(())
    }

    #[derive(Debug)]
    struct SlowCountingDistiller {
        calls: AtomicUsize,
        started: Arc<Notify>,
    }

    #[async_trait]
    impl Distiller for SlowCountingDistiller {
        async fn distill(&self, input: DistillationInput) -> Result<DistillationOutput> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_waiters();
            sleep(Duration::from_millis(50)).await;
            TestSimpleDistiller.distill(input).await
        }
    }

    #[tokio::test]
    async fn concurrent_maintenance_claims_each_batch_once() -> Result<()> {
        let mut deps = build_demo_deps();
        let distiller = Arc::new(SlowCountingDistiller {
            calls: AtomicUsize::new(0),
            started: Arc::new(Notify::new()),
        });
        deps.distiller = distiller.clone();
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let mut node = RawNode::text(
            RawNodeKind::Note,
            Some(session_id),
            Some(loop_id),
            "system",
            "claim once",
            0.5,
            Vec::new(),
        );
        node.overflow.was_pushed_out_of_session = true;
        deps.repository.insert_raw(node).await?;

        let left_deps = deps.clone();
        let right_deps = deps.clone();
        let config = EngineConfig::default();
        let (left, right) = tokio::join!(
            run_maintenance_pass(&config, &left_deps, 8),
            run_maintenance_pass(&config, &right_deps, 8)
        );
        let left = left?;
        let right = right?;
        assert_eq!(distiller.calls.load(Ordering::SeqCst), 1);
        assert_eq!(left.processed_loops + right.processed_loops, 1);
        assert_eq!(left.skipped_loops + right.skipped_loops, 1);
        Ok(())
    }

    #[tokio::test]
    async fn stale_distillation_claim_cannot_release_successor() -> Result<()> {
        let deps = build_demo_deps();
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let old = deps
            .repository
            .try_claim_distillation(
                session_id,
                loop_id,
                "old".to_string(),
                Duration::from_millis(1),
            )
            .await?
            .expect("first claim");
        sleep(Duration::from_millis(5)).await;
        let successor = deps
            .repository
            .try_claim_distillation(
                session_id,
                loop_id,
                "new".to_string(),
                Duration::from_secs(1),
            )
            .await?
            .expect("expired claim should be replaceable");
        assert!(!deps.repository.distillation_claim_is_current(&old).await?);
        deps.repository.release_distillation_claim(&old).await?;
        assert!(
            deps.repository
                .distillation_claim_is_current(&successor)
                .await?
        );
        Ok(())
    }

    // A loop larger than the maintenance batch is processed in stable,
    // independently fenced chunks. Repeated passes eventually drain it without
    // loading the full loop or dropping any raw node.
    #[tokio::test]
    async fn maintenance_distills_large_loop_in_bounded_batches() -> Result<()> {
        let deps = build_demo_deps();
        let session_id = SessionId::new();
        let loop_id = LoopId::new();

        let total = 5usize;
        for i in 0..total {
            let mut node = RawNode::text(
                RawNodeKind::Note,
                Some(session_id),
                Some(loop_id),
                "system",
                format!("backlog item {i}"),
                0.5,
                Vec::new(),
            );
            node.overflow.was_pushed_out_of_session = true;
            node.distillation_state = DistillationState::Undistilled;
            deps.repository.insert_raw(node.clone()).await?;
            deps.vector_index
                .index_raw(
                    node.id,
                    deps.embedder.embed_text(&node.content_text()).await?,
                )
                .await?;
        }

        let first = run_maintenance_pass(&EngineConfig::default(), &deps, 2).await?;
        assert_eq!(first.processed_loops, 1);
        assert_eq!(first.new_abstract_nodes, 1);
        assert_eq!(deps.repository.undistilled_raw(100, true).await?.len(), 3);

        let second = run_maintenance_pass(&EngineConfig::default(), &deps, 2).await?;
        assert_eq!(second.processed_loops, 1);
        assert_eq!(deps.repository.undistilled_raw(100, true).await?.len(), 1);
        let third = run_maintenance_pass(&EngineConfig::default(), &deps, 2).await?;
        assert_eq!(third.processed_loops, 1);
        assert!(deps.repository.undistilled_raw(100, true).await?.is_empty());

        let fourth = run_maintenance_pass(&EngineConfig::default(), &deps, 2).await?;
        assert_eq!(fourth.processed_loops, 0);
        assert_eq!(fourth.new_abstract_nodes, 0);
        Ok(())
    }

    fn temp_object_root(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "takos-agent-engine-{name}-{}",
            uuid::Uuid::new_v4()
        ))
    }

    fn build_object_deps(root: &PathBuf) -> Result<EngineDeps> {
        let store = FileObjectStore::open(root)?;
        let repository = Arc::new(ObjectNodeRepository::new(store.clone()));
        let vector_index = Arc::new(ObjectVectorIndex::new(store.clone()));
        let graph_repository = Arc::new(ObjectGraphRepository::new(store.clone()));
        let loop_state_repository = Arc::new(ObjectLoopStateRepository::new(store));
        let embedder = Arc::new(TestHashEmbedder::default());
        let scoring_policy = Arc::new(DefaultScoringPolicy::default());
        let token_estimator = Arc::new(TestWhitespaceTokenEstimator);
        let model_runner = Arc::new(TestRuleBasedModelRunner);
        let distiller = Arc::new(TestSimpleDistiller);
        let memory_tools = MemoryTools::new(
            repository.clone(),
            vector_index.clone(),
            graph_repository.clone(),
            embedder.clone(),
        );
        let tool_executor = Arc::new(DefaultToolExecutor::new(memory_tools));

        Ok(EngineDeps {
            repository,
            vector_index,
            graph_repository,
            loop_state_repository,
            embedder,
            model_runner,
            tool_executor,
            distiller,
            scoring_policy,
            token_estimator,
        })
    }

    #[tokio::test]
    async fn object_store_persists_session_across_rebuilds() -> Result<()> {
        let root = temp_object_root("object-continuity");
        let deps = build_object_deps(&root)?;
        let first = run_turn(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "Explain object persistence".to_string(),
                plan: None,
            },
        )
        .await?;

        let deps_after_restart = build_object_deps(&root)?;
        let second = run_turn(
            &EngineConfig::default(),
            &deps_after_restart,
            SessionRequest {
                session_id: Some(first.session_id),
                user_message: "timeline: recent".to_string(),
                plan: None,
            },
        )
        .await?;

        assert_eq!(second.session_id, first.session_id);
        assert_eq!(second.status, LoopStatus::Finished);
        let session_raw = deps_after_restart
            .repository
            .session_raw(&first.session_id)
            .await?;
        assert!(session_raw.len() >= 4);

        let _ = std::fs::remove_dir_all(root);
        Ok(())
    }

    #[derive(Debug)]
    struct RepeatingToolModelRunner;

    #[async_trait]
    impl ModelRunner for RepeatingToolModelRunner {
        async fn run(&self, _input: ModelInput) -> Result<ModelOutput> {
            Ok(ModelOutput {
                assistant_message: None,
                tool_calls: vec![crate::model::runner::ToolCallRequest {
                    id: None,
                    name: "timeline_search".to_string(),
                    arguments: serde_json::json!({ "limit": 1 }),
                }],
                usage: None,
            })
        }
    }

    #[derive(Debug)]
    struct BurstToolModelRunner {
        calls: usize,
    }

    #[derive(Debug, Default)]
    struct TranscriptCheckingModelRunner {
        calls: AtomicUsize,
    }

    #[derive(Debug, Default)]
    struct MixedResultModelRunner {
        calls: AtomicUsize,
    }

    #[derive(Debug)]
    struct MixedResultToolExecutor;

    #[async_trait]
    impl ToolExecutor for MixedResultToolExecutor {
        fn execution_kind(&self, _call: &ToolCallRequest) -> ToolExecutionKind {
            ToolExecutionKind::ReadOnly
        }

        async fn execute(&self, call: ToolCallRequest) -> Result<ToolCallResult> {
            if call.name == "fails" {
                return Err(EngineError::Tool("intentional failure".to_string()));
            }
            Ok(ToolCallResult {
                tool_call_id: call.id,
                name: call.name,
                content: serde_json::json!({ "ok": true }),
                summary: "succeeds output=ok".to_string(),
            })
        }
    }

    #[async_trait]
    impl ModelRunner for MixedResultModelRunner {
        async fn run(&self, input: ModelInput) -> Result<ModelOutput> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Ok(ModelOutput {
                    assistant_message: None,
                    tool_calls: vec![
                        ToolCallRequest {
                            id: Some("call-fail".to_string()),
                            name: "fails".to_string(),
                            arguments: serde_json::json!({}),
                        },
                        ToolCallRequest {
                            id: Some("call-success".to_string()),
                            name: "succeeds".to_string(),
                            arguments: serde_json::json!({}),
                        },
                    ],
                    usage: None,
                });
            }
            let tool_messages = input
                .turn_messages
                .iter()
                .filter(|message| message.role == ConversationRole::Tool)
                .collect::<Vec<_>>();
            assert_eq!(tool_messages.len(), 2);
            assert!(tool_messages
                .iter()
                .any(|message| message.content.contains("intentional failure")));
            assert!(tool_messages
                .iter()
                .any(|message| message.content.contains("\"ok\":true")));
            Ok(ModelOutput {
                assistant_message: Some("reconciled".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[async_trait]
    impl ModelRunner for TranscriptCheckingModelRunner {
        async fn run(&self, input: ModelInput) -> Result<ModelOutput> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                assert!(input.turn_messages.is_empty());
                return Ok(ModelOutput {
                    assistant_message: Some("I will inspect the timeline.".to_string()),
                    tool_calls: vec![ToolCallRequest {
                        id: Some("provider-call-1".to_string()),
                        name: "timeline_search".to_string(),
                        arguments: serde_json::json!({ "limit": 1 }),
                    }],
                    usage: None,
                });
            }

            assert_eq!(input.turn_messages.len(), 2);
            assert_eq!(input.turn_messages[0].role, ConversationRole::Assistant);
            assert_eq!(
                input.turn_messages[0].content,
                "I will inspect the timeline."
            );
            assert_eq!(
                input.turn_messages[0].tool_calls[0].id.as_deref(),
                Some("provider-call-1")
            );
            assert_eq!(input.turn_messages[1].role, ConversationRole::Tool);
            assert_eq!(
                input.turn_messages[1].tool_call_id.as_deref(),
                Some("provider-call-1")
            );
            Ok(ModelOutput {
                assistant_message: Some("done".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[async_trait]
    impl ModelRunner for BurstToolModelRunner {
        async fn run(&self, input: ModelInput) -> Result<ModelOutput> {
            let has_current_tool_result = input
                .turn_messages
                .iter()
                .any(|message| message.role == ConversationRole::Tool);
            if input.tool_context.is_empty() && !has_current_tool_result {
                // Opening pass: emit a burst of tool calls in one round.
                let tool_calls = (0..self.calls)
                    .map(|_| crate::model::runner::ToolCallRequest {
                        id: None,
                        name: "timeline_search".to_string(),
                        arguments: serde_json::json!({ "limit": 1 }),
                    })
                    .collect();
                Ok(ModelOutput {
                    assistant_message: None,
                    tool_calls,
                    usage: None,
                })
            } else {
                // After tools ran: finish with a plain answer.
                Ok(ModelOutput {
                    assistant_message: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    usage: None,
                })
            }
        }
    }

    // A model round over the configured call cap fails as a whole. Silently
    // dropping the tail would change provider intent.
    #[tokio::test]
    async fn execute_tools_rejects_calls_over_per_round_cap() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(BurstToolModelRunner { calls: 10 });
        let mut config = EngineConfig::default();
        config.runtime.max_tool_calls_per_round = 3;
        let error = run_turn(
            &config,
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "burst".to_string(),
                plan: None,
            },
        )
        .await
        .expect_err("over-cap model output must fail closed");
        assert!(
            matches!(error, EngineError::Tool(message) if message.contains("exceeding max_tool_calls_per_round"))
        );
        Ok(())
    }

    #[tokio::test]
    async fn engine_preserves_native_tool_call_correlation_for_followup_model() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(TranscriptCheckingModelRunner::default());
        let response = run_turn(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "inspect the timeline".to_string(),
                plan: None,
            },
        )
        .await?;

        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(response.tool_results_count, 1);
        assert_eq!(response.assistant_message.as_deref(), Some("done"));
        Ok(())
    }

    #[derive(Debug, Default)]
    struct ReusedToolIdModelRunner {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl ModelRunner for ReusedToolIdModelRunner {
        async fn run(&self, _input: ModelInput) -> Result<ModelOutput> {
            let round = self.calls.fetch_add(1, Ordering::SeqCst);
            if round < 2 {
                return Ok(ModelOutput {
                    assistant_message: None,
                    tool_calls: vec![ToolCallRequest {
                        id: Some("provider-reused-id".to_string()),
                        name: format!("remote-{round}"),
                        arguments: serde_json::json!({}),
                    }],
                    usage: None,
                });
            }
            Ok(ModelOutput {
                assistant_message: Some("unexpected".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[derive(Debug, Default)]
    struct CountingRemoteToolExecutor {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl ToolExecutor for CountingRemoteToolExecutor {
        fn execution_kind(&self, _call: &ToolCallRequest) -> ToolExecutionKind {
            ToolExecutionKind::ReadOnly
        }

        async fn execute(&self, _call: ToolCallRequest) -> Result<ToolCallResult> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolCallResult {
                tool_call_id: Some("executor-wrong-id".to_string()),
                name: "executor-wrong-name".to_string(),
                content: serde_json::json!({ "ok": true }),
                summary: "ok".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn tool_ids_cannot_be_reused_across_rounds() {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(ReusedToolIdModelRunner::default());
        let executor = Arc::new(CountingRemoteToolExecutor::default());
        deps.tool_executor = executor.clone();

        let error = run_turn_with_options(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "reuse an id".to_string(),
                plan: None,
            },
            RunOptions {
                execution_profile: ExecutionProfile::ExternalContext,
                ..RunOptions::default()
            },
        )
        .await
        .expect_err("a provider tool id must be unique for the entire turn");

        assert!(
            matches!(error, EngineError::Tool(message) if message.contains("reused correlation id"))
        );
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn executor_cannot_replace_provider_tool_correlation() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(TranscriptCheckingModelRunner::default());
        deps.tool_executor = Arc::new(CountingRemoteToolExecutor::default());
        let response = run_turn_with_options(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "inspect remote state".to_string(),
                plan: None,
            },
            RunOptions {
                execution_profile: ExecutionProfile::ExternalContext,
                ..RunOptions::default()
            },
        )
        .await?;

        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(
            response.turn_messages[1].tool_call_id.as_deref(),
            Some("provider-call-1")
        );
        Ok(())
    }

    #[tokio::test]
    async fn mixed_parallel_tool_results_are_all_recorded_before_followup() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(MixedResultModelRunner::default());
        deps.tool_executor = Arc::new(MixedResultToolExecutor);
        let response = run_turn(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "run both tools".to_string(),
                plan: None,
            },
        )
        .await?;

        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(response.tool_results_count, 2);
        assert_eq!(response.assistant_message.as_deref(), Some("reconciled"));
        Ok(())
    }

    #[derive(Debug, Default)]
    struct PolicyOrderModelRunner {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl ModelRunner for PolicyOrderModelRunner {
        async fn run(&self, _input: ModelInput) -> Result<ModelOutput> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Ok(ModelOutput {
                    assistant_message: None,
                    tool_calls: ["read-1", "read-2", "write-1", "write-2"]
                        .into_iter()
                        .map(|name| ToolCallRequest {
                            id: Some(format!("call-{name}")),
                            name: name.to_string(),
                            arguments: serde_json::json!({}),
                        })
                        .collect(),
                    usage: None,
                });
            }
            Ok(ModelOutput {
                assistant_message: Some("done".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[derive(Debug)]
    struct PolicyOrderToolExecutor {
        read_barrier: Arc<Barrier>,
        active_reads: AtomicUsize,
        write_order: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl ToolExecutor for PolicyOrderToolExecutor {
        fn execution_kind(&self, call: &ToolCallRequest) -> ToolExecutionKind {
            if call.name.starts_with("read-") {
                ToolExecutionKind::ReadOnly
            } else {
                ToolExecutionKind::SideEffecting
            }
        }

        async fn execute(&self, call: ToolCallRequest) -> Result<ToolCallResult> {
            if call.name.starts_with("read-") {
                self.active_reads.fetch_add(1, Ordering::SeqCst);
                self.read_barrier.wait().await;
                sleep(Duration::from_millis(10)).await;
                self.active_reads.fetch_sub(1, Ordering::SeqCst);
            } else {
                assert_eq!(
                    self.active_reads.load(Ordering::SeqCst),
                    0,
                    "a side-effecting call must not overlap a read-only phase",
                );
                self.write_order
                    .lock()
                    .expect("write order lock")
                    .push(call.name.clone());
            }
            Ok(ToolCallResult {
                tool_call_id: call.id,
                name: call.name,
                content: serde_json::json!({ "ok": true }),
                summary: "ok".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn read_only_tools_overlap_but_side_effects_run_in_provider_order() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(PolicyOrderModelRunner::default());
        let executor = Arc::new(PolicyOrderToolExecutor {
            read_barrier: Arc::new(Barrier::new(2)),
            active_reads: AtomicUsize::new(0),
            write_order: Mutex::new(Vec::new()),
        });
        deps.tool_executor = executor.clone();

        let response = run_turn(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "read and then write".to_string(),
                plan: None,
            },
        )
        .await?;

        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(response.tool_results_count, 4);
        assert_eq!(
            *executor.write_order.lock().expect("write order lock"),
            vec!["write-1".to_string(), "write-2".to_string()],
        );
        Ok(())
    }

    #[derive(Debug, Default)]
    struct SequentialSideEffectModelRunner {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl ModelRunner for SequentialSideEffectModelRunner {
        async fn run(&self, _input: ModelInput) -> Result<ModelOutput> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Ok(ModelOutput {
                    assistant_message: None,
                    tool_calls: ["write-a", "write-b"]
                        .into_iter()
                        .map(|name| ToolCallRequest {
                            id: Some(format!("call-{name}")),
                            name: name.to_string(),
                            arguments: serde_json::json!({}),
                        })
                        .collect(),
                    usage: None,
                });
            }
            Ok(ModelOutput {
                assistant_message: Some("both writes completed".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[derive(Debug)]
    struct SlowSideEffectToolExecutor {
        completed: AtomicUsize,
    }

    #[async_trait]
    impl ToolExecutor for SlowSideEffectToolExecutor {
        async fn execute(&self, call: ToolCallRequest) -> Result<ToolCallResult> {
            sleep(Duration::from_millis(70)).await;
            self.completed.fetch_add(1, Ordering::SeqCst);
            Ok(ToolCallResult {
                tool_call_id: call.id,
                name: call.name,
                content: serde_json::json!({ "ok": true }),
                summary: "write completed".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn sequential_side_effects_each_receive_the_full_per_call_timeout() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(SequentialSideEffectModelRunner::default());
        let executor = Arc::new(SlowSideEffectToolExecutor {
            completed: AtomicUsize::new(0),
        });
        deps.tool_executor = executor.clone();

        let response = run_turn_with_options(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "perform both writes".to_string(),
                plan: None,
            },
            RunOptions {
                tool_timeout: Some(Duration::from_millis(100)),
                ..RunOptions::default()
            },
        )
        .await?;

        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(response.tool_results_count, 2);
        assert_eq!(executor.completed.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[derive(Debug)]
    struct CommitThenPendingToolExecutor {
        started: Arc<Notify>,
        calls: Mutex<Vec<String>>,
        contexts: Mutex<Vec<ToolExecutionContext>>,
    }

    #[async_trait]
    impl ToolExecutor for CommitThenPendingToolExecutor {
        async fn execute(&self, _call: ToolCallRequest) -> Result<ToolCallResult> {
            unreachable!("the context-aware path must be used")
        }

        async fn execute_with_context(
            &self,
            context: ToolExecutionContext,
            call: ToolCallRequest,
        ) -> Result<ToolCallResult> {
            self.calls.lock().expect("call log").push(call.name.clone());
            self.contexts.lock().expect("context log").push(context);
            // Simulate a remote durable commit before the local response is
            // available. Cancellation from this point is ambiguous.
            self.started.notify_waiters();
            std::future::pending::<()>().await;
            unreachable!("pending side effect should be cancelled")
        }
    }

    #[tokio::test]
    async fn cancellation_after_side_effect_dispatch_is_indeterminate_and_replayable() -> Result<()>
    {
        let mut deps = build_demo_deps();
        let model = Arc::new(SequentialSideEffectModelRunner::default());
        deps.model_runner = model.clone();
        let started = Arc::new(Notify::new());
        let executor = Arc::new(CommitThenPendingToolExecutor {
            started: started.clone(),
            calls: Mutex::new(Vec::new()),
            contexts: Mutex::new(Vec::new()),
        });
        deps.tool_executor = executor.clone();

        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let token = CancellationToken::new();
        let run_token = token.clone();
        let run_deps = deps.clone();
        let handle = tokio::spawn(async move {
            run_turn_with_options(
                &EngineConfig::default(),
                &run_deps,
                SessionRequest {
                    session_id: Some(session_id),
                    user_message: "perform both writes".to_string(),
                    plan: None,
                },
                RunOptions {
                    loop_id: Some(loop_id),
                    execution_profile: ExecutionProfile::ExternalContext,
                    tool_timeout: Some(Duration::from_secs(5)),
                    cancellation_token: Some(run_token),
                    ..RunOptions::default()
                },
            )
            .await
        });

        tokio::time::timeout(Duration::from_secs(1), started.notified())
            .await
            .expect("side effect did not start");
        token.cancel();
        let error = handle
            .await
            .expect("run task panicked")
            .expect_err("post-dispatch cancellation must be indeterminate");
        let expected_key = tool_result_operation_key(loop_id, 1, 0, "write-a");
        assert!(matches!(
            error,
            EngineError::ToolOutcomeIndeterminate {
                ref idempotency_key,
                ..
            } if idempotency_key == &expected_key
        ));
        assert_eq!(
            executor.calls.lock().expect("call log").as_slice(),
            ["write-a"],
            "a later side effect must not dispatch after an ambiguous outcome"
        );
        {
            let contexts = executor.contexts.lock().expect("context log");
            assert_eq!(contexts.len(), 1);
            assert_eq!(contexts[0].idempotency_key, expected_key);
            assert_eq!(
                contexts[0].max_result_bytes,
                EngineConfig::default().tools.max_tool_result_bytes
            );
        }
        assert_eq!(model.calls.load(Ordering::SeqCst), 1);

        let checkpoint = deps
            .loop_state_repository
            .load_checkpoint(&session_id, &loop_id)
            .await?
            .expect("failed checkpoint should remain");
        assert_eq!(checkpoint.status, LoopStatus::Failed);
        let (state, _, _) = ExecutionState::from_checkpoint(checkpoint)?;
        assert_eq!(state.pending_tool_calls.len(), 2);
        assert_eq!(
            tool_result_operation_key(loop_id, 1, 0, &state.pending_tool_calls[0].name),
            expected_key
        );
        Ok(())
    }

    #[tokio::test]
    async fn insufficient_tool_transcript_budget_fails_before_side_effects() {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(SequentialSideEffectModelRunner::default());
        let executor = Arc::new(SlowSideEffectToolExecutor {
            completed: AtomicUsize::new(0),
        });
        deps.tool_executor = executor.clone();
        deps.token_estimator = Arc::new(CharacterTokenEstimator);
        let mut config = EngineConfig::default();
        config.context_budget.reserve_tools = 1;

        let error = run_turn_with_options(
            &config,
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "do not partially execute".to_string(),
                plan: None,
            },
            RunOptions {
                execution_profile: ExecutionProfile::ExternalContext,
                ..RunOptions::default()
            },
        )
        .await
        .expect_err("unrecordable side effects must not execute");

        assert!(matches!(error, EngineError::Configuration(_)));
        assert_eq!(executor.completed.load(Ordering::SeqCst), 0);
    }

    #[derive(Debug)]
    struct RecoveredAfterToolModelRunner;

    #[async_trait]
    impl ModelRunner for RecoveredAfterToolModelRunner {
        async fn run(&self, input: ModelInput) -> Result<ModelOutput> {
            assert_eq!(input.turn_messages.len(), 2);
            assert_eq!(input.turn_messages[0].role, ConversationRole::Assistant);
            assert_eq!(input.turn_messages[1].role, ConversationRole::Tool);
            assert_eq!(
                input.turn_messages[1].tool_call_id.as_deref(),
                Some("recover-call")
            );
            Ok(ModelOutput {
                assistant_message: Some("recovered safely".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[tokio::test]
    async fn interrupted_tool_node_recovers_from_its_pre_node_checkpoint() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(RecoveredAfterToolModelRunner);
        let executor = Arc::new(CountingRemoteToolExecutor::default());
        deps.tool_executor = executor.clone();
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let request = SessionRequest {
            session_id: Some(session_id),
            user_message: "recover the tool round".to_string(),
            plan: None,
        };
        let tool_call = ToolCallRequest {
            id: Some("recover-call".to_string()),
            name: "remote-recover".to_string(),
            arguments: serde_json::json!({ "value": 1 }),
        };
        let mut state = ExecutionState::from_request(request, session_id, loop_id);
        state.execution_profile = ExecutionProfile::ExternalContext;
        state.assembled_context = Some(crate::engine::context_assembler::AssembledContext {
            system_prompt: EngineConfig::default().system_prompt,
            ..crate::engine::context_assembler::AssembledContext::default()
        });
        state.pending_tool_calls = vec![tool_call.clone()];
        state.latest_model_output = Some(ModelOutput {
            assistant_message: Some("running a tool".to_string()),
            tool_calls: vec![tool_call],
            usage: None,
        });
        let checkpoint = state.checkpoint("execute_tools".to_string(), LoopStatus::Running)?;

        let response = recover_interrupted_loop_with_options(
            &EngineConfig::default(),
            &deps,
            checkpoint,
            RunOptions {
                execution_profile: ExecutionProfile::ExternalContext,
                ..RunOptions::default()
            },
        )
        .await?;

        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(
            response.assistant_message.as_deref(),
            Some("recovered safely")
        );
        assert_eq!(executor.calls.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test]
    async fn interrupted_model_node_is_not_automatically_reissued() -> Result<()> {
        let deps = build_demo_deps();
        let session_id = SessionId::new();
        let loop_id = LoopId::new();
        let request = SessionRequest {
            session_id: Some(session_id),
            user_message: "do not double bill".to_string(),
            plan: None,
        };
        let mut state = ExecutionState::from_request(request, session_id, loop_id);
        state.execution_profile = ExecutionProfile::ExternalContext;
        let checkpoint = state.checkpoint(
            "run_model_external_context".to_string(),
            LoopStatus::Running,
        )?;

        let error = recover_interrupted_loop_with_options(
            &EngineConfig::default(),
            &deps,
            checkpoint,
            RunOptions {
                execution_profile: ExecutionProfile::ExternalContext,
                ..RunOptions::default()
            },
        )
        .await
        .expect_err("ambiguous model completions must not be reissued");

        assert!(matches!(error, EngineError::RecoveryUnsafe(_)));
        Ok(())
    }

    #[derive(Debug)]
    struct ForbiddenExternalContextEmbedder;

    #[async_trait]
    impl Embedder for ForbiddenExternalContextEmbedder {
        async fn embed_text(&self, _text: &str) -> Result<Embedding> {
            panic!("external-context execution must not invoke the local embedder")
        }
    }

    #[derive(Debug)]
    struct ForbiddenExternalContextDistiller;

    #[async_trait]
    impl Distiller for ForbiddenExternalContextDistiller {
        async fn distill(&self, _input: DistillationInput) -> Result<DistillationOutput> {
            panic!("external-context execution must not invoke local distillation")
        }
    }

    #[derive(Debug)]
    struct ExternalContextToolExecutor;

    #[async_trait]
    impl ToolExecutor for ExternalContextToolExecutor {
        fn execution_kind(&self, _call: &ToolCallRequest) -> ToolExecutionKind {
            ToolExecutionKind::ReadOnly
        }

        async fn execute(&self, call: ToolCallRequest) -> Result<ToolCallResult> {
            assert_eq!(call.id.as_deref(), Some("provider-external-call"));
            assert_eq!(call.name, "worker_tool");
            Ok(ToolCallResult {
                tool_call_id: call.id,
                name: call.name,
                content: serde_json::json!({ "source": "worker", "ok": true }),
                summary: "worker tool result".to_string(),
            })
        }
    }

    #[derive(Debug)]
    struct ExternalContextModelRunner {
        calls: AtomicUsize,
        expected_history: Vec<ConversationMessage>,
    }

    #[async_trait]
    impl ModelRunner for ExternalContextModelRunner {
        async fn run(&self, input: ModelInput) -> Result<ModelOutput> {
            assert_eq!(input.system_prompt, "worker-owned system prompt");
            assert_eq!(input.conversation_history, self.expected_history);
            assert_eq!(input.user_message, "current worker user message");
            assert!(input.session_context.is_empty());
            assert!(input.memory_context.is_empty());
            assert!(input.tool_context.is_empty());

            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                assert!(input.turn_messages.is_empty());
                return Ok(ModelOutput {
                    assistant_message: None,
                    tool_calls: vec![ToolCallRequest {
                        id: Some("provider-external-call".to_string()),
                        name: "worker_tool".to_string(),
                        arguments: serde_json::json!({ "query": "current" }),
                    }],
                    usage: None,
                });
            }

            assert_eq!(input.turn_messages.len(), 2);
            assert_eq!(input.turn_messages[0].role, ConversationRole::Assistant);
            assert_eq!(input.turn_messages[0].tool_calls.len(), 1);
            assert_eq!(
                input.turn_messages[0].tool_calls[0].id.as_deref(),
                Some("provider-external-call")
            );
            assert_eq!(input.turn_messages[1].role, ConversationRole::Tool);
            assert_eq!(
                input.turn_messages[1].tool_call_id.as_deref(),
                Some("provider-external-call")
            );
            assert!(input.turn_messages[1]
                .content
                .contains("\"source\":\"worker\""));
            Ok(ModelOutput {
                assistant_message: Some("external context complete".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[tokio::test]
    async fn external_context_profile_runs_only_the_bounded_model_tool_loop() -> Result<()> {
        let history = vec![
            ConversationMessage {
                role: ConversationRole::User,
                content: "durable history user".to_string(),
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
            ConversationMessage {
                role: ConversationRole::Assistant,
                content: "durable history assistant".to_string(),
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
        ];
        let mut deps = build_demo_deps();
        deps.embedder = Arc::new(ForbiddenExternalContextEmbedder);
        deps.distiller = Arc::new(ForbiddenExternalContextDistiller);
        deps.model_runner = Arc::new(ExternalContextModelRunner {
            calls: AtomicUsize::new(0),
            expected_history: history.clone(),
        });
        deps.tool_executor = Arc::new(ExternalContextToolExecutor);

        let config = EngineConfig {
            system_prompt: "worker-owned system prompt".to_string(),
            ..EngineConfig::default()
        };
        let response = run_turn_with_options(
            &config,
            &deps,
            SessionRequest {
                session_id: Some(SessionId::new()),
                user_message: "current worker user message".to_string(),
                plan: None,
            },
            RunOptions {
                execution_profile: ExecutionProfile::ExternalContext,
                conversation_history: history,
                ..RunOptions::default()
            },
        )
        .await?;

        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(
            response.assistant_message.as_deref(),
            Some("external context complete")
        );
        assert_eq!(response.activated_raw_count, 0);
        assert_eq!(response.activated_abstract_count, 0);
        assert_eq!(response.tool_results_count, 1);
        assert_eq!(response.tool_rounds_completed, 1);
        assert_eq!(response.turn_messages.len(), 2);
        assert!(deps
            .repository
            .raw_for_loop(&response.loop_id)
            .await?
            .is_empty());
        assert!(deps
            .repository
            .session_raw(&response.session_id)
            .await?
            .is_empty());
        Ok(())
    }

    #[derive(Debug)]
    struct HistoryCaptureModelRunner {
        observed: Arc<Mutex<Option<Vec<ConversationMessage>>>>,
    }

    #[async_trait]
    impl ModelRunner for HistoryCaptureModelRunner {
        async fn run(&self, input: ModelInput) -> Result<ModelOutput> {
            *self.observed.lock().expect("history capture lock") = Some(input.conversation_history);
            Ok(ModelOutput {
                assistant_message: Some("trimmed".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[tokio::test]
    async fn external_history_is_recent_bounded_and_keeps_tool_groups_coherent() -> Result<()> {
        let recent_group = vec![
            ConversationMessage {
                role: ConversationRole::Assistant,
                content: "checking".to_string(),
                tool_call_id: None,
                tool_calls: vec![ToolCallRequest {
                    id: Some("recent-call".to_string()),
                    name: "lookup".to_string(),
                    arguments: serde_json::json!({}),
                }],
            },
            ConversationMessage {
                role: ConversationRole::Tool,
                content: serde_json::json!({ "ok": true }).to_string(),
                tool_call_id: Some("recent-call".to_string()),
                tool_calls: Vec::new(),
            },
            ConversationMessage {
                role: ConversationRole::Assistant,
                content: "recent answer".to_string(),
                tool_call_id: None,
                tool_calls: Vec::new(),
            },
        ];
        let mut history = vec![ConversationMessage {
            role: ConversationRole::User,
            content: "o".repeat(120),
            tool_call_id: None,
            tool_calls: Vec::new(),
        }];
        history.extend(recent_group.clone());

        let observed = Arc::new(Mutex::new(None));
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(HistoryCaptureModelRunner {
            observed: observed.clone(),
        });
        deps.token_estimator = Arc::new(CharacterTokenEstimator);
        let config = EngineConfig {
            system_prompt: "system".to_string(),
            context_budget: ContextBudgetConfig {
                total_tokens: 260,
                reserve_system: 30,
                reserve_tools: 120,
                reserve_working: 20,
                ..ContextBudgetConfig::default()
            },
            ..EngineConfig::default()
        };

        let response = run_turn_with_options(
            &config,
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "now".to_string(),
                plan: None,
            },
            RunOptions {
                execution_profile: ExecutionProfile::ExternalContext,
                conversation_history: history,
                ..RunOptions::default()
            },
        )
        .await?;

        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(
            observed
                .lock()
                .expect("history capture lock")
                .clone()
                .expect("captured history"),
            recent_group
        );
        Ok(())
    }

    #[derive(Debug, Default)]
    struct IntermediateOnlyModelRunner {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl ModelRunner for IntermediateOnlyModelRunner {
        async fn run(&self, input: ModelInput) -> Result<ModelOutput> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Ok(ModelOutput {
                    assistant_message: Some("I will inspect before answering.".to_string()),
                    tool_calls: vec![ToolCallRequest {
                        id: Some("provider-external-call".to_string()),
                        name: "worker_tool".to_string(),
                        arguments: serde_json::json!({}),
                    }],
                    usage: None,
                });
            }
            assert_eq!(
                input.turn_messages[0].content,
                "I will inspect before answering."
            );
            Ok(ModelOutput {
                assistant_message: None,
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[tokio::test]
    async fn intermediate_assistant_tool_content_is_not_reused_as_the_final_answer() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(IntermediateOnlyModelRunner::default());
        deps.tool_executor = Arc::new(ExternalContextToolExecutor);
        let response = run_turn_with_options(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: Some(SessionId::new()),
                user_message: "inspect".to_string(),
                plan: None,
            },
            RunOptions {
                execution_profile: ExecutionProfile::ExternalContext,
                ..RunOptions::default()
            },
        )
        .await?;

        assert_eq!(
            response.turn_messages[0].content,
            "I will inspect before answering."
        );
        assert_eq!(
            response.assistant_message.as_deref(),
            Some("No assistant message generated.")
        );
        Ok(())
    }

    #[derive(Debug, Default)]
    struct ExternalMemoryNameCollisionModelRunner {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl ModelRunner for ExternalMemoryNameCollisionModelRunner {
        async fn run(&self, _input: ModelInput) -> Result<ModelOutput> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Ok(ModelOutput {
                    assistant_message: None,
                    tool_calls: vec![ToolCallRequest {
                        id: Some("remote-timeline-call".to_string()),
                        name: "timeline_search".to_string(),
                        arguments: serde_json::json!({
                            "session_id": "remote-schema-session",
                            "limit": 999,
                            "custom": true,
                        }),
                    }],
                    usage: None,
                });
            }
            Ok(ModelOutput {
                assistant_message: Some("remote collision preserved".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[derive(Debug)]
    struct ExternalMemoryNameCollisionExecutor;

    #[async_trait]
    impl ToolExecutor for ExternalMemoryNameCollisionExecutor {
        async fn execute(&self, call: ToolCallRequest) -> Result<ToolCallResult> {
            assert_eq!(call.id.as_deref(), Some("remote-timeline-call"));
            assert_eq!(call.name, "timeline_search");
            assert_eq!(
                call.arguments,
                serde_json::json!({
                    "session_id": "remote-schema-session",
                    "limit": 999,
                    "custom": true,
                })
            );
            Ok(ToolCallResult {
                tool_call_id: call.id,
                name: call.name,
                content: serde_json::json!({ "ok": true }),
                summary: "remote timeline".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn external_context_does_not_rewrite_remote_memory_tool_name_collisions() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(ExternalMemoryNameCollisionModelRunner::default());
        deps.tool_executor = Arc::new(ExternalMemoryNameCollisionExecutor);
        deps.embedder = Arc::new(ForbiddenExternalContextEmbedder);
        deps.distiller = Arc::new(ForbiddenExternalContextDistiller);
        let response = run_turn_with_options(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: Some(SessionId::new()),
                user_message: "use remote timeline".to_string(),
                plan: None,
            },
            RunOptions {
                execution_profile: ExecutionProfile::ExternalContext,
                ..RunOptions::default()
            },
        )
        .await?;

        assert_eq!(
            response.assistant_message.as_deref(),
            Some("remote collision preserved")
        );
        Ok(())
    }

    #[derive(Debug)]
    struct CharacterTokenEstimator;

    impl crate::engine::context_assembler::TokenEstimator for CharacterTokenEstimator {
        fn estimate_text(&self, text: &str) -> usize {
            text.chars().count()
        }
    }

    #[derive(Debug, Default)]
    struct LargeMultiOutputModelRunner {
        calls: AtomicUsize,
    }

    #[async_trait]
    impl ModelRunner for LargeMultiOutputModelRunner {
        async fn run(&self, input: ModelInput) -> Result<ModelOutput> {
            if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Ok(ModelOutput {
                    assistant_message: None,
                    tool_calls: ["large-a", "large-b"]
                        .into_iter()
                        .map(|name| ToolCallRequest {
                            id: Some(format!("call-{name}")),
                            name: name.to_string(),
                            arguments: serde_json::json!({}),
                        })
                        .collect(),
                    usage: None,
                });
            }

            let results = input
                .turn_messages
                .iter()
                .filter(|message| message.role == ConversationRole::Tool)
                .collect::<Vec<_>>();
            assert_eq!(results.len(), 2);
            assert!(
                results
                    .iter()
                    .map(|message| message.content.len())
                    .sum::<usize>()
                    <= 160
            );
            for result in results {
                let content: serde_json::Value =
                    serde_json::from_str(&result.content).expect("bounded result JSON");
                assert_eq!(content["truncated"], true);
                assert!(content["preview"]
                    .as_str()
                    .is_some_and(|value| !value.is_empty()));
            }
            Ok(ModelOutput {
                assistant_message: Some("bounded".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    #[derive(Debug)]
    struct LargeMultiOutputToolExecutor;

    #[async_trait]
    impl ToolExecutor for LargeMultiOutputToolExecutor {
        fn execution_kind(&self, _call: &ToolCallRequest) -> ToolExecutionKind {
            ToolExecutionKind::ReadOnly
        }

        async fn execute(&self, call: ToolCallRequest) -> Result<ToolCallResult> {
            Ok(ToolCallResult {
                tool_call_id: call.id,
                name: call.name,
                content: serde_json::json!({ "payload": "x".repeat(2_000) }),
                summary: "large result".to_string(),
            })
        }
    }

    #[tokio::test]
    async fn tool_result_transcript_is_aggregate_bounded_in_both_profiles() -> Result<()> {
        for execution_profile in [
            ExecutionProfile::MemoryAware,
            ExecutionProfile::ExternalContext,
        ] {
            let mut deps = build_demo_deps();
            deps.model_runner = Arc::new(LargeMultiOutputModelRunner::default());
            deps.tool_executor = Arc::new(LargeMultiOutputToolExecutor);
            deps.token_estimator = Arc::new(CharacterTokenEstimator);
            let mut config = EngineConfig::default();
            config.context_budget.reserve_tools = 160;
            let response = run_turn_with_options(
                &config,
                &deps,
                SessionRequest {
                    session_id: Some(SessionId::new()),
                    user_message: "large tools".to_string(),
                    plan: None,
                },
                RunOptions {
                    execution_profile,
                    ..RunOptions::default()
                },
            )
            .await?;

            let tool_payload_bytes = response
                .turn_messages
                .iter()
                .filter(|message| message.role == ConversationRole::Tool)
                .map(|message| message.content.len())
                .sum::<usize>();
            assert!(tool_payload_bytes <= config.context_budget.reserve_tools);
            if execution_profile == ExecutionProfile::MemoryAware {
                let persisted = deps.repository.raw_for_loop(&response.loop_id).await?;
                let tool_results = persisted
                    .iter()
                    .filter(|node| node.kind == RawNodeKind::ToolResult)
                    .collect::<Vec<_>>();
                assert_eq!(tool_results.len(), 2);
                assert!(tool_results
                    .iter()
                    .all(|node| node.content_text().contains(&"x".repeat(2_000))));
            }
        }
        Ok(())
    }

    #[derive(Debug)]
    struct SleepyModelRunner {
        delay: Duration,
    }

    #[async_trait]
    impl ModelRunner for SleepyModelRunner {
        async fn run(&self, _input: ModelInput) -> Result<ModelOutput> {
            sleep(self.delay).await;
            Ok(ModelOutput {
                assistant_message: Some("slept and answered".to_string()),
                tool_calls: Vec::new(),
                usage: None,
            })
        }
    }

    // C1 regression: a completion that takes longer than the (small) Standard
    // `node_timeout` but well under `model_timeout` must still succeed, because
    // the model node runs under `NodeRuntimeClass::Model`, not `Standard`.
    #[tokio::test]
    async fn model_node_completes_when_slower_than_node_timeout() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(SleepyModelRunner {
            delay: Duration::from_millis(300),
        });
        let response = run_turn_with_options(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "answer slowly".to_string(),
                plan: None,
            },
            RunOptions {
                // Every Standard node must finish inside this tiny budget; the
                // model node must NOT inherit it.
                node_timeout: Some(Duration::from_millis(80)),
                model_timeout: Some(Duration::from_secs(3)),
                ..RunOptions::default()
            },
        )
        .await?;
        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(
            response.assistant_message.as_deref(),
            Some("slept and answered")
        );
        Ok(())
    }

    // Conversely, the model node IS bounded by `model_timeout` (not the large
    // `node_timeout`): a tiny `model_timeout` aborts a slow completion.
    #[tokio::test]
    async fn model_node_is_bounded_by_model_timeout() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(SleepyModelRunner {
            delay: Duration::from_millis(300),
        });
        let response = run_turn_with_options(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "answer slowly".to_string(),
                plan: None,
            },
            RunOptions {
                node_timeout: Some(Duration::from_secs(3)),
                model_timeout: Some(Duration::from_millis(20)),
                ..RunOptions::default()
            },
        )
        .await?;
        assert_eq!(response.status, LoopStatus::TimedOut);
        Ok(())
    }

    // Locks the builder to the expected execution-graph topology: the opening
    // pass, the post-tool re-activation pass, persist, and the overflow/distill
    // tail must register exactly these node and edge counts.
    #[test]
    fn default_graph_topology_is_stable() {
        let default = build_default_execution_graph();
        assert_eq!(default.node_count(), 14);
        assert_eq!(default.edge_count(), 15);
    }

    #[test]
    fn external_context_graph_is_the_minimal_model_tool_loop() {
        let graph = build_external_context_execution_graph();
        assert_eq!(graph.node_count(), 5);
        assert_eq!(graph.edge_count(), 6);
    }

    #[tokio::test]
    async fn default_graph_bounds_multi_step_tool_rounds() -> Result<()> {
        let mut deps = build_demo_deps();
        deps.model_runner = Arc::new(RepeatingToolModelRunner);
        let response = run_turn_with_options(
            &EngineConfig::default(),
            &deps,
            SessionRequest {
                session_id: None,
                user_message: "keep calling tools".to_string(),
                plan: None,
            },
            RunOptions {
                max_tool_rounds: Some(2),
                ..RunOptions::default()
            },
        )
        .await?;
        assert_eq!(response.status, LoopStatus::Finished);
        assert_eq!(response.tool_rounds_completed, 2);
        assert_eq!(response.tool_results_count, 2);
        Ok(())
    }

    #[test]
    fn default_graph_contains_expected_nodes() {
        let _graph = build_default_execution_graph();
    }

    #[test]
    fn operation_key_formats_are_stable() {
        let loop_id = LoopId::new();

        assert_eq!(
            user_input_operation_key(loop_id),
            format!("loop:{loop_id}:user_input")
        );
        assert_eq!(
            tool_result_operation_key(loop_id, 2, 3, "timeline_search"),
            format!("loop:{loop_id}:tool:2:3:timeline_search")
        );
        assert_eq!(
            assistant_output_operation_key(loop_id),
            format!("loop:{loop_id}:assistant_output")
        );
    }
}
