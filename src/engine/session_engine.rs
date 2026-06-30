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
use crate::domain::{LoopStatus, RawNode};
use crate::engine::context_assembler::{ContextAssembler, TokenEstimator};
use crate::engine::execution_graph::{ExecutionState, GraphRunner, ResolvedRunOptions, RunOptions};
use crate::engine::nodes::{build_response, persist_abstract_node};
use crate::error::{EngineError, Result};
use crate::ids::{LoopId, SessionId};
use crate::memory::{ActivationService, DistillationInput, Distiller, ScoringPolicy};
use crate::model::{Embedder, ModelRunner};
use crate::storage::{GraphRepository, LoopStateRepository, NodeRepository, VectorIndex};
use crate::tools::executor::ToolExecutor;

// Re-export the graph builder so the crate-facing API keeps serving it from
// `crate::engine::session_engine::*` (lib.rs and engine/mod.rs re-export it).
pub use crate::engine::graph_spec::build_default_execution_graph;

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
    options: RunOptions,
) -> Result<SessionResponse> {
    config.validate()?;

    let session_id = request.session_id.unwrap_or_default();
    let loop_id = LoopId::new();
    tracing::Span::current().record("session_id", tracing::field::display(session_id));
    tracing::Span::current().record("loop_id", tracing::field::display(loop_id));

    let resolved_options = ResolvedRunOptions::from_config(config, options);
    let graph = Arc::new(build_default_execution_graph());
    let runner = GraphRunner::new(graph);
    let mut state = ExecutionState::from_request(request, session_id, loop_id);
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
    let graph = Arc::new(build_default_execution_graph());
    let runner = GraphRunner::new(graph);
    let (state, result) = runner
        .resume(checkpoint, config, deps, &resolved_options)
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
    let mut grouped: BTreeMap<(SessionId, LoopId), Vec<RawNode>> = BTreeMap::new();
    for node in backlog {
        if let (Some(session_id), Some(loop_id)) = (node.session_id, node.loop_id) {
            grouped.entry((session_id, loop_id)).or_default().push(node);
        }
    }

    let mut report = MaintenanceReport::default();
    for ((session_id, loop_id), raw_nodes) in grouped {
        let span = info_span!(
            "maintenance_distillation",
            session_id = %session_id,
            loop_id = %loop_id,
            raw_nodes = raw_nodes.len()
        );
        let _guard = span.enter();
        let distilled = deps
            .distiller
            .distill(DistillationInput {
                session_id,
                loop_id,
                raw_nodes: raw_nodes.clone(),
                activated_abstract_ids: Vec::new(),
            })
            .await?;

        let mut created_for_loop = 0usize;
        for node in distilled.new_nodes {
            if persist_abstract_node(deps, node, Some(session_id))
                .await?
                .is_some()
            {
                report.new_abstract_nodes += 1;
                created_for_loop += 1;
            }
        }
        for update in distilled.raw_updates {
            deps.repository
                .update_raw_lifecycle(&[update.raw_node_id], &update.patch)
                .await?;
            report.updated_raw_nodes += 1;
        }
        if created_for_loop == 0 {
            report.skipped_loops += 1;
        } else {
            report.processed_loops += 1;
        }
    }

    Ok(report)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Duration;

    use async_trait::async_trait;
    use tokio::sync::Notify;
    use tokio::time::sleep;
    use tokio_util::sync::CancellationToken;

    use crate::config::{EngineConfig, ToolsConfig};
    use crate::domain::{DistillationState, LoopStatus, RawNode, RawNodeKind};
    use crate::engine::execution_graph::{
        ExecutionGraph, ExecutionState, GraphNode, GraphRunner, NodeOutcome, ResolvedRunOptions,
        RunOptions, DEFAULT_EDGE,
    };
    use crate::engine::nodes::{
        assistant_output_operation_key, prepare_tool_call_for_config, tool_result_operation_key,
        user_input_operation_key,
    };
    use crate::engine::session_engine::{
        build_default_execution_graph, run_maintenance_pass, run_turn, run_turn_with_options,
        EngineDeps, SessionRequest,
    };
    use crate::error::{EngineError, Result};
    use crate::ids::{AbstractNodeId, LoopId, SessionId};
    use crate::memory::scoring::DefaultScoringPolicy;
    use crate::model::{ModelInput, ModelOutput, ModelRunner, ToolCallRequest};
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
    use crate::tools::executor::{DefaultToolExecutor, ToolCallResult, ToolExecutor};
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

        let semantic = prepare_tool_call_for_config(
            ToolCallRequest {
                name: "semantic_search_memory".to_string(),
                arguments: serde_json::json!({
                    "query": "topic",
                    "target": "both",
                    "top_k": 100
                }),
            },
            &tools,
        )?;
        let semantic_params: MemorySearchParams =
            serde_json::from_value(semantic.arguments).expect("semantic args should deserialize");
        assert_eq!(semantic_params.top_k, 2);

        let graph = prepare_tool_call_for_config(
            ToolCallRequest {
                name: "graph_search_memory".to_string(),
                arguments: serde_json::json!({
                    "start_node_id": AbstractNodeId::new().to_string(),
                    "max_depth": 100
                }),
            },
            &tools,
        )?;
        let graph_params: GraphSearchParams =
            serde_json::from_value(graph.arguments).expect("graph args should deserialize");
        assert_eq!(graph_params.max_depth, 1);

        let timeline = prepare_tool_call_for_config(
            ToolCallRequest {
                name: "timeline_search".to_string(),
                arguments: serde_json::json!({
                    "limit": 100
                }),
            },
            &tools,
        )?;
        let timeline_params: TimelineSearchParams =
            serde_json::from_value(timeline.arguments).expect("timeline args should deserialize");
        assert_eq!(timeline_params.limit, 3);

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
                    name: "timeline_search".to_string(),
                    arguments: serde_json::json!({ "limit": 1 }),
                }],
                usage: None,
            })
        }
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
