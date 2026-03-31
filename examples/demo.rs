use std::sync::Arc;

use takos_agent_engine::config::EngineConfig;
use takos_agent_engine::engine::context_assembler::WhitespaceTokenEstimator;
use takos_agent_engine::engine::session_engine::{run_turn, EngineDeps, SessionRequest};
use takos_agent_engine::memory::distillation::SimpleDistiller;
use takos_agent_engine::memory::scoring::DefaultScoringPolicy;
use takos_agent_engine::model::embedding::HashEmbedder;
use takos_agent_engine::model::runner::RuleBasedModelRunner;
use takos_agent_engine::storage::graph::InMemoryGraphRepository;
use takos_agent_engine::storage::in_memory::{InMemoryLoopStateRepository, InMemoryNodeRepository};
use takos_agent_engine::storage::vector::InMemoryVectorIndex;
use takos_agent_engine::tools::executor::DefaultToolExecutor;
use takos_agent_engine::tools::memory_tools::MemoryTools;
use takos_agent_engine::Result;
use tracing::info;

fn build_demo_deps() -> EngineDeps {
    let repository = Arc::new(InMemoryNodeRepository::default());
    let vector_index = Arc::new(InMemoryVectorIndex::default());
    let graph_repository = Arc::new(InMemoryGraphRepository::default());
    let loop_state_repository = Arc::new(InMemoryLoopStateRepository::default());
    let embedder = Arc::new(HashEmbedder::default());
    let scoring_policy = Arc::new(DefaultScoringPolicy::default());
    let token_estimator = Arc::new(WhitespaceTokenEstimator);
    let model_runner = Arc::new(RuleBasedModelRunner);
    let distiller = Arc::new(SimpleDistiller);
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

#[tokio::main]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let response = run_turn(
        &EngineConfig::default(),
        &build_demo_deps(),
        SessionRequest {
            session_id: None,
            user_message:
                "Explain how short-term session context and long-term memory work together."
                    .to_string(),
            plan: Some("Describe the two-layer memory model succinctly.".to_string()),
        },
    )
    .await?;

    info!(
        session_id = %response.session_id,
        loop_id = %response.loop_id,
        status = ?response.status,
        raw_activated = response.activated_raw_count,
        abstract_activated = response.activated_abstract_count,
        tool_results = response.tool_results_count,
        "demo loop completed"
    );
    println!(
        "status={:?}\n{}",
        response.status,
        response
            .assistant_message
            .as_deref()
            .unwrap_or("<no assistant message>")
    );

    Ok(())
}
