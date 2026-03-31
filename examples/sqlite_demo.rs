use std::path::PathBuf;
use std::sync::Arc;

use takos_agent_engine::config::EngineConfig;
use takos_agent_engine::engine::context_assembler::WhitespaceTokenEstimator;
use takos_agent_engine::engine::session_engine::{run_turn, EngineDeps, SessionRequest};
use takos_agent_engine::memory::distillation::SimpleDistiller;
use takos_agent_engine::memory::scoring::DefaultScoringPolicy;
use takos_agent_engine::model::embedding::HashEmbedder;
use takos_agent_engine::model::runner::RuleBasedModelRunner;
use takos_agent_engine::storage::{
    SqliteDatabase, SqliteGraphRepository, SqliteLoopStateRepository, SqliteNodeRepository,
    SqliteVectorIndex,
};
use takos_agent_engine::tools::executor::DefaultToolExecutor;
use takos_agent_engine::tools::memory_tools::MemoryTools;
use takos_agent_engine::Result;

fn build_sqlite_deps(path: &PathBuf) -> Result<EngineDeps> {
    let database = SqliteDatabase::open(path)?;
    let repository = Arc::new(SqliteNodeRepository::new(database.clone()));
    let vector_index = Arc::new(SqliteVectorIndex::new(database.clone()));
    let graph_repository = Arc::new(SqliteGraphRepository::new(database.clone()));
    let loop_state_repository = Arc::new(SqliteLoopStateRepository::new(database.clone()));
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

fn demo_path() -> PathBuf {
    std::env::temp_dir().join("takos-agent-engine-demo.sqlite")
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let path = demo_path();
    let deps = build_sqlite_deps(&path)?;
    let first = run_turn(
        &EngineConfig::default(),
        &deps,
        SessionRequest {
            session_id: None,
            user_message: "Explain how persistence works across agent restarts.".to_string(),
            plan: Some("Focus on unified session and memory storage.".to_string()),
        },
    )
    .await?;

    let deps_after_restart = build_sqlite_deps(&path)?;
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

    println!("db={}", path.display());
    println!("session={}", first.session_id);
    println!(
        "turn1 status={:?}\n{}\n",
        first.status,
        first
            .assistant_message
            .as_deref()
            .unwrap_or("<no assistant message>")
    );
    println!(
        "turn2 status={:?}\n{}",
        second.status,
        second
            .assistant_message
            .as_deref()
            .unwrap_or("<no assistant message>")
    );

    Ok(())
}
