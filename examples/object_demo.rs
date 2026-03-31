use std::path::{Path, PathBuf};
use std::sync::Arc;

use takos_agent_engine::config::EngineConfig;
use takos_agent_engine::engine::context_assembler::WhitespaceTokenEstimator;
use takos_agent_engine::engine::session_engine::{run_turn, EngineDeps, SessionRequest};
use takos_agent_engine::memory::distillation::SimpleDistiller;
use takos_agent_engine::memory::scoring::DefaultScoringPolicy;
use takos_agent_engine::model::embedding::HashEmbedder;
use takos_agent_engine::model::runner::RuleBasedModelRunner;
use takos_agent_engine::storage::{
    FileObjectStore, ObjectGraphRepository, ObjectLoopStateRepository, ObjectNodeRepository,
    ObjectVectorIndex,
};
use takos_agent_engine::tools::executor::DefaultToolExecutor;
use takos_agent_engine::tools::memory_tools::MemoryTools;
use takos_agent_engine::Result;

fn build_object_deps(root: &Path) -> Result<EngineDeps> {
    let store = FileObjectStore::open(root)?;
    let repository = Arc::new(ObjectNodeRepository::new(store.clone()));
    let vector_index = Arc::new(ObjectVectorIndex::new(store.clone()));
    let graph_repository = Arc::new(ObjectGraphRepository::new(store.clone()));
    let loop_state_repository = Arc::new(ObjectLoopStateRepository::new(store));
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

fn demo_root() -> PathBuf {
    std::env::temp_dir().join(format!(
        "takos-agent-engine-object-demo-{}",
        uuid::Uuid::new_v4()
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    let root = demo_root();
    let deps = build_object_deps(&root)?;
    let first = run_turn(
        &EngineConfig::default(),
        &deps,
        SessionRequest {
            session_id: None,
            user_message: "Explain how object-backed persistence survives agent restarts."
                .to_string(),
            plan: Some("Focus on unified session and memory storage.".to_string()),
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

    println!("root={}", root.display());
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
