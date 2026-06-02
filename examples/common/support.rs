use std::path::Path;
use std::sync::Arc;

use takos_agent_engine::config::EngineConfig;
use takos_agent_engine::engine::session_engine::EngineDeps;
use takos_agent_engine::memory::scoring::DefaultScoringPolicy;
use takos_agent_engine::storage::{
    FileObjectStore, ObjectGraphRepository, ObjectLoopStateRepository, ObjectNodeRepository,
    ObjectVectorIndex,
};
use takos_agent_engine::test_support::{
    TestHashEmbedder, TestRuleBasedModelRunner, TestSimpleDistiller, TestWhitespaceTokenEstimator,
};
use takos_agent_engine::tools::executor::DefaultToolExecutor;
use takos_agent_engine::tools::memory_tools::MemoryTools;
use takos_agent_engine::Result;

pub fn build_object_deps(root: &Path) -> Result<EngineDeps> {
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

pub fn default_demo_config() -> EngineConfig {
    EngineConfig::default()
}
