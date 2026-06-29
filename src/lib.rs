pub mod config;
pub mod domain;
pub mod engine;
pub mod error;
pub mod ids;
pub mod memory;
pub mod model;
pub mod storage;
pub mod tools;

pub use engine::execution_graph::{
    ExecutionGraph, ExecutionState, GraphNode, GraphRunResult, GraphRunner, NodeOutcome,
    NodeRuntimeClass, ResolvedRunOptions, RunOptions, DEFAULT_EDGE,
};
pub use engine::session_engine::{
    build_default_execution_graph, resume_loop, run_maintenance_pass, run_turn,
    run_turn_with_options, EngineDeps, MaintenanceReport, SessionRequest, SessionResponse,
};
pub use error::{EngineError, Result};
pub use ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};

// Deterministic stub implementations (hash embedder, whitespace token
// estimator, rule-based model runner, simple distiller) shared between the
// crate's own tests and the bundled examples. Internal tests reach these via
// `pub(crate)`; the bundled examples enable the non-default `test-support`
// feature, which promotes the module to `pub` so `examples/common/support.rs`
// can reuse the same stubs instead of redefining them. Neither the default
// build nor downstream consumers (which do not enable `test-support`) compile
// this module.
#[cfg(all(test, not(feature = "test-support")))]
pub(crate) mod test_support;
#[cfg(feature = "test-support")]
pub mod test_support;
