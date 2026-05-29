pub mod executor;
// Gated behind `federated-memory`: implemented and tested, but not yet wired
// into the engine's tool-dispatch path. See Cargo.toml for the rationale.
#[cfg(feature = "federated-memory")]
pub mod federated_memory_tools;
pub mod memory_tools;

pub use executor::{DefaultToolExecutor, ToolCallResult, ToolExecutor};
#[cfg(feature = "federated-memory")]
pub use federated_memory_tools::{
    FederatedGraphSearchHit, FederatedGraphSearchResult, FederatedMemoryHit, FederatedMemoryNode,
    FederatedMemorySearchParams, FederatedMemorySearchResult, FederatedMemoryTools,
    FederatedTimelineRawNode, FederatedTimelineSearchResult, MemorySource,
};
pub use memory_tools::{
    GraphSearchHit, GraphSearchParams, GraphSearchResult, MemorySearchParams, MemorySearchResult,
    MemorySearchTarget, MemoryTools, ProvenanceLookupParams, ProvenanceLookupResult,
    ScoredAbstractHit, ScoredRawHit, TimelineSearchParams, TimelineSearchResult,
};
