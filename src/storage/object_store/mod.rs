//! File-backed object store and its per-trait repository implementations.
//!
//! `store.rs` holds the [`FileObjectStore`] handle (on-disk layout, index
//! rebuild/reconciliation, atomic JSON IO) plus the shared on-disk record
//! types. Each public `Object*` type lives in its own sibling module and
//! implements exactly one storage trait against that shared handle:
//!
//! - [`ObjectNodeRepository`] (`node.rs`) — [`crate::storage::traits::NodeRepository`]
//! - [`ObjectVectorIndex`] (`vector.rs`) — [`crate::storage::traits::VectorIndex`]
//! - [`ObjectGraphRepository`] (`graph.rs`) — [`crate::storage::traits::GraphRepository`]
//! - [`ObjectLoopStateRepository`] (`loop_state.rs`) — [`crate::storage::traits::LoopStateRepository`]

mod graph;
mod loop_state;
mod node;
mod store;
mod vector;

pub use graph::ObjectGraphRepository;
pub use loop_state::ObjectLoopStateRepository;
pub use node::ObjectNodeRepository;
pub use store::FileObjectStore;
pub use vector::ObjectVectorIndex;
