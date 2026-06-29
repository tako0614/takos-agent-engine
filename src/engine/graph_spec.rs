//! Construction of the agent's execution graph.
//!
//! The engine runs a single linear pipeline: ingest the user input, load the
//! session view, build an activation query, activate memory, assemble context,
//! and call the model. When the model emits tool calls the graph branches into
//! `execute_tools` and runs a second build/activate/assemble/model pass before
//! persisting. The tail marks session overflow and distills the loop. The node
//! implementations these builders register live in
//! [`nodes`](crate::engine::nodes).

use std::sync::Arc;

use crate::engine::execution_graph::{ExecutionGraph, DEFAULT_EDGE};
use crate::engine::nodes::{
    ActivateMemoryNode, AssembleContextNode, BuildActivationQueryNode, DistillCurrentLoopNode,
    ExecuteToolsNode, IngestUserInputNode, LoadSessionViewNode, MarkSessionOverflowNode, ModelNode,
    PersistAssistantOutputNode, DISTILL_CURRENT_LOOP_NODE, EXECUTE_TOOLS_NODE,
    INGEST_USER_INPUT_NODE, LOAD_SESSION_VIEW_NODE, MARK_SESSION_OVERFLOW_NODE,
    PERSIST_ASSISTANT_OUTPUT_NODE,
};

/// Build the agent's execution graph.
///
/// The graph is a linear chain (`DEFAULT_EDGE` between consecutive nodes) with
/// one branch: each model node also wires a `needs_tools` edge into the shared
/// `execute_tools` node, whose default tail runs a re-activation pass before
/// flowing back into persist. After persisting, the tail marks session overflow
/// and distills the loop.
#[must_use]
pub fn build_default_execution_graph() -> ExecutionGraph {
    let mut graph = ExecutionGraph::new(INGEST_USER_INPUT_NODE);

    // Opening pass.
    graph.add_node(Arc::new(IngestUserInputNode));
    graph.add_node(Arc::new(LoadSessionViewNode));
    graph.add_node(Arc::new(BuildActivationQueryNode {
        id: "build_activation_query",
    }));
    graph.add_node(Arc::new(ActivateMemoryNode {
        id: "activate_memory",
    }));
    graph.add_node(Arc::new(AssembleContextNode {
        id: "assemble_context",
        reload_session: false,
    }));
    graph.add_node(Arc::new(ModelNode { id: "run_model" }));

    graph.add_edge(INGEST_USER_INPUT_NODE, DEFAULT_EDGE, LOAD_SESSION_VIEW_NODE);
    graph.add_edge(
        LOAD_SESSION_VIEW_NODE,
        DEFAULT_EDGE,
        "build_activation_query",
    );
    graph.add_edge("build_activation_query", DEFAULT_EDGE, "activate_memory");
    graph.add_edge("activate_memory", DEFAULT_EDGE, "assemble_context");
    graph.add_edge("assemble_context", DEFAULT_EDGE, "run_model");

    // Post-tool re-activation pass. The opening model and the post-tool model
    // both default-edge into persist and branch into `execute_tools` on
    // `needs_tools`.
    graph.add_node(Arc::new(ExecuteToolsNode));
    graph.add_node(Arc::new(BuildActivationQueryNode {
        id: "build_followup_activation_query",
    }));
    graph.add_node(Arc::new(ActivateMemoryNode {
        id: "reactivate_memory",
    }));
    graph.add_node(Arc::new(AssembleContextNode {
        id: "reassemble_context",
        reload_session: true,
    }));
    graph.add_node(Arc::new(ModelNode {
        id: "run_model_after_tools",
    }));

    graph.add_edge("run_model", "needs_tools", EXECUTE_TOOLS_NODE);
    graph.add_edge(
        EXECUTE_TOOLS_NODE,
        DEFAULT_EDGE,
        "build_followup_activation_query",
    );
    graph.add_edge(
        "build_followup_activation_query",
        DEFAULT_EDGE,
        "reactivate_memory",
    );
    graph.add_edge("reactivate_memory", DEFAULT_EDGE, "reassemble_context");
    graph.add_edge("reassemble_context", DEFAULT_EDGE, "run_model_after_tools");
    graph.add_edge(
        "run_model_after_tools",
        DEFAULT_EDGE,
        PERSIST_ASSISTANT_OUTPUT_NODE,
    );
    graph.add_edge("run_model_after_tools", "needs_tools", EXECUTE_TOOLS_NODE);

    // Opening model flows straight into persist when it emits no tool calls.
    graph.add_edge("run_model", DEFAULT_EDGE, PERSIST_ASSISTANT_OUTPUT_NODE);

    graph.add_node(Arc::new(PersistAssistantOutputNode));

    // Tail: mark overflow, then distill the loop.
    graph.add_node(Arc::new(MarkSessionOverflowNode));
    graph.add_node(Arc::new(DistillCurrentLoopNode));
    graph.add_edge(
        PERSIST_ASSISTANT_OUTPUT_NODE,
        DEFAULT_EDGE,
        MARK_SESSION_OVERFLOW_NODE,
    );
    graph.add_edge(
        MARK_SESSION_OVERFLOW_NODE,
        DEFAULT_EDGE,
        DISTILL_CURRENT_LOOP_NODE,
    );

    graph
}
