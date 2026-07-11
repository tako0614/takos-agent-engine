//! Construction of the agent's execution graph.
//!
//! The memory-aware default ingests the user input, activates local memory,
//! assembles context, and distills the completed loop. The external-context
//! profile skips those local-memory nodes and runs a minimal model/tool loop for
//! wrappers that supply canonical durable context. The node implementations
//! these builders register live in
//! [`nodes`](crate::engine::nodes).

use std::sync::Arc;

use crate::engine::execution_graph::{ExecutionGraph, DEFAULT_EDGE};
use crate::engine::nodes::{
    ActivateMemoryNode, AssembleContextNode, AssembleExternalContextNode, BuildActivationQueryNode,
    DistillCurrentLoopNode, ExecuteToolsNode, FinalizeExternalResponseNode, IngestUserInputNode,
    LoadSessionViewNode, MarkSessionOverflowNode, ModelNode, PersistAssistantOutputNode,
    ASSEMBLE_EXTERNAL_CONTEXT_NODE, DISTILL_CURRENT_LOOP_NODE, EXECUTE_TOOLS_NODE,
    FINALIZE_EXTERNAL_RESPONSE_NODE, INGEST_USER_INPUT_NODE, LOAD_SESSION_VIEW_NODE,
    MARK_SESSION_OVERFLOW_NODE, PERSIST_ASSISTANT_OUTPUT_NODE,
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

/// Build the lean graph for wrappers that own canonical history and memory.
///
/// This topology intentionally contains no ingest, local session load, memory
/// activation, context-window overflow, assistant persistence, or distillation
/// nodes. It only prepares the external-context model input, runs the bounded
/// model/tool loop, and returns the structured current-turn transcript.
#[must_use]
pub fn build_external_context_execution_graph() -> ExecutionGraph {
    const RUN_MODEL: &str = "run_model_external_context";
    const RUN_MODEL_AFTER_TOOLS: &str = "run_model_external_context_after_tools";

    let mut graph = ExecutionGraph::new(ASSEMBLE_EXTERNAL_CONTEXT_NODE);
    graph.add_node(Arc::new(AssembleExternalContextNode));
    graph.add_node(Arc::new(ModelNode { id: RUN_MODEL }));
    graph.add_node(Arc::new(ExecuteToolsNode));
    graph.add_node(Arc::new(ModelNode {
        id: RUN_MODEL_AFTER_TOOLS,
    }));
    graph.add_node(Arc::new(FinalizeExternalResponseNode));

    graph.add_edge(ASSEMBLE_EXTERNAL_CONTEXT_NODE, DEFAULT_EDGE, RUN_MODEL);
    graph.add_edge(RUN_MODEL, "needs_tools", EXECUTE_TOOLS_NODE);
    graph.add_edge(RUN_MODEL, DEFAULT_EDGE, FINALIZE_EXTERNAL_RESPONSE_NODE);
    graph.add_edge(EXECUTE_TOOLS_NODE, DEFAULT_EDGE, RUN_MODEL_AFTER_TOOLS);
    graph.add_edge(RUN_MODEL_AFTER_TOOLS, "needs_tools", EXECUTE_TOOLS_NODE);
    graph.add_edge(
        RUN_MODEL_AFTER_TOOLS,
        DEFAULT_EDGE,
        FINALIZE_EXTERNAL_RESPONSE_NODE,
    );

    graph
}
