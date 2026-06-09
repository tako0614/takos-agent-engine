//! Declarative, table-driven construction of the execution-graph presets.
//!
//! The three [`ExecutionGraphPreset`] variants are the same linear pipeline
//! with a few axes flipped, so a single internal graph-spec table drives the
//! graph builder instead of hand-wiring nodes and edges per preset.
//! The node implementations these builders register live in
//! [`nodes`](crate::engine::nodes).

use std::sync::Arc;

use crate::engine::execution_graph::{ExecutionGraph, DEFAULT_EDGE};
use crate::engine::nodes::{
    ActivateMemoryNode, AssembleContextNode, BuildActivationQueryNode, CapturePlannerOutputNode,
    DistillCurrentLoopNode, ExecuteToolsNode, IngestUserInputNode, LoadSessionViewNode,
    MarkSessionOverflowNode, ModelNode, PersistAssistantOutputNode, CAPTURE_PLANNER_OUTPUT_NODE,
    DISTILL_CURRENT_LOOP_NODE, EXECUTE_TOOLS_NODE, INGEST_USER_INPUT_NODE, LOAD_SESSION_VIEW_NODE,
    MARK_SESSION_OVERFLOW_NODE, PERSIST_ASSISTANT_OUTPUT_NODE,
};
use crate::engine::session_engine::ExecutionGraphPreset;

#[must_use]
pub fn build_execution_graph_preset(preset: ExecutionGraphPreset) -> ExecutionGraph {
    match preset {
        ExecutionGraphPreset::Default => build_default_execution_graph(),
        ExecutionGraphPreset::Planner => build_planner_execution_graph(),
        ExecutionGraphPreset::Subgoal => build_subgoal_execution_graph(),
    }
}

/// Ids for the four prefix nodes that vary per preset: build-query, activate,
/// assemble, and model. Every preset runs `ingest -> load -> <these four>`.
struct PrefixIds {
    build_query: &'static str,
    activate: &'static str,
    assemble: &'static str,
    model: &'static str,
}

/// Ids for the post-tool re-activation pass that presets running the tool loop
/// share: `execute_tools -> <these four> -> persist`.
struct ToolLoopIds {
    build_query: &'static str,
    activate: &'static str,
    assemble: &'static str,
    model: &'static str,
}

/// Declarative description of one [`ExecutionGraphPreset`]'s topology. The three
/// presets are the same linear pipeline with a few axes flipped, so the shared
/// [`build_graph_from_spec`] reads this table instead of hand-wiring nodes and
/// edges. Node ids stay `&'static str` so [`crate::engine::execution_graph::GraphNode::id`]
/// keeps returning a `'static` key and `ExecutionGraph` registry slots remain
/// distinct per preset.
struct GraphSpec {
    /// build-query/activate/assemble/model ids for the opening pass.
    prefix: PrefixIds,
    /// `true` when the opening model node may emit tool calls.
    prefix_allow_tools: bool,
    /// `Some` when the preset runs a tool loop after the opening model node.
    tool_loop: Option<ToolLoopIds>,
    /// `true` for the planner, which captures the model output as the plan
    /// before persisting.
    capture_planner: bool,
    /// `true` when persisting the assistant output finishes the loop.
    persist_finish: bool,
    /// `true` for the default preset, whose tail marks overflow and distills.
    overflow_distill_tail: bool,
}

impl GraphSpec {
    fn for_preset(preset: ExecutionGraphPreset) -> Self {
        match preset {
            ExecutionGraphPreset::Default => Self {
                prefix: PrefixIds {
                    build_query: "build_activation_query",
                    activate: "activate_memory",
                    assemble: "assemble_context",
                    model: "run_model",
                },
                prefix_allow_tools: true,
                tool_loop: Some(ToolLoopIds {
                    build_query: "build_followup_activation_query",
                    activate: "reactivate_memory",
                    assemble: "reassemble_context",
                    model: "run_model_after_tools",
                }),
                capture_planner: false,
                persist_finish: false,
                overflow_distill_tail: true,
            },
            ExecutionGraphPreset::Planner => Self {
                prefix: PrefixIds {
                    build_query: "build_planner_activation_query",
                    activate: "activate_planner_memory",
                    assemble: "assemble_planner_context",
                    model: "run_planner_model",
                },
                prefix_allow_tools: false,
                tool_loop: None,
                capture_planner: true,
                persist_finish: true,
                overflow_distill_tail: false,
            },
            ExecutionGraphPreset::Subgoal => Self {
                prefix: PrefixIds {
                    build_query: "build_subgoal_activation_query",
                    activate: "activate_subgoal_memory",
                    assemble: "assemble_subgoal_context",
                    model: "run_subgoal_model",
                },
                prefix_allow_tools: true,
                tool_loop: Some(ToolLoopIds {
                    build_query: "build_subgoal_followup_activation_query",
                    activate: "reactivate_subgoal_memory",
                    assemble: "reassemble_subgoal_context",
                    model: "run_subgoal_model_after_tools",
                }),
                capture_planner: false,
                persist_finish: true,
                overflow_distill_tail: false,
            },
        }
    }
}

/// Build an [`ExecutionGraph`] from a [`GraphSpec`].
///
/// The graph is a linear chain (`DEFAULT_EDGE` between consecutive nodes) with
/// one optional branch: every model node that allows tools also wires a
/// `needs_tools` edge into the shared `execute_tools` node, whose default tail
/// runs the re-activation pass before flowing back into persist.
fn build_graph_from_spec(spec: &GraphSpec) -> ExecutionGraph {
    let mut graph = ExecutionGraph::new(INGEST_USER_INPUT_NODE);

    // Opening pass, shared by every preset.
    graph.add_node(Arc::new(IngestUserInputNode));
    graph.add_node(Arc::new(LoadSessionViewNode));
    graph.add_node(Arc::new(BuildActivationQueryNode {
        id: spec.prefix.build_query,
    }));
    graph.add_node(Arc::new(ActivateMemoryNode {
        id: spec.prefix.activate,
    }));
    graph.add_node(Arc::new(AssembleContextNode {
        id: spec.prefix.assemble,
        reload_session: false,
    }));
    graph.add_node(Arc::new(ModelNode {
        id: spec.prefix.model,
        allow_tools: spec.prefix_allow_tools,
    }));

    graph.add_edge(INGEST_USER_INPUT_NODE, DEFAULT_EDGE, LOAD_SESSION_VIEW_NODE);
    graph.add_edge(
        LOAD_SESSION_VIEW_NODE,
        DEFAULT_EDGE,
        spec.prefix.build_query,
    );
    graph.add_edge(spec.prefix.build_query, DEFAULT_EDGE, spec.prefix.activate);
    graph.add_edge(spec.prefix.activate, DEFAULT_EDGE, spec.prefix.assemble);
    graph.add_edge(spec.prefix.assemble, DEFAULT_EDGE, spec.prefix.model);

    // Optional tool loop. The opening model and the post-tool model both
    // default-edge into persist and branch into `execute_tools` on `needs_tools`.
    if let Some(tool_loop) = &spec.tool_loop {
        graph.add_node(Arc::new(ExecuteToolsNode));
        graph.add_node(Arc::new(BuildActivationQueryNode {
            id: tool_loop.build_query,
        }));
        graph.add_node(Arc::new(ActivateMemoryNode {
            id: tool_loop.activate,
        }));
        graph.add_node(Arc::new(AssembleContextNode {
            id: tool_loop.assemble,
            reload_session: true,
        }));
        graph.add_node(Arc::new(ModelNode {
            id: tool_loop.model,
            allow_tools: true,
        }));

        graph.add_edge(spec.prefix.model, "needs_tools", EXECUTE_TOOLS_NODE);
        graph.add_edge(EXECUTE_TOOLS_NODE, DEFAULT_EDGE, tool_loop.build_query);
        graph.add_edge(tool_loop.build_query, DEFAULT_EDGE, tool_loop.activate);
        graph.add_edge(tool_loop.activate, DEFAULT_EDGE, tool_loop.assemble);
        graph.add_edge(tool_loop.assemble, DEFAULT_EDGE, tool_loop.model);
        graph.add_edge(tool_loop.model, DEFAULT_EDGE, PERSIST_ASSISTANT_OUTPUT_NODE);
        graph.add_edge(tool_loop.model, "needs_tools", EXECUTE_TOOLS_NODE);
    }

    // Optional planner capture between the model and persist.
    let into_persist = if spec.capture_planner {
        graph.add_node(Arc::new(CapturePlannerOutputNode));
        graph.add_edge(spec.prefix.model, DEFAULT_EDGE, CAPTURE_PLANNER_OUTPUT_NODE);
        CAPTURE_PLANNER_OUTPUT_NODE
    } else {
        spec.prefix.model
    };
    graph.add_edge(into_persist, DEFAULT_EDGE, PERSIST_ASSISTANT_OUTPUT_NODE);

    graph.add_node(Arc::new(PersistAssistantOutputNode {
        finish_after_persist: spec.persist_finish,
    }));

    // Default preset tail: mark overflow, then distill the loop.
    if spec.overflow_distill_tail {
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
    }

    graph
}

#[must_use]
pub fn build_default_execution_graph() -> ExecutionGraph {
    build_graph_from_spec(&GraphSpec::for_preset(ExecutionGraphPreset::Default))
}

#[must_use]
pub fn build_planner_execution_graph() -> ExecutionGraph {
    build_graph_from_spec(&GraphSpec::for_preset(ExecutionGraphPreset::Planner))
}

#[must_use]
pub fn build_subgoal_execution_graph() -> ExecutionGraph {
    build_graph_from_spec(&GraphSpec::for_preset(ExecutionGraphPreset::Subgoal))
}
