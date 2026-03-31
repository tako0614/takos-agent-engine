use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::domain::{
    AbstractNode, AbstractNodeMetadata, DistillationState, EntityRef, GraphFragment, RawNode,
    RawNodeKind, References, Relation,
};
use crate::error::Result;
use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};
use crate::storage::RawLifecyclePatch;

#[derive(Debug, Clone)]
pub struct DistillationInput {
    pub session_id: SessionId,
    pub loop_id: LoopId,
    pub raw_nodes: Vec<RawNode>,
    pub activated_abstract_ids: Vec<AbstractNodeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawLifecycleUpdate {
    pub raw_node_id: RawNodeId,
    pub patch: RawLifecyclePatch,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DistillationOutput {
    pub new_nodes: Vec<AbstractNode>,
    pub raw_updates: Vec<RawLifecycleUpdate>,
}

#[async_trait]
pub trait Distiller: Send + Sync {
    async fn distill(&self, input: DistillationInput) -> Result<DistillationOutput>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleDistiller;

#[async_trait]
impl Distiller for SimpleDistiller {
    async fn distill(&self, input: DistillationInput) -> Result<DistillationOutput> {
        if input.raw_nodes.is_empty() {
            return Ok(DistillationOutput::default());
        }

        let user_request = input
            .raw_nodes
            .iter()
            .find(|node| node.kind == RawNodeKind::UserUtterance)
            .map(|node| node.content_text())
            .unwrap_or_else(|| "Untitled session".to_string());

        let assistant_summary = input
            .raw_nodes
            .iter()
            .find(|node| node.kind == RawNodeKind::AssistantUtterance)
            .map(|node| node.content_text())
            .unwrap_or_else(|| "No assistant output yet.".to_string());

        let mut entities = vec![
            EntityRef {
                id: input.session_id.to_string(),
                label: "session".to_string(),
            },
            EntityRef {
                id: input.loop_id.to_string(),
                label: "loop".to_string(),
            },
        ];

        let mut relations = vec![Relation {
            subject: input.session_id.to_string(),
            predicate: "contains_loop".to_string(),
            object: input.loop_id.to_string(),
            weight: 1.0,
            provenance_raw_node_ids: input.raw_nodes.iter().map(|node| node.id).collect(),
        }];

        for node in &input.raw_nodes {
            entities.push(EntityRef {
                id: node.id.to_string(),
                label: format!("raw:{:?}", node.kind),
            });
            relations.push(Relation {
                subject: input.loop_id.to_string(),
                predicate: match node.kind {
                    RawNodeKind::UserUtterance => "captures_request".to_string(),
                    RawNodeKind::AssistantUtterance => "captures_response".to_string(),
                    RawNodeKind::ToolResult => "records_tool_result".to_string(),
                    RawNodeKind::Note => "records_note".to_string(),
                    RawNodeKind::Event => "records_event".to_string(),
                },
                object: node.id.to_string(),
                weight: 0.8,
                provenance_raw_node_ids: vec![node.id],
            });

            if node.kind == RawNodeKind::ToolResult {
                relations.push(Relation {
                    subject: node.metadata.source.clone(),
                    predicate: "produced".to_string(),
                    object: node.id.to_string(),
                    weight: 0.7,
                    provenance_raw_node_ids: vec![node.id],
                });
            }
        }

        for abstract_id in &input.activated_abstract_ids {
            relations.push(Relation {
                subject: input.loop_id.to_string(),
                predicate: "informed_by".to_string(),
                object: abstract_id.to_string(),
                weight: 0.85,
                provenance_raw_node_ids: input.raw_nodes.iter().map(|node| node.id).collect(),
            });
        }

        entities.sort_by(|left, right| {
            left.id
                .cmp(&right.id)
                .then_with(|| left.label.cmp(&right.label))
        });
        relations.sort_by(|left, right| {
            left.subject
                .cmp(&right.subject)
                .then_with(|| left.predicate.cmp(&right.predicate))
                .then_with(|| left.object.cmp(&right.object))
        });

        let abstract_node = AbstractNode::new(
            truncate_title(&user_request),
            assistant_summary,
            References {
                abstract_node_ids: input.activated_abstract_ids.clone(),
                raw_node_ids: input.raw_nodes.iter().map(|node| node.id).collect(),
            },
            GraphFragment {
                entities,
                relations,
            },
            AbstractNodeMetadata {
                abstraction_level: 1,
                confidence: 0.68,
                importance: 0.72,
                tags: vec!["distilled".to_string(), "loop".to_string()],
            },
        )
        .with_operation_key(format!("loop:{}:abstract:primary", input.loop_id));

        let raw_updates = input
            .raw_nodes
            .iter()
            .map(|node| RawLifecycleUpdate {
                raw_node_id: node.id,
                patch: RawLifecyclePatch {
                    distillation_state: Some(DistillationState::Distilled),
                    overflow: Some(crate::domain::OverflowPolicy {
                        was_pushed_out_of_session: false,
                        relax_retrieval_until: None,
                    }),
                },
            })
            .collect();

        Ok(DistillationOutput {
            new_nodes: vec![abstract_node],
            raw_updates,
        })
    }
}

fn truncate_title(source: &str) -> String {
    let trimmed = source.trim();
    if trimmed.len() <= 64 {
        trimmed.to_string()
    } else {
        format!("{}...", &trimmed[..61])
    }
}
