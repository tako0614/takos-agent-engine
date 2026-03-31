use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::error::Result;
use crate::ids::{LoopId, SessionId};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallRequest {
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ModelInput {
    pub session_id: SessionId,
    pub loop_id: LoopId,
    pub system_prompt: String,
    pub session_context: Vec<String>,
    pub memory_context: Vec<String>,
    pub tool_context: Vec<String>,
    pub user_message: String,
    pub plan: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelOutput {
    pub assistant_message: Option<String>,
    pub tool_calls: Vec<ToolCallRequest>,
}

#[async_trait]
pub trait ModelRunner: Send + Sync {
    async fn run(&self, input: ModelInput) -> Result<ModelOutput>;
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RuleBasedModelRunner;

#[async_trait]
impl ModelRunner for RuleBasedModelRunner {
    async fn run(&self, input: ModelInput) -> Result<ModelOutput> {
        if input.tool_context.is_empty() {
            if let Some(query) = input.user_message.strip_prefix("memory:") {
                return Ok(ModelOutput {
                    assistant_message: None,
                    tool_calls: vec![ToolCallRequest {
                        name: "semantic_search_memory".to_string(),
                        arguments: json!({
                            "query": query.trim(),
                            "target": "both",
                            "top_k": 4
                        }),
                    }],
                });
            }

            if input.user_message.starts_with("timeline:") {
                return Ok(ModelOutput {
                    assistant_message: None,
                    tool_calls: vec![ToolCallRequest {
                        name: "timeline_search".to_string(),
                        arguments: json!({
                            "limit": 8
                        }),
                    }],
                });
            }
        }

        let mut lines = Vec::new();
        lines.push(format!(
            "session={} loop={}",
            input.session_id, input.loop_id
        ));
        lines.push(format!(
            "system_prompt_tokens={}",
            input.system_prompt.split_whitespace().count()
        ));
        lines.push(format!(
            "recent_session_items={}",
            input.session_context.len()
        ));
        if let Some(plan) = &input.plan {
            lines.push(format!("plan={plan}"));
        }
        if !input.memory_context.is_empty() {
            lines.push(format!("memory_hits={}", input.memory_context.len()));
        }
        if !input.tool_context.is_empty() {
            lines.push(format!("tool_findings={}", input.tool_context.join(" | ")));
        }
        lines.push(format!("user={}", input.user_message));

        Ok(ModelOutput {
            assistant_message: Some(lines.join("\n")),
            tool_calls: Vec::new(),
        })
    }
}
