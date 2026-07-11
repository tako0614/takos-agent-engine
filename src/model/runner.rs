use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ids::{LoopId, SessionId};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolCallRequest {
    /// Provider or engine assigned correlation id. Keeping this id across the
    /// assistant tool-call and tool-result messages is required by native tool
    /// protocols and prevents parallel calls from being matched by name/order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    #[serde(default)]
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConversationRole {
    System,
    User,
    Assistant,
    Tool,
}

/// Provider-neutral conversation item supplied by the product-owned durable
/// history. The engine transports it but does not own or persist that history.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ConversationMessage {
    pub role: ConversationRole,
    #[serde(default)]
    pub content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallRequest>,
}

#[derive(Debug, Clone)]
pub struct ModelInput {
    pub session_id: SessionId,
    pub loop_id: LoopId,
    pub system_prompt: String,
    pub session_context: Vec<String>,
    pub memory_context: Vec<String>,
    pub tool_context: Vec<String>,
    /// Durable history supplied by the product control plane. This excludes
    /// the current user message, which remains in `user_message`.
    pub conversation_history: Vec<ConversationMessage>,
    /// Structured assistant tool-call/tool-result items produced during the
    /// current turn. Providers receive these after the current user message.
    pub turn_messages: Vec<ConversationMessage>,
    pub user_message: String,
    pub plan: Option<String>,
}

/// Provider-reported token usage for one model call. `input_tokens` is the TOTAL
/// prompt tokens (cached + uncached); `cached_input_tokens` is the cached subset.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ModelUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cached_input_tokens: u32,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelOutput {
    pub assistant_message: Option<String>,
    pub tool_calls: Vec<ToolCallRequest>,
    /// Provider-reported usage for this call, when available. Lets the engine
    /// reconcile its pre-send token estimate against ground truth instead of
    /// flying blind.
    pub usage: Option<ModelUsage>,
}

#[async_trait]
pub trait ModelRunner: Send + Sync {
    async fn run(&self, input: ModelInput) -> Result<ModelOutput>;
}
