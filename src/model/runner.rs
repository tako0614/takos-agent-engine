use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::ids::{LoopId, SessionId};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
