use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio_util::sync::CancellationToken;

use crate::error::{EngineError, Result};
use crate::ids::{LoopId, SessionId};
use crate::model::runner::ToolCallRequest;

use super::memory_tools::{
    GraphSearchParams, MemorySearchParams, MemoryTools, ProvenanceLookupParams,
    TimelineSearchParams,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    pub name: String,
    pub content: serde_json::Value,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExecutionKind {
    /// Safe to overlap with adjacent read-only calls from the same model turn.
    ReadOnly,
    /// Must run alone and in provider order.
    SideEffecting,
}

/// Stable execution identity supplied to every tool invocation.
///
/// Side-effecting executors must forward `idempotency_key` to their durable
/// boundary and should observe `cancellation_token` when they own a child
/// process or request. The key is identical when an explicitly-safe recovery
/// retries the same checkpoint.
#[derive(Clone)]
pub struct ToolExecutionContext {
    pub session_id: SessionId,
    pub loop_id: LoopId,
    pub idempotency_key: String,
    pub timeout: Duration,
    /// Maximum serialized result envelope accepted by the engine.
    ///
    /// Remote/process executors should enforce this before materializing a
    /// `serde_json::Value` in this process, returning an artifact reference and
    /// bounded preview when the complete result is larger.
    pub max_result_bytes: usize,
    pub cancellation_token: Option<CancellationToken>,
}

impl std::fmt::Debug for ToolExecutionContext {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ToolExecutionContext")
            .field("session_id", &self.session_id)
            .field("loop_id", &self.loop_id)
            .field("idempotency_key", &self.idempotency_key)
            .field("timeout", &self.timeout)
            .field("max_result_bytes", &self.max_result_bytes)
            .field(
                "cancellation_requested",
                &self
                    .cancellation_token
                    .as_ref()
                    .is_some_and(CancellationToken::is_cancelled),
            )
            .finish()
    }
}

#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// Fail closed: executors must explicitly classify calls as read-only
    /// before the engine will overlap them.
    fn execution_kind(&self, _call: &ToolCallRequest) -> ToolExecutionKind {
        ToolExecutionKind::SideEffecting
    }

    /// Whether a `Running` checkpoint may safely invoke this call again with
    /// the same [`ToolExecutionContext::idempotency_key`]. Read-only calls are
    /// safe by definition; side-effecting executors must opt in only after
    /// their remote boundary durably fences that key.
    fn recovery_is_idempotent(&self, call: &ToolCallRequest) -> bool {
        self.execution_kind(call) == ToolExecutionKind::ReadOnly
    }

    async fn execute(&self, call: ToolCallRequest) -> Result<ToolCallResult>;

    /// Context-aware execution path. Existing executors remain source
    /// compatible through the default delegation, while production executors
    /// can override this to propagate idempotency and cancellation.
    async fn execute_with_context(
        &self,
        _context: ToolExecutionContext,
        call: ToolCallRequest,
    ) -> Result<ToolCallResult> {
        self.execute(call).await
    }
}

pub struct DefaultToolExecutor {
    memory_tools: MemoryTools,
}

impl DefaultToolExecutor {
    #[must_use]
    pub const fn new(memory_tools: MemoryTools) -> Self {
        Self { memory_tools }
    }
}

#[async_trait]
impl ToolExecutor for DefaultToolExecutor {
    fn execution_kind(&self, _call: &ToolCallRequest) -> ToolExecutionKind {
        ToolExecutionKind::ReadOnly
    }

    async fn execute(&self, call: ToolCallRequest) -> Result<ToolCallResult> {
        let tool_call_id = call.id.clone();
        match call.name.as_str() {
            "semantic_search_memory" => {
                let params: MemorySearchParams =
                    serde_json::from_value(call.arguments).map_err(|err| {
                        EngineError::Tool(format!("invalid semantic search args: {err}"))
                    })?;
                let result = self.memory_tools.semantic_search(params).await?;
                let summary = format!(
                    "semantic_search_memory raw_hits={} abstract_hits={}",
                    result.raw_hits.len(),
                    result.abstract_hits.len()
                );
                Ok(ToolCallResult {
                    tool_call_id,
                    name: call.name,
                    content: serde_json::to_value(&result).map_err(|err| {
                        EngineError::Tool(format!("failed to serialize result: {err}"))
                    })?,
                    summary,
                })
            }
            "graph_search_memory" => {
                let params: GraphSearchParams =
                    serde_json::from_value(call.arguments).map_err(|err| {
                        EngineError::Tool(format!("invalid graph search args: {err}"))
                    })?;
                let result = self.memory_tools.graph_search(params).await?;
                let summary = format!("graph_search_memory hits={}", result.hits.len());
                Ok(ToolCallResult {
                    tool_call_id,
                    name: call.name,
                    content: serde_json::to_value(&result).map_err(|err| {
                        EngineError::Tool(format!("failed to serialize result: {err}"))
                    })?,
                    summary,
                })
            }
            "provenance_lookup" => {
                let params: ProvenanceLookupParams = serde_json::from_value(call.arguments)
                    .map_err(|err| EngineError::Tool(format!("invalid provenance args: {err}")))?;
                let result = self.memory_tools.provenance_lookup(params).await?;
                let summary = format!("provenance_lookup raw_nodes={}", result.raw_nodes.len());
                Ok(ToolCallResult {
                    tool_call_id,
                    name: call.name,
                    content: serde_json::to_value(&result).map_err(|err| {
                        EngineError::Tool(format!("failed to serialize result: {err}"))
                    })?,
                    summary,
                })
            }
            "timeline_search" => {
                let params: TimelineSearchParams = serde_json::from_value(call.arguments)
                    .map_err(|err| EngineError::Tool(format!("invalid timeline args: {err}")))?;
                let result = self.memory_tools.timeline_search(params).await?;
                let summary = format!("timeline_search raw_nodes={}", result.raw_nodes.len());
                Ok(ToolCallResult {
                    tool_call_id,
                    name: call.name,
                    content: serde_json::to_value(&result).map_err(|err| {
                        EngineError::Tool(format!("failed to serialize result: {err}"))
                    })?,
                    summary,
                })
            }
            other => Err(EngineError::Tool(format!("unknown tool: {other}"))),
        }
    }
}
