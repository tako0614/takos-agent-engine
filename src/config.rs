use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};

const MAX_COLLECTION_BOUND: usize = 4_096;
const MAX_CONTEXT_TOKENS: usize = 1_000_000;
const MAX_TOOL_PAYLOAD_BYTES: usize = 16 * 1024 * 1024;
const MIN_TOOL_RESULT_BYTES: usize = 4 * 1024;
const MAX_TOOL_CALLS_PER_ROUND: usize = 256;
const MAX_TOOL_ROUNDS: u32 = 256;
const MAX_GRAPH_STEPS: u32 = 100_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub system_prompt: String,
    pub memory: MemoryConfig,
    pub context_budget: ContextBudgetConfig,
    pub tools: ToolsConfig,
    pub runtime: RuntimeConfig,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            system_prompt: "You are the Rust-based Takos agent engine.".to_string(),
            memory: MemoryConfig::default(),
            context_budget: ContextBudgetConfig::default(),
            tools: ToolsConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }
}

impl EngineConfig {
    /// # Errors
    ///
    /// Returns an [`EngineError::Configuration`] when any field falls outside
    /// its allowed range — empty system prompt, ratios outside `0..=1`, ratios
    /// that do not sum to 1, or any of the integer minimums set to zero.
    pub fn validate(&self) -> Result<()> {
        if self.system_prompt.trim().is_empty() {
            return Err(EngineError::Configuration(
                "system_prompt must not be empty".to_string(),
            ));
        }
        let split = self.context_budget.session_ratio + self.context_budget.memory_ratio;
        if !(0.0..=1.0).contains(&self.context_budget.session_ratio) {
            return Err(EngineError::Configuration(
                "context_budget.session_ratio must be between 0 and 1".to_string(),
            ));
        }
        if !(0.0..=1.0).contains(&self.context_budget.memory_ratio) {
            return Err(EngineError::Configuration(
                "context_budget.memory_ratio must be between 0 and 1".to_string(),
            ));
        }
        if (split - 1.0).abs() > 0.001 {
            return Err(EngineError::Configuration(
                "context_budget.session_ratio + memory_ratio must equal 1".to_string(),
            ));
        }
        self.memory.activation.validated_budgets()?;
        validate_threshold(
            "memory.retrieval.similarity_threshold.raw",
            self.memory.retrieval.similarity_threshold.raw,
        )?;
        validate_threshold(
            "memory.retrieval.similarity_threshold.abstract",
            self.memory.retrieval.similarity_threshold.abstract_nodes,
        )?;
        validate_threshold(
            "memory.retrieval.relaxed_threshold_for_pushed_raw",
            self.memory.retrieval.relaxed_threshold_for_pushed_raw,
        )?;
        if self.memory.retrieval.relaxed_threshold_for_pushed_raw
            > self.memory.retrieval.similarity_threshold.raw
        {
            return Err(EngineError::Configuration(
                "memory.retrieval.relaxed_threshold_for_pushed_raw must not exceed the normal raw threshold"
                    .to_string(),
            ));
        }
        if self.context_budget.total_tokens == 0
            || self.context_budget.total_tokens > MAX_CONTEXT_TOKENS
        {
            return Err(EngineError::Configuration(format!(
                "context_budget.total_tokens must be between 1 and {MAX_CONTEXT_TOKENS}"
            )));
        }
        let reserved = self
            .context_budget
            .reserve_system
            .checked_add(self.context_budget.reserve_tools)
            .and_then(|value| value.checked_add(self.context_budget.reserve_working))
            .ok_or_else(|| {
                EngineError::Configuration(
                    "context_budget reserve sum must not overflow".to_string(),
                )
            })?;
        if reserved > self.context_budget.total_tokens {
            return Err(EngineError::Configuration(
                "context_budget reserves must not exceed total_tokens".to_string(),
            ));
        }
        if self.runtime.max_graph_steps == 0 {
            return Err(EngineError::Configuration(
                "runtime.max_graph_steps must be greater than 0".to_string(),
            ));
        }
        if self.runtime.max_graph_steps > MAX_GRAPH_STEPS {
            return Err(EngineError::Configuration(format!(
                "runtime.max_graph_steps must not exceed {MAX_GRAPH_STEPS}"
            )));
        }
        if self.runtime.max_tool_rounds == 0 {
            return Err(EngineError::Configuration(
                "runtime.max_tool_rounds must be greater than 0".to_string(),
            ));
        }
        if self.runtime.max_tool_rounds > MAX_TOOL_ROUNDS {
            return Err(EngineError::Configuration(format!(
                "runtime.max_tool_rounds must not exceed {MAX_TOOL_ROUNDS}"
            )));
        }
        if self.runtime.maintenance_batch_size == 0 {
            return Err(EngineError::Configuration(
                "runtime.maintenance_batch_size must be greater than 0".to_string(),
            ));
        }
        for (name, value) in [
            (
                "runtime.maintenance_batch_size",
                self.runtime.maintenance_batch_size,
            ),
            (
                "runtime.max_tool_calls_per_round",
                self.runtime.max_tool_calls_per_round,
            ),
            ("runtime.max_session_nodes", self.runtime.max_session_nodes),
            ("runtime.max_loop_nodes", self.runtime.max_loop_nodes),
        ] {
            validate_collection_bound(name, value)?;
        }
        if self.runtime.max_tool_calls_per_round > MAX_TOOL_CALLS_PER_ROUND {
            return Err(EngineError::Configuration(format!(
                "runtime.max_tool_calls_per_round must not exceed {MAX_TOOL_CALLS_PER_ROUND}"
            )));
        }
        if self.tools.max_memory_search_top_k == 0 {
            return Err(EngineError::Configuration(
                "tools.max_memory_search_top_k must be greater than 0".to_string(),
            ));
        }
        if self.tools.max_graph_search_depth == 0 {
            return Err(EngineError::Configuration(
                "tools.max_graph_search_depth must be greater than 0".to_string(),
            ));
        }
        if self.tools.max_timeline_search_limit == 0 {
            return Err(EngineError::Configuration(
                "tools.max_timeline_search_limit must be greater than 0".to_string(),
            ));
        }
        for (name, value) in [
            (
                "tools.max_memory_search_top_k",
                self.tools.max_memory_search_top_k,
            ),
            (
                "tools.max_graph_search_depth",
                self.tools.max_graph_search_depth,
            ),
            (
                "tools.max_graph_search_hits",
                self.tools.max_graph_search_hits,
            ),
            (
                "tools.max_provenance_raw_nodes",
                self.tools.max_provenance_raw_nodes,
            ),
            (
                "tools.max_timeline_search_limit",
                self.tools.max_timeline_search_limit,
            ),
        ] {
            validate_collection_bound(name, value)?;
        }
        for (name, value) in [
            (
                "tools.max_tool_argument_bytes",
                self.tools.max_tool_argument_bytes,
            ),
            (
                "tools.max_tool_result_bytes",
                self.tools.max_tool_result_bytes,
            ),
        ] {
            let minimum = if name == "tools.max_tool_result_bytes" {
                MIN_TOOL_RESULT_BYTES
            } else {
                1
            };
            if value < minimum || value > MAX_TOOL_PAYLOAD_BYTES {
                return Err(EngineError::Configuration(format!(
                    "{name} must be between {minimum} and {MAX_TOOL_PAYLOAD_BYTES}"
                )));
            }
        }
        if self.runtime.model_timeout_ms == 0 {
            return Err(EngineError::Configuration(
                "runtime.model_timeout_ms must be greater than 0".to_string(),
            ));
        }
        for (name, value) in [
            ("runtime.node_timeout_ms", self.runtime.node_timeout_ms),
            ("runtime.tool_timeout_ms", self.runtime.tool_timeout_ms),
            (
                "runtime.distillation_timeout_ms",
                self.runtime.distillation_timeout_ms,
            ),
        ] {
            if value == 0 {
                return Err(EngineError::Configuration(format!(
                    "{name} must be greater than 0"
                )));
            }
        }
        if self.runtime.max_tool_calls_per_round == 0 {
            return Err(EngineError::Configuration(
                "runtime.max_tool_calls_per_round must be greater than 0".to_string(),
            ));
        }
        Ok(())
    }
}

fn validate_collection_bound(name: &str, value: usize) -> Result<()> {
    if value == 0 || value > MAX_COLLECTION_BOUND {
        return Err(EngineError::Configuration(format!(
            "{name} must be between 1 and {MAX_COLLECTION_BOUND}"
        )));
    }
    Ok(())
}

fn validate_threshold(name: &str, value: f32) -> Result<()> {
    if !value.is_finite() || !(-1.0..=1.0).contains(&value) {
        return Err(EngineError::Configuration(format!(
            "{name} must be finite and between -1 and 1"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub activation: ActivationConfig,
    pub retrieval: RetrievalConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationConfig {
    pub target_ratio: ActivationTargetRatio,
    pub top_k_total: usize,
    pub use_time_decay: bool,
    pub overflow_raw_threshold_relaxation: bool,
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            target_ratio: ActivationTargetRatio::default(),
            top_k_total: 20,
            use_time_decay: true,
            overflow_raw_threshold_relaxation: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActivationBudgets {
    pub raw: usize,
    pub abstract_nodes: usize,
    pub search_window: usize,
}

impl ActivationConfig {
    pub(crate) fn validated_budgets(&self) -> Result<ActivationBudgets> {
        let raw_ratio = self.target_ratio.raw;
        let abstract_ratio = self.target_ratio.abstract_nodes;
        if raw_ratio == 0 || abstract_ratio == 0 {
            return Err(EngineError::Configuration(
                "memory.activation target ratios must both be greater than 0".to_string(),
            ));
        }
        if self.top_k_total < 2 || self.top_k_total > MAX_COLLECTION_BOUND {
            return Err(EngineError::Configuration(format!(
                "memory.activation.top_k_total must be between 2 and {MAX_COLLECTION_BOUND}"
            )));
        }
        let total_ratio = raw_ratio.checked_add(abstract_ratio).ok_or_else(|| {
            EngineError::Configuration(
                "memory.activation target ratio sum must not overflow".to_string(),
            )
        })?;
        let weighted_raw = self.top_k_total.checked_mul(raw_ratio).ok_or_else(|| {
            EngineError::Configuration(
                "memory.activation weighted raw budget must not overflow".to_string(),
            )
        })?;
        let raw = (weighted_raw / total_ratio)
            .max(1)
            .min(self.top_k_total - 1);
        let abstract_nodes = self.top_k_total - raw;
        let search_window = self.top_k_total.checked_mul(2).ok_or_else(|| {
            EngineError::Configuration(
                "memory.activation search window must not overflow".to_string(),
            )
        })?;
        Ok(ActivationBudgets {
            raw,
            abstract_nodes,
            search_window,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationTargetRatio {
    pub raw: usize,
    #[serde(rename = "abstract")]
    pub abstract_nodes: usize,
}

impl Default for ActivationTargetRatio {
    fn default() -> Self {
        Self {
            raw: 1,
            abstract_nodes: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalConfig {
    pub similarity_threshold: SimilarityThresholdConfig,
    pub relaxed_threshold_for_pushed_raw: f32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: SimilarityThresholdConfig::default(),
            relaxed_threshold_for_pushed_raw: 0.63,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarityThresholdConfig {
    pub raw: f32,
    #[serde(rename = "abstract")]
    pub abstract_nodes: f32,
}

impl Default for SimilarityThresholdConfig {
    fn default() -> Self {
        Self {
            raw: 0.72,
            abstract_nodes: 0.74,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBudgetConfig {
    pub total_tokens: usize,
    pub reserve_system: usize,
    pub reserve_tools: usize,
    pub reserve_working: usize,
    pub session_ratio: f32,
    pub memory_ratio: f32,
}

impl Default for ContextBudgetConfig {
    fn default() -> Self {
        Self {
            total_tokens: 64_000,
            reserve_system: 4_000,
            reserve_tools: 12_000,
            reserve_working: 8_000,
            session_ratio: 0.5,
            memory_ratio: 0.5,
        }
    }
}

impl ContextBudgetConfig {
    #[must_use]
    pub const fn remaining_tokens(&self) -> usize {
        self.total_tokens
            .saturating_sub(self.reserve_system)
            .saturating_sub(self.reserve_tools)
            .saturating_sub(self.reserve_working)
    }
}

// Each bool toggles an independent tool; serde-mapped from TOML scalars, so
// a bitflags struct would obscure the operator-facing configuration shape.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolsConfig {
    pub memory_search: bool,
    pub graph_search: bool,
    pub provenance_lookup: bool,
    pub timeline_search: bool,
    #[serde(default = "default_max_memory_search_top_k")]
    pub max_memory_search_top_k: usize,
    #[serde(default = "default_max_graph_search_depth")]
    pub max_graph_search_depth: usize,
    #[serde(default = "default_max_graph_search_hits")]
    pub max_graph_search_hits: usize,
    #[serde(default = "default_max_provenance_raw_nodes")]
    pub max_provenance_raw_nodes: usize,
    #[serde(default = "default_max_timeline_search_limit")]
    pub max_timeline_search_limit: usize,
    #[serde(default = "default_max_tool_argument_bytes")]
    pub max_tool_argument_bytes: usize,
    #[serde(default = "default_max_tool_result_bytes")]
    pub max_tool_result_bytes: usize,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        Self {
            memory_search: true,
            graph_search: true,
            provenance_lookup: true,
            timeline_search: true,
            max_memory_search_top_k: default_max_memory_search_top_k(),
            max_graph_search_depth: default_max_graph_search_depth(),
            max_graph_search_hits: default_max_graph_search_hits(),
            max_provenance_raw_nodes: default_max_provenance_raw_nodes(),
            max_timeline_search_limit: default_max_timeline_search_limit(),
            max_tool_argument_bytes: default_max_tool_argument_bytes(),
            max_tool_result_bytes: default_max_tool_result_bytes(),
        }
    }
}

pub(crate) const fn default_max_memory_search_top_k() -> usize {
    32
}

pub(crate) const fn default_max_graph_search_depth() -> usize {
    4
}

pub(crate) const fn default_max_graph_search_hits() -> usize {
    128
}

pub(crate) const fn default_max_provenance_raw_nodes() -> usize {
    128
}

pub(crate) const fn default_max_timeline_search_limit() -> usize {
    100
}

pub(crate) const fn default_max_tool_argument_bytes() -> usize {
    256 * 1024
}

pub(crate) const fn default_max_tool_result_bytes() -> usize {
    1024 * 1024
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub max_graph_steps: u32,
    pub max_tool_rounds: u32,
    pub node_timeout_ms: u64,
    /// Per-node budget for model inference. Model completions routinely take
    /// longer than a `Standard` node, so embedders can align this dedicated
    /// timeout with their provider transport. `#[serde(default)]` keeps older
    /// serialized configs compatible with the library default.
    #[serde(default = "default_model_timeout_ms")]
    pub model_timeout_ms: u64,
    pub tool_timeout_ms: u64,
    pub distillation_timeout_ms: u64,
    pub maintenance_batch_size: usize,
    /// Hard cap on the number of tool calls processed in a single model round.
    /// The model's tool-call list is influenced by injectable content (memory,
    /// prior tool results, the user message), so an unbounded list could fan out
    /// thousands of concurrent tasks / control-plane RPCs. Excess calls beyond
    /// this cap are dropped for the round. `#[serde(default)]` keeps older
    /// configs deserializing onto the default.
    #[serde(default = "default_max_tool_calls_per_round")]
    pub max_tool_calls_per_round: usize,
    #[serde(default = "default_max_session_nodes")]
    pub max_session_nodes: usize,
    #[serde(default = "default_max_loop_nodes")]
    pub max_loop_nodes: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_graph_steps: 64,
            max_tool_rounds: 8,
            node_timeout_ms: 10_000,
            model_timeout_ms: default_model_timeout_ms(),
            tool_timeout_ms: 30_000,
            distillation_timeout_ms: 15_000,
            maintenance_batch_size: 32,
            max_tool_calls_per_round: default_max_tool_calls_per_round(),
            max_session_nodes: default_max_session_nodes(),
            max_loop_nodes: default_max_loop_nodes(),
        }
    }
}

pub(crate) const fn default_model_timeout_ms() -> u64 {
    60_000
}

pub(crate) const fn default_max_tool_calls_per_round() -> usize {
    16
}

pub(crate) const fn default_max_session_nodes() -> usize {
    512
}

pub(crate) const fn default_max_loop_nodes() -> usize {
    256
}

impl RuntimeConfig {
    #[must_use]
    pub const fn node_timeout(&self) -> Duration {
        Duration::from_millis(self.node_timeout_ms)
    }

    #[must_use]
    pub const fn model_timeout(&self) -> Duration {
        Duration::from_millis(self.model_timeout_ms)
    }

    #[must_use]
    pub const fn tool_timeout(&self) -> Duration {
        Duration::from_millis(self.tool_timeout_ms)
    }

    #[must_use]
    pub const fn distillation_timeout(&self) -> Duration {
        Duration::from_millis(self.distillation_timeout_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::EngineConfig;

    #[test]
    fn default_config_is_valid() {
        let config = EngineConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn config_rejects_zero_tool_bounds() {
        let mut config = EngineConfig::default();
        config.tools.max_memory_search_top_k = 0;
        assert!(config.validate().is_err());

        let mut config = EngineConfig::default();
        config.tools.max_graph_search_depth = 0;
        assert!(config.validate().is_err());

        let mut config = EngineConfig::default();
        config.tools.max_timeline_search_limit = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_rejects_overflowing_activation_and_context_budgets() {
        let mut config = EngineConfig::default();
        config.memory.activation.target_ratio.raw = usize::MAX;
        assert!(config.validate().is_err());

        let mut config = EngineConfig::default();
        config.memory.activation.top_k_total = usize::MAX;
        assert!(config.validate().is_err());

        let mut config = EngineConfig::default();
        config.context_budget.total_tokens = 100;
        config.context_budget.reserve_system = usize::MAX;
        assert!(config.validate().is_err());

        let mut config = EngineConfig::default();
        config.context_budget.total_tokens = 100;
        config.context_budget.reserve_system = 40;
        config.context_budget.reserve_tools = 40;
        config.context_budget.reserve_working = 40;
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_rejects_invalid_thresholds_and_zero_timeouts() {
        let mut config = EngineConfig::default();
        config.memory.retrieval.similarity_threshold.raw = f32::NAN;
        assert!(config.validate().is_err());

        let mut config = EngineConfig::default();
        config.memory.retrieval.relaxed_threshold_for_pushed_raw = 0.9;
        assert!(config.validate().is_err());

        for clear in [
            |runtime: &mut super::RuntimeConfig| runtime.node_timeout_ms = 0,
            |runtime: &mut super::RuntimeConfig| runtime.tool_timeout_ms = 0,
            |runtime: &mut super::RuntimeConfig| runtime.distillation_timeout_ms = 0,
        ] {
            let mut config = EngineConfig::default();
            clear(&mut config.runtime);
            assert!(config.validate().is_err());
        }
    }
}
