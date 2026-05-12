use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::{LoopId, RawNodeId, SessionId};
use crate::model::embedding::EmbeddingRef;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RawNodeKind {
    UserUtterance,
    AssistantUtterance,
    ToolResult,
    Note,
    Event,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum RawContent {
    Text(String),
    Json(serde_json::Value),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    Normal,
    Hidden,
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DistillationState {
    Undistilled,
    Distilled,
    PartiallyDistilled,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OverflowPolicy {
    pub was_pushed_out_of_session: bool,
    pub relax_retrieval_until: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawNodeMetadata {
    pub source: String,
    pub tags: Vec<String>,
    pub importance: f32,
    pub visibility: Visibility,
}

impl Default for RawNodeMetadata {
    fn default() -> Self {
        Self {
            source: "system".to_string(),
            tags: Vec::new(),
            importance: 0.5,
            visibility: Visibility::Normal,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RawNode {
    pub id: RawNodeId,
    pub operation_key: Option<String>,
    pub session_id: Option<SessionId>,
    pub loop_id: Option<LoopId>,
    pub timestamp: DateTime<Utc>,
    pub kind: RawNodeKind,
    pub content: RawContent,
    pub embedding_ref: EmbeddingRef,
    pub metadata: RawNodeMetadata,
    pub distillation_state: DistillationState,
    pub overflow: OverflowPolicy,
}

impl RawNode {
    pub fn text(
        kind: RawNodeKind,
        session_id: Option<SessionId>,
        loop_id: Option<LoopId>,
        source: impl Into<String>,
        text: impl Into<String>,
        importance: f32,
        tags: Vec<String>,
    ) -> Self {
        let id = RawNodeId::new();
        Self {
            id,
            operation_key: None,
            session_id,
            loop_id,
            timestamp: Utc::now(),
            kind,
            content: RawContent::Text(text.into()),
            embedding_ref: EmbeddingRef::for_node("raw", id.to_string()),
            metadata: RawNodeMetadata {
                source: source.into(),
                tags,
                importance,
                visibility: Visibility::Normal,
            },
            distillation_state: DistillationState::Undistilled,
            overflow: OverflowPolicy::default(),
        }
    }

    pub fn json(
        kind: RawNodeKind,
        session_id: Option<SessionId>,
        loop_id: Option<LoopId>,
        source: impl Into<String>,
        value: serde_json::Value,
        importance: f32,
        tags: Vec<String>,
    ) -> Self {
        let id = RawNodeId::new();
        Self {
            id,
            operation_key: None,
            session_id,
            loop_id,
            timestamp: Utc::now(),
            kind,
            content: RawContent::Json(value),
            embedding_ref: EmbeddingRef::for_node("raw", id.to_string()),
            metadata: RawNodeMetadata {
                source: source.into(),
                tags,
                importance,
                visibility: Visibility::Normal,
            },
            distillation_state: DistillationState::Undistilled,
            overflow: OverflowPolicy::default(),
        }
    }

    #[must_use]
    pub fn content_text(&self) -> String {
        match &self.content {
            RawContent::Text(text) => text.clone(),
            RawContent::Json(value) => value.to_string(),
        }
    }

    pub fn with_operation_key(mut self, operation_key: impl Into<String>) -> Self {
        self.operation_key = Some(operation_key.into());
        self
    }

    #[must_use]
    pub fn context_text(&self) -> String {
        format!("{:?}: {}", self.kind, self.content_text())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_constructor_user_utterance() {
        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "hello world",
            0.8,
            vec!["greeting".to_string()],
        );
        assert_eq!(node.kind, RawNodeKind::UserUtterance);
        assert_eq!(node.content, RawContent::Text("hello world".to_string()));
        assert_eq!(node.metadata.source, "user");
        assert!((node.metadata.importance - 0.8).abs() < f32::EPSILON);
        assert_eq!(node.metadata.tags, vec!["greeting".to_string()]);
        assert_eq!(node.metadata.visibility, Visibility::Normal);
        assert_eq!(node.distillation_state, DistillationState::Undistilled);
        assert!(!node.overflow.was_pushed_out_of_session);
        assert!(node.overflow.relax_retrieval_until.is_none());
        assert!(node.session_id.is_none());
        assert!(node.loop_id.is_none());
        assert!(node.operation_key.is_none());
    }

    #[test]
    fn text_constructor_assistant_utterance() {
        let sid = SessionId::new();
        let lid = LoopId::new();
        let node = RawNode::text(
            RawNodeKind::AssistantUtterance,
            Some(sid),
            Some(lid),
            "assistant",
            "I can help",
            0.6,
            Vec::new(),
        );
        assert_eq!(node.kind, RawNodeKind::AssistantUtterance);
        assert_eq!(node.session_id, Some(sid));
        assert_eq!(node.loop_id, Some(lid));
    }

    #[test]
    fn text_constructor_tool_result() {
        let node = RawNode::text(
            RawNodeKind::ToolResult,
            None,
            None,
            "search_tool",
            "found 3 results",
            0.7,
            vec!["tool".to_string()],
        );
        assert_eq!(node.kind, RawNodeKind::ToolResult);
        assert_eq!(node.metadata.source, "search_tool");
    }

    #[test]
    fn text_constructor_note() {
        let node = RawNode::text(
            RawNodeKind::Note,
            None,
            None,
            "system",
            "internal note",
            0.3,
            Vec::new(),
        );
        assert_eq!(node.kind, RawNodeKind::Note);
    }

    #[test]
    fn text_constructor_event() {
        let node = RawNode::text(
            RawNodeKind::Event,
            None,
            None,
            "system",
            "session started",
            0.5,
            Vec::new(),
        );
        assert_eq!(node.kind, RawNodeKind::Event);
    }

    #[test]
    fn json_constructor() {
        let value = json!({"key": "value", "count": 42});
        let node = RawNode::json(
            RawNodeKind::ToolResult,
            None,
            None,
            "api",
            value.clone(),
            0.9,
            vec!["json".to_string()],
        );
        assert_eq!(node.kind, RawNodeKind::ToolResult);
        assert_eq!(node.content, RawContent::Json(value));
        assert_eq!(node.metadata.source, "api");
        assert!((node.metadata.importance - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn content_text_for_text_node() {
        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "hello",
            0.5,
            Vec::new(),
        );
        assert_eq!(node.content_text(), "hello");
    }

    #[test]
    fn content_text_for_json_node() {
        let value = json!({"answer": 42});
        let node = RawNode::json(
            RawNodeKind::ToolResult,
            None,
            None,
            "tool",
            value.clone(),
            0.5,
            Vec::new(),
        );
        assert_eq!(node.content_text(), value.to_string());
    }

    #[test]
    fn context_text_format() {
        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "hi",
            0.5,
            Vec::new(),
        );
        let ctx = node.context_text();
        assert!(ctx.starts_with("UserUtterance: "));
        assert!(ctx.contains("hi"));
    }

    #[test]
    fn with_operation_key() {
        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "msg",
            0.5,
            Vec::new(),
        )
        .with_operation_key("op-123");
        assert_eq!(node.operation_key, Some("op-123".to_string()));
    }

    #[test]
    fn embedding_ref_is_set() {
        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "msg",
            0.5,
            Vec::new(),
        );
        assert!(node.embedding_ref.0.starts_with("raw:"));
        assert!(node.embedding_ref.0.contains(&node.id.to_string()));
    }

    #[test]
    fn default_metadata() {
        let meta = RawNodeMetadata::default();
        assert_eq!(meta.source, "system");
        assert!(meta.tags.is_empty());
        assert!((meta.importance - 0.5).abs() < f32::EPSILON);
        assert_eq!(meta.visibility, Visibility::Normal);
    }

    #[test]
    fn distillation_state_transitions() {
        let mut node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "msg",
            0.5,
            Vec::new(),
        );
        assert_eq!(node.distillation_state, DistillationState::Undistilled);

        node.distillation_state = DistillationState::PartiallyDistilled;
        assert_eq!(
            node.distillation_state,
            DistillationState::PartiallyDistilled
        );

        node.distillation_state = DistillationState::Distilled;
        assert_eq!(node.distillation_state, DistillationState::Distilled);
    }

    #[test]
    fn overflow_policy_default() {
        let policy = OverflowPolicy::default();
        assert!(!policy.was_pushed_out_of_session);
        assert!(policy.relax_retrieval_until.is_none());
    }

    #[test]
    fn raw_node_kind_serde_roundtrip() {
        let kinds = vec![
            RawNodeKind::UserUtterance,
            RawNodeKind::AssistantUtterance,
            RawNodeKind::ToolResult,
            RawNodeKind::Note,
            RawNodeKind::Event,
        ];
        for kind in kinds {
            let json = serde_json::to_string(&kind).unwrap();
            let deserialized: RawNodeKind = serde_json::from_str(&json).unwrap();
            assert_eq!(kind, deserialized);
        }
    }

    #[test]
    fn raw_content_text_serde_roundtrip() {
        let content = RawContent::Text("hello world".to_string());
        let json = serde_json::to_string(&content).unwrap();
        let deserialized: RawContent = serde_json::from_str(&json).unwrap();
        assert_eq!(content, deserialized);
    }

    #[test]
    fn raw_content_json_serde_roundtrip() {
        let content = RawContent::Json(json!({"a": 1}));
        let json = serde_json::to_string(&content).unwrap();
        let deserialized: RawContent = serde_json::from_str(&json).unwrap();
        assert_eq!(content, deserialized);
    }

    #[test]
    fn raw_node_serde_roundtrip() {
        let node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "hello",
            0.5,
            Vec::new(),
        );
        let json = serde_json::to_string(&node).unwrap();
        let deserialized: RawNode = serde_json::from_str(&json).unwrap();
        assert_eq!(node, deserialized);
    }
}
