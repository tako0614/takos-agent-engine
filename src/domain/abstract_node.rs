use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::domain::Relation;
use crate::ids::{AbstractNodeId, RawNodeId};
use crate::model::embedding::EmbeddingRef;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EntityRef {
    pub id: String,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct GraphFragment {
    pub entities: Vec<EntityRef>,
    pub relations: Vec<Relation>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct References {
    pub abstract_node_ids: Vec<AbstractNodeId>,
    pub raw_node_ids: Vec<RawNodeId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AbstractNodeMetadata {
    pub abstraction_level: u8,
    pub confidence: f32,
    pub importance: f32,
    pub tags: Vec<String>,
}

impl Default for AbstractNodeMetadata {
    fn default() -> Self {
        Self {
            abstraction_level: 1,
            confidence: 0.5,
            importance: 0.5,
            tags: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AbstractNode {
    pub id: AbstractNodeId,
    pub operation_key: Option<String>,
    pub timestamp: DateTime<Utc>,
    pub title: String,
    pub summary: String,
    pub embedding_ref: EmbeddingRef,
    pub graph: GraphFragment,
    pub references: References,
    pub metadata: AbstractNodeMetadata,
    pub version: u32,
}

impl AbstractNode {
    pub fn new(
        title: impl Into<String>,
        summary: impl Into<String>,
        references: References,
        graph: GraphFragment,
        metadata: AbstractNodeMetadata,
    ) -> Self {
        let id = AbstractNodeId::new();
        Self {
            id,
            operation_key: None,
            timestamp: Utc::now(),
            title: title.into(),
            summary: summary.into(),
            embedding_ref: EmbeddingRef::for_node("abstract", id.to_string()),
            graph,
            references,
            metadata,
            version: 1,
        }
    }

    #[must_use]
    pub fn context_text(&self) -> String {
        format!("{}: {}", self.title, self.summary)
    }

    pub fn with_operation_key(mut self, operation_key: impl Into<String>) -> Self {
        self.operation_key = Some(operation_key.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_basic_node() -> AbstractNode {
        AbstractNode::new(
            "Test Title",
            "Test summary content",
            References::default(),
            GraphFragment::default(),
            AbstractNodeMetadata::default(),
        )
    }

    #[test]
    fn new_node_has_correct_fields() {
        let node = make_basic_node();
        assert_eq!(node.title, "Test Title");
        assert_eq!(node.summary, "Test summary content");
        assert_eq!(node.version, 1);
        assert!(node.operation_key.is_none());
    }

    #[test]
    fn new_node_has_embedding_ref() {
        let node = make_basic_node();
        assert!(node.embedding_ref.0.starts_with("abstract:"));
        assert!(node.embedding_ref.0.contains(&node.id.to_string()));
    }

    #[test]
    fn default_metadata() {
        let meta = AbstractNodeMetadata::default();
        assert_eq!(meta.abstraction_level, 1);
        assert!((meta.confidence - 0.5).abs() < f32::EPSILON);
        assert!((meta.importance - 0.5).abs() < f32::EPSILON);
        assert!(meta.tags.is_empty());
    }

    #[test]
    fn custom_metadata() {
        let meta = AbstractNodeMetadata {
            abstraction_level: 3,
            confidence: 0.9,
            importance: 0.8,
            tags: vec!["important".to_string()],
        };
        let node = AbstractNode::new(
            "Custom",
            "summary",
            References::default(),
            GraphFragment::default(),
            meta,
        );
        assert_eq!(node.metadata.abstraction_level, 3);
        assert!((node.metadata.confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn version_incrementing() {
        let mut node = make_basic_node();
        assert_eq!(node.version, 1);
        node.version += 1;
        assert_eq!(node.version, 2);
        node.version += 1;
        assert_eq!(node.version, 3);
    }

    #[test]
    fn relation_management_via_graph_fragment() {
        let relation = Relation {
            subject: "entity-a".to_string(),
            predicate: "related_to".to_string(),
            object: "entity-b".to_string(),
            weight: 0.9,
            provenance_raw_node_ids: Vec::new(),
        };
        let entity_a = EntityRef {
            id: "entity-a".to_string(),
            label: "Entity A".to_string(),
        };
        let entity_b = EntityRef {
            id: "entity-b".to_string(),
            label: "Entity B".to_string(),
        };
        let graph = GraphFragment {
            entities: vec![entity_a.clone(), entity_b.clone()],
            relations: vec![relation.clone()],
        };
        let node = AbstractNode::new(
            "With graph",
            "summary",
            References::default(),
            graph,
            AbstractNodeMetadata::default(),
        );
        assert_eq!(node.graph.entities.len(), 2);
        assert_eq!(node.graph.relations.len(), 1);
        assert_eq!(node.graph.relations[0].predicate, "related_to");
    }

    #[test]
    fn references_management() {
        let raw_id = RawNodeId::new();
        let abstract_id = AbstractNodeId::new();
        let refs = References {
            abstract_node_ids: vec![abstract_id],
            raw_node_ids: vec![raw_id],
        };
        let node = AbstractNode::new(
            "Referenced",
            "summary",
            refs,
            GraphFragment::default(),
            AbstractNodeMetadata::default(),
        );
        assert_eq!(node.references.raw_node_ids.len(), 1);
        assert_eq!(node.references.abstract_node_ids.len(), 1);
        assert_eq!(node.references.raw_node_ids[0], raw_id);
        assert_eq!(node.references.abstract_node_ids[0], abstract_id);
    }

    #[test]
    fn context_text_format() {
        let node = make_basic_node();
        let ctx = node.context_text();
        assert_eq!(ctx, "Test Title: Test summary content");
    }

    #[test]
    fn with_operation_key() {
        let node = make_basic_node().with_operation_key("loop:123:abstract:primary");
        assert_eq!(
            node.operation_key,
            Some("loop:123:abstract:primary".to_string())
        );
    }

    #[test]
    fn default_references_are_empty() {
        let refs = References::default();
        assert!(refs.abstract_node_ids.is_empty());
        assert!(refs.raw_node_ids.is_empty());
    }

    #[test]
    fn default_graph_fragment_is_empty() {
        let graph = GraphFragment::default();
        assert!(graph.entities.is_empty());
        assert!(graph.relations.is_empty());
    }

    #[test]
    fn serde_roundtrip() {
        let node = make_basic_node().with_operation_key("test-op");
        let json = serde_json::to_string(&node).unwrap();
        let deserialized: AbstractNode = serde_json::from_str(&json).unwrap();
        assert_eq!(node, deserialized);
    }

    #[test]
    fn unique_ids_per_node() {
        let a = make_basic_node();
        let b = make_basic_node();
        assert_ne!(a.id, b.id);
    }
}
