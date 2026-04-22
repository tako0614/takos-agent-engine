use serde::{Deserialize, Serialize};

use crate::ids::RawNodeId;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Relation {
    pub subject: String,
    pub predicate: String,
    pub object: String,
    pub weight: f32,
    pub provenance_raw_node_ids: Vec<RawNodeId>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_relation() {
        let relation = Relation {
            subject: "entity-a".to_string(),
            predicate: "causes".to_string(),
            object: "entity-b".to_string(),
            weight: 0.85,
            provenance_raw_node_ids: Vec::new(),
        };
        assert_eq!(relation.subject, "entity-a");
        assert_eq!(relation.predicate, "causes");
        assert_eq!(relation.object, "entity-b");
        assert!((relation.weight - 0.85).abs() < f32::EPSILON);
        assert!(relation.provenance_raw_node_ids.is_empty());
    }

    #[test]
    fn relation_with_provenance() {
        let raw_id_1 = RawNodeId::new();
        let raw_id_2 = RawNodeId::new();
        let relation = Relation {
            subject: "user".to_string(),
            predicate: "asked".to_string(),
            object: "question".to_string(),
            weight: 1.0,
            provenance_raw_node_ids: vec![raw_id_1, raw_id_2],
        };
        assert_eq!(relation.provenance_raw_node_ids.len(), 2);
        assert_eq!(relation.provenance_raw_node_ids[0], raw_id_1);
        assert_eq!(relation.provenance_raw_node_ids[1], raw_id_2);
    }

    #[test]
    fn weight_bounds_zero() {
        let relation = Relation {
            subject: "a".to_string(),
            predicate: "b".to_string(),
            object: "c".to_string(),
            weight: 0.0,
            provenance_raw_node_ids: Vec::new(),
        };
        assert!((relation.weight - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn weight_bounds_one() {
        let relation = Relation {
            subject: "a".to_string(),
            predicate: "b".to_string(),
            object: "c".to_string(),
            weight: 1.0,
            provenance_raw_node_ids: Vec::new(),
        };
        assert!((relation.weight - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn weight_fractional() {
        let relation = Relation {
            subject: "a".to_string(),
            predicate: "b".to_string(),
            object: "c".to_string(),
            weight: 0.42,
            provenance_raw_node_ids: Vec::new(),
        };
        assert!((relation.weight - 0.42).abs() < f32::EPSILON);
    }

    #[test]
    fn serde_roundtrip() {
        let relation = Relation {
            subject: "subj".to_string(),
            predicate: "pred".to_string(),
            object: "obj".to_string(),
            weight: 0.75,
            provenance_raw_node_ids: vec![RawNodeId::new()],
        };
        let json = serde_json::to_string(&relation).unwrap();
        let deserialized: Relation = serde_json::from_str(&json).unwrap();
        assert_eq!(relation, deserialized);
    }

    #[test]
    fn equality() {
        let id = RawNodeId::new();
        let a = Relation {
            subject: "s".to_string(),
            predicate: "p".to_string(),
            object: "o".to_string(),
            weight: 0.5,
            provenance_raw_node_ids: vec![id],
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
