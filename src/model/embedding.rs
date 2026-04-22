use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Embedding(pub Vec<f32>);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct EmbeddingRef(pub String);

impl EmbeddingRef {
    pub fn for_node(prefix: &str, node_id: String) -> Self {
        Self(format!("{prefix}:{node_id}"))
    }
}

pub fn cosine_similarity(left: &Embedding, right: &Embedding) -> f32 {
    if left.0.len() != right.0.len() || left.0.is_empty() {
        return 0.0;
    }

    let dot = left
        .0
        .iter()
        .zip(right.0.iter())
        .map(|(a, b)| a * b)
        .sum::<f32>();
    let left_norm = left.0.iter().map(|v| v * v).sum::<f32>().sqrt();
    let right_norm = right.0.iter().map(|v| v * v).sum::<f32>().sqrt();

    if left_norm == 0.0 || right_norm == 0.0 {
        0.0
    } else {
        dot / (left_norm * right_norm)
    }
}

#[async_trait]
pub trait Embedder: Send + Sync {
    async fn embed_text(&self, text: &str) -> Result<Embedding>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_similarity_identical_vectors() {
        let a = Embedding(vec![1.0, 0.0, 0.0]);
        let b = Embedding(vec![1.0, 0.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical vectors should have similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_opposite_vectors() {
        let a = Embedding(vec![1.0, 0.0, 0.0]);
        let b = Embedding(vec![-1.0, 0.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "opposite vectors should have similarity ~-1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_orthogonal_vectors() {
        let a = Embedding(vec![1.0, 0.0, 0.0]);
        let b = Embedding(vec![0.0, 1.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal vectors should have similarity ~0.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_same_direction_different_magnitude() {
        let a = Embedding(vec![1.0, 2.0, 3.0]);
        let b = Embedding(vec![2.0, 4.0, 6.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "parallel vectors should have similarity ~1.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_empty_vectors() {
        let a = Embedding(Vec::new());
        let b = Embedding(Vec::new());
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 0.0).abs() < 1e-6,
            "empty vectors should return 0.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_mismatched_lengths() {
        let a = Embedding(vec![1.0, 0.0]);
        let b = Embedding(vec![1.0, 0.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 0.0).abs() < 1e-6,
            "mismatched lengths should return 0.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_zero_vector() {
        let a = Embedding(vec![0.0, 0.0, 0.0]);
        let b = Embedding(vec![1.0, 2.0, 3.0]);
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim - 0.0).abs() < 1e-6,
            "zero vector should return 0.0, got {sim}"
        );
    }

    #[test]
    fn cosine_similarity_partial_overlap() {
        let a = Embedding(vec![1.0, 1.0, 0.0]);
        let b = Embedding(vec![1.0, 0.0, 0.0]);
        let sim = cosine_similarity(&a, &b);
        // cos(45 degrees) ~ 0.7071
        let expected = 1.0 / 2.0_f32.sqrt();
        assert!(
            (sim - expected).abs() < 1e-5,
            "expected ~{expected}, got {sim}"
        );
    }

    #[test]
    fn embedding_ref_for_node_format() {
        let eref = EmbeddingRef::for_node("raw", "abc-123".to_string());
        assert_eq!(eref.0, "raw:abc-123");
    }

    #[test]
    fn embedding_ref_for_node_abstract() {
        let eref = EmbeddingRef::for_node("abstract", "xyz".to_string());
        assert_eq!(eref.0, "abstract:xyz");
    }

    #[test]
    fn embedding_ref_equality() {
        let a = EmbeddingRef::for_node("raw", "id1".to_string());
        let b = EmbeddingRef::for_node("raw", "id1".to_string());
        assert_eq!(a, b);
    }

    #[test]
    fn embedding_ref_inequality() {
        let a = EmbeddingRef::for_node("raw", "id1".to_string());
        let b = EmbeddingRef::for_node("raw", "id2".to_string());
        assert_ne!(a, b);
    }

    #[test]
    fn embedding_ref_is_hashable() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let eref = EmbeddingRef::for_node("raw", "id1".to_string());
        set.insert(eref.clone());
        set.insert(eref.clone());
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn embedding_serde_roundtrip() {
        let emb = Embedding(vec![0.1, 0.2, 0.3]);
        let json = serde_json::to_string(&emb).unwrap();
        let deserialized: Embedding = serde_json::from_str(&json).unwrap();
        assert_eq!(emb, deserialized);
    }

    #[test]
    fn embedding_ref_serde_roundtrip() {
        let eref = EmbeddingRef::for_node("raw", "node-1".to_string());
        let json = serde_json::to_string(&eref).unwrap();
        let deserialized: EmbeddingRef = serde_json::from_str(&json).unwrap();
        assert_eq!(eref, deserialized);
    }
}
