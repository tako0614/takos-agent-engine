use std::cmp::Ordering;
use std::collections::HashMap;
use std::hash::Hash;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::error::Result;
use crate::ids::{AbstractNodeId, RawNodeId};
use crate::model::embedding::{cosine_similarity, Embedding};
use crate::storage::traits::{ScoredAbstractRef, ScoredRawRef, VectorIndex};

const DEFAULT_PROJECTION_BITS: usize = 16;
const DEFAULT_CANDIDATE_MULTIPLIER: usize = 8;
const DEFAULT_MIN_CANDIDATES: usize = 64;
const DEFAULT_PROJECTION_SEED: u64 = 0x7461_6b6f_735f_616e;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnnVectorIndexConfig {
    pub projection_bits: usize,
    pub candidate_multiplier: usize,
    pub min_candidates: usize,
    pub projection_seed: u64,
}

impl Default for AnnVectorIndexConfig {
    fn default() -> Self {
        Self {
            projection_bits: DEFAULT_PROJECTION_BITS,
            candidate_multiplier: DEFAULT_CANDIDATE_MULTIPLIER,
            min_candidates: DEFAULT_MIN_CANDIDATES,
            projection_seed: DEFAULT_PROJECTION_SEED,
        }
    }
}

impl AnnVectorIndexConfig {
    fn normalized(mut self) -> Self {
        self.projection_bits = self.projection_bits.clamp(1, 63);
        self.candidate_multiplier = self.candidate_multiplier.max(1);
        self
    }

    fn candidate_limit(&self, top_k: usize, total: usize) -> usize {
        top_k
            .saturating_mul(self.candidate_multiplier)
            .max(self.min_candidates)
            .min(total)
    }
}

#[derive(Debug)]
pub struct AnnVectorIndex {
    config: AnnVectorIndexConfig,
    raw_index: RwLock<LshEmbeddingIndex<RawNodeId>>,
    abstract_index: RwLock<LshEmbeddingIndex<AbstractNodeId>>,
}

impl Default for AnnVectorIndex {
    fn default() -> Self {
        Self::new(AnnVectorIndexConfig::default())
    }
}

impl AnnVectorIndex {
    #[must_use]
    pub fn new(config: AnnVectorIndexConfig) -> Self {
        let config = config.normalized();
        Self {
            raw_index: RwLock::new(LshEmbeddingIndex::new(config.clone())),
            abstract_index: RwLock::new(LshEmbeddingIndex::new(config.clone())),
            config,
        }
    }

    pub fn config(&self) -> &AnnVectorIndexConfig {
        &self.config
    }
}

#[async_trait]
impl VectorIndex for AnnVectorIndex {
    async fn index_raw(&self, id: RawNodeId, embedding: Embedding) -> Result<()> {
        self.raw_index.write().await.index(id, embedding);
        Ok(())
    }

    async fn index_abstract(&self, id: AbstractNodeId, embedding: Embedding) -> Result<()> {
        self.abstract_index.write().await.index(id, embedding);
        Ok(())
    }

    async fn search_raw(&self, query: &Embedding, top_k: usize) -> Result<Vec<ScoredRawRef>> {
        Ok(self
            .raw_index
            .read()
            .await
            .search(query, top_k)
            .into_iter()
            .map(|(id, score)| ScoredRawRef { id, score })
            .collect())
    }

    async fn search_abstract(
        &self,
        query: &Embedding,
        top_k: usize,
    ) -> Result<Vec<ScoredAbstractRef>> {
        Ok(self
            .abstract_index
            .read()
            .await
            .search(query, top_k)
            .into_iter()
            .map(|(id, score)| ScoredAbstractRef { id, score })
            .collect())
    }
}

#[derive(Debug)]
struct LshEmbeddingIndex<ID> {
    config: AnnVectorIndexConfig,
    embeddings: HashMap<ID, IndexedEmbedding>,
    buckets: HashMap<u64, Vec<ID>>,
}

impl<ID> LshEmbeddingIndex<ID>
where
    ID: Copy + Eq + Hash + Ord,
{
    fn new(config: AnnVectorIndexConfig) -> Self {
        Self {
            config,
            embeddings: HashMap::new(),
            buckets: HashMap::new(),
        }
    }

    fn index(&mut self, id: ID, embedding: Embedding) {
        if let Some(previous) = self.embeddings.remove(&id) {
            self.remove_from_bucket(previous.signature, id);
        }

        let signature = lsh_signature(&embedding, &self.config);
        self.embeddings.insert(
            id,
            IndexedEmbedding {
                embedding,
                signature,
            },
        );
        let bucket = self.buckets.entry(signature).or_default();
        bucket.push(id);
        bucket.sort();
    }

    fn search(&self, query: &Embedding, top_k: usize) -> Vec<(ID, f32)> {
        if top_k == 0 || self.embeddings.is_empty() {
            return Vec::new();
        }

        let candidate_limit = self.config.candidate_limit(top_k, self.embeddings.len());
        let candidate_ids = self.candidate_ids(query, candidate_limit);
        let mut scored: Vec<_> = candidate_ids
            .into_iter()
            .filter_map(|id| {
                self.embeddings
                    .get(&id)
                    .map(|record| (id, cosine_similarity(query, &record.embedding)))
            })
            .collect();

        sort_scored_by_score_then_id(&mut scored);
        scored.truncate(top_k);
        scored
    }

    fn candidate_ids(&self, query: &Embedding, candidate_limit: usize) -> Vec<ID> {
        if candidate_limit >= self.embeddings.len() {
            let mut ids: Vec<_> = self.embeddings.keys().copied().collect();
            ids.sort();
            return ids;
        }

        let query_signature = lsh_signature(query, &self.config);
        let mut buckets: Vec<_> = self.buckets.keys().copied().collect();
        buckets.sort_by(|left, right| {
            hamming_distance(*left, query_signature)
                .cmp(&hamming_distance(*right, query_signature))
                .then_with(|| left.cmp(right))
        });

        let mut candidates = Vec::with_capacity(candidate_limit);
        for signature in buckets {
            if let Some(bucket) = self.buckets.get(&signature) {
                for id in bucket {
                    candidates.push(*id);
                    if candidates.len() >= candidate_limit {
                        return candidates;
                    }
                }
            }
        }
        candidates
    }

    fn remove_from_bucket(&mut self, signature: u64, id: ID) {
        if let Some(bucket) = self.buckets.get_mut(&signature) {
            bucket.retain(|candidate| *candidate != id);
            if bucket.is_empty() {
                self.buckets.remove(&signature);
            }
        }
    }
}

#[derive(Debug)]
struct IndexedEmbedding {
    embedding: Embedding,
    signature: u64,
}

fn sort_scored_by_score_then_id<ID: Ord>(scored: &mut [(ID, f32)]) {
    scored.sort_by(|(left_id, left_score), (right_id, right_score)| {
        right_score
            .partial_cmp(left_score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_id.cmp(right_id))
    });
}

fn hamming_distance(left: u64, right: u64) -> u32 {
    (left ^ right).count_ones()
}

fn lsh_signature(embedding: &Embedding, config: &AnnVectorIndexConfig) -> u64 {
    let mut signature = 0_u64;
    for bit in 0..config.projection_bits {
        let dot = embedding
            .0
            .iter()
            .enumerate()
            .map(|(dimension, value)| {
                *value * projection_sign(config.projection_seed, bit, dimension)
            })
            .sum::<f32>();
        if dot >= 0.0 {
            signature |= 1_u64 << bit;
        }
    }
    signature
}

fn projection_sign(seed: u64, bit: usize, dimension: usize) -> f32 {
    let hash = splitmix64(
        seed ^ ((bit as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15))
            ^ ((dimension as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9)),
    );
    if hash & 1 == 0 {
        -1.0
    } else {
        1.0
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[derive(Debug, Default)]
#[cfg(test)]
pub struct InMemoryVectorIndex {
    raw_embeddings: RwLock<HashMap<RawNodeId, Embedding>>,
    abstract_embeddings: RwLock<HashMap<AbstractNodeId, Embedding>>,
}

#[async_trait]
#[cfg(test)]
impl VectorIndex for InMemoryVectorIndex {
    async fn index_raw(&self, id: RawNodeId, embedding: Embedding) -> Result<()> {
        self.raw_embeddings.write().await.insert(id, embedding);
        Ok(())
    }

    async fn index_abstract(&self, id: AbstractNodeId, embedding: Embedding) -> Result<()> {
        self.abstract_embeddings.write().await.insert(id, embedding);
        Ok(())
    }

    async fn search_raw(&self, query: &Embedding, top_k: usize) -> Result<Vec<ScoredRawRef>> {
        let guard = self.raw_embeddings.read().await;
        let mut scored: Vec<_> = guard
            .iter()
            .map(|(id, embedding)| ScoredRawRef {
                id: *id,
                score: cosine_similarity(query, embedding),
            })
            .collect();
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        scored.truncate(top_k);
        Ok(scored)
    }

    async fn search_abstract(
        &self,
        query: &Embedding,
        top_k: usize,
    ) -> Result<Vec<ScoredAbstractRef>> {
        let guard = self.abstract_embeddings.read().await;
        let mut scored: Vec<_> = guard
            .iter()
            .map(|(id, embedding)| ScoredAbstractRef {
                id: *id,
                score: cosine_similarity(query, embedding),
            })
            .collect();
        scored.sort_by(|left, right| {
            right
                .score
                .partial_cmp(&left.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left.id.to_string().cmp(&right.id.to_string()))
        });
        scored.truncate(top_k);
        Ok(scored)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn raw_id(value: u128) -> RawNodeId {
        RawNodeId(Uuid::from_u128(value))
    }

    fn abstract_id(value: u128) -> AbstractNodeId {
        AbstractNodeId(Uuid::from_u128(value))
    }

    fn exact_config() -> AnnVectorIndexConfig {
        AnnVectorIndexConfig {
            min_candidates: usize::MAX,
            ..AnnVectorIndexConfig::default()
        }
    }

    #[tokio::test]
    async fn ann_vector_index_reranks_raw_candidates_with_exact_ordering() -> Result<()> {
        let index = AnnVectorIndex::new(exact_config());
        let first = raw_id(1);
        let second = raw_id(2);
        let third = raw_id(3);

        index.index_raw(third, Embedding(vec![0.8, 0.6])).await?;
        index.index_raw(second, Embedding(vec![1.0, 0.0])).await?;
        index.index_raw(first, Embedding(vec![1.0, 0.0])).await?;

        let scored = index.search_raw(&Embedding(vec![1.0, 0.0]), 3).await?;

        assert_eq!(scored.len(), 3);
        assert_eq!(scored[0].id, first);
        assert_eq!(scored[1].id, second);
        assert_eq!(scored[2].id, third);
        assert!((scored[0].score - 1.0).abs() < 1e-6);
        assert!((scored[1].score - 1.0).abs() < 1e-6);
        assert!((scored[2].score - 0.8).abs() < 1e-6);
        Ok(())
    }

    #[tokio::test]
    async fn ann_vector_index_indexes_abstract_nodes_independently() -> Result<()> {
        let index = AnnVectorIndex::new(exact_config());
        let raw = raw_id(1);
        let abstract_node = abstract_id(1);

        index.index_raw(raw, Embedding(vec![0.0, 1.0])).await?;
        index
            .index_abstract(abstract_node, Embedding(vec![1.0, 0.0]))
            .await?;

        let raw_scored = index.search_raw(&Embedding(vec![1.0, 0.0]), 8).await?;
        let abstract_scored = index.search_abstract(&Embedding(vec![1.0, 0.0]), 8).await?;

        assert_eq!(raw_scored.len(), 1);
        assert_eq!(raw_scored[0].id, raw);
        assert!((raw_scored[0].score - 0.0).abs() < 1e-6);
        assert_eq!(abstract_scored.len(), 1);
        assert_eq!(abstract_scored[0].id, abstract_node);
        assert!((abstract_scored[0].score - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[tokio::test]
    async fn ann_vector_index_reindex_replaces_previous_bucket_membership() -> Result<()> {
        let config = AnnVectorIndexConfig {
            projection_bits: 8,
            candidate_multiplier: 1,
            min_candidates: 1,
            projection_seed: DEFAULT_PROJECTION_SEED,
        };
        let index = AnnVectorIndex::new(config.clone());
        let id = raw_id(1);
        let original = Embedding(vec![-1.0, 0.0]);
        let replacement = Embedding(vec![1.0, 0.0]);
        let normalized_config = config.normalized();
        let original_signature = lsh_signature(&original, &normalized_config);
        let replacement_signature = lsh_signature(&replacement, &normalized_config);

        index.index_raw(id, original).await?;
        index.index_raw(id, replacement).await?;

        let guard = index.raw_index.read().await;
        let bucket_occurrences = guard
            .buckets
            .values()
            .flat_map(|bucket| bucket.iter())
            .filter(|candidate| **candidate == id)
            .count();
        assert_eq!(bucket_occurrences, 1);
        if original_signature != replacement_signature {
            assert!(!guard
                .buckets
                .get(&original_signature)
                .is_some_and(|bucket| bucket.contains(&id)));
        }
        assert!(guard
            .buckets
            .get(&replacement_signature)
            .is_some_and(|bucket| bucket.contains(&id)));
        Ok(())
    }

    #[tokio::test]
    async fn ann_vector_index_respects_top_k_zero() -> Result<()> {
        let index = AnnVectorIndex::default();
        index.index_raw(raw_id(1), Embedding(vec![1.0])).await?;

        let scored = index.search_raw(&Embedding(vec![1.0]), 0).await?;

        assert!(scored.is_empty());
        Ok(())
    }
}
