use chrono::{DateTime, Utc};

use crate::domain::{AbstractNode, RawNode};
use crate::memory::activation::overflow_relaxation_active;

pub trait ScoringPolicy: Send + Sync {
    fn score_raw(&self, base_similarity: f32, node: &RawNode, now: DateTime<Utc>) -> f32;
    fn score_abstract(&self, base_similarity: f32, node: &AbstractNode, now: DateTime<Utc>) -> f32;
}

#[derive(Debug, Clone, Copy)]
pub struct DefaultScoringPolicy {
    semantic_weight: f32,
    importance_weight: f32,
    decay_per_day: f32,
    overflow_bonus: f32,
}

impl Default for DefaultScoringPolicy {
    fn default() -> Self {
        Self {
            semantic_weight: 1.0,
            importance_weight: 0.2,
            decay_per_day: 0.015,
            overflow_bonus: 0.12,
        }
    }
}

impl DefaultScoringPolicy {
    fn age_in_days(timestamp: DateTime<Utc>, now: DateTime<Utc>) -> f32 {
        // Memory age fits well within f32's representable seconds for any
        // realistic timestamp delta; we want days as f32 for the scoring math.
        #[allow(clippy::cast_precision_loss)]
        let seconds = (now - timestamp).num_seconds().max(0) as f32;
        seconds / 86_400.0
    }
}

impl ScoringPolicy for DefaultScoringPolicy {
    fn score_raw(&self, base_similarity: f32, node: &RawNode, now: DateTime<Utc>) -> f32 {
        let age_penalty = Self::age_in_days(node.timestamp, now) * self.decay_per_day;
        // The overflow bonus expires with the same deadline as the threshold
        // relaxation (see `overflow_relaxation_active`); `now` here is the
        // activation reference instant. [C7]
        let overflow_bonus = if overflow_relaxation_active(node, now) {
            self.overflow_bonus
        } else {
            0.0
        };

        base_similarity.mul_add(
            self.semantic_weight,
            node.metadata.importance * self.importance_weight,
        ) - age_penalty
            + overflow_bonus
    }

    fn score_abstract(&self, base_similarity: f32, node: &AbstractNode, now: DateTime<Utc>) -> f32 {
        let age_penalty = Self::age_in_days(node.timestamp, now) * self.decay_per_day;
        base_similarity.mul_add(
            self.semantic_weight,
            node.metadata.importance * self.importance_weight,
        ) - age_penalty
    }
}

#[cfg(test)]
mod tests {
    use chrono::{Duration, Utc};

    use crate::domain::{
        AbstractNode, AbstractNodeMetadata, GraphFragment, RawNode, RawNodeKind, References,
    };

    use super::{DefaultScoringPolicy, ScoringPolicy};

    #[test]
    fn overflow_bonus_lifts_raw_score() {
        let scorer = DefaultScoringPolicy::default();
        let mut node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "hello",
            0.5,
            Vec::new(),
        );
        let now = Utc::now();
        node.overflow.was_pushed_out_of_session = true;
        // Bonus applies only within the relaxation deadline.
        node.overflow.relax_retrieval_until = Some(now + Duration::hours(1));
        let boosted = scorer.score_raw(0.5, &node, now);
        node.overflow.was_pushed_out_of_session = false;
        let normal = scorer.score_raw(0.5, &node, now);
        assert!(boosted > normal);
    }

    #[test]
    fn overflow_bonus_expires_after_deadline() {
        let scorer = DefaultScoringPolicy::default();
        let mut node = RawNode::text(
            RawNodeKind::UserUtterance,
            None,
            None,
            "user",
            "hello",
            0.5,
            Vec::new(),
        );
        let now = Utc::now();
        node.overflow.was_pushed_out_of_session = true;
        // Deadline already elapsed -> no bonus (same score as a non-overflow
        // node), proving the relaxation window is enforced.
        node.overflow.relax_retrieval_until = Some(now - Duration::hours(1));
        let expired = scorer.score_raw(0.5, &node, now);
        node.overflow.was_pushed_out_of_session = false;
        let normal = scorer.score_raw(0.5, &node, now);
        assert!((expired - normal).abs() < f32::EPSILON);
    }

    #[test]
    fn old_abstract_nodes_decay() {
        let scorer = DefaultScoringPolicy::default();
        let mut node = AbstractNode::new(
            "title",
            "summary",
            References::default(),
            GraphFragment::default(),
            AbstractNodeMetadata::default(),
        );
        let now = Utc::now();
        node.timestamp = now - Duration::days(7);
        let old_score = scorer.score_abstract(0.8, &node, now);
        node.timestamp = now;
        let fresh_score = scorer.score_abstract(0.8, &node, now);
        assert!(fresh_score > old_score);
    }
}
