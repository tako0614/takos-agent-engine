use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::ids::SessionId;

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Session {
    pub id: SessionId,
    pub goal: String,
    pub created_at: DateTime<Utc>,
}

impl Session {
    #[allow(dead_code)]
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            id: SessionId::new(),
            goal: goal.into(),
            created_at: Utc::now(),
        }
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionPlan {
    pub summary: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_creation() {
        let session = Session::new("Find all bugs in the codebase");
        assert_eq!(session.goal, "Find all bugs in the codebase");
        assert!(session.created_at <= Utc::now());
    }

    #[test]
    fn session_has_unique_id() {
        let a = Session::new("goal a");
        let b = Session::new("goal b");
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn session_goal_from_string_type() {
        let goal = String::from("dynamic goal");
        let session = Session::new(goal);
        assert_eq!(session.goal, "dynamic goal");
    }

    #[test]
    fn session_serde_roundtrip() {
        let session = Session::new("test goal");
        let json = serde_json::to_string(&session).unwrap();
        let deserialized: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(session, deserialized);
    }

    #[test]
    fn session_created_at_is_recent() {
        let before = Utc::now();
        let session = Session::new("goal");
        let after = Utc::now();
        assert!(session.created_at >= before);
        assert!(session.created_at <= after);
    }

    #[test]
    fn session_plan_creation() {
        let plan = SessionPlan {
            summary: "Step 1: analyze. Step 2: fix.".to_string(),
        };
        assert_eq!(plan.summary, "Step 1: analyze. Step 2: fix.");
    }

    #[test]
    fn session_plan_serde_roundtrip() {
        let plan = SessionPlan {
            summary: "plan summary".to_string(),
        };
        let json = serde_json::to_string(&plan).unwrap();
        let deserialized: SessionPlan = serde_json::from_str(&json).unwrap();
        assert_eq!(plan, deserialized);
    }
}
