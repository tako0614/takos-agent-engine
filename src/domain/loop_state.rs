use serde::{Deserialize, Serialize};

use crate::ids::{LoopId, SessionId};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LoopStatus {
    Running,
    Paused,
    Finished,
    Cancelled,
    TimedOut,
    Failed,
}

/// A durable checkpoint of an in-flight loop.
///
/// `state_json` is the complete serialized `ExecutionState` and is the ONLY
/// payload read back on resume (`ExecutionState::from_checkpoint`). The struct
/// therefore carries just the routing/identity it needs alongside it -- the
/// session/loop key, the node to resume at, and the status. Earlier versions
/// also stored a large denormalized projection (recent_events, activated_*,
/// session_window, pushed_out_raw, tool_result_ids, assistant_message,
/// last_effect_key, ...) that nothing ever read; it was pure write amplification
/// (re-serialized on every node step) and a drift risk against `state_json`, so
/// it was dropped. Old on-disk checkpoints still deserialize (serde ignores the
/// now-unknown fields). [Q3]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoopState {
    pub session_id: SessionId,
    pub loop_id: LoopId,
    pub current_node: String,
    pub status: LoopStatus,
    pub state_json: serde_json::Value,
}

impl LoopState {
    #[cfg(test)]
    pub(crate) fn new_for_test(session_id: SessionId, loop_id: LoopId, _goal: &str) -> Self {
        Self {
            session_id,
            loop_id,
            current_node: "start".to_string(),
            status: LoopStatus::Running,
            state_json: serde_json::Value::Null,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_loop_state() -> LoopState {
        LoopState::new_for_test(SessionId::new(), LoopId::new(), "test goal")
    }

    #[test]
    fn initial_status_is_running() {
        let state = make_loop_state();
        assert_eq!(state.status, LoopStatus::Running);
    }

    #[test]
    fn running_to_paused() {
        let mut state = make_loop_state();
        assert_eq!(state.status, LoopStatus::Running);
        state.status = LoopStatus::Paused;
        assert_eq!(state.status, LoopStatus::Paused);
    }

    #[test]
    fn paused_to_running() {
        let mut state = make_loop_state();
        state.status = LoopStatus::Paused;
        state.status = LoopStatus::Running;
        assert_eq!(state.status, LoopStatus::Running);
    }

    #[test]
    fn running_to_finished() {
        let mut state = make_loop_state();
        state.status = LoopStatus::Finished;
        assert_eq!(state.status, LoopStatus::Finished);
    }

    #[test]
    fn running_to_cancelled() {
        let mut state = make_loop_state();
        state.status = LoopStatus::Cancelled;
        assert_eq!(state.status, LoopStatus::Cancelled);
    }

    #[test]
    fn running_to_timed_out() {
        let mut state = make_loop_state();
        state.status = LoopStatus::TimedOut;
        assert_eq!(state.status, LoopStatus::TimedOut);
    }

    #[test]
    fn running_to_failed() {
        let mut state = make_loop_state();
        state.status = LoopStatus::Failed;
        assert_eq!(state.status, LoopStatus::Failed);
    }

    #[test]
    fn paused_to_finished() {
        let mut state = make_loop_state();
        state.status = LoopStatus::Paused;
        state.status = LoopStatus::Finished;
        assert_eq!(state.status, LoopStatus::Finished);
    }

    #[test]
    fn checkpoint_serialization_roundtrip() {
        let mut state = make_loop_state();
        state.current_node = "run_model".to_string();
        state.status = LoopStatus::Paused;
        state.state_json = json!({"progress": 0.5});

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: LoopState = serde_json::from_str(&json).unwrap();
        assert_eq!(state, deserialized);
    }

    #[test]
    fn unknown_fields_from_old_checkpoints_are_ignored() {
        // A pre-Q3 checkpoint carried the now-removed projection fields. Such an
        // on-disk payload must still deserialize (serde drops unknown fields).
        let legacy = json!({
            "session_id": SessionId::new().to_string(),
            "loop_id": LoopId::new().to_string(),
            "current_node": "run_model",
            "status": "paused",
            "state_json": {"progress": 0.5},
            "iteration": 7,
            "recent_events": ["00000000-0000-0000-0000-000000000001"],
            "last_effect_key": "loop:x:tool:1:0:t"
        });
        let parsed: LoopState = serde_json::from_value(legacy).unwrap();
        assert_eq!(parsed.current_node, "run_model");
        assert_eq!(parsed.status, LoopStatus::Paused);
    }

    #[test]
    fn loop_status_serde_roundtrip() {
        let statuses = vec![
            LoopStatus::Running,
            LoopStatus::Paused,
            LoopStatus::Finished,
            LoopStatus::Cancelled,
            LoopStatus::TimedOut,
            LoopStatus::Failed,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let deserialized: LoopStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(status, deserialized);
        }
    }

    #[test]
    fn loop_status_serde_snake_case() {
        let json = serde_json::to_string(&LoopStatus::TimedOut).unwrap();
        assert_eq!(json, "\"timed_out\"");
    }
}
