use serde::{Deserialize, Serialize};

use crate::ids::{AbstractNodeId, LoopId, RawNodeId, SessionId};

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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoopState {
    pub session_id: SessionId,
    pub loop_id: LoopId,
    pub user_goal: String,
    pub plan: Option<String>,
    pub current_node: String,
    pub iteration: u32,
    pub tool_rounds_completed: u32,
    pub model_invocations: u32,
    pub status: LoopStatus,
    pub last_completed_node: Option<String>,
    pub last_effect_key: Option<String>,
    pub recent_events: Vec<RawNodeId>,
    pub activated_raw: Vec<RawNodeId>,
    pub activated_abstract: Vec<AbstractNodeId>,
    pub session_window: Vec<RawNodeId>,
    pub pushed_out_raw: Vec<RawNodeId>,
    pub tool_result_ids: Vec<RawNodeId>,
    pub assistant_message: Option<String>,
    pub state_json: serde_json::Value,
}

impl LoopState {
    #[cfg(test)]
    pub(crate) fn new_for_test(session_id: SessionId, loop_id: LoopId, goal: &str) -> Self {
        Self {
            session_id,
            loop_id,
            user_goal: goal.to_string(),
            plan: None,
            current_node: "start".to_string(),
            iteration: 0,
            tool_rounds_completed: 0,
            model_invocations: 0,
            status: LoopStatus::Running,
            last_completed_node: None,
            last_effect_key: None,
            recent_events: Vec::new(),
            activated_raw: Vec::new(),
            activated_abstract: Vec::new(),
            session_window: Vec::new(),
            pushed_out_raw: Vec::new(),
            tool_result_ids: Vec::new(),
            assistant_message: None,
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
    fn iteration_tracking() {
        let mut state = make_loop_state();
        assert_eq!(state.iteration, 0);
        state.iteration += 1;
        assert_eq!(state.iteration, 1);
        state.iteration += 1;
        assert_eq!(state.iteration, 2);
    }

    #[test]
    fn tool_rounds_and_model_invocations() {
        let mut state = make_loop_state();
        state.tool_rounds_completed = 3;
        state.model_invocations = 5;
        assert_eq!(state.tool_rounds_completed, 3);
        assert_eq!(state.model_invocations, 5);
    }

    #[test]
    fn checkpoint_serialization_roundtrip() {
        let mut state = make_loop_state();
        state.plan = Some("step 1, step 2".to_string());
        state.iteration = 3;
        state.status = LoopStatus::Paused;
        state.assistant_message = Some("partial response".to_string());
        state.state_json = json!({"progress": 0.5});

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: LoopState = serde_json::from_str(&json).unwrap();
        assert_eq!(state, deserialized);
    }

    #[test]
    fn checkpoint_with_node_ids() {
        let mut state = make_loop_state();
        let raw_id = RawNodeId::new();
        let abstract_id = AbstractNodeId::new();
        state.recent_events.push(raw_id);
        state.activated_raw.push(raw_id);
        state.activated_abstract.push(abstract_id);
        state.session_window.push(raw_id);

        let json = serde_json::to_string(&state).unwrap();
        let deserialized: LoopState = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.recent_events, vec![raw_id]);
        assert_eq!(deserialized.activated_abstract, vec![abstract_id]);
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
