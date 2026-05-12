use thiserror::Error;

use crate::domain::LoopStatus;

pub type Result<T> = std::result::Result<T, EngineError>;

#[allow(dead_code)]
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("configuration error: {0}")]
    Configuration(String),
    #[error("storage error: {0}")]
    Storage(String),
    #[error("model error: {0}")]
    Model(String),
    #[error("tool error: {0}")]
    Tool(String),
    #[error("timeout while executing {0}")]
    Timeout(String),
    #[error("unsupported backend: {0}")]
    Unsupported(&'static str),
    #[error("operation cancelled")]
    Cancelled,
    #[error("loop checkpoint not found for session={session_id} loop={loop_id}")]
    CheckpointNotFound { session_id: String, loop_id: String },
    #[error("loop terminated with status {0:?}")]
    LoopTerminated(LoopStatus),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_error_display() {
        let err = EngineError::Configuration("bad config".to_string());
        assert_eq!(err.to_string(), "configuration error: bad config");
    }

    #[test]
    fn storage_error_display() {
        let err = EngineError::Storage("disk full".to_string());
        assert_eq!(err.to_string(), "storage error: disk full");
    }

    #[test]
    fn model_error_display() {
        let err = EngineError::Model("inference failed".to_string());
        assert_eq!(err.to_string(), "model error: inference failed");
    }

    #[test]
    fn tool_error_display() {
        let err = EngineError::Tool("tool crashed".to_string());
        assert_eq!(err.to_string(), "tool error: tool crashed");
    }

    #[test]
    fn timeout_error_display() {
        let err = EngineError::Timeout("model call".to_string());
        assert_eq!(err.to_string(), "timeout while executing model call");
    }

    #[test]
    fn unsupported_error_display() {
        let err = EngineError::Unsupported("llama");
        assert_eq!(err.to_string(), "unsupported backend: llama");
    }

    #[test]
    fn cancelled_error_display() {
        let err = EngineError::Cancelled;
        assert_eq!(err.to_string(), "operation cancelled");
    }

    #[test]
    fn checkpoint_not_found_display() {
        let err = EngineError::CheckpointNotFound {
            session_id: "sess-1".to_string(),
            loop_id: "loop-1".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "loop checkpoint not found for session=sess-1 loop=loop-1"
        );
    }

    #[test]
    fn loop_terminated_display() {
        let err = EngineError::LoopTerminated(LoopStatus::Failed);
        assert_eq!(err.to_string(), "loop terminated with status Failed");
    }

    #[test]
    fn result_type_alias_works() {
        let ok: Result<i32> = Ok(42);
        match ok {
            Ok(value) => assert_eq!(value, 42),
            Err(err) => panic!("expected Ok(42), got Err({err:?})"),
        }

        let err: Result<i32> = Err(EngineError::Cancelled);
        assert!(err.is_err());
    }

    #[test]
    fn error_is_debug() {
        let err = EngineError::Storage("test".to_string());
        let debug = format!("{err:?}");
        assert!(debug.contains("Storage"));
    }
}
