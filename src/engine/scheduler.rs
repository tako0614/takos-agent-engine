use std::sync::Arc;

use tokio::task::JoinHandle;

use crate::config::EngineConfig;
use crate::error::Result;

use super::session_engine::{run_turn, EngineDeps, SessionRequest, SessionResponse};

#[allow(dead_code)]
#[derive(Clone)]
pub struct SessionScheduler {
    config: Arc<EngineConfig>,
    deps: Arc<EngineDeps>,
}

#[allow(dead_code)]
impl SessionScheduler {
    pub fn new(config: EngineConfig, deps: Arc<EngineDeps>) -> Self {
        Self {
            config: Arc::new(config),
            deps,
        }
    }

    pub fn spawn(&self, request: SessionRequest) -> JoinHandle<Result<SessionResponse>> {
        let config = self.config.clone();
        let deps = self.deps.clone();
        tokio::spawn(async move { run_turn(&config, deps.as_ref(), request).await })
    }
}
