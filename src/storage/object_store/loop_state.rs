use async_trait::async_trait;

use crate::domain::LoopState;
use crate::error::Result;
use crate::ids::{LoopId, SessionId};

use crate::storage::traits::LoopStateRepository;

use super::store::FileObjectStore;

#[derive(Debug, Clone)]
pub struct ObjectLoopStateRepository {
    store: FileObjectStore,
}

impl ObjectLoopStateRepository {
    #[must_use]
    pub const fn new(store: FileObjectStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl LoopStateRepository for ObjectLoopStateRepository {
    async fn save_checkpoint(&self, state: LoopState) -> Result<()> {
        let _guard = self.store.lock().await;
        self.store
            .write_json(
                &self
                    .store
                    .checkpoint_path(&state.session_id, &state.loop_id),
                &state,
            )
            .await?;
        self.store.touch_metadata_unlocked().await
    }

    async fn load_checkpoint(
        &self,
        session_id: &SessionId,
        loop_id: &LoopId,
    ) -> Result<Option<LoopState>> {
        let _guard = self.store.lock().await;
        self.store
            .try_read_json(&self.store.checkpoint_path(session_id, loop_id))
            .await
    }

    async fn clear_checkpoint(&self, session_id: &SessionId, loop_id: &LoopId) -> Result<()> {
        let _guard = self.store.lock().await;
        self.store
            .remove_file_if_exists(&self.store.checkpoint_path(session_id, loop_id))
            .await?;
        self.store.touch_metadata_unlocked().await
    }
}
