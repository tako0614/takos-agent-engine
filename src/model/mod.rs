pub mod embedding;
#[cfg(feature = "openai-chat")]
pub mod openai_chat;
#[cfg(feature = "openai-embeddings")]
pub mod openai_embedding;
pub mod runner;

pub use embedding::{cosine_similarity, Embedder, Embedding, EmbeddingRef};
#[cfg(feature = "openai-chat")]
pub use openai_chat::{
    OpenAiChatConfig, OpenAiChatFunctionTool, OpenAiChatToolDefinition,
    OpenAiCompatibleChatModelRunner,
};
#[cfg(feature = "openai-embeddings")]
pub use openai_embedding::{OpenAiCompatibleEmbedder, OpenAiEmbeddingConfig};
pub use runner::{ModelInput, ModelOutput, ModelRunner, ModelUsage, ToolCallRequest};
