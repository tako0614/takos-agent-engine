pub mod embedding;
#[cfg(feature = "openai-embeddings")]
pub mod openai_embedding;
#[cfg(feature = "openai-embeddings")]
mod openai_http;
pub mod runner;

pub use embedding::{cosine_similarity, Embedder, Embedding, EmbeddingRef};
#[cfg(feature = "openai-embeddings")]
pub use openai_embedding::{OpenAiCompatibleEmbedder, OpenAiEmbeddingConfig};
pub use runner::{ModelInput, ModelOutput, ModelRunner, ModelUsage, ToolCallRequest};
