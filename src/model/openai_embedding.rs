use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};
use crate::model::embedding::{Embedder, Embedding};
use crate::model::openai_http::send_with_retry;

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone)]
pub struct OpenAiEmbeddingConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    pub dimensions: Option<u32>,
    pub timeout: Duration,
}

impl OpenAiEmbeddingConfig {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            api_key: api_key.into(),
            dimensions: None,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_dimensions(mut self, dimensions: u32) -> Self {
        self.dimensions = Some(dimensions);
        self
    }
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleEmbedder {
    client: reqwest::Client,
    config: OpenAiEmbeddingConfig,
}

impl OpenAiCompatibleEmbedder {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        Self::with_config(OpenAiEmbeddingConfig::new(model, api_key))
    }

    pub fn with_config(config: OpenAiEmbeddingConfig) -> Result<Self> {
        validate_config(&config)?;
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|err| {
                EngineError::Configuration(format!("failed to build HTTP client: {err}"))
            })?;
        Ok(Self { client, config })
    }

    fn embeddings_url(&self) -> String {
        format!("{}/embeddings", self.config.base_url.trim_end_matches('/'))
    }
}

#[async_trait]
impl Embedder for OpenAiCompatibleEmbedder {
    async fn embed_text(&self, text: &str) -> Result<Embedding> {
        let request = EmbeddingsRequest {
            model: &self.config.model,
            input: text,
            dimensions: self.config.dimensions,
        };

        // Transport policy (retry / backoff / jitter / `Retry-After`) lives in
        // `openai_http::send_with_retry`; this client keeps only its request
        // and response shapes.
        let response = send_with_retry("embedding request failed", || {
            self.client
                .post(self.embeddings_url())
                .bearer_auth(&self.config.api_key)
                .json(&request)
        })
        .await?;

        let response = response.json::<EmbeddingsResponse>().await.map_err(|err| {
            EngineError::Model(format!("failed to parse embedding response: {err}"))
        })?;

        let mut data = response.data.into_iter();
        let first = data.next();
        data.find(|item| item.index == Some(0))
            .or(first)
            .map(|item| Embedding(item.embedding))
            .ok_or_else(|| {
                EngineError::Model("embedding response did not contain data".to_string())
            })
    }
}

fn validate_config(config: &OpenAiEmbeddingConfig) -> Result<()> {
    if config.base_url.trim().is_empty() {
        return Err(EngineError::Configuration(
            "OpenAI-compatible embedding base_url must not be empty".to_string(),
        ));
    }
    if config.model.trim().is_empty() {
        return Err(EngineError::Configuration(
            "OpenAI-compatible embedding model must not be empty".to_string(),
        ));
    }
    if config.api_key.trim().is_empty() {
        return Err(EngineError::Configuration(
            "OpenAI-compatible embedding api_key must not be empty".to_string(),
        ));
    }
    Ok(())
}

#[derive(Debug, Serialize)]
struct EmbeddingsRequest<'a> {
    model: &'a str,
    input: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingsResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
    index: Option<usize>,
}

#[cfg(test)]
mod tests {
    use std::str;

    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;

    use super::*;

    #[tokio::test]
    async fn embed_text_sends_openai_compatible_request_and_parses_response() {
        let server = FakeEmbeddingServer::spawn(
            200,
            r#"{"object":"list","data":[{"object":"embedding","index":0,"embedding":[0.1,0.2,0.3]}],"model":"text-embedding-3-small"}"#,
        )
        .await;

        let embedder = OpenAiCompatibleEmbedder::with_config(
            OpenAiEmbeddingConfig::new("text-embedding-3-small", "test-key")
                .with_base_url(server.base_url())
                .with_dimensions(256),
        )
        .unwrap();

        let embedding = embedder.embed_text("hello world").await.unwrap();

        assert_eq!(embedding, Embedding(vec![0.1, 0.2, 0.3]));

        let request = server.request().await;
        let lower_request = request.to_lowercase();
        assert!(request.starts_with("POST /embeddings HTTP/1.1"));
        assert!(lower_request.contains("authorization: bearer test-key"));
        assert!(lower_request.contains("content-type: application/json"));

        let body = request_body(&request);
        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(
            body,
            json!({
                "model": "text-embedding-3-small",
                "input": "hello world",
                "dimensions": 256
            })
        );
    }

    #[tokio::test]
    async fn embed_text_parses_first_embedding_when_index_is_absent() {
        let server = FakeEmbeddingServer::spawn(
            200,
            r#"{"data":[{"embedding":[1.0,2.0]},{"embedding":[3.0,4.0]}]}"#,
        )
        .await;
        let embedder = OpenAiCompatibleEmbedder::with_config(
            OpenAiEmbeddingConfig::new("embedding-model", "test-key")
                .with_base_url(server.base_url()),
        )
        .unwrap();

        let embedding = embedder.embed_text("input").await.unwrap();

        assert_eq!(embedding, Embedding(vec![1.0, 2.0]));
        let _ = server.request().await;
    }

    #[tokio::test]
    async fn embed_text_returns_model_error_for_non_retryable_http_error() {
        // 401 is non-retryable (auth failure), so the embedder must surface
        // the error on the first attempt without consuming any retry budget.
        let server = FakeEmbeddingServer::spawn(
            401,
            r#"{"error":{"message":"missing api key","type":"invalid_request_error"}}"#,
        )
        .await;
        let embedder = OpenAiCompatibleEmbedder::with_config(
            OpenAiEmbeddingConfig::new("embedding-model", "test-key")
                .with_base_url(server.base_url()),
        )
        .unwrap();

        let err = embedder.embed_text("input").await.unwrap_err();

        assert!(matches!(err, EngineError::Model(_)));
        let message = err.to_string();
        assert!(message.contains("HTTP 401"));
        assert!(message.contains("missing api key"));
        let _ = server.request().await;
    }

    #[tokio::test]
    async fn embed_text_retries_on_429_and_succeeds() {
        // First connection sees 429 (retryable), second connection sees 200.
        // We confirm the embedder consumes both responses and ultimately
        // returns the successful embedding payload.
        let scripted = ScriptedFakeEmbeddingServer::spawn(vec![
            (
                429,
                r#"{"error":{"message":"rate limited","type":"rate_limit_error"}}"#,
                Some("0"),
            ),
            (200, r#"{"data":[{"embedding":[0.5,0.5],"index":0}]}"#, None),
        ])
        .await;

        let embedder = OpenAiCompatibleEmbedder::with_config(
            OpenAiEmbeddingConfig::new("embedding-model", "test-key")
                .with_base_url(scripted.base_url()),
        )
        .unwrap();

        let embedding = embedder.embed_text("hello").await.unwrap();

        assert_eq!(embedding, Embedding(vec![0.5, 0.5]));
        let request_count = scripted.shutdown().await;
        assert_eq!(request_count, 2, "embedder should have retried once");
    }

    #[test]
    fn empty_api_key_is_rejected() {
        let err = OpenAiCompatibleEmbedder::new("embedding-model", "").unwrap_err();
        assert!(matches!(err, EngineError::Configuration(_)));
        assert!(err.to_string().contains("api_key"));
    }

    struct FakeEmbeddingServer {
        address: std::net::SocketAddr,
        handle: JoinHandle<String>,
    }

    impl FakeEmbeddingServer {
        async fn spawn(status: u16, response_body: &'static str) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let handle = tokio::spawn(async move {
                let (mut stream, _) = listener.accept().await.unwrap();
                let request = read_http_request(&mut stream).await;
                let response = format!(
                    "HTTP/1.1 {status} OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{response_body}",
                    response_body.len()
                );
                stream.write_all(response.as_bytes()).await.unwrap();
                request
            });
            Self { address, handle }
        }

        fn base_url(&self) -> String {
            format!("http://{}", self.address)
        }

        async fn request(self) -> String {
            self.handle.await.unwrap()
        }
    }

    /// Multi-shot fake server: serves one canned response per connection,
    /// in the order supplied. Each item is `(status, body, retry_after)`.
    /// The `retry_after` value (if `Some`) is sent as a `Retry-After`
    /// header so the client's backoff path is exercised end-to-end.
    struct ScriptedFakeEmbeddingServer {
        address: std::net::SocketAddr,
        handle: JoinHandle<usize>,
    }

    impl ScriptedFakeEmbeddingServer {
        async fn spawn(script: Vec<(u16, &'static str, Option<&'static str>)>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let address = listener.local_addr().unwrap();
            let handle = tokio::spawn(async move {
                let mut served: usize = 0;
                for (status, body, retry_after) in script {
                    let (mut stream, _) = listener.accept().await.unwrap();
                    let _ = read_http_request(&mut stream).await;
                    let retry_after_header = retry_after
                        .map(|value| format!("retry-after: {value}\r\n"))
                        .unwrap_or_default();
                    let response = format!(
                        "HTTP/1.1 {status} OK\r\ncontent-type: application/json\r\n{retry}content-length: {}\r\nconnection: close\r\n\r\n{body}",
                        body.len(),
                        retry = retry_after_header,
                    );
                    stream.write_all(response.as_bytes()).await.unwrap();
                    served += 1;
                }
                served
            });
            Self { address, handle }
        }

        fn base_url(&self) -> String {
            format!("http://{}", self.address)
        }

        async fn shutdown(self) -> usize {
            self.handle.await.unwrap()
        }
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> String {
        let mut buffer = Vec::new();
        let mut temp = [0; 1024];
        let header_end;
        loop {
            let read = stream.read(&mut temp).await.unwrap();
            assert_ne!(read, 0, "client closed before sending request headers");
            buffer.extend_from_slice(&temp[..read]);
            if let Some(position) = find_header_end(&buffer) {
                header_end = position;
                break;
            }
        }

        let headers = str::from_utf8(&buffer[..header_end]).unwrap();
        let content_length = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                if name.eq_ignore_ascii_case("content-length") {
                    Some(value.trim().parse::<usize>().unwrap())
                } else {
                    None
                }
            })
            .unwrap_or(0);
        let body_start = header_end + 4;
        while buffer.len() < body_start + content_length {
            let read = stream.read(&mut temp).await.unwrap();
            assert_ne!(read, 0, "client closed before sending request body");
            buffer.extend_from_slice(&temp[..read]);
        }

        String::from_utf8(buffer[..body_start + content_length].to_vec()).unwrap()
    }

    fn find_header_end(buffer: &[u8]) -> Option<usize> {
        buffer.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn request_body(request: &str) -> &str {
        request.split_once("\r\n\r\n").unwrap().1
    }
}
