use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{EngineError, Result};
use crate::model::openai_http::send_with_retry;
use crate::model::runner::{ModelInput, ModelOutput, ModelRunner, ModelUsage, ToolCallRequest};

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";
const DEFAULT_API_KEY_ENV: &str = "OPENAI_API_KEY";

#[derive(Debug, Clone)]
pub struct OpenAiChatConfig {
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    pub tools: Vec<OpenAiChatToolDefinition>,
    pub tool_choice: Option<serde_json::Value>,
    pub timeout: Duration,
}

impl OpenAiChatConfig {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            model: model.into(),
            api_key: api_key.into(),
            temperature: None,
            max_tokens: None,
            tools: Vec::new(),
            tool_choice: None,
            timeout: Duration::from_secs(60),
        }
    }

    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        let api_key = std::env::var(DEFAULT_API_KEY_ENV).map_err(|_| {
            EngineError::Configuration(format!(
                "missing {DEFAULT_API_KEY_ENV} for OpenAI-compatible chat"
            ))
        })?;
        Ok(Self::new(model, api_key))
    }

    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    pub fn with_tools(mut self, tools: impl IntoIterator<Item = OpenAiChatToolDefinition>) -> Self {
        self.tools = tools.into_iter().collect();
        self
    }

    pub fn with_tool_choice(mut self, tool_choice: impl Into<serde_json::Value>) -> Self {
        self.tool_choice = Some(tool_choice.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAiChatToolDefinition {
    #[serde(rename = "type")]
    pub kind: String,
    pub function: OpenAiChatFunctionTool,
}

impl OpenAiChatToolDefinition {
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: serde_json::Value,
    ) -> Self {
        Self {
            kind: "function".to_string(),
            function: OpenAiChatFunctionTool {
                name: name.into(),
                description: Some(description.into()),
                parameters,
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAiChatFunctionTool {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct OpenAiCompatibleChatModelRunner {
    client: reqwest::Client,
    config: OpenAiChatConfig,
}

impl OpenAiCompatibleChatModelRunner {
    pub fn new(model: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        Self::with_config(OpenAiChatConfig::new(model, api_key))
    }

    pub fn from_env(model: impl Into<String>) -> Result<Self> {
        Self::with_config(OpenAiChatConfig::from_env(model)?)
    }

    pub fn with_config(config: OpenAiChatConfig) -> Result<Self> {
        validate_config(&config)?;
        let client = reqwest::Client::builder()
            .timeout(config.timeout)
            .build()
            .map_err(|err| {
                EngineError::Configuration(format!("failed to build HTTP client: {err}"))
            })?;
        Ok(Self { client, config })
    }

    fn chat_completions_url(&self) -> String {
        format!(
            "{}/chat/completions",
            self.config.base_url.trim_end_matches('/')
        )
    }
}

#[async_trait]
impl ModelRunner for OpenAiCompatibleChatModelRunner {
    async fn run(&self, input: ModelInput) -> Result<ModelOutput> {
        let request = ChatCompletionsRequest {
            model: &self.config.model,
            messages: messages_for_input(input),
            temperature: self.config.temperature,
            max_tokens: self.config.max_tokens,
            tools: &self.config.tools,
            tool_choice: self.config.tool_choice.as_ref(),
        };

        // Transport policy (retry / backoff / jitter / `Retry-After`) lives in
        // `openai_http::send_with_retry`; this client keeps only its request
        // and response shapes.
        let response = send_with_retry("chat completion request failed", || {
            self.client
                .post(self.chat_completions_url())
                .bearer_auth(&self.config.api_key)
                .json(&request)
        })
        .await?;

        let response = response
            .json::<ChatCompletionsResponse>()
            .await
            .map_err(|err| {
                EngineError::Model(format!("failed to parse chat completion response: {err}"))
            })?;
        let usage = response.usage.as_ref().map(|u| ModelUsage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cached_input_tokens: u
                .prompt_tokens_details
                .as_ref()
                .map_or(0, |d| d.cached_tokens),
        });
        let message = response
            .choices
            .into_iter()
            .next()
            .map(|choice| choice.message)
            .ok_or_else(|| {
                EngineError::Model("chat completion response did not contain choices".to_string())
            })?;

        Ok(ModelOutput {
            assistant_message: message.content.filter(|content| !content.is_empty()),
            tool_calls: parse_tool_calls(message.tool_calls, message.function_call)?,
            usage,
        })
    }
}

fn validate_config(config: &OpenAiChatConfig) -> Result<()> {
    if config.base_url.trim().is_empty() {
        return Err(EngineError::Configuration(
            "OpenAI-compatible chat base_url must not be empty".to_string(),
        ));
    }
    if config.model.trim().is_empty() {
        return Err(EngineError::Configuration(
            "OpenAI-compatible chat model must not be empty".to_string(),
        ));
    }
    if config.api_key.trim().is_empty() {
        return Err(EngineError::Configuration(
            "OpenAI-compatible chat api_key must not be empty".to_string(),
        ));
    }
    Ok(())
}

fn messages_for_input(input: ModelInput) -> Vec<ChatMessage> {
    let mut messages = Vec::new();
    if !input.system_prompt.trim().is_empty() {
        messages.push(ChatMessage {
            role: "system",
            content: input.system_prompt,
        });
    }

    let mut user_content = Vec::new();
    user_content.push(format!("Session ID: {}", input.session_id));
    user_content.push(format!("Loop ID: {}", input.loop_id));
    if let Some(plan) = input.plan {
        push_section(&mut user_content, "Plan", &[plan]);
    }
    push_section(&mut user_content, "Session context", &input.session_context);
    push_section(&mut user_content, "Memory context", &input.memory_context);
    push_section(&mut user_content, "Tool context", &input.tool_context);
    push_section(&mut user_content, "User message", &[input.user_message]);

    messages.push(ChatMessage {
        role: "user",
        content: user_content.join("\n\n"),
    });
    messages
}

fn push_section(lines: &mut Vec<String>, title: &str, values: &[String]) {
    if values.is_empty() {
        return;
    }
    let mut section = String::from(title);
    section.push_str(":\n");
    section.push_str(&values.join("\n"));
    lines.push(section);
}

fn parse_tool_calls(
    tool_calls: Option<Vec<OpenAiToolCall>>,
    function_call: Option<OpenAiFunctionCall>,
) -> Result<Vec<ToolCallRequest>> {
    let mut parsed = Vec::new();
    if let Some(tool_calls) = tool_calls {
        for call in tool_calls {
            let function = call.function.ok_or_else(|| {
                EngineError::Model("tool call response did not contain function".to_string())
            })?;
            parsed.push(parse_function_call(function)?);
        }
    }
    if let Some(function_call) = function_call {
        parsed.push(parse_function_call(function_call)?);
    }
    Ok(parsed)
}

fn parse_function_call(function: OpenAiFunctionCall) -> Result<ToolCallRequest> {
    if function.name.trim().is_empty() {
        return Err(EngineError::Model(
            "tool call response contained empty function name".to_string(),
        ));
    }
    let arguments = match function.arguments {
        None => serde_json::Value::Object(serde_json::Map::new()),
        Some(serde_json::Value::String(arguments)) if arguments.trim().is_empty() => {
            serde_json::Value::Object(serde_json::Map::new())
        }
        Some(serde_json::Value::String(arguments)) => {
            serde_json::from_str(&arguments).map_err(|err| {
                EngineError::Model(format!(
                    "failed to parse tool call arguments for {}: {err}",
                    function.name
                ))
            })?
        }
        Some(arguments) => arguments,
    };

    Ok(ToolCallRequest {
        name: function.name,
        arguments,
    })
}

#[derive(Debug, Serialize)]
struct ChatCompletionsRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "is_empty_tools")]
    tools: &'a [OpenAiChatToolDefinition],
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a serde_json::Value>,
}

fn is_empty_tools(tools: &[OpenAiChatToolDefinition]) -> bool {
    tools.is_empty()
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatCompletionsResponse {
    choices: Vec<ChatChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    prompt_tokens_details: Option<OpenAiPromptTokensDetails>,
}

#[derive(Debug, Deserialize, Default)]
struct OpenAiPromptTokensDetails {
    #[serde(default)]
    cached_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatChoiceMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAiToolCall>>,
    function_call: Option<OpenAiFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct OpenAiToolCall {
    function: Option<OpenAiFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct OpenAiFunctionCall {
    name: String,
    arguments: Option<serde_json::Value>,
}

#[cfg(test)]
mod tests {
    use std::str;

    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::task::JoinHandle;

    use super::*;
    use crate::ids::{LoopId, SessionId};

    #[tokio::test]
    async fn run_sends_openai_compatible_request_and_parses_text_response() {
        let server = FakeChatServer::spawn(
            200,
            r#"{"choices":[{"message":{"role":"assistant","content":"hello from model"}}]}"#,
        )
        .await;
        let runner = OpenAiCompatibleChatModelRunner::with_config(
            OpenAiChatConfig::new("chat-model", "test-key")
                .with_base_url(server.base_url())
                .with_temperature(0.2)
                .with_max_tokens(128),
        )
        .unwrap();

        let output = runner.run(sample_input()).await.unwrap();

        assert_eq!(
            output.assistant_message.as_deref(),
            Some("hello from model")
        );
        assert!(output.tool_calls.is_empty());

        let request = server.request().await;
        let lower_request = request.to_lowercase();
        assert!(request.starts_with("POST /chat/completions HTTP/1.1"));
        assert!(lower_request.contains("authorization: bearer test-key"));
        assert!(lower_request.contains("content-type: application/json"));

        let body = request_body(&request);
        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["model"], "chat-model");
        assert_eq!(body["temperature"], json!(0.2));
        assert_eq!(body["max_tokens"], json!(128));
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["messages"][0]["content"], "system prompt");
        assert_eq!(body["messages"][1]["role"], "user");
        let user_content = body["messages"][1]["content"].as_str().unwrap();
        assert!(user_content.contains("Session ID:"));
        assert!(user_content.contains("Plan:\nplan text"));
        assert!(user_content.contains("Session context:\nrecent message"));
        assert!(user_content.contains("Memory context:\nremembered fact"));
        assert!(user_content.contains("Tool context:\ntool result"));
        assert!(user_content.contains("User message:\nwhat next?"));
    }

    #[tokio::test]
    async fn run_parses_tool_calls() {
        let server = FakeChatServer::spawn(
            200,
            r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"semantic_search_memory","arguments":"{\"query\":\"rust\",\"top_k\":2}"}}]}}]}"#,
        )
        .await;
        let runner = OpenAiCompatibleChatModelRunner::with_config(
            OpenAiChatConfig::new("chat-model", "test-key").with_base_url(server.base_url()),
        )
        .unwrap();

        let output = runner.run(sample_input()).await.unwrap();

        assert_eq!(output.assistant_message, None);
        assert_eq!(
            output.tool_calls,
            vec![ToolCallRequest {
                name: "semantic_search_memory".to_string(),
                arguments: json!({ "query": "rust", "top_k": 2 })
            }]
        );
        let _ = server.request().await;
    }

    #[tokio::test]
    async fn run_sends_tools_and_parses_tool_call_response() {
        let server = FakeChatServer::spawn(
            200,
            r#"{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"semantic_search_memory","arguments":"{\"query\":\"rust\",\"top_k\":2}"}}]}}]}"#,
        )
        .await;
        let tool = OpenAiChatToolDefinition::function(
            "semantic_search_memory",
            "Search activated and long-term memory.",
            json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string" },
                    "top_k": { "type": "integer" }
                },
                "required": ["query"]
            }),
        );
        let runner = OpenAiCompatibleChatModelRunner::with_config(
            OpenAiChatConfig::new("chat-model", "test-key")
                .with_base_url(server.base_url())
                .with_tools([tool])
                .with_tool_choice("auto"),
        )
        .unwrap();

        let output = runner.run(sample_input()).await.unwrap();

        assert_eq!(
            output.tool_calls,
            vec![ToolCallRequest {
                name: "semantic_search_memory".to_string(),
                arguments: json!({ "query": "rust", "top_k": 2 })
            }]
        );

        let request = server.request().await;
        let body = request_body(&request);
        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["tool_choice"], json!("auto"));
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(
            body["tools"][0]["function"]["name"],
            "semantic_search_memory"
        );
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["required"],
            json!(["query"])
        );
    }

    #[tokio::test]
    async fn run_returns_model_error_for_http_error() {
        let server = FakeChatServer::spawn(
            500,
            r#"{"error":{"message":"backend unavailable","type":"server_error"}}"#,
        )
        .await;
        let runner = OpenAiCompatibleChatModelRunner::with_config(
            OpenAiChatConfig::new("chat-model", "test-key").with_base_url(server.base_url()),
        )
        .unwrap();

        let err = runner.run(sample_input()).await.unwrap_err();

        assert!(matches!(err, EngineError::Model(_)));
        let message = err.to_string();
        assert!(message.contains("HTTP 500"));
        assert!(message.contains("backend unavailable"));
        let _ = server.request().await;
    }

    #[tokio::test]
    async fn run_retries_on_503_and_recovers() {
        // First request gets 503 (retryable transient backend error), second
        // gets a successful payload. Confirms both the retry and that the
        // final parsed output reflects the second response.
        let scripted = ScriptedFakeChatServer::spawn(vec![
            (
                503,
                r#"{"error":{"message":"backend unavailable","type":"server_error"}}"#,
                Some("0"),
            ),
            (
                200,
                r#"{"choices":[{"message":{"role":"assistant","content":"recovered"}}]}"#,
                None,
            ),
        ])
        .await;
        let runner = OpenAiCompatibleChatModelRunner::with_config(
            OpenAiChatConfig::new("chat-model", "test-key").with_base_url(scripted.base_url()),
        )
        .unwrap();

        let output = runner.run(sample_input()).await.unwrap();
        assert_eq!(output.assistant_message.as_deref(), Some("recovered"));

        let served = scripted.shutdown().await;
        assert_eq!(served, 2, "chat runner should have retried once");
    }

    #[test]
    fn empty_model_is_rejected() {
        let err = OpenAiCompatibleChatModelRunner::new("", "test-key").unwrap_err();
        assert!(matches!(err, EngineError::Configuration(_)));
        assert!(err.to_string().contains("model"));
    }

    fn sample_input() -> ModelInput {
        ModelInput {
            session_id: SessionId::new(),
            loop_id: LoopId::new(),
            system_prompt: "system prompt".to_string(),
            session_context: vec!["recent message".to_string()],
            memory_context: vec!["remembered fact".to_string()],
            tool_context: vec!["tool result".to_string()],
            user_message: "what next?".to_string(),
            plan: Some("plan text".to_string()),
        }
    }

    struct FakeChatServer {
        address: std::net::SocketAddr,
        handle: JoinHandle<String>,
    }

    impl FakeChatServer {
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

    /// Multi-shot fake server, mirrors the helper in `openai_embedding.rs`
    /// tests. Each tuple in `script` is `(status, body, retry_after)`.
    struct ScriptedFakeChatServer {
        address: std::net::SocketAddr,
        handle: JoinHandle<usize>,
    }

    impl ScriptedFakeChatServer {
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
