//! Shared HTTP transport policy for the OpenAI-compatible chat and embedding
//! clients. Both clients post a JSON request to an OpenAI-style endpoint and
//! share an identical retry / backoff / jitter / `Retry-After` policy; only
//! their request and response shapes differ. This module owns the transport
//! concern so the two clients keep only their wire shapes.

use std::time::Duration;

use crate::error::{EngineError, Result};

/// Maximum number of additional attempts after the first failure when the
/// remote side returns a transient status (429 / 502 / 503 / 504). This
/// caps total backoff at roughly 1 + 2 + 4 + 8 = 15 seconds plus jitter.
const RETRY_MAX_ATTEMPTS: u32 = 4;

/// Send a request with the shared retry policy and return the first successful
/// HTTP response. `build_request` is called once per attempt to produce a fresh
/// `RequestBuilder` (each attempt consumes one via `send`). `label` is the
/// human-readable request description used in error messages (e.g.
/// `"chat completion request failed"`).
///
/// Retry loop: re-send the request on transient upstream failures (`429` rate
/// limit / `502` / `503` / `504`). Each retry waits `1s, 2s, 4s, 8s` plus
/// jitter (max 4 retries). When the server returns a `Retry-After` header that
/// exceeds the computed backoff, we honour it. Non-retryable HTTP errors and
/// 4xx-other-than-429 still surface immediately, matching the pre-retry
/// behaviour.
pub(crate) async fn send_with_retry<F>(
    label: &str,
    mut build_request: F,
) -> Result<reqwest::Response>
where
    F: FnMut() -> reqwest::RequestBuilder,
{
    let mut attempt: u32 = 0;
    loop {
        let send_result = build_request().send().await;

        let response = match send_result {
            Ok(response) => response,
            Err(err) => {
                return Err(EngineError::Model(format!("{label}: {err}")));
            }
        };

        let status = response.status();
        if status.is_success() {
            return Ok(response);
        }

        let retry_after_header = response
            .headers()
            .get(reqwest::header::RETRY_AFTER)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);

        let is_retryable_status = status.as_u16() == 429 || matches!(status.as_u16(), 502..=504);
        if !is_retryable_status || attempt >= RETRY_MAX_ATTEMPTS {
            let body = response.text().await.unwrap_or_default();
            return Err(EngineError::Model(format!(
                "{label} with HTTP {status}: {body}"
            )));
        }

        // Drain the body so the connection can be reused for the retry.
        let _ = response.text().await;
        let backoff = retry_backoff(attempt, retry_after_header.as_deref());
        tokio::time::sleep(backoff).await;
        attempt += 1;
    }
}

/// Compute the backoff duration before the next retry attempt.
///
/// Sequence: `attempt=0 -> 1s`, `attempt=1 -> 2s`, `attempt=2 -> 4s`,
/// `attempt=3 -> 8s`. Jitter adds 0-25% on top so concurrent clients do not
/// re-synchronise after a shared rate-limit event. When the server supplies a
/// `Retry-After` value larger than the computed backoff we use the server
/// value instead.
fn retry_backoff(attempt: u32, retry_after_header: Option<&str>) -> Duration {
    let base_secs = 1u64.checked_shl(attempt).unwrap_or(u64::MAX).min(8);
    let base = Duration::from_secs(base_secs);
    let jitter_ms = retry_jitter_ms_for_attempt(attempt, base_secs);
    let with_jitter = base.saturating_add(Duration::from_millis(jitter_ms));
    if let Some(server_value) = retry_after_header.and_then(parse_retry_after_seconds) {
        let server_duration = Duration::from_secs(server_value);
        if server_duration > with_jitter {
            return server_duration;
        }
    }
    with_jitter
}

/// Pseudo-random jitter of up to 25% of the base backoff, sourced from the
/// system clock so we do not need a `rand` dependency. The exact value does
/// not need to be cryptographic — its only job is to break up synchronised
/// retry storms across multiple clients.
fn retry_jitter_ms_for_attempt(attempt: u32, base_secs: u64) -> u64 {
    let now_nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.subsec_nanos() as u64)
        .unwrap_or(0);
    let entropy = now_nanos
        .wrapping_mul(0x9e37_79b9_7f4a_7c15)
        .wrapping_add(attempt as u64);
    let max_jitter_ms = base_secs.saturating_mul(250); // 25% of base, in ms.
    if max_jitter_ms == 0 {
        0
    } else {
        entropy % max_jitter_ms
    }
}

/// Parse a `Retry-After: <seconds>` value. We intentionally do not support
/// the HTTP-date form because OpenAI-compatible endpoints only send integer
/// seconds and adding a date parser would pull in a heavier dependency.
fn parse_retry_after_seconds(value: &str) -> Option<u64> {
    value.trim().parse::<u64>().ok()
}
