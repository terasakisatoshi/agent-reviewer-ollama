// Copyright 2026- SiLeader (Cerussite).
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use genai::chat::ToolCall;
use serde_json::Value;
use std::sync::atomic::{AtomicU64, Ordering};

static TEXT_TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Some local models (e.g. Ollama-hosted Qwen) emit tool calls as JSON text instead of
/// structured `tool_calls`. Parse those payloads when the provider did not normalize them.
pub(crate) fn extract_text_tool_calls(text: &str) -> Vec<ToolCall> {
    let trimmed = strip_markdown_fence(text.trim());
    let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
        return Vec::new();
    };

    match value {
        Value::Object(map) => parse_tool_call_object(&map)
            .into_iter()
            .collect(),
        Value::Array(items) => items
            .iter()
            .filter_map(Value::as_object)
            .filter_map(parse_tool_call_object)
            .collect(),
        _ => Vec::new(),
    }
}

fn strip_markdown_fence(text: &str) -> &str {
    let text = text
        .strip_prefix("```json")
        .or_else(|| text.strip_prefix("```"))
        .unwrap_or(text);
    text.strip_suffix("```").unwrap_or(text).trim()
}

fn parse_tool_call_object(map: &serde_json::Map<String, Value>) -> Option<ToolCall> {
    let (fn_name, fn_arguments) = if let (Some(name), Some(arguments)) =
        (map.get("name"), map.get("arguments"))
    {
        (name.as_str()?, arguments.clone())
    } else if let Some(function) = map.get("function").and_then(Value::as_object) {
        (
            function.get("name")?.as_str()?,
            function
                .get("arguments")
                .cloned()
                .unwrap_or(Value::Object(serde_json::Map::new())),
        )
    } else {
        return None;
    };

    Some(ToolCall {
        call_id: format!(
            "call_{}",
            TEXT_TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed)
        ),
        fn_name: fn_name.to_string(),
        fn_arguments,
        thought_signatures: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_json_tool_call() {
        let calls = extract_text_tool_calls(
            r#"{"name":"submit_triage","arguments":{"review_units":[]}}"#,
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].fn_name, "submit_triage");
    }

    #[test]
    fn parses_fenced_json_tool_call() {
        let calls = extract_text_tool_calls(
            "```json\n{\"name\":\"read_file\",\"arguments\":{\"path\":\"Cargo.toml\"}}\n```",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].fn_name, "read_file");
    }
}
