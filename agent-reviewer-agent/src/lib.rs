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

pub mod builder;
mod concurrency;
mod text_tool_calls;
pub mod tools;

use agent_reviewer_tools::CompoundAgentTools;
pub use concurrency::ConcurrencyLimiter;
use genai::Client;
use genai::chat::{ChatMessage, ChatOptions, ChatRequest, MessageContent, ToolCall};
use text_tool_calls::extract_text_tool_calls;
use tracing::{debug, info};

pub struct ReActAgent {
    model_name: String,
    client: Client,
    tools: CompoundAgentTools,
    max_loop_count: usize,
    options: Option<ChatOptions>,
    submit_tool_name: String,
    concurrency_limiter: ConcurrencyLimiter,
}

impl ReActAgent {
    pub fn builder(concurrency_limiter: ConcurrencyLimiter) -> builder::ReActAgentBuilder {
        builder::ReActAgentBuilder::new(concurrency_limiter)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_name: String,
        client: Client,
        tools: CompoundAgentTools,
        max_loop_count: usize,
        submit_tool_name: String,
        options: Option<ChatOptions>,
        concurrency_limiter: ConcurrencyLimiter,
    ) -> Self {
        Self {
            model_name,
            client,
            tools,
            max_loop_count,
            options,
            submit_tool_name,
            concurrency_limiter,
        }
    }

    fn create_request(
        &self,
        system: String,
        messages: Vec<ChatMessage>,
        is_last_turn: bool,
    ) -> ChatRequest {
        ChatRequest {
            system: Some(system),
            messages,
            tools: if is_last_turn {
                Some(
                    self.tools
                        .get_tool_description_by_name(&self.submit_tool_name)
                        .into_iter()
                        .collect(),
                )
            } else {
                Some(self.tools.description())
            },
            previous_response_id: None,
            store: Some(false),
        }
    }

    fn step_worder(&self, current: usize) -> String {
        format!("You are in step {} of {}.", current, self.max_loop_count)
    }

    async fn try_recover_marker(
        &self,
        system_prompt: &str,
        messages: &[ChatMessage],
    ) -> anyhow::Result<Option<serde_json::Value>> {
        let mut recovery_messages = messages.to_vec();
        recovery_messages.push(ChatMessage::user(format!(
            "This is the final recovery step. You MUST call `{}` now. \
             Respond with ONLY a JSON object shaped like \
             {{\"name\":\"{}\",\"arguments\":{{...}}}} using the context gathered so far.",
            self.submit_tool_name, self.submit_tool_name
        )));

        let request = ChatRequest {
            system: Some(system_prompt.to_string()),
            messages: recovery_messages,
            tools: Some(
                self.tools
                    .get_tool_description_by_name(&self.submit_tool_name)
                    .into_iter()
                    .collect(),
            ),
            previous_response_id: None,
            store: Some(false),
        };

        let response = {
            let _permit = self.concurrency_limiter.acquire().await?;
            self.client
                .exec_chat(&self.model_name, request, self.options.as_ref())
                .await?
        };

        let mut tool_calls: Vec<ToolCall> = response.content.tool_calls().into_iter().cloned().collect();
        if tool_calls.is_empty()
            && let Some(text) = response.first_text()
        {
            tool_calls = extract_text_tool_calls(text);
        }

        Ok(tool_calls
            .iter()
            .find(|call| call.fn_name == self.submit_tool_name)
            .map(|call| call.fn_arguments.clone()))
    }

    pub async fn run(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> anyhow::Result<serde_json::Value> {
        let mut messages = Vec::new();

        for i in 1..=self.max_loop_count {
            info!(
                "Starting step {}/{} with model {}",
                i, self.max_loop_count, self.model_name
            );

            let is_last_turn = i == self.max_loop_count;

            let user = if i == 1 {
                ChatMessage::user(format!("{}\n{user_prompt}", self.step_worder(1)))
            } else if is_last_turn {
                ChatMessage::user(format!(
                    "{}\nThis is the final step. Make sure to call the '{}' tool if you haven't already.",
                    self.step_worder(i),
                    self.submit_tool_name
                ))
            } else {
                ChatMessage::user(self.step_worder(i))
            };
            debug!("User message: {:?}", user);
            messages.push(user);

            let request =
                self.create_request(system_prompt.to_string(), messages.clone(), is_last_turn);
            let response = {
                let _permit = self.concurrency_limiter.acquire().await?;
                self.client
                    .exec_chat(&self.model_name, request, self.options.as_ref())
                    .await?
            };
            debug!("Model response: {:?}", response);

            let structured_tool_calls = !response.content.tool_calls().is_empty();
            let mut tool_calls: Vec<ToolCall> =
                response.content.tool_calls().into_iter().cloned().collect();
            if tool_calls.is_empty()
                && let Some(text) = response.first_text()
            {
                tool_calls = extract_text_tool_calls(text);
                if !tool_calls.is_empty() {
                    debug!("Recovered {} tool call(s) from text content", tool_calls.len());
                }
            }

            let assistant_message = if !tool_calls.is_empty() && !structured_tool_calls {
                ChatMessage::assistant(MessageContent::from_tool_calls(tool_calls.clone()))
            } else {
                ChatMessage::assistant(response.content.clone())
            };
            messages.push(assistant_message);

            let tool_call_refs: Vec<&ToolCall> = tool_calls.iter().collect();
            let (markers, non_markers) = self
                .tools
                .separate_marker_and_non_marker(tool_call_refs);
            if let Some(call) = markers
                .iter()
                .find(|call| call.fn_name == self.submit_tool_name)
            {
                debug!("Marker tool call: {:?}", call);
                return Ok(call.fn_arguments.clone());
            }

            if is_last_turn {
                if let Some(args) = self.try_recover_marker(system_prompt, &messages).await? {
                    debug!("Recovered marker arguments on final recovery step");
                    return Ok(args);
                }
            }

            messages.push(self.tools.run_all_non_markers(non_markers).await);
        }
        anyhow::bail!("Exceeded max loop count")
    }
}
