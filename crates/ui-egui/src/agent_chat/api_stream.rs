//! Bounded Chat Completions event streams. Tool arguments are assembled completely before
//! execution; metadata such as Gemini's thought signatures travels with the tool call.

use std::collections::BTreeMap;

use serde_json::{Value, json};

const MAX_BYTES: usize = 16 << 20;
const MAX_EVENT: usize = 1 << 20;

#[derive(Default)]
pub(super) struct Reply {
    pending: Vec<u8>,
    data: Vec<u8>,
    bytes: usize,
    content: String,
    thought: String,
    calls: BTreeMap<usize, Value>,
    finished: bool,
    usage: Value,
}

impl Reply {
    /// Provider-supplied reasoning for display; never infer reasoning from a delay.
    pub fn take_thought(&mut self) -> String {
        std::mem::take(&mut self.thought)
    }

    pub fn tool_preview(&self) -> String {
        self.calls.values().filter_map(|call| call.pointer("/function/name").and_then(Value::as_str)).collect::<Vec<_>>().join(", ")
    }

    pub fn push(&mut self, chunk: &[u8], text: &mut dyn FnMut(&str) -> Result<(), String>) -> Result<(), String> {
        self.bytes = self.bytes.saturating_add(chunk.len());
        if self.bytes > MAX_BYTES {
            return Err("model stream exceeded 16 MiB".into());
        }
        for &byte in chunk {
            if byte == b'\n' {
                let line = std::mem::take(&mut self.pending);
                self.line(line.strip_suffix(b"\r").unwrap_or(&line), text)?;
            } else {
                if self.pending.len() >= MAX_EVENT {
                    return Err("model stream event exceeded 1 MiB".into());
                }
                self.pending.push(byte);
            }
        }
        Ok(())
    }

    fn line(&mut self, line: &[u8], text: &mut dyn FnMut(&str) -> Result<(), String>) -> Result<(), String> {
        if line.is_empty() {
            if self.data.is_empty() {
                return Ok(());
            }
            let data = std::mem::take(&mut self.data);
            if data == b"[DONE]" {
                self.finished = true;
                return Ok(());
            }
            let value: Value = serde_json::from_slice(&data).map_err(|e| format!("invalid model stream: {e}"))?;
            if value.get("usage").is_some_and(Value::is_object) {
                self.usage = value["usage"].clone();
            }
            if let Some(error) = value.get("error") {
                return Err(error.get("message").and_then(Value::as_str).unwrap_or("model stream failed").chars().take(300).collect());
            }
            if let Some(choice) = value.get("choices").and_then(Value::as_array).and_then(|c| c.first()) {
                if let Some(reason) = choice.get("finish_reason").and_then(Value::as_str) {
                    if !matches!(reason, "stop" | "tool_calls") {
                        return Err(format!("model response stopped: {reason}; no partial tool calls were run"));
                    }
                    self.finished = true;
                }
                if let Some(delta) = choice.get("delta") {
                    if let Some(part) = delta.get("reasoning_content").or_else(|| delta.get("reasoning")).and_then(Value::as_str) {
                        self.thought.push_str(part);
                    }
                    if let Some(part) = delta.get("content").and_then(Value::as_str).filter(|s| !s.is_empty()) {
                        self.content.push_str(part);
                        text(part)?;
                    }
                    let calls = delta.get("tool_calls").and_then(Value::as_array).map(Vec::as_slice).unwrap_or(&[]);
                    if calls.len() > 1 && calls.iter().any(|call| call.get("index").is_none()) {
                        return Err("model stream omitted indices for multiple tool calls".into());
                    }
                    for call in calls {
                        let index = if let Some(index) = call.get("index") {
                            index.as_u64().filter(|i| *i < 32).map(|i| i as usize)
                        } else {
                            let id = call.get("id").and_then(Value::as_str);
                            let matching_id = id.and_then(|id| {
                                self.calls
                                    .iter()
                                    .find(|(_, existing)| existing.get("id").and_then(Value::as_str) == Some(id))
                                    .map(|(index, _)| *index)
                            });
                            matching_id.or_else(|| {
                                if self.calls.is_empty() {
                                    Some(0)
                                } else if id.is_none() && self.calls.len() == 1 {
                                    self.calls.keys().next().copied()
                                } else {
                                    None
                                }
                            })
                        }
                        .ok_or_else(|| "invalid or ambiguous model stream tool index".to_string())?;
                        let target = self.calls.entry(index).or_insert_with(|| json!({"type": "function", "function": {}}));
                        merge(target, call);
                        if target.pointer("/function/name").and_then(Value::as_str).is_some_and(|s| s.len() > 128)
                            || target.get("id").and_then(Value::as_str).is_some_and(|s| s.len() > 256)
                            || target.pointer("/function/arguments").and_then(Value::as_str).is_some_and(|s| s.len() > MAX_EVENT)
                        {
                            return Err("model stream tool call exceeded its size limit".into());
                        }
                    }
                }
            }
        } else if let Some(data) = line.strip_prefix(b"data:") {
            let data = data.strip_prefix(b" ").unwrap_or(data);
            if self.data.len().saturating_add(data.len()).saturating_add(1) > MAX_EVENT {
                return Err("model stream event exceeded 1 MiB".into());
            }
            if !self.data.is_empty() {
                self.data.push(b'\n');
            }
            self.data.extend_from_slice(data);
        }
        Ok(())
    }

    pub fn finish(mut self, text: &mut dyn FnMut(&str) -> Result<(), String>) -> Result<Value, String> {
        // SSE may end without a final blank line.
        self.push(b"\n\n", text)?;
        if !self.finished {
            return Err("model stream ended before the response was complete; no partial tool calls were run".into());
        }
        let mut message = json!({"role": "assistant", "content": self.content});
        if !self.calls.is_empty() {
            message["tool_calls"] = Value::Array(self.calls.into_values().collect());
        }
        Ok(json!({"choices": [{"message": message}], "usage": self.usage}))
    }
}

fn merge(target: &mut Value, delta: &Value) {
    if let (Some(target), Some(delta)) = (target.as_object_mut(), delta.as_object()) {
        for (key, value) in delta {
            match key.as_str() {
                "index" => {}
                "function" => merge(target.entry(key).or_insert_with(|| json!({})), value),
                "name" | "arguments" => {
                    if let Some(part) = value.as_str() {
                        let entry = target.entry(key).or_insert_with(|| json!(""));
                        let mut combined = entry.as_str().unwrap_or("").to_string();
                        combined.push_str(part);
                        *entry = Value::String(combined);
                    }
                }
                _ => {
                    target.insert(key.clone(), value.clone());
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_stream_is_separate_from_the_answer() {
        let mut reply = Reply::default();
        let mut answer = String::new();
        let wire = format!(
            "data: {}\n\ndata: {}\n\n",
            json!({"choices":[{"delta":{"reasoning_content":"Check dimensions."}}]}),
            json!({"choices":[{"delta":{"content":"Done."},"finish_reason":"stop"}]})
        );
        for chunk in wire.as_bytes().chunks(3) {
            reply
                .push(chunk, &mut |s| {
                    answer.push_str(s);
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(reply.take_thought(), "Check dimensions.");
        assert!(reply.take_thought().is_empty());
        assert_eq!(answer, "Done.");
        assert_eq!(reply.finish(&mut |_| Ok(())).unwrap()["choices"][0]["message"]["content"], "Done.");
    }

    #[test]
    fn fragmented_utf8_text_and_tool_arguments_keep_signatures() {
        let frames = [
            json!({"choices":[{"delta":{"content":"Checking π…","tool_calls":[{"index":0,"id":"call-a","type":"function","function":{"name":"inspect_","arguments":"{"},"extra_content":{"google":{"thought_signature":"sig"}}}]}}]}),
            json!({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"design","arguments":"\"measure\":false}"}}]},"finish_reason":"tool_calls"}]}),
            json!({"choices":[],"usage":{"prompt_tokens":123,"completion_tokens":7,"prompt_tokens_details":{"cached_tokens":100}}}),
        ];
        let wire = frames.iter().map(|v| format!("data: {v}\r\n\r\n")).collect::<String>();
        let mut reply = Reply::default();
        let mut shown = String::new();
        for byte in wire.as_bytes() {
            reply
                .push(&[*byte], &mut |text| {
                    shown.push_str(text);
                    Ok(())
                })
                .unwrap();
        }
        assert_eq!(shown, "Checking π…", "visible before the stream is closed");
        let result = reply.finish(&mut |_| Ok(())).unwrap();
        let call = &result["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["function"]["name"], "inspect_design");
        assert_eq!(call["function"]["arguments"], "{\"measure\":false}");
        assert_eq!(call["extra_content"]["google"]["thought_signature"], "sig");
        assert_eq!(result["usage"]["prompt_tokens_details"]["cached_tokens"], 100);
    }

    #[test]
    fn truncated_or_hostile_calls_are_never_accepted() {
        let mut partial = Reply::default();
        partial.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n", &mut |_| Ok(())).unwrap();
        assert!(partial.finish(&mut |_| Ok(())).is_err());
        let mut hostile = Reply::default();
        assert!(hostile.push(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":999999999}]}}]}\n\n", &mut |_| Ok(())).is_err());
    }

    #[test]
    fn single_unindexed_tool_call_streams_but_ambiguous_calls_fail() {
        let mut reply = Reply::default();
        reply.push(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"call-a\",\"type\":\"function\",\"function\":{\"name\":\"solid.extrude\",\"arguments\":\"{\\\"distance\\\":\"}}]}}]}\n\n", &mut |_| Ok(())).unwrap();
        reply
            .push(
                b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"20}\"}}]},\"finish_reason\":\"tool_calls\"}]}\n\n",
                &mut |_| Ok(()),
            )
            .unwrap();
        let result = reply.finish(&mut |_| Ok(())).unwrap();
        assert_eq!(result["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"], "{\"distance\":20}");

        let mut ambiguous = Reply::default();
        assert!(ambiguous.push(b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"id\":\"a\"},{\"id\":\"b\"}]}}]}\n\n", &mut |_| Ok(())).is_err());
    }

    #[test]
    fn final_event_without_newlines_still_displays_its_text() {
        let mut reply = Reply::default();
        reply.push(b"data: {\"choices\":[{\"delta\":{\"content\":\"Done\"},\"finish_reason\":\"stop\"}]}", &mut |_| Ok(())).unwrap();
        let mut shown = String::new();
        reply
            .finish(&mut |text| {
                shown.push_str(text);
                Ok(())
            })
            .unwrap();
        assert_eq!(shown, "Done");
    }
}
