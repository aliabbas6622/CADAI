//! Gemini's static prefix cache. Creation runs in the background; chat never waits for it.

use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use super::api_client::ApiConfig;
use super::backend::HttpRequest;

#[derive(Default)]
pub(super) struct Cache {
    model: String,
    name: Option<String>,
    pending: Option<Receiver<Option<String>>>,
    attempted: Option<Instant>,
}

impl Cache {
    pub fn prepare(&mut self, cfg: &ApiConfig, system: &str, tools: &[Value]) -> Option<String> {
        if !cfg.base_url.starts_with("https://generativelanguage.googleapis.com/") || !cfg.model.starts_with("gemini-3") {
            return None;
        }
        if self.model != cfg.model {
            *self = Self { model: cfg.model.clone(), ..Self::default() };
        }
        if self.attempted.is_some_and(|at| at.elapsed() >= Duration::from_secs(270)) {
            self.name = None;
            self.attempted = None;
            self.pending = None;
        }
        if let Some(pending) = &self.pending {
            match pending.try_recv() {
                Ok(name) => {
                    self.name = name;
                    self.pending = None;
                }
                Err(std::sync::mpsc::TryRecvError::Disconnected) => self.pending = None,
                Err(std::sync::mpsc::TryRecvError::Empty) => {}
            }
        }
        // Avoid a likely below-minimum request. The API remains authoritative about token count.
        if self.attempted.is_none() && system.len() + tools.iter().map(|t| t.to_string().len()).sum::<usize>() >= 16_384 {
            self.attempted = Some(Instant::now());
            let (tx, rx) = channel();
            self.pending = Some(rx);
            let http = cfg.http.clone();
            let key = cfg.key.clone();
            let body = cache_body(&cfg.model, system, tools).to_string().into_bytes();
            std::thread::spawn(move || {
                let mut headers = vec![("Content-Type".into(), "application/json".into())];
                if let Some(key) = key {
                    headers.push(("x-goog-api-key".into(), key));
                }
                let request = HttpRequest {
                    method: "POST",
                    url: "https://generativelanguage.googleapis.com/v1beta/cachedContents".into(),
                    headers,
                    body: Some(body),
                    timeout_s: 10,
                };
                let name = http(&request)
                    .ok()
                    .filter(|(status, _)| *status == 200)
                    .and_then(|(_, bytes)| serde_json::from_slice::<Value>(&bytes).ok())
                    .and_then(|v| {
                        v.get("name").and_then(Value::as_str).filter(|s| s.starts_with("cachedContents/") && s.len() < 256).map(str::to_string)
                    });
                let _ = tx.send(name);
            });
        }
        self.name.clone()
    }

    pub fn reject(&mut self) {
        self.name = None;
        self.pending = None;
        // Back off after refusal; do not create/retry a cache on every model request.
        self.attempted = Some(Instant::now());
    }
}

fn cache_body(model: &str, system: &str, tools: &[Value]) -> Value {
    let declarations: Vec<Value> = tools.iter().filter_map(|tool| {
        let function = tool.get("function")?;
        Some(json!({"name": function.get("name")?, "description": function.get("description")?, "parametersJsonSchema": function.get("parameters")?}))
    }).collect();
    json!({"model": format!("models/{model}"), "systemInstruction": {"parts": [{"text": system}]}, "tools": [{"functionDeclarations": declarations}], "ttl": "300s"})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creation_is_nonblocking_reused_and_rejection_backs_off() {
        use std::sync::{Arc, Mutex};
        let (release_tx, release_rx) = channel();
        let release_rx = Mutex::new(release_rx);
        let cfg = ApiConfig {
            name: "Gemini".into(),
            base_url: "https://generativelanguage.googleapis.com/v1beta/openai".into(),
            model: "gemini-3.8-flash".into(),
            key: None,
            stream: None,
            http: Arc::new(move |request| {
                assert!(request.url.ends_with("/cachedContents"));
                release_rx.lock().unwrap().recv_timeout(Duration::from_secs(2)).unwrap();
                Ok((200, br#"{"name":"cachedContents/test"}"#.to_vec()))
            }),
        };
        let system = "static ".repeat(3000);
        let mut cache = Cache::default();
        assert!(cache.prepare(&cfg, &system, &[]).is_none());
        // No completion is required before the caller can make its normal model request.
        release_tx.send(()).unwrap();
        let deadline = Instant::now() + Duration::from_secs(2);
        let mut name = None;
        while name.is_none() && Instant::now() < deadline {
            name = cache.prepare(&cfg, &system, &[]);
            std::thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(name.as_deref(), Some("cachedContents/test"));
        cache.reject();
        assert!(cache.prepare(&cfg, &system, &[]).is_none());
        assert!(cache.pending.is_none(), "a rejected reference cannot loop through cache creation");
    }

    #[test]
    fn cache_contains_only_static_instructions_and_native_tool_schema() {
        let body = cache_body(
            "gemini-3.8-flash",
            "Static",
            &[json!({"type":"function","function":{"name":"execute","description":"Run","parameters":{"type":"object","properties":{}}}})],
        );
        assert_eq!(body["model"], "models/gemini-3.8-flash");
        assert_eq!(body["systemInstruction"]["parts"][0]["text"], "Static");
        assert_eq!(body["tools"][0]["functionDeclarations"][0]["parametersJsonSchema"]["type"], "object");
        assert!(body.get("messages").is_none());
        assert_eq!(body["ttl"], "300s");
    }
}
