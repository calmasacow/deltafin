//! Bounded, exact response memoization for the native deterministic server.
//!
//! A memo hit is permitted only for the same endpoint kind and structurally
//! identical validated target request. JSON whitespace, streaming transport,
//! and other wire-only details do not prevent reuse, but no merely similar
//! prompt can ever reuse another prompt's output.

use std::collections::VecDeque;

use serde_json::Value;

use super::types::{TargetOutput, TargetPrompt, TargetRequest};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(super) enum ResponseMode {
    Completion,
    Chat,
}

#[derive(Debug, Clone)]
struct Entry {
    key: Box<[u8]>,
    output: TargetOutput,
    bytes: usize,
}

/// A byte-bounded LRU containing only full-K3-certified responses.
///
/// The earlier Python implementation bounded entry count but not memory. A
/// long prompt or response could therefore make a nominally small cache very
/// large. The native version enforces both limits and declines an entry that
/// cannot fit by itself.
#[derive(Debug)]
pub(super) struct DeterministicResponseMemo {
    max_entries: usize,
    max_bytes: usize,
    used_bytes: usize,
    entries: VecDeque<Entry>,
    hits: u64,
    misses: u64,
}

impl DeterministicResponseMemo {
    pub(super) fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            max_entries,
            max_bytes,
            used_bytes: 0,
            entries: VecDeque::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub(super) fn get(
        &mut self,
        mode: ResponseMode,
        request: &TargetRequest,
    ) -> Option<TargetOutput> {
        if !self.enabled() {
            return None;
        }
        let key = request_key(mode, request, self.max_bytes)?;
        let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.key.as_ref() == key.as_slice())
        else {
            self.misses = self.misses.saturating_add(1);
            return None;
        };
        let entry = self
            .entries
            .remove(index)
            .expect("response memo index came from the same deque");
        let output = entry.output.clone();
        self.entries.push_back(entry);
        self.hits = self.hits.saturating_add(1);
        Some(output)
    }

    pub(super) fn put(
        &mut self,
        mode: ResponseMode,
        request: &TargetRequest,
        output: &TargetOutput,
    ) {
        if !self.enabled() {
            return;
        }
        let Some(key) = request_key(mode, request, self.max_bytes) else {
            return;
        };
        let bytes = entry_bytes(&key, output);
        if bytes > self.max_bytes {
            return;
        }
        if let Some(index) = self
            .entries
            .iter()
            .position(|entry| entry.key.as_ref() == key.as_slice())
            && let Some(old) = self.entries.remove(index)
        {
            self.used_bytes = self.used_bytes.saturating_sub(old.bytes);
        }
        while self.entries.len() >= self.max_entries
            || self.used_bytes.saturating_add(bytes) > self.max_bytes
        {
            let Some(oldest) = self.entries.pop_front() else {
                break;
            };
            self.used_bytes = self.used_bytes.saturating_sub(oldest.bytes);
        }
        self.entries.push_back(Entry {
            key: key.into_boxed_slice(),
            output: output.clone(),
            bytes,
        });
        self.used_bytes = self.used_bytes.saturating_add(bytes);
    }

    fn enabled(&self) -> bool {
        self.max_entries != 0 && self.max_bytes != 0
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    fn used_bytes(&self) -> usize {
        self.used_bytes
    }
}

fn entry_bytes(key: &[u8], output: &TargetOutput) -> usize {
    key.len()
        .saturating_add(output.text().len())
        .saturating_add(output.reasoning_content().map_or(0, str::len))
        .saturating_add(std::mem::size_of::<Entry>())
}

fn request_key(mode: ResponseMode, request: &TargetRequest, maximum: usize) -> Option<Vec<u8>> {
    let mut key = Vec::with_capacity(maximum.min(4096));
    append_byte(
        &mut key,
        match mode {
            ResponseMode::Completion => 0,
            ResponseMode::Chat => 1,
        },
        maximum,
    )?;
    append_bytes(&mut key, &request.max_new_tokens.to_le_bytes(), maximum)?;
    match &request.prompt {
        TargetPrompt::Completion(prompt) => {
            append_byte(&mut key, 0, maximum)?;
            append_bytes(&mut key, prompt.as_bytes(), maximum)?;
        }
        TargetPrompt::Chat(messages) => {
            append_byte(&mut key, 1, maximum)?;
            append_bytes(&mut key, &messages.len().to_le_bytes(), maximum)?;
            for message in messages {
                append_bytes(&mut key, message.role.as_bytes(), maximum)?;
                append_value(&mut key, &message.content, maximum)?;
                append_byte(&mut key, 6, maximum)?;
                append_bytes(
                    &mut key,
                    &message.additional_fields.len().to_le_bytes(),
                    maximum,
                )?;
                for (name, value) in &message.additional_fields {
                    append_bytes(&mut key, name.as_bytes(), maximum)?;
                    append_value(&mut key, value, maximum)?;
                }
            }
        }
    }
    Some(key)
}

fn append_value(target: &mut Vec<u8>, value: &Value, maximum: usize) -> Option<()> {
    match value {
        Value::Null => append_byte(target, 0, maximum),
        Value::Bool(boolean) => append_byte(target, if *boolean { 2 } else { 1 }, maximum),
        Value::Number(number) => {
            append_byte(target, 3, maximum)?;
            append_bytes(target, number.to_string().as_bytes(), maximum)
        }
        Value::String(text) => {
            append_byte(target, 4, maximum)?;
            append_bytes(target, text.as_bytes(), maximum)
        }
        Value::Array(values) => {
            append_byte(target, 5, maximum)?;
            append_bytes(target, &values.len().to_le_bytes(), maximum)?;
            for value in values {
                append_value(target, value, maximum)?;
            }
            Some(())
        }
        Value::Object(fields) => {
            append_byte(target, 6, maximum)?;
            append_bytes(target, &fields.len().to_le_bytes(), maximum)?;
            for (name, value) in fields {
                append_bytes(target, name.as_bytes(), maximum)?;
                append_value(target, value, maximum)?;
            }
            Some(())
        }
    }
}

fn append_byte(target: &mut Vec<u8>, byte: u8, maximum: usize) -> Option<()> {
    if target.len() >= maximum {
        return None;
    }
    target.push(byte);
    Some(())
}

fn append_bytes(target: &mut Vec<u8>, bytes: &[u8], maximum: usize) -> Option<()> {
    let length = bytes.len().to_le_bytes();
    let required = length.len().checked_add(bytes.len())?;
    if target.len().checked_add(required)? > maximum {
        return None;
    }
    target.extend_from_slice(&length);
    target.extend_from_slice(bytes);
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::openai::FinishReason;

    fn output(text: &str) -> TargetOutput {
        TargetOutput::target_verified(text, FinishReason::Length)
    }

    fn request(prompt: &str, max_new_tokens: usize) -> TargetRequest {
        TargetRequest {
            prompt: TargetPrompt::Completion(prompt.to_owned()),
            max_new_tokens,
        }
    }

    #[test]
    fn equality_includes_mode_prompt_and_limit_but_not_wire_formatting() {
        let mut memo = DeterministicResponseMemo::new(4, 4096);
        let exact = request("x", 2);
        memo.put(ResponseMode::Completion, &exact, &output("one"));
        assert!(
            memo.get(ResponseMode::Completion, &request("x", 3))
                .is_none()
        );
        assert!(memo.get(ResponseMode::Chat, &exact).is_none());
        assert_eq!(
            memo.get(ResponseMode::Completion, &exact).unwrap().text(),
            "one"
        );
    }

    #[test]
    fn entry_and_byte_limits_evict_least_recently_used() {
        let overhead = std::mem::size_of::<Entry>();
        let mut memo = DeterministicResponseMemo::new(2, overhead * 2 + 128);
        let a = request("a", 1);
        let b = request("b", 1);
        let c = request("c", 1);
        memo.put(ResponseMode::Completion, &a, &output("1"));
        memo.put(ResponseMode::Completion, &b, &output("2"));
        assert!(memo.get(ResponseMode::Completion, &a).is_some());
        memo.put(ResponseMode::Completion, &c, &output("3"));
        assert!(memo.get(ResponseMode::Completion, &b).is_none());
        assert!(memo.get(ResponseMode::Completion, &a).is_some());
        assert!(memo.get(ResponseMode::Completion, &c).is_some());
        assert!(memo.used_bytes() <= overhead * 2 + 128);
    }

    #[test]
    fn oversized_or_disabled_entries_are_never_retained() {
        let exact = request("request", 1);
        let mut small = DeterministicResponseMemo::new(32, 8);
        small.put(ResponseMode::Completion, &exact, &output("response"));
        assert_eq!(small.len(), 0);

        let mut disabled = DeterministicResponseMemo::new(0, usize::MAX);
        disabled.put(ResponseMode::Completion, &exact, &output("y"));
        assert!(disabled.get(ResponseMode::Completion, &exact).is_none());
    }
}
