//! Exact Kimi K3 XTML chat rendering.
//!
//! Rendering and tokenization remain separate: this module labels every
//! segment as either trusted structure or ordinary user/tool text.  The native
//! tokenizer may recognize special tokens only in trusted structural segments,
//! preventing literal control-token text in a request from changing the XTML
//! tree.

use serde_json::{Map, Value};

use crate::error::{DeltafinError, Result};
use crate::tokenizer::K3Tokenizer;

const OPEN_TOKEN: &str = "<|open|>";
const CLOSE_TOKEN: &str = "<|close|>";
const SEP_TOKEN: &str = "<|sep|>";
const END_OF_MSG_TOKEN: &str = "<|end_of_msg|>";
const IMAGE_PLACEHOLDER: &str = "<|kimi_image_placeholder|>";

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct EncodeSegment {
    pub text: String,
    pub allow_special: bool,
}

#[derive(Debug, Clone)]
pub struct ChatOptions<'a> {
    pub tools: Option<&'a Value>,
    pub add_generation_prompt: bool,
    pub thinking: bool,
    pub image_prompts: Option<&'a [String]>,
    pub thinking_effort: Option<&'a str>,
    pub tool_choice: Option<&'a str>,
    pub response_format: Option<&'a Value>,
    pub response_schema: Option<&'a Value>,
}

impl Default for ChatOptions<'_> {
    fn default() -> Self {
        Self {
            tools: None,
            add_generation_prompt: true,
            thinking: true,
            image_prompts: None,
            // TikTokenTokenizer.apply_chat_template supplies `max` when its
            // caller does not specify an effort.
            thinking_effort: Some("max"),
            tool_choice: None,
            response_format: None,
            response_schema: None,
        }
    }
}

pub fn encode_chat(
    tokenizer: &K3Tokenizer,
    messages: &[Value],
    options: &ChatOptions<'_>,
) -> Result<Vec<u32>> {
    let segments = build_chat_segments(messages, options)?;
    let total_bytes = segments.iter().try_fold(0usize, |total, segment| {
        total
            .checked_add(segment.text.len())
            .ok_or_else(|| DeltafinError::new("rendered chat byte length overflows usize"))
    })?;
    if tokenizer.should_parallelize_segments(segments.len(), total_bytes) {
        let inputs = segments
            .iter()
            .map(|segment| (segment.text.as_str(), segment.allow_special))
            .collect::<Vec<_>>();
        return tokenizer.encode_segments(&inputs);
    }
    encode_chat_segments_sequential(tokenizer, &segments)
}

fn encode_chat_segments_sequential(
    tokenizer: &K3Tokenizer,
    segments: &[EncodeSegment],
) -> Result<Vec<u32>> {
    let mut tokens = Vec::new();
    for segment in segments {
        tokenizer.encode_into(&segment.text, segment.allow_special, &mut tokens)?;
    }
    Ok(tokens)
}

pub fn build_chat_segments(
    messages: &[Value],
    options: &ChatOptions<'_>,
) -> Result<Vec<EncodeSegment>> {
    let messages = normalize_conversation(messages)?;
    let tools = options.tools.map(deep_sort);
    let mut response_schema = options
        .response_schema
        .cloned()
        .or_else(|| options.response_format.and_then(extract_response_schema));
    if let Some(schema) = response_schema.as_mut() {
        *schema = deep_sort(schema);
    }

    let mut image_state = ImagePromptState::new(options.image_prompts);
    let mut segments = Vec::new();
    let mut previous_tool_calls: Option<Vec<Value>> = None;
    let mut tool_index = 0usize;

    if tools.as_ref().is_some_and(json_truthy) {
        render_tool_declare(tools.as_ref().expect("checked above"), false, &mut segments)?;
    }

    if options.thinking
        && let Some(effort) = options.thinking_effort
    {
        if !matches!(effort, "low" | "high" | "max") {
            return Err(DeltafinError::new(format!(
                "unsupported thinking_effort={effort:?}; supported values are low, high, and max"
            )));
        }
        internal_system_message(
            "thinking-effort",
            &format!(
                "`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort={effort}`."
            ),
            &mut segments,
        );
    }

    for message in &messages {
        let Some(message) = message.as_object() else {
            continue;
        };
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| DeltafinError::new("K3 chat message lacks a string role"))?;
        match role {
            "user" => {
                let mut attrs = vec![("role", AttributeValue::Text("user"))];
                if message.get("name").is_some_and(json_truthy) {
                    attrs.push((
                        "name",
                        AttributeValue::Json(message.get("name").expect("checked above")),
                    ));
                }
                open_tag("message", &attrs, &mut segments);
                render_content_segments(message.get("content"), &mut image_state, &mut segments)?;
                close_tag("message", &mut segments);
                end_of_message(&mut segments);
            }
            "system" if message.get("tools").is_some_and(json_truthy) => {
                render_tool_declare(
                    message.get("tools").expect("checked above"),
                    true,
                    &mut segments,
                )?;
            }
            "system" => {
                let mut attrs = vec![("role", AttributeValue::Text("system"))];
                if message.get("name").is_some_and(json_truthy) {
                    attrs.push((
                        "name",
                        AttributeValue::Json(message.get("name").expect("checked above")),
                    ));
                }
                open_tag("message", &attrs, &mut segments);
                render_content_segments(message.get("content"), &mut image_state, &mut segments)?;
                close_tag("message", &mut segments);
                end_of_message(&mut segments);
            }
            "tool" => {
                tool_index += 1;
                let tool_name_value = if message.contains_key("tool") {
                    message.get("tool")
                } else {
                    message.get("name")
                };
                let tool_name = tool_name_value
                    .filter(|value| !value.is_null())
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| {
                        previous_tool_calls
                            .as_ref()
                            .and_then(|calls| calls.get(tool_index - 1))
                            .and_then(tool_function)
                            .and_then(|function| function.get("name"))
                            .and_then(Value::as_str)
                            .map(str::to_owned)
                    })
                    .ok_or_else(|| {
                        DeltafinError::new(
                            "Kimi K3 tool messages need a resolvable tool name: carry tool/name, or match a preceding assistant tool_call by order",
                        )
                    })?;
                let index = tool_index.to_string();
                open_tag(
                    "message",
                    &[
                        ("role", AttributeValue::Text("tool")),
                        ("tool", AttributeValue::Text(&tool_name)),
                        ("index", AttributeValue::Text(&index)),
                    ],
                    &mut segments,
                );
                render_content_segments(message.get("content"), &mut image_state, &mut segments)?;
                close_tag("message", &mut segments);
                end_of_message(&mut segments);
            }
            "assistant" => {
                previous_tool_calls = message.get("tool_calls").and_then(Value::as_array).cloned();
                tool_index = 0;
                let mut attrs = vec![("role", AttributeValue::Text("assistant"))];
                if message.get("name").is_some_and(json_truthy) {
                    attrs.push((
                        "name",
                        AttributeValue::Json(message.get("name").expect("checked above")),
                    ));
                }
                open_tag("message", &attrs, &mut segments);
                render_assistant_segments(
                    message,
                    &mut image_state,
                    options.thinking,
                    &mut segments,
                )?;
                close_tag("message", &mut segments);
                end_of_message(&mut segments);
            }
            _ => {}
        }
    }

    match options.tool_choice {
        Some("required") => internal_system_message(
            "tool-choice",
            "The system is invoked with `tool_choice=required`.\nYou MUST call tools in the next message.",
            &mut segments,
        ),
        Some("none") => internal_system_message(
            "tool-choice",
            "The system is invoked with `tool_choice=none`.\nYou MUST NOT call any tools in the next message.",
            &mut segments,
        ),
        _ => {}
    }

    let response_type = options.response_format.and_then(|format| {
        format
            .as_object()
            .and_then(|object| object.get("type"))
            .or(Some(format))
            .and_then(Value::as_str)
    });
    match response_type {
        Some("json_object") => internal_system_message(
            "response-format",
            "The system is invoked with `response_format=json_object`.\nYour response must be raw JSON data without markdown code blocks (```json) or any additional formatting.",
            &mut segments,
        ),
        Some("json_schema") => {
            let schema = json_compact(response_schema.as_ref().unwrap_or(&Value::Null))?;
            internal_system_message(
                "response-format",
                &format!(
                    "The system is invoked with `response_format=json_schema`.\nYour response must be raw JSON data without markdown code blocks (```json) or any additional formatting.\nThe JSON data must match the following schema:\n```json\n{schema}\n```"
                ),
                &mut segments,
            );
        }
        _ => {}
    }

    if options.add_generation_prompt {
        open_tag(
            "message",
            &[("role", AttributeValue::Text("assistant"))],
            &mut segments,
        );
        open_tag(
            if options.thinking {
                "think"
            } else {
                "response"
            },
            &[],
            &mut segments,
        );
    }

    image_state.assert_consumed()?;
    Ok(segments)
}

pub fn render_segments(segments: &[EncodeSegment]) -> String {
    let size = segments.iter().map(|segment| segment.text.len()).sum();
    let mut rendered = String::with_capacity(size);
    for segment in segments {
        rendered.push_str(&segment.text);
    }
    rendered
}

fn segment(text: impl Into<String>, allow_special: bool, output: &mut Vec<EncodeSegment>) {
    let text = text.into();
    if !text.is_empty() {
        output.push(EncodeSegment {
            text,
            allow_special,
        });
    }
}

fn control(text: &str, output: &mut Vec<EncodeSegment>) {
    segment(text, true, output);
}

fn ordinary(text: impl Into<String>, output: &mut Vec<EncodeSegment>) {
    segment(text, false, output);
}

enum AttributeValue<'a> {
    Text(&'a str),
    Json(&'a Value),
}

fn open_tag(tag: &str, attrs: &[(&str, AttributeValue<'_>)], output: &mut Vec<EncodeSegment>) {
    control(OPEN_TOKEN, output);
    ordinary(tag, output);
    for (key, value) in attrs {
        ordinary(format!(" {key}"), output);
        ordinary("=\"", output);
        let value = match value {
            AttributeValue::Text(value) => (*value).to_owned(),
            AttributeValue::Json(value) => python_string(value),
        };
        ordinary(escape_attribute(&value), output);
        ordinary("\"", output);
    }
    control(SEP_TOKEN, output);
}

fn close_tag(tag: &str, output: &mut Vec<EncodeSegment>) {
    control(CLOSE_TOKEN, output);
    ordinary(tag, output);
    control(SEP_TOKEN, output);
}

fn end_of_message(output: &mut Vec<EncodeSegment>) {
    control(END_OF_MSG_TOKEN, output);
}

fn escape_attribute(value: &str) -> String {
    value.replace('&', "&amp;").replace('"', "&quot;")
}

struct ImagePromptState<'a> {
    prompts: Option<&'a [String]>,
    index: usize,
}

impl<'a> ImagePromptState<'a> {
    fn new(prompts: Option<&'a [String]>) -> Self {
        Self { prompts, index: 0 }
    }

    fn next_prompt(&mut self) -> Result<&str> {
        match self.prompts {
            None => Ok(IMAGE_PLACEHOLDER),
            Some(prompts) => {
                let prompt = prompts.get(self.index).ok_or_else(|| {
                    DeltafinError::new("more image placeholders than image prompts")
                })?;
                self.index += 1;
                Ok(prompt)
            }
        }
    }

    fn assert_consumed(&self) -> Result<()> {
        if let Some(prompts) = self.prompts
            && self.index != prompts.len()
        {
            return Err(DeltafinError::new(format!(
                "image prompt count {} != consumed placeholder count {}",
                prompts.len(),
                self.index
            )));
        }
        Ok(())
    }
}

fn append_text(
    value: &Value,
    image_state: &mut ImagePromptState<'_>,
    output: &mut Vec<EncodeSegment>,
) -> Result<()> {
    append_string(&python_string(value), image_state, output)
}

fn append_string(
    text: &str,
    image_state: &mut ImagePromptState<'_>,
    output: &mut Vec<EncodeSegment>,
) -> Result<()> {
    if text.is_empty() {
        return Ok(());
    }
    if image_state.prompts.is_none() || !text.contains(IMAGE_PLACEHOLDER) {
        ordinary(text, output);
        return Ok(());
    }

    let mut pieces = text.split(IMAGE_PLACEHOLDER).peekable();
    while let Some(piece) = pieces.next() {
        ordinary(piece, output);
        if pieces.peek().is_some() {
            let prompt = image_state.next_prompt()?;
            control(prompt, output);
        }
    }
    Ok(())
}

fn render_content_segments(
    content: Option<&Value>,
    image_state: &mut ImagePromptState<'_>,
    output: &mut Vec<EncodeSegment>,
) -> Result<()> {
    match content {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(text)) => append_string(text, image_state, output),
        Some(Value::Array(parts)) => {
            for part in parts {
                let part = part.as_object().ok_or_else(|| {
                    DeltafinError::new("K3 multimodal content entries must be JSON objects")
                })?;
                match part.get("type").and_then(Value::as_str) {
                    Some("image" | "image_url") => {
                        let prompt = image_state.next_prompt()?;
                        control(prompt, output);
                    }
                    _ => {
                        let text = part
                            .get("text")
                            .ok_or_else(|| DeltafinError::new("K3 text content part lacks text"))?;
                        append_text(text, image_state, output)?;
                    }
                }
            }
            Ok(())
        }
        Some(_) => Err(DeltafinError::new(
            "K3 message content must be a string, array, or null",
        )),
    }
}

fn internal_system_message(message_type: &str, body: &str, output: &mut Vec<EncodeSegment>) {
    open_tag(
        "message",
        &[
            ("role", AttributeValue::Text("system")),
            ("type", AttributeValue::Text(message_type)),
        ],
        output,
    );
    ordinary(body.trim(), output);
    close_tag("message", output);
    end_of_message(output);
}

fn render_assistant_segments(
    message: &Map<String, Value>,
    image_state: &mut ImagePromptState<'_>,
    thinking: bool,
    output: &mut Vec<EncodeSegment>,
) -> Result<()> {
    if thinking {
        let reasoning = message
            .get("reasoning_content")
            .filter(|value| json_truthy(value))
            .or_else(|| message.get("reasoning"));
        open_tag("think", &[], output);
        if let Some(reasoning) = reasoning {
            let rendered = python_string(reasoning);
            if !rendered.trim().is_empty() {
                append_string(&rendered, image_state, output)?;
            }
        }
        close_tag("think", output);
    }

    open_tag("response", &[], output);
    render_content_segments(message.get("content"), image_state, output)?;
    close_tag("response", output);

    if message.get("tool_calls").is_some_and(json_truthy) {
        let tool_calls = message
            .get("tool_calls")
            .and_then(Value::as_array)
            .ok_or_else(|| DeltafinError::new("assistant tool_calls must be an array"))?;
        open_tag("tools", &[], output);
        for (offset, tool_call) in tool_calls.iter().enumerate() {
            let function = tool_function(tool_call)
                .ok_or_else(|| DeltafinError::new("assistant tool call must be a JSON object"))?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| DeltafinError::new("assistant tool call lacks a string name"))?;
            let index = (offset + 1).to_string();
            open_tag(
                "call",
                &[
                    ("tool", AttributeValue::Text(name)),
                    ("index", AttributeValue::Text(&index)),
                ],
                output,
            );
            let arguments = function.get("arguments");
            if let Some(json_block) = function.get("_xtml_json_block") {
                open_tag("json", &[("type", AttributeValue::Text("object"))], output);
                append_text(json_block, image_state, output)?;
                close_tag("json", output);
            } else if let Some(arguments) = arguments.and_then(Value::as_object) {
                for (key, value) in arguments {
                    let value_type = xtml_type(value);
                    open_tag(
                        "argument",
                        &[
                            ("key", AttributeValue::Text(key)),
                            ("type", AttributeValue::Text(value_type)),
                        ],
                        output,
                    );
                    append_string(&xtml_value(value)?, image_state, output)?;
                    close_tag("argument", output);
                }
            }
            close_tag("call", output);
        }
        close_tag("tools", output);
    }
    Ok(())
}

fn render_tool_declare(
    tools: &Value,
    dynamic: bool,
    output: &mut Vec<EncodeSegment>,
) -> Result<()> {
    let tools = deep_sort(tools);
    let tools = json_compact(&tools)?;
    let body = if dynamic {
        format!(
            "## New Tools Available\nThe system dynamically extends the toolset via lazy-loading.\nYou have access to all existing and extended tools.\nHere are the specs for the extended tools.\n\n```json\n{tools}\n```"
        )
    } else {
        format!(
            "# Tools\nHere are the available tools, described in JSONSchema.\n\n```json\n{tools}\n```"
        )
    };
    open_tag(
        "message",
        &[
            ("role", AttributeValue::Text("system")),
            ("type", AttributeValue::Text("tool-declare")),
        ],
        output,
    );
    ordinary(body, output);
    close_tag("message", output);
    end_of_message(output);
    Ok(())
}

fn normalize_conversation(messages: &[Value]) -> Result<Vec<Value>> {
    let normalized = messages
        .iter()
        .map(normalize_message)
        .collect::<Result<Vec<_>>>()?;
    normalize_tool_result_messages(&normalized)
}

fn normalize_message(message: &Value) -> Result<Value> {
    let Some(message) = message.as_object() else {
        return Ok(message.clone());
    };
    let mut normalized = message.clone();
    if let Some(tools) = normalized.get_mut("tools") {
        *tools = deep_sort(tools);
    }
    let Some(tool_calls) = normalized.get_mut("tool_calls") else {
        return Ok(Value::Object(normalized));
    };
    if !json_truthy(tool_calls) {
        return Ok(Value::Object(normalized));
    }
    let calls = tool_calls
        .as_array_mut()
        .ok_or_else(|| DeltafinError::new("assistant tool_calls must be an array"))?;
    for tool_call in calls {
        let Some(call) = tool_call.as_object_mut() else {
            continue;
        };
        if call.get("function").is_some_and(Value::is_object) {
            let function = call
                .get_mut("function")
                .and_then(Value::as_object_mut)
                .expect("checked above");
            normalize_arguments(function)?;
        } else {
            normalize_arguments(call)?;
        }
    }
    Ok(Value::Object(normalized))
}

fn normalize_arguments(function: &mut Map<String, Value>) -> Result<()> {
    let arguments = function.get("arguments").cloned().unwrap_or(Value::Null);
    match arguments {
        Value::Null => {
            function.insert("arguments".into(), Value::Object(Map::new()));
            function.remove("_xtml_json_block");
        }
        Value::Object(_) => {
            function.remove("_xtml_json_block");
        }
        Value::String(raw) if raw.trim().is_empty() => {
            function.insert("arguments".into(), Value::Object(Map::new()));
            function.remove("_xtml_json_block");
        }
        Value::String(raw) => match serde_json::from_str::<Value>(&raw) {
            Ok(Value::Object(object)) => {
                function.insert("arguments".into(), Value::Object(object));
                function.remove("_xtml_json_block");
            }
            Ok(_) => {
                return Err(DeltafinError::new(
                    "Kimi K3 tool call arguments must be a JSON object",
                ));
            }
            Err(_) => {
                function.insert("arguments".into(), Value::Object(Map::new()));
                function.insert("_xtml_json_block".into(), Value::String(raw));
            }
        },
        _ => {
            return Err(DeltafinError::new(
                "Kimi K3 tool call arguments must be an object or JSON object string",
            ));
        }
    }
    Ok(())
}

fn normalize_tool_result_messages(messages: &[Value]) -> Result<Vec<Value>> {
    let mut output = Vec::with_capacity(messages.len());
    let mut current_index: FxToolIndex = FxToolIndex::default();
    let mut cursor = 0;
    while cursor < messages.len() {
        let message = &messages[cursor];
        if role(message) == Some("assistant") {
            current_index = message
                .get("tool_calls")
                .filter(|calls| json_truthy(calls))
                .map(tool_call_id_index)
                .transpose()?
                .unwrap_or_default();
            output.push(message.clone());
            cursor += 1;
            continue;
        }
        if role(message) != Some("tool") {
            output.push(message.clone());
            cursor += 1;
            continue;
        }

        let mut run = Vec::new();
        let mut unresolved = false;
        let mut offset = 0usize;
        while cursor < messages.len() && role(&messages[cursor]) == Some("tool") {
            let tool_message = &messages[cursor];
            let call_id = tool_message
                .get("tool_call_id")
                .or_else(|| tool_message.get("id"));
            let matched = call_id
                .filter(|id| !id.is_null())
                .map(python_string)
                .and_then(|id| current_index.get(&id).cloned());
            if matched.is_none() {
                unresolved = true;
            }
            run.push((matched, offset, tool_message.clone()));
            cursor += 1;
            offset += 1;
        }

        if unresolved {
            output.extend(run.into_iter().map(|(_, _, message)| message));
            continue;
        }
        run.sort_by_key(|(matched, offset, _)| {
            (
                matched.as_ref().map_or(usize::MAX, |value| value.0),
                *offset,
            )
        });
        for (matched, _, mut message) in run {
            let (_, name) = matched.expect("unresolved run handled above");
            if let Some(name) = name {
                let object = message.as_object_mut().ok_or_else(|| {
                    DeltafinError::new("tool result message must be a JSON object")
                })?;
                object.insert("tool".into(), Value::String(name.clone()));
                if object.contains_key("name") {
                    object.insert("name".into(), Value::String(name));
                }
            }
            output.push(message);
        }
    }
    Ok(output)
}

type FxToolIndex = rustc_hash::FxHashMap<String, (usize, Option<String>)>;

fn tool_call_id_index(tool_calls: &Value) -> Result<FxToolIndex> {
    let Some(tool_calls) = tool_calls.as_array() else {
        return Ok(FxToolIndex::default());
    };
    let mut index = FxToolIndex::default();
    for (offset, tool_call) in tool_calls.iter().enumerate() {
        let Some(call) = tool_call.as_object() else {
            continue;
        };
        let Some(call_id) = call.get("id").filter(|id| !id.is_null()) else {
            continue;
        };
        let key = python_string(call_id);
        if index.contains_key(&key) {
            continue;
        }
        let name = tool_function(tool_call)
            .and_then(|function| function.get("name"))
            .filter(|name| !name.is_null())
            .map(python_string);
        index.insert(key, (offset + 1, name));
    }
    Ok(index)
}

fn tool_function(tool_call: &Value) -> Option<&Map<String, Value>> {
    let call = tool_call.as_object()?;
    match call.get("function") {
        Some(function) => function.as_object(),
        None => Some(call),
    }
}

fn role(message: &Value) -> Option<&str> {
    message.get("role").and_then(Value::as_str)
}

fn deep_sort(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let mut entries: Vec<_> = object.iter().collect();
            entries.sort_unstable_by(|left, right| left.0.cmp(right.0));
            let mut sorted = Map::new();
            for (key, value) in entries {
                sorted.insert(key.clone(), deep_sort(value));
            }
            Value::Object(sorted)
        }
        Value::Array(values) => Value::Array(values.iter().map(deep_sort).collect()),
        _ => value.clone(),
    }
}

fn extract_response_schema(response_format: &Value) -> Option<Value> {
    let json_schema = response_format.get("json_schema")?;
    if let Some(object) = json_schema.as_object() {
        return Some(
            object
                .get("schema")
                .or_else(|| object.get("json_schema"))
                .unwrap_or(json_schema)
                .clone(),
        );
    }
    Some(json_schema.clone())
}

fn xtml_type(value: &Value) -> &'static str {
    match value {
        Value::Bool(_) => "boolean",
        Value::Null => "null",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Object(_) => "object",
        Value::Array(_) => "array",
    }
}

fn xtml_value(value: &Value) -> Result<String> {
    match value {
        Value::String(value) => Ok(value.clone()),
        _ => json_python(value, false),
    }
}

fn json_compact(value: &Value) -> Result<String> {
    json_python(value, true)
}

fn json_python(value: &Value, compact: bool) -> Result<String> {
    match value {
        Value::Null => Ok("null".into()),
        Value::Bool(value) => Ok(if *value { "true" } else { "false" }.into()),
        Value::Number(value) => Ok(value.to_string()),
        Value::String(_) => serde_json::to_string(value)
            .map_err(|error| DeltafinError::new(format!("serialize JSON string: {error}"))),
        Value::Array(values) => {
            let separator = if compact { "," } else { ", " };
            let values = values
                .iter()
                .map(|value| json_python(value, compact))
                .collect::<Result<Vec<_>>>()?;
            Ok(format!("[{}]", values.join(separator)))
        }
        Value::Object(object) => {
            let separators = if compact { (",", ":") } else { (", ", ": ") };
            let mut fields = Vec::with_capacity(object.len());
            for (key, value) in object {
                let key = serde_json::to_string(key).map_err(|error| {
                    DeltafinError::new(format!("serialize JSON object key: {error}"))
                })?;
                fields.push(format!(
                    "{key}{}{}",
                    separators.1,
                    json_python(value, compact)?
                ));
            }
            Ok(format!("{{{}}}", fields.join(separators.0)))
        }
    }
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        // Structured attribute values are outside the documented chat schema;
        // retain deterministic diagnostics instead of invoking code or losing
        // the value. Valid K3 attributes are strings or generated integers.
        Value::Array(_) | Value::Object(_) => {
            json_python(value, false).unwrap_or_else(|_| "<invalid-json>".into())
        }
    }
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(number) => number.as_f64() != Some(0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn repository_root() -> PathBuf {
        if let Some(root) = std::env::var_os("DELTAFIN_TEST_ROOT") {
            return PathBuf::from(root);
        }
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn parse(json: &str) -> Value {
        serde_json::from_str(json).unwrap()
    }

    fn messages(value: &Value) -> &[Value] {
        value.as_array().unwrap()
    }

    fn assert_gold(
        tokenizer: &K3Tokenizer,
        messages: &[Value],
        options: &ChatOptions<'_>,
        expected_segments: usize,
        expected_tokens: usize,
        expected_token_hash: u64,
        expected_segment_hash: u64,
    ) -> Vec<EncodeSegment> {
        let segments = build_chat_segments(messages, options).unwrap();
        let ids = encode_chat(tokenizer, messages, options).unwrap();
        assert_eq!(segments.len(), expected_segments);
        assert_eq!(ids.len(), expected_tokens);
        assert_eq!(u32_hash(ids.iter().copied()), expected_token_hash);
        assert_eq!(
            u32_hash(segments.iter().map(|segment| {
                u32::try_from(segment.text.len()).unwrap()
                    | if segment.allow_special { 1 << 31 } else { 0 }
            })),
            expected_segment_hash
        );
        segments
    }

    #[test]
    #[ignore = "requires an installed k3-meta fixture (deltafin setup); run with --include-ignored on a machine with K3 metadata present"]
    fn simple_public_chat_template_matches_python_gold() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        let conversation = parse(r#"[{"role":"user","content":"Hello <|open|> & goodbye"}]"#);
        let segments = assert_gold(
            &tokenizer,
            messages(&conversation),
            &ChatOptions::default(),
            38,
            95,
            0x14e3_84b3_7a03_643f,
            0x8d67_f740_27f3_f8f4,
        );
        assert_eq!(
            render_segments(&segments),
            "<|open|>message role=\"system\" type=\"thinking-effort\"<|sep|>`thinking_effort` guides on how much to think in your thinking channel (not including the response channel), supported values include `low`, `medium`, `high`, and `max`.\nNow the system is invoked with `thinking_effort=max`.<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"user\"<|sep|>Hello <|open|> & goodbye<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|><|open|>think<|sep|>"
        );
        // The marker embedded in user content is ordinary BPE, not token
        // 163587. Only the four trusted structural opens use that ID.
        let ids =
            encode_chat(&tokenizer, messages(&conversation), &ChatOptions::default()).unwrap();
        assert_eq!(ids.iter().filter(|&&token| token == 163_587).count(), 4);
    }

    #[test]
    #[ignore = "requires an installed k3-meta fixture (deltafin setup); run with --include-ignored on a machine with K3 metadata present"]
    fn nonthinking_multimodal_template_matches_python_gold() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        let conversation = parse(
            r#"[
              {"role":"system","name":"ops&\"x","content":"Rules."},
              {"role":"user","content":[
                {"type":"text","text":"Look: "},
                {"type":"image_url","image_url":{"url":"ignored"}},
                {"type":"text","text":" then <|kimi_image_placeholder|>."}
              ]}
            ]"#,
        );
        let image_prompts = [
            "<|media_begin|>cat<|media_end|>".to_owned(),
            "<|media_begin|>dog<|media_end|>".to_owned(),
        ];
        let options = ChatOptions {
            thinking: false,
            thinking_effort: None,
            image_prompts: Some(&image_prompts),
            ..ChatOptions::default()
        };
        let segments = assert_gold(
            &tokenizer,
            messages(&conversation),
            &options,
            42,
            54,
            0xfb1c_c459_19cb_3313,
            0x35c4_3bb5_88fe_ecd9,
        );
        assert_eq!(
            render_segments(&segments),
            "<|open|>message role=\"system\" name=\"ops&amp;&quot;x\"<|sep|>Rules.<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"user\"<|sep|>Look: <|media_begin|>cat<|media_end|> then <|media_begin|>dog<|media_end|>.<|close|>message<|sep|><|end_of_msg|><|open|>message role=\"assistant\"<|sep|><|open|>response<|sep|>"
        );
    }

    #[test]
    #[ignore = "requires an installed k3-meta fixture (deltafin setup); run with --include-ignored on a machine with K3 metadata present"]
    fn tools_arguments_and_result_reordering_match_python_gold() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        let tools = parse(
            r#"[
              {"type":"function","function":{"description":"B tool","name":"beta","parameters":{"z":{"type":"number"},"a":{"type":"string"}}}},
              {"function":{"name":"alpha","description":"A tool","parameters":{"type":"object"}},"type":"function"}
            ]"#,
        );
        let conversation = parse(
            r#"[
              {"role":"assistant","reasoning_content":"Check.","content":"Calling.","tool_calls":[
                {"id":"b","function":{"name":"beta","arguments":"{\"z\":1,\"a\":\"x\"}"}},
                {"id":"a","function":{"name":"alpha","arguments":"not valid <|open|>"}}
              ]},
              {"role":"tool","tool_call_id":"a","name":"stale","content":"alpha result"},
              {"role":"tool","tool_call_id":"b","tool":"wrong","content":"beta result"}
            ]"#,
        );
        let options = ChatOptions {
            tools: Some(&tools),
            thinking_effort: Some("low"),
            ..ChatOptions::default()
        };
        let segments = assert_gold(
            &tokenizer,
            messages(&conversation),
            &options,
            182,
            316,
            0x38e7_54cb_40cd_d010,
            0xb060_b0cf_4e2a_9500,
        );
        let rendered = render_segments(&segments);
        assert!(rendered.contains("parameters\":{\"a\":{\"type\":\"string\"},\"z\""));
        assert!(rendered.contains("argument key=\"z\" type=\"number\""));
        let beta_result = rendered.find("tool=\"beta\" index=\"1\"").unwrap();
        let alpha_result = rendered.rfind("tool=\"alpha\" index=\"2\"").unwrap();
        assert!(beta_result < alpha_result);
        // Invalid JSON remains ordinary text inside a structural <json> node.
        assert!(rendered.contains("not valid <|open|><|close|>json"));
    }

    #[test]
    #[ignore = "requires an installed k3-meta fixture (deltafin setup); run with --include-ignored on a machine with K3 metadata present"]
    fn response_controls_and_deep_sorted_schema_match_python_gold() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        let conversation = parse(r#"[{"role":"user","name":"A&B\"","content":"Return JSON."}]"#);
        let response_format = parse(
            r#"{"type":"json_schema","json_schema":{"name":"answer","schema":{"required":["b","a"],"properties":{"b":{"type":"number"},"a":{"type":"string"}},"type":"object"}}}"#,
        );
        let options = ChatOptions {
            thinking_effort: Some("high"),
            tool_choice: Some("required"),
            response_format: Some(&response_format),
            ..ChatOptions::default()
        };
        let segments = assert_gold(
            &tokenizer,
            messages(&conversation),
            &options,
            74,
            223,
            0x5974_61e6_ffe6_8c32,
            0xc65d_123d_688d_d863,
        );
        let rendered = render_segments(&segments);
        assert!(rendered.contains("name=\"A&amp;B&quot;\""));
        assert!(rendered.contains(
            r#"{"properties":{"a":{"type":"string"},"b":{"type":"number"}},"required":["b","a"],"type":"object"}"#
        ));
    }

    fn deterministic_history(message_count: usize, fragments_per_message: usize) -> Vec<Value> {
        let fragments = [
            "alpha beta ",
            "漢字かな ",
            "🙂🚀 ",
            "1234567 ",
            "line one\r\nline two ",
            "literal <|open|> marker ",
            "café naïve ",
            "!?'s ",
        ];
        let mut state = 0xa341_316c_u32;
        (0..message_count)
            .map(|index| {
                let mut content = String::with_capacity(fragments_per_message * 16);
                for _ in 0..fragments_per_message {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    content.push_str(fragments[state as usize % fragments.len()]);
                }
                serde_json::json!({
                    "role": if index % 2 == 0 { "user" } else { "assistant" },
                    "content": content,
                    "reasoning_content": (index % 2 == 1).then_some("reviewed thought"),
                })
            })
            .collect()
    }

    #[test]
    #[ignore = "requires an installed k3-meta fixture (deltafin setup); run with --include-ignored on a machine with K3 metadata present"]
    fn native_batched_chat_is_bit_exact_for_fuzzed_large_histories() {
        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        for (messages, fragments) in [(64, 160), (96, 192), (160, 192)] {
            let history = deterministic_history(messages, fragments);
            let options = ChatOptions::default();
            let segments = build_chat_segments(&history, &options).unwrap();
            let total_bytes: usize = segments.iter().map(|segment| segment.text.len()).sum();
            if std::thread::available_parallelism().is_ok_and(|workers| workers.get() > 1) {
                assert!(tokenizer.should_parallelize_segments(segments.len(), total_bytes));
            }
            let expected = encode_chat_segments_sequential(&tokenizer, &segments).unwrap();
            let actual = encode_chat(&tokenizer, &history, &options).unwrap();
            assert_eq!(actual, expected, "{messages} messages");
        }
    }

    #[test]
    #[ignore = "manual model-free tokenizer benchmark"]
    fn benchmark_native_chat_tokenization_small_and_large() {
        fn fastest(mut operation: impl FnMut(), iterations: usize) -> std::time::Duration {
            let mut fastest = std::time::Duration::MAX;
            for _ in 0..iterations {
                let started = std::time::Instant::now();
                operation();
                fastest = fastest.min(started.elapsed());
            }
            fastest
        }

        fn sequential_chat(
            tokenizer: &K3Tokenizer,
            history: &[Value],
            options: &ChatOptions<'_>,
        ) -> Vec<u32> {
            let segments = build_chat_segments(history, options).unwrap();
            encode_chat_segments_sequential(tokenizer, &segments).unwrap()
        }

        let tokenizer = K3Tokenizer::load_from_root(&repository_root()).unwrap();
        let options = ChatOptions::default();
        let small = deterministic_history(2, 8);
        let large = deterministic_history(128, 192);

        let small_sequential = fastest(
            || {
                std::hint::black_box(sequential_chat(&tokenizer, &small, &options));
            },
            40,
        );
        let small_native = fastest(
            || {
                std::hint::black_box(encode_chat(&tokenizer, &small, &options).unwrap());
            },
            40,
        );
        let large_sequential = fastest(
            || {
                std::hint::black_box(sequential_chat(&tokenizer, &large, &options));
            },
            5,
        );
        let large_native = fastest(
            || {
                std::hint::black_box(encode_chat(&tokenizer, &large, &options).unwrap());
            },
            5,
        );
        eprintln!(
            "native_chat_tokenization small sequential={small_sequential:?} native={small_native:?}; large sequential={large_sequential:?} native={large_native:?}"
        );
        assert!(small_native <= small_sequential.mul_f64(1.20));
        assert!(large_native < large_sequential);
    }

    #[test]
    fn malformed_inputs_fail_closed() {
        let bad_arguments = parse(
            r#"[{"role":"assistant","tool_calls":[{"function":{"name":"x","arguments":"[1]"}}]}]"#,
        );
        assert!(build_chat_segments(messages(&bad_arguments), &ChatOptions::default()).is_err());

        let images = ["only one".to_owned()];
        let no_image = parse(r#"[{"role":"user","content":"none"}]"#);
        let options = ChatOptions {
            image_prompts: Some(&images),
            ..ChatOptions::default()
        };
        assert!(build_chat_segments(messages(&no_image), &options).is_err());

        let medium = ChatOptions {
            thinking_effort: Some("medium"),
            ..ChatOptions::default()
        };
        assert!(build_chat_segments(messages(&no_image), &medium).is_err());
    }

    fn u32_hash(values: impl IntoIterator<Item = u32>) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in values {
            for byte in value.to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        }
        hash
    }
}
