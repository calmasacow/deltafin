//! Minimal OpenAI-compatible HTTP boundary for the native runtime.
//!
//! This module deliberately knows nothing about draft models.  Its only
//! generation dependency is an [`AuthoritativeTarget`], and it serializes all
//! calls into that target.  Speculation belongs behind the target adapter,
//! where full K3 verification can remain the sole authority for emitted text.

mod response_memo;
mod server;
mod types;

pub use server::{
    DEFAULT_MAX_REQUEST_BODY_BYTES, DEFAULT_RESPONSE_MEMO_BYTES, DEFAULT_RESPONSE_MEMO_ENTRIES,
    MAX_REQUEST_BODY_BYTES, MAX_RESPONSE_MEMO_BYTES, OpenAiHttpServer, OpenAiService, ServerConfig,
};
pub use types::{
    AuthoritativeTarget, ChatMessage, FinishReason, StreamGenerationError, StreamPublication,
    TargetDelta, TargetDeltaChannel, TargetDeltaSink, TargetOutput, TargetPrompt, TargetRequest,
    TargetStreamSummary, TokenUsage,
};
