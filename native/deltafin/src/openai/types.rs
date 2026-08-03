use std::error::Error;
use std::fmt::{self, Display, Formatter};
use std::io;

use serde_json::{Map, Value};

use crate::error::{DeltafinError, Result};

/// One validated OpenAI chat message.
///
/// `content` remains JSON because OpenAI permits text, `null`, or a typed part
/// array. The production request boundary currently admits only string and
/// text-part content; multimodal parts fail before target entry because the
/// native vision/projector path is not implemented. The native target adapter
/// applies K3's exact chat template. `additional_fields` contains only the
/// role-specific history fields that the native template implements; the HTTP
/// boundary rejects unknown message semantics instead of retaining and then
/// silently ignoring them.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    pub content: Value,
    pub additional_fields: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TargetPrompt {
    Completion(String),
    Chat(Vec<ChatMessage>),
}

/// A request that may be handed only to the authoritative target adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetRequest {
    pub prompt: TargetPrompt,
    pub max_new_tokens: usize,
    /// Chat-only thinking depth (`low`, `high`, or `max`), already
    /// normalized. `None` defers to the engine's configured default and
    /// ultimately the chat template's own `max`.
    pub reasoning_effort: Option<String>,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl FinishReason {
    pub(crate) fn as_openai_str(self) -> &'static str {
        match self {
            Self::Stop => "stop",
            Self::Length => "length",
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TokenUsage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TargetDeltaChannel {
    Content,
    ReasoningContent,
}

/// One text delta already certified by full K3.
///
/// The HTTP layer has no constructor for unverified draft output.  A target
/// adapter may create these values only after the corresponding tokens have
/// passed exact target verification.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TargetDelta {
    channel: TargetDeltaChannel,
    text: String,
}

impl TargetDelta {
    pub fn target_verified_content(text: impl Into<String>) -> Self {
        Self {
            channel: TargetDeltaChannel::Content,
            text: text.into(),
        }
    }

    pub fn target_verified_reasoning(text: impl Into<String>) -> Self {
        Self {
            channel: TargetDeltaChannel::ReasoningContent,
            text: text.into(),
        }
    }

    pub fn channel(&self) -> TargetDeltaChannel {
        self.channel
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Callback supplied by the OpenAI SSE writer to the authoritative target.
pub trait TargetDeltaSink {
    fn publish_target_delta(&mut self, delta: TargetDelta) -> io::Result<()>;
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct TargetStreamSummary {
    finish_reason: FinishReason,
    usage: Option<TokenUsage>,
}

impl TargetStreamSummary {
    pub fn target_verified(finish_reason: FinishReason) -> Self {
        Self {
            finish_reason,
            usage: None,
        }
    }

    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn finish_reason(self) -> FinishReason {
        self.finish_reason
    }

    pub fn usage(self) -> Option<TokenUsage> {
        self.usage
    }
}

#[derive(Debug)]
pub enum StreamGenerationError {
    Target(DeltafinError),
    Publication(io::Error),
}

impl Display for StreamGenerationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Target(error) => error.fmt(formatter),
            Self::Publication(error) => error.fmt(formatter),
        }
    }
}

impl Error for StreamGenerationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Target(error) => Some(error),
            Self::Publication(error) => Some(error),
        }
    }
}

impl From<DeltafinError> for StreamGenerationError {
    fn from(error: DeltafinError) -> Self {
        Self::Target(error)
    }
}

impl From<io::Error> for StreamGenerationError {
    fn from(error: io::Error) -> Self {
        Self::Publication(error)
    }
}

/// Whether the complete externally visible SSE response was published.
///
/// A target adapter should publish staged KV/DSpark state only for
/// `Complete`.  `Aborted` must discard it.  The server calls this hook after
/// the terminal finish frame, `[DONE]`, and the response writer all flush.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum StreamPublication {
    Complete,
    Aborted,
}

/// Text certified by the full target model.
///
/// The constructor is intentionally named `target_verified`: an adapter must
/// not create this value from a draft model's proposal.  Drafts may influence
/// scheduling only; the adapter must compare every emitted token with K3.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetOutput {
    text: String,
    reasoning_content: Option<String>,
    finish_reason: FinishReason,
    usage: Option<TokenUsage>,
}

impl TargetOutput {
    pub fn target_verified(text: impl Into<String>, finish_reason: FinishReason) -> Self {
        Self {
            text: text.into(),
            reasoning_content: None,
            finish_reason,
            usage: None,
        }
    }

    pub fn with_reasoning_content(mut self, reasoning_content: impl Into<String>) -> Self {
        self.reasoning_content = Some(reasoning_content.into());
        self
    }

    pub fn with_usage(mut self, usage: TokenUsage) -> Self {
        self.usage = Some(usage);
        self
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn reasoning_content(&self) -> Option<&str> {
        self.reasoning_content.as_deref()
    }

    pub fn finish_reason(&self) -> FinishReason {
        self.finish_reason
    }

    pub fn usage(&self) -> Option<TokenUsage> {
        self.usage
    }
}

/// Liveness of the one HTTP client waiting on a generation.
///
/// A generation on this hardware can run for hours, so an adapter is told how
/// to ask whether anyone is still listening and may stop cooperatively when
/// the answer is provably no.  The probe contract is asymmetric: a false
/// "still present" merely spends compute the old behavior also spent, while a
/// false "disconnected" abandons a live client's request — so probes must
/// report disconnection only on positive evidence that the connection has
/// left its established state, never on doubt.  Fabricated test requests and
/// transports without an observable socket use
/// [`ClientPresence::assumed_present`].
pub struct ClientPresence {
    probe: Option<Box<dyn Fn() -> bool + Send + Sync>>,
}

impl ClientPresence {
    /// A client that can never be observed to disconnect.
    pub fn assumed_present() -> Self {
        Self { probe: None }
    }

    /// A client observed through `probe`, which returns true only once the
    /// client has provably gone away.  The probe runs on the generation
    /// thread between decode transactions and must never block.
    pub fn from_probe(probe: Box<dyn Fn() -> bool + Send + Sync>) -> Self {
        Self { probe: Some(probe) }
    }

    pub fn disconnected(&self) -> bool {
        self.probe.as_ref().is_some_and(|probe| probe())
    }
}

impl fmt::Debug for ClientPresence {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ClientPresence")
            .field("probed", &self.probe.is_some())
            .finish()
    }
}

/// The sole generation capability visible to the HTTP service.
///
/// Implementations may internally use speculative work, but must return only
/// text already certified by a full K3 target pass.  Requiring `&mut self`
/// also makes exclusive ownership explicit; [`super::OpenAiService`] adds a
/// mutex so concurrent HTTP handlers cannot enter the target together.
pub trait AuthoritativeTarget: Send {
    /// Generate one complete K3-certified response.
    ///
    /// `client` reports whether the requesting HTTP client is still
    /// connected.  An adapter that honors it must fail the request rather
    /// than return partial text: a disconnect-truncated result is not a
    /// certified response and must never be published or replayed.
    fn generate_target(
        &mut self,
        request: &TargetRequest,
        client: &ClientPresence,
    ) -> Result<TargetOutput>;

    /// Resolve state staged by a non-streaming response after the HTTP body is
    /// completely written. Implementations must treat failure as rollback of
    /// optional reuse only; K3-certified output cannot be retroactively changed.
    fn finish_target_response(&mut self, _publication: StreamPublication) {}

    /// Generate target-certified deltas.
    ///
    /// Existing adapters remain source-compatible: the default implementation
    /// performs ordinary authoritative generation and emits it as at most two
    /// deltas.  A real native decoder should override this method and invoke
    /// the sink immediately after each decoded token becomes K3-certified.
    fn generate_target_stream(
        &mut self,
        request: &TargetRequest,
        sink: &mut dyn TargetDeltaSink,
        client: &ClientPresence,
    ) -> std::result::Result<TargetStreamSummary, StreamGenerationError> {
        let output = self.generate_target(request, client)?;
        if let Some(reasoning) = output.reasoning_content().filter(|text| !text.is_empty()) {
            sink.publish_target_delta(TargetDelta::target_verified_reasoning(reasoning))?;
        }
        if !output.text().is_empty() {
            sink.publish_target_delta(TargetDelta::target_verified_content(output.text()))?;
        }
        let mut summary = TargetStreamSummary::target_verified(output.finish_reason());
        if let Some(usage) = output.usage() {
            summary = summary.with_usage(usage);
        }
        Ok(summary)
    }

    /// Resolve any response-boundary state staged during streaming.
    ///
    /// This hook is intentionally infallible: publication or cleanup trouble
    /// may disable an optimization, but cannot retroactively invalidate text
    /// already certified by K3 and written to the client.
    fn finish_target_stream(&mut self, _publication: StreamPublication) {}
}
