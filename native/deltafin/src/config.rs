use std::path::PathBuf;

use crate::cli::{RunArgs, ServeArgs};
use crate::dspark_runtime::Mode as DSparkRuntimeMode;
use crate::error::{DeltafinError, Result};
use crate::platform::DeviceRequest;
use crate::program::K3_LAYER_COUNT;
use crate::quality::{QualityPolicy, ResidentWeightAuthority};
use crate::router_trace::RouterTraceMode;

pub const MAX_SPINE_READ_THREADS: usize = 16;
pub const MAX_EXPERT_READ_THREADS: usize = 16;

pub fn parse_spine_read_threads(raw: &str) -> Result<usize> {
    let workers = raw
        .parse::<usize>()
        .map_err(|_| DeltafinError::new("spine read threads must be an integer in 1..=16"))?;
    if !(1..=MAX_SPINE_READ_THREADS).contains(&workers) {
        return Err(DeltafinError::new(
            "spine read threads must be an integer in 1..=16",
        ));
    }
    Ok(workers)
}

pub fn parse_expert_read_threads(raw: &str) -> Result<usize> {
    let workers = raw
        .parse::<usize>()
        .map_err(|_| DeltafinError::new("K3_EXPERT_READ_THREADS must be an integer in 1..=16"))?;
    if !(1..=MAX_EXPERT_READ_THREADS).contains(&workers) {
        return Err(DeltafinError::new(
            "K3_EXPERT_READ_THREADS must be an integer in 1..=16",
        ));
    }
    Ok(workers)
}

fn parse_spine_fd_cache(raw: &str) -> Result<Option<bool>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "1" | "true" | "on" | "yes" | "enabled" => Ok(Some(true)),
        "0" | "false" | "off" | "no" | "disabled" => Ok(Some(false)),
        _ => Err(DeltafinError::new(
            "K3_SPINE_FDCACHE must be auto, 0/1, false/true, or off/on",
        )),
    }
}

fn parse_spine_stream_nocache(raw: &str) -> Result<Option<bool>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(None),
        "1" | "true" | "on" | "yes" | "enabled" => Ok(Some(true)),
        "0" | "false" | "off" | "no" | "disabled" => Ok(Some(false)),
        _ => Err(DeltafinError::new(
            "K3_SPINE_STREAM_NOCACHE must be auto, 0/1, false/true, or off/on",
        )),
    }
}

fn parse_spine_resident_gb(raw: &str) -> Result<u64> {
    let gigabytes = if raw.trim().is_empty() {
        0.0
    } else {
        raw.trim().parse::<f64>().map_err(|_| {
            DeltafinError::new("K3_SPINE_RESIDENT_GB must be a finite non-negative number")
        })?
    };
    let bytes = gigabytes * 1_000_000_000.0;
    if !gigabytes.is_finite() || gigabytes < 0.0 || bytes > u64::MAX as f64 {
        return Err(DeltafinError::new(
            "K3_SPINE_RESIDENT_GB must be a finite non-negative number",
        ));
    }
    Ok(bytes as u64)
}

fn parse_provider_resident_layers(raw: &str) -> Result<usize> {
    let layers = raw.parse::<usize>().map_err(|_| {
        DeltafinError::new(format!(
            "K3_PROVIDER_RESIDENT_LAYERS must be an integer in 0..={K3_LAYER_COUNT}"
        ))
    })?;
    if layers > K3_LAYER_COUNT {
        return Err(DeltafinError::new(format!(
            "K3_PROVIDER_RESIDENT_LAYERS must be an integer in 0..={K3_LAYER_COUNT}"
        )));
    }
    Ok(layers)
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum SpineRequest {
    Auto,
    Bf16,
    Int8,
}

impl SpineRequest {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let value = value.unwrap_or("auto").trim().to_ascii_lowercase();
        match value.as_str() {
            "" | "auto" => Ok(Self::Auto),
            "bf16" => Ok(Self::Bf16),
            "int8" => Ok(Self::Int8),
            "mixed" => Err(DeltafinError::new(
                "K3_SPINE=mixed is unsupported because its research codecs change target weights",
            )),
            _ => Err(DeltafinError::new(
                "spine must be auto/bf16 (original weights) or explicit int8 (quantized and non-weight-exact)",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExpertBackendRequest {
    Auto,
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ExpertScale4Request {
    Off,
    Auto,
    Require,
}

impl ExpertScale4Request {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
            "off" => Ok(Self::Off),
            "" | "auto" => Ok(Self::Auto),
            "require" => Ok(Self::Require),
            _ => Err(DeltafinError::new(
                "K3_EXPERT_SCALE4 must be auto, off, or require",
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DSparkRequest {
    Off,
    Auto,
    On,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum QwenRequest {
    Off,
    Auto,
    On,
}

/// Product surface which owns the native engine.
///
/// A direct run has one immutable request shape, so automatic draft models
/// that cannot serve that shape must not consume unified memory. The server
/// accepts both chat and raw-completion requests over its lifetime and must
/// retain both automatic proposal paths when they are installed.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum RuntimeSurface {
    DirectRun,
    Server,
}

impl QwenRequest {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" | "no" => Ok(Self::Off),
            "" | "auto" => Ok(Self::Auto),
            "on" | "1" | "true" | "yes" => Ok(Self::On),
            _ => Err(DeltafinError::new("K3_UAG_DRAFT must be off, auto, or on")),
        }
    }
}

impl DSparkRequest {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("auto").trim().to_ascii_lowercase().as_str() {
            "off" | "0" => Ok(Self::Off),
            "" | "auto" => Ok(Self::Auto),
            "on" | "1" => Ok(Self::On),
            _ => Err(DeltafinError::new("K3_DSPARK must be off, auto, or on")),
        }
    }

    pub const fn runtime_mode(self) -> DSparkRuntimeMode {
        match self {
            Self::Off => DSparkRuntimeMode::Off,
            Self::Auto => DSparkRuntimeMode::Auto,
            Self::On => DSparkRuntimeMode::On,
        }
    }
}

impl ExpertBackendRequest {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        let value = value.unwrap_or("auto").trim().to_ascii_lowercase();
        match value.as_str() {
            "" | "auto" => Ok(Self::Auto),
            "cpu" => Ok(Self::Cpu),
            "metal" => Ok(Self::Metal),
            "cuda" => Ok(Self::Cuda),
            _ => Err(DeltafinError::new(
                "expert backend must be auto, cpu, metal, or cuda",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuntimeConfig {
    pub surface: RuntimeSurface,
    pub prompt: String,
    pub max_new: Option<u64>,
    pub chat: bool,
    pub stats: bool,
    pub events_jsonl: Option<PathBuf>,
    pub router_trace_mode: RouterTraceMode,
    pub router_trace_path: Option<PathBuf>,
    pub model_root: PathBuf,
    pub device: DeviceRequest,
    pub spine: SpineRequest,
    pub spine_read_threads: Option<usize>,
    /// `None` selects the capability-qualified automatic loose-spine policy.
    /// Explicit enablement remains all-or-nothing and fails if the complete
    /// immutable source roster cannot fit with descriptor headroom.
    pub spine_fd_cache: Option<bool>,
    /// `None` is the capability-qualified automatic policy. `Some` preserves
    /// the established K3_SPINE_STREAM_NOCACHE explicit override.
    pub spine_stream_nocache: Option<bool>,
    pub spine_resident_bytes: Option<u64>,
    /// Diagnostic ceiling for provider-owned layer storage. `None` preserves
    /// automatic safe-prefix selection; every explicit value remains subject
    /// to the same host/device safety envelope.
    pub provider_resident_layers: Option<usize>,
    pub expert_read_threads: Option<usize>,
    pub expert_backend: ExpertBackendRequest,
    pub expert_scale4: ExpertScale4Request,
    pub quality: QualityPolicy,
    pub dspark: DSparkRequest,
    pub dspark_max_context: Option<usize>,
    pub dspark_min_auto_speedup: f64,
    pub qwen: QwenRequest,
}

impl RuntimeConfig {
    pub fn resolve<F>(arguments: RunArgs, mut environment: F) -> Result<Self>
    where
        F: FnMut(&str) -> Option<String>,
    {
        let device_value = arguments.device.or_else(|| environment("K3_DEV"));
        let spine_value = arguments.spine.or_else(|| environment("K3_SPINE"));
        let spine_read_threads = arguments
            .spine_read_threads
            .map(Ok)
            .or_else(|| {
                environment("K3_SPINE_READ_THREADS").map(|raw| parse_spine_read_threads(&raw))
            })
            .transpose()?;
        let spine_fd_cache = environment("K3_SPINE_FDCACHE")
            .as_deref()
            .map(parse_spine_fd_cache)
            .transpose()?
            .flatten();
        let spine_stream_nocache = environment("K3_SPINE_STREAM_NOCACHE")
            .as_deref()
            .map(parse_spine_stream_nocache)
            .transpose()?
            .flatten();
        let spine_resident_bytes = environment("K3_SPINE_RESIDENT_GB")
            .as_deref()
            .map(parse_spine_resident_gb)
            .transpose()?;
        let provider_resident_layers = environment("K3_PROVIDER_RESIDENT_LAYERS")
            .as_deref()
            .map(parse_provider_resident_layers)
            .transpose()?;
        let expert_read_threads = environment("K3_EXPERT_READ_THREADS")
            .as_deref()
            .map(parse_expert_read_threads)
            .transpose()?;
        let backend_value = arguments.expert_backend.or_else(|| environment("K3_MOE"));
        let router_trace_path = arguments
            .router_trace
            .or_else(|| environment("K3_TRACE_PATH").map(PathBuf::from));
        let router_trace_mode_value = arguments
            .router_trace_mode
            .or_else(|| environment("K3_TRACE"));
        let router_trace_mode = if router_trace_mode_value.is_none() && router_trace_path.is_some()
        {
            RouterTraceMode::Buffered
        } else {
            RouterTraceMode::parse(router_trace_mode_value.as_deref())?
        };
        if router_trace_mode == RouterTraceMode::Off && router_trace_path.is_some() {
            return Err(DeltafinError::new(
                "router-trace path was supplied while tracing is explicitly off",
            ));
        }
        let spine = SpineRequest::parse(spine_value.as_deref())?;
        let resident_weights = match spine {
            // The default resident spine is the measured row-int8
            // representation; setup prepares it automatically. The original
            // BF16 checkpoint remains on disk as the conversion source and
            // verification authority, and stays selectable with --spine bf16.
            SpineRequest::Auto | SpineRequest::Int8 => ResidentWeightAuthority::QuantizedInt8,
            SpineRequest::Bf16 => ResidentWeightAuthority::OriginalBf16,
        };
        let quality = QualityPolicy::from_legacy_environment(
            environment("K3_APPROX").as_deref(),
            environment("K3_DTYPE").as_deref(),
            environment("K3_MOE_TOP_K").as_deref(),
        )?
        .with_resident_weights(resident_weights);
        let dspark = DSparkRequest::parse(environment("K3_DSPARK").as_deref())?;
        let qwen = QwenRequest::parse(environment("K3_UAG_DRAFT").as_deref())?;
        let expert_scale4 = ExpertScale4Request::parse(environment("K3_EXPERT_SCALE4").as_deref())?;
        let dspark_max_context = parse_optional_positive_usize(
            environment("K3_DSPARK_MAX_CONTEXT").as_deref(),
            8_192,
            "K3_DSPARK_MAX_CONTEXT",
        )?;
        let dspark_min_auto_speedup = environment("K3_DSPARK_AUTO_MIN_SPEEDUP")
            .as_deref()
            .unwrap_or("0.03")
            .parse::<f64>()
            .map_err(|_| {
                DeltafinError::new("K3_DSPARK_AUTO_MIN_SPEEDUP must be a finite number in [0,1)")
            })?;
        if !dspark_min_auto_speedup.is_finite() || !(0.0..1.0).contains(&dspark_min_auto_speedup) {
            return Err(DeltafinError::new(
                "K3_DSPARK_AUTO_MIN_SPEEDUP must be a finite number in [0,1)",
            ));
        }

        Ok(Self {
            surface: RuntimeSurface::DirectRun,
            prompt: arguments.prompt,
            max_new: arguments.max_new,
            chat: arguments.chat,
            stats: arguments.stats,
            events_jsonl: arguments.events_jsonl,
            router_trace_mode,
            router_trace_path,
            model_root: arguments.model_root,
            device: DeviceRequest::parse(device_value.as_deref())?,
            spine,
            spine_read_threads,
            spine_fd_cache,
            spine_stream_nocache,
            spine_resident_bytes,
            provider_resident_layers,
            expert_read_threads,
            expert_backend: ExpertBackendRequest::parse(backend_value.as_deref())?,
            expert_scale4,
            quality,
            dspark,
            dspark_max_context,
            dspark_min_auto_speedup,
            qwen,
        })
    }

    pub fn from_process(arguments: RunArgs) -> Result<Self> {
        Self::resolve(arguments, |name| std::env::var(name).ok())
    }

    pub fn from_server(arguments: &ServeArgs) -> Result<Self> {
        let mut config = Self::resolve(
            RunArgs {
                prompt: String::new(),
                max_new: None,
                chat: false,
                stats: false,
                events_jsonl: None,
                router_trace: arguments.router_trace.clone(),
                router_trace_mode: arguments.router_trace_mode.clone(),
                model_root: arguments.model_root.clone(),
                device: arguments.device.clone(),
                spine: arguments.spine.clone(),
                spine_read_threads: arguments.spine_read_threads,
                expert_backend: arguments.expert_backend.clone(),
            },
            |name| std::env::var(name).ok(),
        )?;
        config.surface = RuntimeSurface::Server;
        Ok(config)
    }
}

fn parse_optional_positive_usize(
    value: Option<&str>,
    default: usize,
    name: &str,
) -> Result<Option<usize>> {
    let parsed = value
        .unwrap_or("")
        .trim()
        .parse::<usize>()
        .or_else(|_| value.is_none().then_some(default).ok_or(()))
        .map_err(|_| DeltafinError::new(format!("{name} must be a non-negative integer")))?;
    Ok((parsed != 0).then_some(parsed))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments() -> RunArgs {
        RunArgs {
            prompt: "hello".into(),
            max_new: Some(4),
            chat: true,
            stats: false,
            events_jsonl: None,
            router_trace: None,
            router_trace_mode: None,
            model_root: PathBuf::from("."),
            device: None,
            spine: None,
            spine_read_threads: None,
            expert_backend: None,
        }
    }

    #[test]
    fn cli_overrides_environment_without_mutating_process_state() {
        let mut arguments = arguments();
        arguments.device = Some("cpu".into());
        arguments.spine = Some("int8".into());
        let config = RuntimeConfig::resolve(arguments, |name| match name {
            "K3_DEV" => Some("cuda:2".into()),
            "K3_SPINE" => Some("bf16".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(config.device, DeviceRequest::Cpu);
        assert_eq!(config.spine, SpineRequest::Int8);
        assert_eq!(
            config.quality.resident_weights,
            ResidentWeightAuthority::QuantizedInt8
        );
    }

    #[test]
    fn auto_and_bf16_preserve_original_checkpoint_authority() {
        let auto = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        assert_eq!(auto.surface, RuntimeSurface::DirectRun);
        assert_eq!(auto.spine, SpineRequest::Auto);
        assert_eq!(auto.spine_fd_cache, None);
        assert_eq!(auto.provider_resident_layers, None);
        assert_eq!(auto.expert_read_threads, None);
        assert!(auto.quality.is_weight_exact());
        assert_eq!(auto.dspark, DSparkRequest::Auto);
        assert_eq!(auto.dspark_max_context, Some(8_192));
        assert_eq!(auto.dspark_min_auto_speedup, 0.03);
        assert_eq!(auto.router_trace_mode, RouterTraceMode::Off);
        assert!(auto.router_trace_path.is_none());

        let mut explicit = arguments();
        explicit.spine = Some("bf16".into());
        let bf16 = RuntimeConfig::resolve(explicit, |_| None).unwrap();
        assert_eq!(bf16.spine, SpineRequest::Bf16);
        assert!(bf16.quality.is_weight_exact());
    }

    #[test]
    fn spine_reader_cli_override_wins_and_environment_is_bounded() {
        let from_environment = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_SPINE_READ_THREADS").then(|| "6".into())
        })
        .unwrap();
        assert_eq!(from_environment.spine_read_threads, Some(6));

        let mut explicit = arguments();
        explicit.spine_read_threads = Some(3);
        let from_cli = RuntimeConfig::resolve(explicit, |name| {
            (name == "K3_SPINE_READ_THREADS").then(|| "6".into())
        })
        .unwrap();
        assert_eq!(from_cli.spine_read_threads, Some(3));

        for invalid in ["0", "17", "many"] {
            assert!(
                RuntimeConfig::resolve(arguments(), |name| {
                    (name == "K3_SPINE_READ_THREADS").then(|| invalid.into())
                })
                .is_err()
            );
        }
    }

    #[test]
    fn established_spine_cache_environment_overrides_are_preserved() {
        let config = RuntimeConfig::resolve(arguments(), |name| match name {
            "K3_SPINE_STREAM_NOCACHE" => Some("on".into()),
            "K3_SPINE_RESIDENT_GB" => Some("2.1".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(config.spine_stream_nocache, Some(true));
        assert_eq!(config.spine_resident_bytes, Some(2_100_000_000));

        let auto = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_SPINE_STREAM_NOCACHE").then(|| "auto".into())
        })
        .unwrap();
        assert_eq!(auto.spine_stream_nocache, None);

        for (name, value) in [
            ("K3_SPINE_STREAM_NOCACHE", ""),
            ("K3_SPINE_STREAM_NOCACHE", "sometimes"),
            ("K3_SPINE_RESIDENT_GB", "-1"),
            ("K3_SPINE_RESIDENT_GB", "nan"),
        ] {
            assert!(
                RuntimeConfig::resolve(arguments(), |candidate| {
                    (candidate == name).then(|| value.into())
                })
                .is_err()
            );
        }
    }

    #[test]
    fn loose_spine_descriptor_cache_is_explicit_or_auto() {
        for (raw, expected) in [
            ("auto", None),
            ("1", Some(true)),
            ("on", Some(true)),
            ("0", Some(false)),
            ("off", Some(false)),
        ] {
            let config = RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_SPINE_FDCACHE").then(|| raw.into())
            })
            .unwrap();
            assert_eq!(config.spine_fd_cache, expected);
        }
        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_SPINE_FDCACHE").then(|| "perhaps".into())
            })
            .is_err()
        );
    }

    #[test]
    fn expert_reader_environment_override_is_bounded() {
        let config = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_EXPERT_READ_THREADS").then(|| "8".into())
        })
        .unwrap();
        assert_eq!(config.expert_read_threads, Some(8));

        for invalid in ["0", "17", "many"] {
            assert!(
                RuntimeConfig::resolve(arguments(), |name| {
                    (name == "K3_EXPERT_READ_THREADS").then(|| invalid.into())
                })
                .is_err()
            );
        }
    }

    #[test]
    fn provider_resident_layer_control_is_env_only_and_bounded() {
        for (raw, expected) in [("0", 0), ("13", 13), ("93", 93)] {
            let config = RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_PROVIDER_RESIDENT_LAYERS").then(|| raw.into())
            })
            .unwrap();
            assert_eq!(config.provider_resident_layers, Some(expected));
        }

        for invalid in ["", "-1", "94", "many"] {
            let error = RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_PROVIDER_RESIDENT_LAYERS").then(|| invalid.into())
            })
            .unwrap_err();
            assert!(error.to_string().contains("K3_PROVIDER_RESIDENT_LAYERS"));
        }
    }

    #[test]
    fn router_trace_path_enables_buffering_and_explicit_off_fails_closed() {
        let mut explicit_path = arguments();
        explicit_path.router_trace = Some(PathBuf::from("k3-meta/native-routes.jsonl"));
        let config = RuntimeConfig::resolve(explicit_path, |_| None).unwrap();
        assert_eq!(config.router_trace_mode, RouterTraceMode::Buffered);
        assert_eq!(
            config.router_trace_path,
            Some(PathBuf::from("k3-meta/native-routes.jsonl"))
        );

        let mut conflict = arguments();
        conflict.router_trace = Some(PathBuf::from("routes.jsonl"));
        conflict.router_trace_mode = Some("off".into());
        assert!(RuntimeConfig::resolve(conflict, |_| None).is_err());
    }

    #[test]
    fn legacy_quality_environment_is_still_fail_closed() {
        let error = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_MOE_TOP_K").then(|| "15".into())
        })
        .unwrap_err();
        assert!(error.to_string().contains("all 16 routed experts"));
    }

    #[test]
    fn dspark_environment_is_bounded_and_fail_closed() {
        let disabled = RuntimeConfig::resolve(arguments(), |name| match name {
            "K3_DSPARK" => Some("off".into()),
            "K3_DSPARK_MAX_CONTEXT" => Some("0".into()),
            "K3_DSPARK_AUTO_MIN_SPEEDUP" => Some("0.2".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(disabled.dspark, DSparkRequest::Off);
        assert_eq!(disabled.dspark_max_context, None);
        assert_eq!(disabled.dspark_min_auto_speedup, 0.2);

        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_DSPARK_AUTO_MIN_SPEEDUP").then(|| "1".into())
            })
            .unwrap_err()
            .to_string()
            .contains("[0,1)")
        );
    }

    #[test]
    fn qwen_environment_preserves_auto_and_explicit_modes() {
        let automatic = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        assert_eq!(automatic.qwen, QwenRequest::Auto);
        let enabled = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_UAG_DRAFT").then(|| "true".into())
        })
        .unwrap();
        assert_eq!(enabled.qwen, QwenRequest::On);
        let disabled = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_UAG_DRAFT").then(|| "off".into())
        })
        .unwrap();
        assert_eq!(disabled.qwen, QwenRequest::Off);
        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_UAG_DRAFT").then(|| "maybe".into())
            })
            .is_err()
        );
    }

    #[test]
    fn scale4_environment_is_explicit_and_fail_closed() {
        let automatic = RuntimeConfig::resolve(arguments(), |_| None).unwrap();
        assert_eq!(automatic.expert_scale4, ExpertScale4Request::Auto);
        let raw = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_EXPERT_SCALE4").then(|| "off".into())
        })
        .unwrap();
        assert_eq!(raw.expert_scale4, ExpertScale4Request::Off);
        let required = RuntimeConfig::resolve(arguments(), |name| {
            (name == "K3_EXPERT_SCALE4").then(|| "require".into())
        })
        .unwrap();
        assert_eq!(required.expert_scale4, ExpertScale4Request::Require);
        assert!(
            RuntimeConfig::resolve(arguments(), |name| {
                (name == "K3_EXPERT_SCALE4").then(|| "on".into())
            })
            .unwrap_err()
            .to_string()
            .contains("auto, off, or require")
        );
    }
}
