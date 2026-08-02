//! Non-negotiable target-model quality policy.
//!
//! Draft models may propose token IDs and providers may schedule work
//! differently. None of those components may author output: full K3, with all
//! 16 routed experts, remains the sole authority for every emitted token. The
//! original BF16 resident checkpoint is canonical; any explicitly requested
//! quantized derivative is labeled non-weight-exact rather than being passed
//! off as a quality-preserving speed path.

use crate::error::{DeltafinError, Result};

pub const K3_ROUTED_EXPERTS: u16 = 16;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ActivationPrecision {
    Fp32,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum TargetAuthority {
    FullK3,
}

/// Fidelity of Deltafin's resident, non-expert weight representation.
///
/// K3's routed experts are released as MXFP4 and remain authoritative in both
/// modes. This field describes only the additional resident-spine conversion:
/// row-int8 is useful, but it is not the original BF16 checkpoint.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ResidentWeightAuthority {
    OriginalBf16,
    QuantizedInt8,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct QualityPolicy {
    pub target_authority: TargetAuthority,
    pub routed_experts: u16,
    pub activation_precision: ActivationPrecision,
    pub resident_weights: ResidentWeightAuthority,
}

impl QualityPolicy {
    pub fn exact() -> Self {
        Self {
            target_authority: TargetAuthority::FullK3,
            routed_experts: K3_ROUTED_EXPERTS,
            activation_precision: ActivationPrecision::Fp32,
            resident_weights: ResidentWeightAuthority::OriginalBf16,
        }
    }

    pub fn with_resident_weights(mut self, resident_weights: ResidentWeightAuthority) -> Self {
        self.resident_weights = resident_weights;
        self
    }

    #[cfg(test)]
    pub const fn is_weight_exact(self) -> bool {
        matches!(self.resident_weights, ResidentWeightAuthority::OriginalBf16)
    }

    pub fn from_legacy_environment(
        approximate: Option<&str>,
        dtype: Option<&str>,
        moe_top_k: Option<&str>,
    ) -> Result<Self> {
        reject_approximate(approximate)?;
        require_fp32(dtype)?;
        require_all_experts(moe_top_k)?;
        Ok(Self::exact())
    }
}

fn normalized(value: Option<&str>, default: &str) -> String {
    value.unwrap_or(default).trim().to_ascii_lowercase()
}

fn reject_approximate(value: Option<&str>) -> Result<()> {
    let value = normalized(value, "0");
    match value.as_str() {
        "" | "0" | "false" | "off" | "no" | "disabled" => Ok(()),
        "1" | "true" | "on" | "yes" | "enabled" => Err(DeltafinError::new(
            "K3_APPROX is unsupported: Deltafin does not trade target quality for speed",
        )),
        _ => Err(DeltafinError::new(
            "K3_APPROX must be off/0/false; approximate target inference is unsupported",
        )),
    }
}

fn require_fp32(value: Option<&str>) -> Result<()> {
    let value = normalized(value, "fp32");
    match value.as_str() {
        "" | "fp32" | "float32" => Ok(()),
        _ => Err(DeltafinError::new(format!(
            "K3_DTYPE={value:?} is unsupported: target activations must remain fp32"
        ))),
    }
}

fn require_all_experts(value: Option<&str>) -> Result<()> {
    let Some(raw) = value.filter(|raw| !raw.trim().is_empty()) else {
        return Ok(());
    };
    let selected = raw.trim().parse::<u16>().map_err(|_| {
        DeltafinError::new(format!(
            "K3_MOE_TOP_K must remain {K3_ROUTED_EXPERTS}; got {raw:?}"
        ))
    })?;
    if selected != K3_ROUTED_EXPERTS {
        return Err(DeltafinError::new(format!(
            "K3_MOE_TOP_K={selected} is unsupported: all {K3_ROUTED_EXPERTS} routed experts are required"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_is_full_k3_fp32_top_16() {
        assert_eq!(
            QualityPolicy::from_legacy_environment(None, None, None).unwrap(),
            QualityPolicy::exact()
        );
    }

    #[test]
    fn accepts_equivalent_exact_controls() {
        assert!(
            QualityPolicy::from_legacy_environment(Some("disabled"), Some("float32"), Some("16"))
                .is_ok()
        );
        assert!(QualityPolicy::from_legacy_environment(None, Some(""), None).is_ok());
    }

    #[test]
    fn rejects_every_quality_reduction() {
        assert!(QualityPolicy::from_legacy_environment(Some("on"), None, None).is_err());
        assert!(QualityPolicy::from_legacy_environment(None, Some("fp16"), None).is_err());
        assert!(QualityPolicy::from_legacy_environment(None, None, Some("8")).is_err());
    }
}
