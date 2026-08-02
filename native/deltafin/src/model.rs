//! Strict native ownership of the target-model architecture contract.
//!
//! The Python runtime currently imports executable model code just to discover
//! these fields.  The native process reads inert JSON, validates every
//! performance-critical K3 dimension, and fails closed before opening any
//! weight payload.  A different model can never be mistaken for the full K3
//! target merely because some tensor names happen to match.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use serde_json::{Map, Value};

use crate::error::{DeltafinError, Result};
use crate::quality::K3_ROUTED_EXPERTS;

const MAX_CONFIG_BYTES: u64 = 1024 * 1024;
const K3_MLA_LAYERS_ONE_BASED: [usize; 24] = [
    4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93,
];

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum LayerKind {
    Kda,
    Mla,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ModelSpec {
    pub bos_token_id: usize,
    pub eos_token_id: usize,
    pub pad_token_id: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub routed_expert_hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_experts: usize,
    pub num_experts_per_token: usize,
    pub num_shared_experts: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub linear_num_heads: usize,
    pub short_conv_kernel_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub attn_res_block_size: usize,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub value_head_dim: usize,
    pub first_k_dense_replace: usize,
    pub moe_layer_frequency: usize,
    pub num_expert_groups: usize,
    pub topk_groups: usize,
    pub layers: Vec<LayerKind>,
}

impl ModelSpec {
    pub fn load_from_root(root: &Path) -> Result<Self> {
        let direct = root.join("config.json");
        let nested = root.join("k3-meta/config.json");
        let path = if direct.is_file() { direct } else { nested };
        Self::load(&path)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let metadata = path
            .metadata()
            .map_err(|error| io_error("inspect", path, error))?;
        if !metadata.is_file() {
            return Err(DeltafinError::new(format!(
                "model config is not a regular file: {}",
                path.display()
            )));
        }
        if metadata.len() > MAX_CONFIG_BYTES {
            return Err(DeltafinError::new(format!(
                "model config exceeds the {}-byte safety limit: {}",
                MAX_CONFIG_BYTES,
                path.display()
            )));
        }
        let file = File::open(path).map_err(|error| io_error("open", path, error))?;
        let mut reader = BufReader::new(file.take(MAX_CONFIG_BYTES + 1));
        let value: Value = serde_json::from_reader(&mut reader).map_err(|error| {
            DeltafinError::new(format!("parse model config {}: {error}", path.display()))
        })?;
        let root = object(&value, "root")?;
        require_string(root, "model_type", "kimi_k3")?;
        require_string(root, "dtype", "bfloat16")?;
        require_bool(root, "tie_word_embeddings", false)?;
        let text = object(field(root, "text_config")?, "text_config")?;
        let bos_token_id = required_usize(root, "bos_token_id")?;
        let eos_token_id = required_usize(root, "eos_token_id")?;
        let pad_token_id = required_usize(root, "pad_token_id")?;
        require_usize(text, "bos_token_id", bos_token_id)?;
        require_usize(text, "eos_token_id", eos_token_id)?;
        require_usize(text, "pad_token_id", pad_token_id)?;
        require_string(text, "model_type", "kimi_linear")?;
        require_string(text, "dtype", "bfloat16")?;
        require_string(text, "hidden_act", "situ")?;
        require_string(text, "moe_router_activation_func", "sigmoid")?;
        require_string(text, "topk_method", "noaux_tc")?;
        require_bool(text, "moe_renormalize", true)?;
        require_bool(text, "latent_moe_use_norm", true)?;
        require_bool(text, "mla_use_nope", true)?;
        require_bool(text, "mla_use_output_gate", true)?;
        require_bool(text, "tie_word_embeddings", false)?;
        require_bool(text, "use_cache", true)?;
        require_bool(text, "use_grouped_topk", true)?;
        require_number(text, "activation_situ_beta", 4.0)?;
        require_number(text, "activation_situ_linear_beta", 25.0)?;
        require_number(text, "rms_norm_eps", 1.0e-5)?;
        require_number(text, "routed_scaling_factor", 1.0)?;
        require_usize(text, "num_nextn_predict_layers", 0)?;

        let linear = object(field(text, "linear_attn_config")?, "linear_attn_config")?;
        require_bool(linear, "use_full_rank_gate", true)?;
        require_number(linear, "gate_lower_bound", -5.0)?;

        let quantization = object(
            field(text, "quantization_config")?,
            "text_config.quantization_config",
        )?;
        require_string(quantization, "format", "mxfp4-pack-quantized")?;
        require_string(quantization, "quant_method", "compressed-tensors")?;
        require_string(quantization, "quantization_status", "compressed")?;
        let groups = object(
            field(quantization, "config_groups")?,
            "quantization_config.config_groups",
        )?;
        let group = object(field(groups, "group_0")?, "config_groups.group_0")?;
        require_string(group, "format", "mxfp4-pack-quantized")?;
        let weights = object(field(group, "weights")?, "config_groups.group_0.weights")?;
        require_usize(weights, "group_size", 32)?;
        require_usize(weights, "num_bits", 4)?;
        require_bool(weights, "dynamic", false)?;
        require_bool(weights, "symmetric", true)?;
        require_string(weights, "observer", "minmax")?;
        require_string(weights, "scale_dtype", "torch.uint8")?;
        require_string(weights, "strategy", "group")?;
        require_string(weights, "type", "float")?;

        let spec = Self {
            bos_token_id,
            eos_token_id,
            pad_token_id,
            hidden_size: required_usize(text, "hidden_size")?,
            intermediate_size: required_usize(text, "intermediate_size")?,
            moe_intermediate_size: required_usize(text, "moe_intermediate_size")?,
            routed_expert_hidden_size: required_usize(text, "routed_expert_hidden_size")?,
            num_hidden_layers: required_usize(text, "num_hidden_layers")?,
            num_experts: required_usize(text, "num_experts")?,
            num_experts_per_token: required_usize(text, "num_experts_per_token")?,
            num_shared_experts: required_usize(text, "num_shared_experts")?,
            num_attention_heads: required_usize(text, "num_attention_heads")?,
            num_key_value_heads: required_usize(text, "num_key_value_heads")?,
            head_dim: required_usize(linear, "head_dim")?,
            linear_num_heads: required_usize(linear, "num_heads")?,
            short_conv_kernel_size: required_usize(linear, "short_conv_kernel_size")?,
            vocab_size: required_usize(text, "vocab_size")?,
            max_position_embeddings: required_usize(text, "max_position_embeddings")?,
            attn_res_block_size: required_usize(text, "attn_res_block_size")?,
            q_lora_rank: required_usize(text, "q_lora_rank")?,
            kv_lora_rank: required_usize(text, "kv_lora_rank")?,
            qk_nope_head_dim: required_usize(text, "qk_nope_head_dim")?,
            qk_rope_head_dim: required_usize(text, "qk_rope_head_dim")?,
            value_head_dim: required_usize(text, "v_head_dim")?,
            first_k_dense_replace: required_usize(text, "first_k_dense_replace")?,
            moe_layer_frequency: required_usize(text, "moe_layer_freq")?,
            num_expert_groups: required_usize(text, "num_expert_group")?,
            topk_groups: required_usize(text, "topk_group")?,
            layers: layer_kinds(linear, required_usize(text, "num_hidden_layers")?)?,
        };
        spec.validate_exact_k3()?;
        Ok(spec)
    }

    pub fn kda_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|kind| **kind == LayerKind::Kda)
            .count()
    }

    pub fn mla_layers(&self) -> usize {
        self.layers.len() - self.kda_layers()
    }

    pub(crate) fn validate_exact_k3(&self) -> Result<()> {
        let expected = [
            ("bos_token_id", self.bos_token_id, 163_584),
            ("eos_token_id", self.eos_token_id, 163_586),
            ("pad_token_id", self.pad_token_id, 163_839),
            ("hidden_size", self.hidden_size, 7168),
            ("intermediate_size", self.intermediate_size, 33792),
            ("moe_intermediate_size", self.moe_intermediate_size, 3072),
            (
                "routed_expert_hidden_size",
                self.routed_expert_hidden_size,
                3584,
            ),
            ("num_hidden_layers", self.num_hidden_layers, 93),
            ("num_experts", self.num_experts, 896),
            (
                "num_experts_per_token",
                self.num_experts_per_token,
                usize::from(K3_ROUTED_EXPERTS),
            ),
            ("num_shared_experts", self.num_shared_experts, 2),
            ("num_attention_heads", self.num_attention_heads, 96),
            ("num_key_value_heads", self.num_key_value_heads, 96),
            ("head_dim", self.head_dim, 128),
            ("linear_num_heads", self.linear_num_heads, 96),
            ("short_conv_kernel_size", self.short_conv_kernel_size, 4),
            ("vocab_size", self.vocab_size, 163_840),
            (
                "max_position_embeddings",
                self.max_position_embeddings,
                1_048_576,
            ),
            ("attn_res_block_size", self.attn_res_block_size, 12),
            ("q_lora_rank", self.q_lora_rank, 1536),
            ("kv_lora_rank", self.kv_lora_rank, 512),
            ("qk_nope_head_dim", self.qk_nope_head_dim, 128),
            ("qk_rope_head_dim", self.qk_rope_head_dim, 64),
            ("v_head_dim", self.value_head_dim, 128),
            ("first_k_dense_replace", self.first_k_dense_replace, 1),
            ("moe_layer_freq", self.moe_layer_frequency, 1),
            ("num_expert_group", self.num_expert_groups, 1),
            ("topk_group", self.topk_groups, 1),
        ];
        for (name, actual, wanted) in expected {
            if actual != wanted {
                return Err(DeltafinError::new(format!(
                    "target-model contract mismatch: {name}={actual}, expected full K3 value {wanted}"
                )));
            }
        }
        if self.kda_layers() != 69 || self.mla_layers() != 24 {
            return Err(DeltafinError::new(format!(
                "target-model layer schedule mismatch: {} KDA and {} MLA, expected 69 and 24",
                self.kda_layers(),
                self.mla_layers()
            )));
        }
        for (zero_based, &actual) in self.layers.iter().enumerate() {
            let one_based = zero_based + 1;
            let wanted = if K3_MLA_LAYERS_ONE_BASED.contains(&one_based) {
                LayerKind::Mla
            } else {
                LayerKind::Kda
            };
            if actual != wanted {
                return Err(DeltafinError::new(format!(
                    "target-model layer schedule mismatch: one-based layer {one_based} is {actual:?}, expected {wanted:?}"
                )));
            }
        }
        Ok(())
    }
}

fn layer_kinds(linear: &Map<String, Value>, layers: usize) -> Result<Vec<LayerKind>> {
    let mut kinds = vec![None; layers];
    assign_layers(linear, "kda_layers", LayerKind::Kda, &mut kinds)?;
    assign_layers(linear, "full_attn_layers", LayerKind::Mla, &mut kinds)?;
    kinds
        .into_iter()
        .enumerate()
        .map(|(index, kind)| {
            kind.ok_or_else(|| {
                DeltafinError::new(format!(
                    "layer schedule omits one-based layer {}",
                    index + 1
                ))
            })
        })
        .collect()
}

fn assign_layers(
    object: &Map<String, Value>,
    name: &str,
    kind: LayerKind,
    output: &mut [Option<LayerKind>],
) -> Result<()> {
    let values = field(object, name)?
        .as_array()
        .ok_or_else(|| DeltafinError::new(format!("{name} must be an array")))?;
    for value in values {
        let one_based = value
            .as_u64()
            .and_then(|number| usize::try_from(number).ok())
            .filter(|number| *number > 0 && *number <= output.len())
            .ok_or_else(|| {
                DeltafinError::new(format!(
                    "{name} contains an invalid one-based layer index: {value}"
                ))
            })?;
        let slot = &mut output[one_based - 1];
        if slot.replace(kind).is_some() {
            return Err(DeltafinError::new(format!(
                "layer schedule repeats one-based layer {one_based}"
            )));
        }
    }
    Ok(())
}

fn field<'a>(object: &'a Map<String, Value>, name: &str) -> Result<&'a Value> {
    object
        .get(name)
        .ok_or_else(|| DeltafinError::new(format!("model config is missing {name}")))
}

fn object<'a>(value: &'a Value, name: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| DeltafinError::new(format!("{name} must be a JSON object")))
}

fn required_usize(object: &Map<String, Value>, name: &str) -> Result<usize> {
    field(object, name)?
        .as_u64()
        .and_then(|number| usize::try_from(number).ok())
        .ok_or_else(|| DeltafinError::new(format!("{name} must be a non-negative integer")))
}

fn require_string(object: &Map<String, Value>, name: &str, expected: &str) -> Result<()> {
    let actual = field(object, name)?.as_str();
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(DeltafinError::new(format!(
            "target-model contract mismatch: {name}={actual:?}, expected {expected:?}"
        )))
    }
}

fn require_bool(object: &Map<String, Value>, name: &str, expected: bool) -> Result<()> {
    let actual = field(object, name)?.as_bool();
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(DeltafinError::new(format!(
            "target-model contract mismatch: {name}={actual:?}, expected {expected}"
        )))
    }
}

fn require_number(object: &Map<String, Value>, name: &str, expected: f64) -> Result<()> {
    let actual = field(object, name)?.as_f64();
    if actual == Some(expected) {
        Ok(())
    } else {
        Err(DeltafinError::new(format!(
            "target-model contract mismatch: {name}={actual:?}, expected {expected}"
        )))
    }
}

fn require_usize(object: &Map<String, Value>, name: &str, expected: usize) -> Result<()> {
    let actual = required_usize(object, name)?;
    if actual == expected {
        Ok(())
    } else {
        Err(DeltafinError::new(format!(
            "target-model contract mismatch: {name}={actual}, expected {expected}"
        )))
    }
}

fn io_error(operation: &str, path: &Path, error: std::io::Error) -> DeltafinError {
    DeltafinError::new(format!(
        "{operation} model config {}: {error}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_ID: AtomicU64 = AtomicU64::new(1);

    fn repository_config() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../k3-meta/config.json")
    }

    #[test]
    fn validates_the_installed_full_k3_contract_without_python() {
        let spec = ModelSpec::load(&repository_config()).unwrap();
        assert_eq!(spec.kda_layers(), 69);
        assert_eq!(spec.mla_layers(), 24);
        assert_eq!(spec.layers[0], LayerKind::Kda);
        assert_eq!(spec.layers[3], LayerKind::Mla);
        assert_eq!(spec.layers[92], LayerKind::Mla);
    }

    #[test]
    fn rejects_a_quality_reducing_top_k_before_weights_open() {
        let original = fs::read_to_string(repository_config()).unwrap();
        let changed = original.replace(
            "\"num_experts_per_token\": 16",
            "\"num_experts_per_token\": 8",
        );
        assert_ne!(changed, original);
        let path = temporary_path();
        fs::write(&path, changed).unwrap();
        let error = ModelSpec::load(&path).unwrap_err();
        fs::remove_file(path).unwrap();
        assert!(error.to_string().contains("num_experts_per_token=8"));
    }

    #[test]
    fn rejects_duplicate_or_missing_layer_assignments() {
        let original = fs::read_to_string(repository_config()).unwrap();
        let changed = original.replace(
            "\"full_attn_layers\": [\n        4,",
            "\"full_attn_layers\": [\n        1,",
        );
        assert_ne!(changed, original);
        let path = temporary_path();
        fs::write(&path, changed).unwrap();
        let error = ModelSpec::load(&path).unwrap_err();
        fs::remove_file(path).unwrap();
        assert!(error.to_string().contains("repeats one-based layer 1"));
    }

    #[test]
    fn rejects_a_different_layer_permutation_with_the_same_counts() {
        let original = fs::read_to_string(repository_config()).unwrap();
        let mut document: Value = serde_json::from_str(&original).unwrap();
        let linear = document["text_config"]["linear_attn_config"]
            .as_object_mut()
            .unwrap();
        let kda = linear["kda_layers"].as_array_mut().unwrap();
        *kda.iter_mut()
            .find(|value| value.as_u64() == Some(3))
            .unwrap() = Value::from(4);
        let mla = linear["full_attn_layers"].as_array_mut().unwrap();
        *mla.iter_mut()
            .find(|value| value.as_u64() == Some(4))
            .unwrap() = Value::from(3);
        let path = temporary_path();
        fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        let error = ModelSpec::load(&path).unwrap_err();
        fs::remove_file(path).unwrap();
        assert!(
            error
                .to_string()
                .contains("one-based layer 3 is Mla, expected Kda")
        );
    }

    #[test]
    fn rejects_text_token_ids_that_disagree_with_the_root_contract() {
        let original = fs::read_to_string(repository_config()).unwrap();
        let mut document: Value = serde_json::from_str(&original).unwrap();
        document["text_config"]["eos_token_id"] = Value::from(1);
        let path = temporary_path();
        fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        let error = ModelSpec::load(&path).unwrap_err();
        fs::remove_file(path).unwrap();
        let message = error.to_string();
        assert!(message.contains("eos_token_id=1"));
        assert!(message.contains("expected 163586"));
    }

    fn temporary_path() -> PathBuf {
        std::env::temp_dir().join(format!(
            "deltafin-model-spec-{}-{}-{}.json",
            std::process::id(),
            NEXT_ID.fetch_add(1, Ordering::Relaxed),
            std::thread::current().name().unwrap_or("test")
        ))
    }
}
