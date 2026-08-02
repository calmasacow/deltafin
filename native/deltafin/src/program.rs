//! Representation-aware, allocation-once resident-spine roster for Kimi K3.
//!
//! Python currently discovers modules and walks parameter dictionaries while
//! every layer is being streamed.  The native runtime instead compiles the
//! validated [`ModelSpec`] into a fixed set of typed weight slots once.  The
//! same roster drives pack construction, provider binding, and parity tests;
//! those three stages therefore cannot silently disagree about a tensor name,
//! shape, storage encoding, or layer kind.
//!
//! Routed MXFP4 expert payloads are demand-loaded rather than resident and use
//! a separate schema; they are intentionally not counted by the spine totals
//! in this module.

use std::path::{Path, PathBuf};
use std::{fmt, fmt::Display};

use crate::error::{DeltafinError, Result};
use crate::model::{LayerKind, ModelSpec};
use crate::packfile::{
    BuildTensor, Codec, ComponentSource, DType, PackBuilder, PackFile, PackIdentity, digest_bytes,
    digest_file,
};
use crate::storage::{BufferKind, BufferLengths, CachePolicy, Extent, ReadPlan};

pub const K3_LAYER_COUNT: usize = 93;
pub const K3_LOGICAL_SPINE_TENSORS: usize = 2_455;
/// Tensor/component totals for the optional, non-weight-exact row-int8 spine.
pub const K3_Q8_SPINE_TENSORS: usize = 1_251;
pub const K3_RAW_SPINE_TENSORS: usize = 1_204;
pub const K3_SPINE_SOURCE_COMPONENTS: usize = 3_706;
/// The canonical checkpoint stores every resident-spine tensor directly.
pub const K3_BF16_RAW_SPINE_TENSORS: usize = K3_LOGICAL_SPINE_TENSORS;
pub const K3_BF16_SPINE_SOURCE_COMPONENTS: usize = K3_LOGICAL_SPINE_TENSORS;
/// Provider-resident tail/head tensors. The exact BF16 embedding table is a
/// separate row-addressed source and is deliberately not counted here.
pub const K3_GLOBAL_LOGICAL_TENSORS: usize = 4;
pub const K3_GLOBAL_SOURCE_COMPONENTS: usize = 5;
pub const K3_BF16_GLOBAL_SOURCE_COMPONENTS: usize = K3_GLOBAL_LOGICAL_TENSORS;
pub const K3_GLOBAL_TRANSFER_GROUPS: usize = 2;
pub const SPINE_COMPONENT_ALIGNMENT: usize = 256;
pub const DEFAULT_SPINE_CHUNK_BYTES: usize = 16 * 1024 * 1024;

pub const SPINE_ENCODING_RAW_BF16: u32 = 1;
pub const SPINE_ENCODING_RAW_F32: u32 = 2;
pub const SPINE_ENCODING_ROW_I8_F16_SCALE: u32 = 3;
pub const SPINE_BUFFER_NONE: u32 = 0;
pub const SPINE_BUFFER_QUANTIZED: u32 = 1;
pub const SPINE_BUFFER_SCALES: u32 = 2;
pub const SPINE_BUFFER_OTHER: u32 = 3;

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum SpineRepresentation {
    OriginalBf16,
    QuantizedInt8,
}

impl SpineRepresentation {
    pub const fn is_weight_exact(self) -> bool {
        matches!(self, Self::OriginalBf16)
    }

    pub const fn pack_directory_name(self) -> &'static str {
        match self {
            Self::OriginalBf16 => "k3-resident-packs-bf16",
            // Preserve the established optional-int8 pack location.
            Self::QuantizedInt8 => "k3-resident-packs",
        }
    }
}

impl Display for SpineRepresentation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OriginalBf16 => formatter.write_str("original-bf16"),
            Self::QuantizedInt8 => formatter.write_str("quantized-int8 (non-weight-exact)"),
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WeightDType {
    Bf16,
    F32,
}

impl WeightDType {
    pub const fn byte_width(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WeightStorage {
    Raw(WeightDType),
    RowI8F16Scale { logical: WeightDType },
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub enum WeightSlot {
    InputLayerNorm = 1,
    PostAttentionLayerNorm = 2,
    SelfAttentionResidualNorm = 3,
    SelfAttentionResidualProjection = 4,
    MlpResidualNorm = 5,
    MlpResidualProjection = 6,
    KdaALog = 7,
    KdaDtBias = 8,
    KdaQueryConvolution = 9,
    KdaKeyConvolution = 10,
    KdaValueConvolution = 11,
    KdaOutputNorm = 12,
    KdaQueryProjection = 13,
    KdaKeyProjection = 14,
    KdaValueProjection = 15,
    KdaGateProjection = 16,
    KdaFeatureAProjection = 17,
    KdaFeatureBProjection = 18,
    KdaBetaProjection = 19,
    KdaOutputProjection = 20,
    MlaQueryAProjection = 21,
    MlaQueryANorm = 22,
    MlaQueryBProjection = 23,
    MlaKeyValueAProjection = 24,
    MlaKeyValueANorm = 25,
    MlaKeyValueBProjection = 26,
    MlaGateProjection = 27,
    MlaOutputProjection = 28,
    DenseGateProjection = 29,
    DenseUpProjection = 30,
    DenseDownProjection = 31,
    MoeGateWeight = 32,
    MoeGateCorrectionBias = 33,
    MoeRoutedDownProjection = 34,
    MoeRoutedNorm = 35,
    MoeRoutedUpProjection = 36,
    MoeSharedGateProjection = 37,
    MoeSharedUpProjection = 38,
    MoeSharedDownProjection = 39,
    TokenEmbedding = 40,
    FinalNorm = 41,
    OutputAttentionResidualNorm = 42,
    OutputAttentionResidualProjection = 43,
    LanguageModelHead = 44,
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightSpec {
    pub slot: WeightSlot,
    pub name: String,
    pub shape: Box<[u64]>,
    pub storage: WeightStorage,
    pub upload_group: u16,
    pub upload_order: u16,
}

impl WeightSpec {
    /// Large rank-two projections have a dedicated exact-BF16 provider path.
    /// Vectors, norms, and the `[1, H]` residual-score projections continue to
    /// use the established lossless fp32 operator path. Keeping this predicate
    /// beside the authenticated roster prevents RAM admission from silently
    /// treating every BF16 value as four bytes again.
    pub fn is_large_projection_matrix(&self) -> bool {
        self.shape.len() == 2 && self.shape[0] > 1
    }

    pub fn element_count(&self) -> Result<u64> {
        self.shape.iter().try_fold(1_u64, |product, &dimension| {
            product.checked_mul(dimension).ok_or_else(|| {
                DeltafinError::new(format!("tensor shape overflows u64: {}", self.name))
            })
        })
    }

    pub fn expected_data_bytes(&self) -> Result<u64> {
        let elements = self.element_count()?;
        match self.storage {
            WeightStorage::Raw(dtype) => {
                elements.checked_mul(dtype.byte_width()).ok_or_else(|| {
                    DeltafinError::new(format!("tensor byte length overflows u64: {}", self.name))
                })
            }
            WeightStorage::RowI8F16Scale { .. } => Ok(elements),
        }
    }

    pub fn expected_scale_bytes(&self) -> Result<Option<u64>> {
        match self.storage {
            WeightStorage::Raw(_) => Ok(None),
            WeightStorage::RowI8F16Scale { .. } => {
                if self.shape.len() != 2 {
                    return Err(DeltafinError::new(format!(
                        "row-int8 tensor is not a matrix: {}",
                        self.name
                    )));
                }
                Ok(Some(self.shape[0].checked_mul(2).ok_or_else(|| {
                    DeltafinError::new(format!("scale byte length overflows u64: {}", self.name))
                })?))
            }
        }
    }

    /// Bytes retained by the native provider after this weight is bound.
    ///
    /// The on-disk and provider representations are deliberately accounted
    /// separately. Large raw-BF16 projection matrices remain exact BF16 in
    /// provider-owned storage. Small vectors/norms and `[1, H]` residual
    /// projections retain the established lossless fp32 operator storage.
    /// Row-int8 payloads remain int8 while their fp16 row scales are promoted
    /// to fp32. Bind-time bundles are views over these allocations, so they do
    /// not add a second copy here.
    pub fn provider_resident_bytes(&self) -> Result<u64> {
        match self.storage {
            WeightStorage::Raw(WeightDType::Bf16) if self.is_large_projection_matrix() => {
                self.expected_data_bytes()
            }
            WeightStorage::Raw(WeightDType::Bf16) => {
                self.element_count()?.checked_mul(4).ok_or_else(|| {
                    DeltafinError::new(format!(
                        "provider-resident byte length overflows u64: {}",
                        self.name
                    ))
                })
            }
            WeightStorage::Raw(WeightDType::F32) => self.expected_data_bytes(),
            WeightStorage::RowI8F16Scale { .. } => {
                let data = self.expected_data_bytes()?;
                let scale_file_bytes = self.expected_scale_bytes()?.ok_or_else(|| {
                    DeltafinError::new(format!(
                        "row-int8 provider accounting lacks scales: {}",
                        self.name
                    ))
                })?;
                let promoted_scales = scale_file_bytes.checked_mul(2).ok_or_else(|| {
                    DeltafinError::new(format!(
                        "provider scale byte length overflows u64: {}",
                        self.name
                    ))
                })?;
                data.checked_add(promoted_scales).ok_or_else(|| {
                    DeltafinError::new(format!(
                        "provider-resident byte length overflows u64: {}",
                        self.name
                    ))
                })
            }
        }
    }

    pub fn source_paths(&self, layout: &SourceLayout) -> WeightSourcePaths {
        match self.storage {
            WeightStorage::Raw(_) => WeightSourcePaths {
                data: layout.resident_tensors.join(&self.name),
                scales: None,
            },
            WeightStorage::RowI8F16Scale { .. } => WeightSourcePaths {
                data: layout.int8_tensors.join(format!("{}.i8", self.name)),
                scales: Some(layout.int8_tensors.join(format!("{}.sc", self.name))),
            },
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct WeightSourcePaths {
    pub data: PathBuf,
    pub scales: Option<PathBuf>,
}

/// Fixed-layout descriptor consumed by the linked C++/ATen provider. Offsets
/// address one of the three leased [`LayerBuffers`](crate::storage::LayerBuffers)
/// slabs and never contain a process pointer.
#[repr(C)]
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct SpineTensorDescriptorV1 {
    pub slot: u32,
    pub encoding: u32,
    pub rank: u32,
    pub data_buffer: u32,
    pub auxiliary_buffer: u32,
    pub reserved0: u32,
    pub shape: [u64; 8],
    pub data_offset: u64,
    pub data_length: u64,
    pub auxiliary_offset: u64,
    pub auxiliary_length: u64,
    pub reserved: [u64; 4],
}

#[derive(Debug)]
pub struct LayerSpinePlan {
    pub(crate) layer: u32,
    pub(crate) kind: LayerKind,
    pub(crate) descriptors: Box<[SpineTensorDescriptorV1]>,
    pub(crate) buffer_lengths: BufferLengths,
    pub(crate) read_plan: ReadPlan,
}

/// One bounded startup transfer for provider-owned global weights.
///
/// The small final-normalization tail and the vocabulary head use separate
/// groups. This lets the same aligned reader arena be reused instead of
/// transiently holding the head alongside its provider-owned copy. Embedding
/// rows are read from the original BF16 table through the dedicated row path.
#[derive(Debug)]
pub struct GlobalSpinePlan {
    group: u16,
    descriptors: Box<[SpineTensorDescriptorV1]>,
    buffer_lengths: BufferLengths,
    read_plan: ReadPlan,
}

impl GlobalSpinePlan {
    pub const fn group(&self) -> u16 {
        self.group
    }

    pub fn descriptors(&self) -> &[SpineTensorDescriptorV1] {
        &self.descriptors
    }

    pub const fn buffer_lengths(&self) -> BufferLengths {
        self.buffer_lengths
    }

    pub const fn read_plan(&self) -> &ReadPlan {
        &self.read_plan
    }
}

impl LayerSpinePlan {
    pub const fn layer(&self) -> u32 {
        self.layer
    }

    pub const fn kind(&self) -> LayerKind {
        self.kind
    }

    pub fn descriptors(&self) -> &[SpineTensorDescriptorV1] {
        &self.descriptors
    }

    pub const fn buffer_lengths(&self) -> BufferLengths {
        self.buffer_lengths
    }

    pub const fn read_plan(&self) -> &ReadPlan {
        &self.read_plan
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct SourceLayout {
    pub resident_tensors: PathBuf,
    pub int8_tensors: PathBuf,
}

impl SourceLayout {
    pub fn under(root: &Path) -> Self {
        Self {
            resident_tensors: root.join("k3-resident/tensors"),
            int8_tensors: root.join("k3-resident-int8/tensors"),
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct LayerProgram {
    pub index: u32,
    pub kind: LayerKind,
    pub weights: Box<[WeightSpec]>,
}

impl LayerProgram {
    pub fn quantized_tensors(&self) -> usize {
        self.weights
            .iter()
            .filter(|weight| matches!(weight.storage, WeightStorage::RowI8F16Scale { .. }))
            .count()
    }

    pub fn raw_tensors(&self) -> usize {
        self.weights.len() - self.quantized_tensors()
    }

    pub fn source_components(&self) -> usize {
        self.raw_tensors() + 2 * self.quantized_tensors()
    }

    pub fn provider_resident_bytes(&self) -> Result<u64> {
        self.weights.iter().try_fold(0_u64, |total, weight| {
            total
                .checked_add(weight.provider_resident_bytes()?)
                .ok_or_else(|| {
                    DeltafinError::new(format!(
                        "provider-resident layer {} byte total overflows u64",
                        self.index
                    ))
                })
        })
    }

    /// Exact row-int8 data/scale bytes used by the established automatic
    /// stream-tier policy. Raw BF16 side tensors are deliberately excluded to
    /// match `spine_cache.layer_bytes`, which sizes this policy from the int8
    /// tensor tree before applying one cache decision to the complete layer.
    pub fn int8_stream_source_bytes(&self) -> Result<u64> {
        self.weights
            .iter()
            .filter(|weight| matches!(weight.storage, WeightStorage::RowI8F16Scale { .. }))
            .try_fold(0_u64, |total, weight| {
                let data = weight.expected_data_bytes()?;
                let auxiliary = weight.expected_scale_bytes()?.unwrap_or(0);
                total
                    .checked_add(data)
                    .and_then(|bytes| bytes.checked_add(auxiliary))
                    .ok_or_else(|| {
                        DeltafinError::new(format!(
                            "source layer {} byte total overflows u64",
                            self.index
                        ))
                    })
            })
    }

    /// FP32 bytes needed to expose every large projection in this layer as
    /// aligned views of one serial execution arena.  The source representation
    /// is deliberately irrelevant: original BF16 and optional row-int8 both
    /// describe the same logical matrices and may target the same arena.
    fn fp32_projection_arena_bytes(&self) -> Result<u64> {
        self.weights
            .iter()
            .filter(|weight| weight.is_large_projection_matrix())
            .try_fold(0_u64, |offset, weight| {
                let aligned = offset
                    .checked_add(SPINE_COMPONENT_ALIGNMENT as u64 - 1)
                    .map(|value| {
                        value / SPINE_COMPONENT_ALIGNMENT as u64 * SPINE_COMPONENT_ALIGNMENT as u64
                    })
                    .ok_or_else(|| {
                        DeltafinError::new(format!(
                            "layer {} FP32 execution-arena alignment overflows u64",
                            self.index
                        ))
                    })?;
                let bytes = weight
                    .element_count()?
                    .checked_mul(WeightDType::F32.byte_width())
                    .ok_or_else(|| {
                        DeltafinError::new(format!(
                            "FP32 execution-arena tensor size overflows u64: {}",
                            weight.name
                        ))
                    })?;
                aligned.checked_add(bytes).ok_or_else(|| {
                    DeltafinError::new(format!(
                        "layer {} FP32 execution-arena size overflows u64",
                        self.index
                    ))
                })
            })
    }

    pub fn pack_builder(
        &self,
        layout: &SourceLayout,
        identity: PackIdentity,
    ) -> Result<PackBuilder> {
        let mut builder = PackBuilder::new(self.index, identity)
            .map_err(|error| DeltafinError::new(format!("create layer pack: {error}")))?;
        for weight in &self.weights {
            let paths = weight.source_paths(layout);
            let data_bytes = weight.expected_data_bytes()?;
            require_source_length(&paths.data, data_bytes)?;
            let tensor = match weight.storage {
                WeightStorage::Raw(dtype) => BuildTensor::raw(
                    weight.name.clone(),
                    pack_dtype(dtype),
                    weight.shape.to_vec(),
                    ComponentSource::file(paths.data, 0, data_bytes),
                    weight.upload_group,
                    weight.upload_order,
                ),
                WeightStorage::RowI8F16Scale { logical } => {
                    let scale_path = paths.scales.ok_or_else(|| {
                        DeltafinError::new(format!("missing scale path for {}", weight.name))
                    })?;
                    let scale_bytes = weight.expected_scale_bytes()?.ok_or_else(|| {
                        DeltafinError::new(format!("missing scale length for {}", weight.name))
                    })?;
                    require_source_length(&scale_path, scale_bytes)?;
                    let shape: [u64; 2] = weight.shape.as_ref().try_into().map_err(|_| {
                        DeltafinError::new(format!(
                            "row-int8 tensor is not rank two: {}",
                            weight.name
                        ))
                    })?;
                    BuildTensor::row_i8_f16_scale(
                        weight.name.clone(),
                        pack_dtype(logical),
                        shape,
                        ComponentSource::file(paths.data, 0, data_bytes),
                        ComponentSource::file(scale_path, 0, scale_bytes),
                        weight.upload_group,
                        weight.upload_order,
                    )
                }
            };
            builder.push(tensor);
        }
        Ok(builder)
    }

    /// Build one immutable direct-read plan for the current loose-file layout.
    /// The cache policy is explicit because automatic mixed admission is a
    /// per-layer decision made before this immutable plan is published.
    pub fn loose_read_plan_with_cache_policy(
        &self,
        layout: &SourceLayout,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
    ) -> Result<LayerSpinePlan> {
        self.loose_read_plan_with_cache_policy_and_descriptors(
            layout,
            chunk_bytes,
            cache_policy,
            false,
        )
    }

    pub(crate) fn loose_read_plan_with_cache_policy_and_descriptors(
        &self,
        layout: &SourceLayout,
        chunk_bytes: usize,
        cache_policy: CachePolicy,
        persistent_descriptors: bool,
    ) -> Result<LayerSpinePlan> {
        let LooseWeightReadPlan {
            descriptors,
            buffer_lengths,
            read_plan,
        } = loose_weight_read_plan(
            &self.weights,
            layout,
            chunk_bytes,
            cache_policy,
            persistent_descriptors,
        )?;
        Ok(LayerSpinePlan {
            layer: self.index,
            kind: self.kind,
            descriptors,
            buffer_lengths,
            read_plan,
        })
    }

    #[cfg(test)]
    pub fn loose_read_plan_default(&self, layout: &SourceLayout) -> Result<LayerSpinePlan> {
        self.loose_read_plan_with_cache_policy(
            layout,
            DEFAULT_SPINE_CHUNK_BYTES,
            CachePolicy::Streaming,
        )
    }

    /// Open one authenticated DFSP file as the complete layer-spine source.
    ///
    /// Unlike the loose compatibility layout, this keeps exactly one source
    /// descriptor for the layer. Pack chunks are copied into one aligned
    /// payload slab and SHA-256-qualified on their first real read; subsequent
    /// passes through this immutable `ReadPlan` do no checksum work. Provider
    /// descriptors point directly into that slab, so no component restaging
    /// or per-tensor host allocation is introduced.
    pub fn packed_read_plan_with_cache_policy(
        &self,
        path: &Path,
        expected_identity: PackIdentity,
        cache_policy: CachePolicy,
    ) -> Result<LayerSpinePlan> {
        let pack = PackFile::open_for(path, self.index, expected_identity).map_err(|error| {
            DeltafinError::new(format!(
                "open authenticated layer pack {}: {error}",
                path.display()
            ))
        })?;
        self.packed_read_plan_from(&pack, cache_policy)
    }

    fn packed_read_plan_from(
        &self,
        pack: &PackFile,
        cache_policy: CachePolicy,
    ) -> Result<LayerSpinePlan> {
        if pack.header().layer != self.index {
            return Err(DeltafinError::new(format!(
                "layer pack {} belongs to layer {}, expected {}",
                pack.path().display(),
                pack.header().layer,
                self.index,
            )));
        }
        if pack.tensors().len() != self.weights.len() {
            return Err(DeltafinError::new(format!(
                "layer {} pack has {} tensors, expected {}",
                self.index,
                pack.tensors().len(),
                self.weights.len(),
            )));
        }

        let payload_bytes = usize::try_from(pack.header().payload_bytes).map_err(|_| {
            DeltafinError::new(format!(
                "layer {} pack payload is too large for this host",
                self.index
            ))
        })?;
        let mut descriptors = Vec::with_capacity(self.weights.len());
        for weight in &self.weights {
            let record_index = pack
                .tensors()
                .binary_search_by(|record| record.name.as_bytes().cmp(weight.name.as_bytes()))
                .map_err(|_| {
                    DeltafinError::new(format!(
                        "layer {} pack is missing required tensor {}",
                        self.index, weight.name
                    ))
                })?;
            let record = &pack.tensors()[record_index];
            validate_packed_weight(weight, record)?;
            let (encoding, auxiliary_buffer) = match weight.storage {
                WeightStorage::Raw(WeightDType::Bf16) => {
                    (SPINE_ENCODING_RAW_BF16, SPINE_BUFFER_NONE)
                }
                WeightStorage::Raw(WeightDType::F32) => (SPINE_ENCODING_RAW_F32, SPINE_BUFFER_NONE),
                WeightStorage::RowI8F16Scale { .. } => {
                    (SPINE_ENCODING_ROW_I8_F16_SCALE, SPINE_BUFFER_OTHER)
                }
            };
            descriptors.push(SpineTensorDescriptorV1 {
                slot: weight.slot as u32,
                encoding,
                rank: u32::from(record.rank),
                data_buffer: SPINE_BUFFER_OTHER,
                auxiliary_buffer,
                reserved0: 0,
                shape: record.shape,
                data_offset: record.data_offset,
                data_length: record.data_length,
                auxiliary_offset: record.auxiliary_offset,
                auxiliary_length: record.auxiliary_length,
                reserved: [0; 4],
            });
        }

        let extents = pack
            .read_extents()
            .into_iter()
            .map(|extent| {
                Ok(Extent::verified(
                    pack.path(),
                    extent.file_offset,
                    BufferKind::Other,
                    usize::try_from(extent.destination_offset).map_err(|_| {
                        DeltafinError::new("pack destination offset does not fit in usize")
                    })?,
                    extent.length as usize,
                    extent.expected_digest,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let buffer_lengths = BufferLengths::new(0, 0, payload_bytes);
        let read_plan = ReadPlan::open(extents, buffer_lengths, 0, cache_policy)?;
        if read_plan.source_count() != 1 {
            return Err(DeltafinError::new(
                "authenticated layer pack did not compile to exactly one source descriptor",
            ));
        }
        Ok(LayerSpinePlan {
            layer: self.index,
            kind: self.kind,
            descriptors: descriptors.into_boxed_slice(),
            buffer_lengths,
            read_plan,
        })
    }
}

struct LooseWeightReadPlan {
    descriptors: Box<[SpineTensorDescriptorV1]>,
    buffer_lengths: BufferLengths,
    read_plan: ReadPlan,
}

/// Compile loose immutable tensor files into the one descriptor/slab contract
/// shared by layer and global startup transfers.  Keeping this in one routine
/// is more than source cleanup: it guarantees both paths use identical
/// alignment, exact-size admission, int8-scale association, and read batching.
fn loose_weight_read_plan(
    weights: &[WeightSpec],
    layout: &SourceLayout,
    chunk_bytes: usize,
    cache_policy: CachePolicy,
    persistent_descriptors: bool,
) -> Result<LooseWeightReadPlan> {
    let source_components = weights.iter().try_fold(0_usize, |total, weight| {
        total
            .checked_add(
                if matches!(weight.storage, WeightStorage::RowI8F16Scale { .. }) {
                    2
                } else {
                    1
                },
            )
            .ok_or_else(|| DeltafinError::new("spine source-component count overflows usize"))
    })?;
    let mut extents = Vec::with_capacity(source_components.saturating_mul(2));
    let mut descriptors = Vec::with_capacity(weights.len());
    let mut cursors = BufferLengths::default();
    for weight in weights {
        let paths = weight.source_paths(layout);
        let data_bytes_u64 = weight.expected_data_bytes()?;
        let data_bytes = usize::try_from(data_bytes_u64).map_err(|_| {
            DeltafinError::new(format!(
                "tensor is too large for this host: {}",
                weight.name
            ))
        })?;
        let mut shape = [0_u64; 8];
        shape[..weight.shape.len()].copy_from_slice(&weight.shape);

        let (
            encoding,
            data_buffer,
            data_offset,
            auxiliary_buffer,
            auxiliary_offset,
            auxiliary_length,
        ) = match weight.storage {
            WeightStorage::Raw(dtype) => {
                let aligned = aligned_cursor(cursors.other)?;
                push_padding(&mut extents, BufferKind::Other, cursors.other, aligned);
                cursors.other = aligned
                    .checked_add(data_bytes)
                    .ok_or_else(|| DeltafinError::new("raw spine buffer length overflows usize"))?;
                extents.push(Extent::new(
                    paths.data,
                    0,
                    BufferKind::Other,
                    aligned,
                    data_bytes,
                ));
                (
                    match dtype {
                        WeightDType::Bf16 => SPINE_ENCODING_RAW_BF16,
                        WeightDType::F32 => SPINE_ENCODING_RAW_F32,
                    },
                    SPINE_BUFFER_OTHER,
                    aligned,
                    SPINE_BUFFER_NONE,
                    0,
                    0,
                )
            }
            WeightStorage::RowI8F16Scale { .. } => {
                let data_aligned = aligned_cursor(cursors.quantized)?;
                push_padding(
                    &mut extents,
                    BufferKind::Quantized,
                    cursors.quantized,
                    data_aligned,
                );
                cursors.quantized = data_aligned.checked_add(data_bytes).ok_or_else(|| {
                    DeltafinError::new("quantized spine buffer length overflows usize")
                })?;
                extents.push(Extent::new(
                    paths.data,
                    0,
                    BufferKind::Quantized,
                    data_aligned,
                    data_bytes,
                ));

                let scale_path = paths.scales.ok_or_else(|| {
                    DeltafinError::new(format!("missing scale path for {}", weight.name))
                })?;
                let scale_bytes_u64 = weight.expected_scale_bytes()?.ok_or_else(|| {
                    DeltafinError::new(format!("missing scale length for {}", weight.name))
                })?;
                let scale_bytes = usize::try_from(scale_bytes_u64).map_err(|_| {
                    DeltafinError::new(format!(
                        "scale tensor is too large for this host: {}",
                        weight.name
                    ))
                })?;
                let scale_aligned = aligned_cursor(cursors.scales)?;
                push_padding(
                    &mut extents,
                    BufferKind::Scales,
                    cursors.scales,
                    scale_aligned,
                );
                cursors.scales = scale_aligned.checked_add(scale_bytes).ok_or_else(|| {
                    DeltafinError::new("scale spine buffer length overflows usize")
                })?;
                extents.push(Extent::new(
                    scale_path,
                    0,
                    BufferKind::Scales,
                    scale_aligned,
                    scale_bytes,
                ));
                (
                    SPINE_ENCODING_ROW_I8_F16_SCALE,
                    SPINE_BUFFER_QUANTIZED,
                    data_aligned,
                    SPINE_BUFFER_SCALES,
                    scale_aligned,
                    scale_bytes_u64,
                )
            }
        };
        descriptors.push(SpineTensorDescriptorV1 {
            slot: weight.slot as u32,
            encoding,
            rank: weight.shape.len() as u32,
            data_buffer,
            auxiliary_buffer,
            reserved0: 0,
            shape,
            data_offset: data_offset as u64,
            data_length: data_bytes_u64,
            auxiliary_offset: auxiliary_offset as u64,
            auxiliary_length,
            reserved: [0; 4],
        });
    }
    let read_plan = if persistent_descriptors {
        ReadPlan::open_persistent_deferred_manifest(extents, cursors, chunk_bytes, cache_policy)?
    } else {
        ReadPlan::open_deferred_manifest(extents, cursors, chunk_bytes, cache_policy)?
    };
    Ok(LooseWeightReadPlan {
        descriptors: descriptors.into_boxed_slice(),
        buffer_lengths: cursors,
        read_plan,
    })
}

fn validate_packed_weight(
    weight: &WeightSpec,
    record: &crate::packfile::TensorRecord,
) -> Result<()> {
    let rank = weight.shape.len();
    let shape_matches = record.rank as usize == rank
        && record.shape[..rank] == weight.shape[..]
        && record.shape[rank..].iter().all(|dimension| *dimension == 0);
    let (codec, logical_dtype, data_dtype, auxiliary_dtype) = match weight.storage {
        WeightStorage::Raw(dtype) => {
            let dtype = pack_dtype(dtype);
            (Codec::Raw, dtype, dtype, DType::None)
        }
        WeightStorage::RowI8F16Scale { logical } => (
            Codec::RowI8F16Scale,
            pack_dtype(logical),
            DType::I8,
            DType::F16,
        ),
    };
    if !shape_matches
        || record.codec != codec
        || record.logical_dtype != logical_dtype
        || record.data_dtype != data_dtype
        || record.auxiliary_dtype != auxiliary_dtype
        || record.data_length != weight.expected_data_bytes()?
        || record.auxiliary_length != weight.expected_scale_bytes()?.unwrap_or(0)
        || record.upload_group != weight.upload_group
        || record.upload_order != weight.upload_order
    {
        return Err(DeltafinError::new(format!(
            "authenticated pack metadata differs from the compiled target roster for {}",
            weight.name
        )));
    }
    Ok(())
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct TargetProgram {
    /// Original checkpoint BF16 table, consumed one exact row at a time. It is
    /// never materialized or quantized as a provider-resident matrix.
    pub embedding_weight: WeightSpec,
    pub global_weights: Box<[WeightSpec]>,
    pub layers: Box<[LayerProgram]>,
    representation: SpineRepresentation,
}

/// All 93 authenticated resident-spine plans admitted transactionally.
/// Holding this catalog keeps one descriptor per layer pack open, replacing
/// the thousands of loose component descriptors and open/close operations in
/// the legacy hot path.
#[derive(Debug)]
pub struct PackedSpineCatalog {
    layers: Box<[LayerSpinePlan]>,
}

impl PackedSpineCatalog {
    pub fn layers(&self) -> &[LayerSpinePlan] {
        &self.layers
    }
}

impl TargetProgram {
    #[cfg(test)]
    pub fn compile(model: &ModelSpec) -> Result<Self> {
        Self::compile_with_representation(model, SpineRepresentation::OriginalBf16)
    }

    pub fn compile_with_representation(
        model: &ModelSpec,
        representation: SpineRepresentation,
    ) -> Result<Self> {
        // Do not rely on every caller having obtained the spec through
        // `ModelSpec::load`; this type is intentionally inspectable and can be
        // assembled or changed by other native modules during migration.
        model.validate_exact_k3()?;
        if model.layers.len() != K3_LAYER_COUNT {
            return Err(DeltafinError::new(format!(
                "native target program requires {K3_LAYER_COUNT} layers, got {}",
                model.layers.len()
            )));
        }
        let mut layers = Vec::with_capacity(model.layers.len());
        for (index, &kind) in model.layers.iter().enumerate() {
            layers.push(compile_layer(index, kind, representation)?);
        }
        let program = Self {
            embedding_weight: compile_embedding_weight(),
            global_weights: compile_global_weights(representation),
            layers: layers.into_boxed_slice(),
            representation,
        };
        program.validate_totals()?;
        Ok(program)
    }

    pub const fn representation(&self) -> SpineRepresentation {
        self.representation
    }

    pub fn logical_tensors(&self) -> usize {
        self.layers.iter().map(|layer| layer.weights.len()).sum()
    }

    pub fn quantized_tensors(&self) -> usize {
        self.layers
            .iter()
            .map(LayerProgram::quantized_tensors)
            .sum()
    }

    pub fn raw_tensors(&self) -> usize {
        self.layers.iter().map(LayerProgram::raw_tensors).sum()
    }

    pub fn source_components(&self) -> usize {
        self.layers
            .iter()
            .map(LayerProgram::source_components)
            .sum()
    }

    /// Exact provider allocation cost of each layer, in execution order.
    /// This is computed from the validated model roster without opening the
    /// 54 GiB payload, so the RAM policy can make its decision before I/O.
    pub fn provider_layer_bytes(&self) -> Result<Box<[u64]>> {
        self.layers
            .iter()
            .map(LayerProgram::provider_resident_bytes)
            .collect::<Result<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    }

    pub fn int8_stream_layer_bytes(&self) -> Result<Box<[u64]>> {
        self.layers
            .iter()
            .map(LayerProgram::int8_stream_source_bytes)
            .collect::<Result<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    }

    /// Maximum matrix-only FP32 arena needed by any one streamed layer.
    /// Layers execute serially, so charging a sum here would incorrectly make
    /// the residency planner reserve 93 dense copies.  Keeping the calculation
    /// in the authenticated roster also prevents the native provider and Rust
    /// memory policy from drifting onto different K3 shapes.
    pub fn fp32_spine_execution_arena_bytes(&self) -> Result<u64> {
        self.layers
            .iter()
            .map(LayerProgram::fp32_projection_arena_bytes)
            .try_fold(0_u64, |maximum, bytes| {
                bytes.map(|value| maximum.max(value))
            })
    }

    pub fn provider_global_bytes(&self) -> Result<u64> {
        self.global_weights.iter().try_fold(0_u64, |total, weight| {
            total
                .checked_add(weight.provider_resident_bytes()?)
                .ok_or_else(|| {
                    DeltafinError::new("provider-resident global byte total overflows u64")
                })
        })
    }

    /// Compile the provider-resident globals into two bounded startup reads:
    /// the small final tail and the vocabulary head. The exact embedding table
    /// is intentionally absent; [`Self::embedding_weight`] feeds the bounded
    /// BF16 row reader instead.
    pub fn global_loose_read_plans(
        &self,
        layout: &SourceLayout,
        chunk_bytes: usize,
    ) -> Result<Box<[GlobalSpinePlan]>> {
        let mut plans = Vec::with_capacity(K3_GLOBAL_TRANSFER_GROUPS);
        let mut start = 0_usize;
        for group in [1_u16, 2_u16] {
            let end = self.global_weights[start..]
                .iter()
                .position(|weight| weight.upload_group != group)
                .map_or(self.global_weights.len(), |offset| start + offset);
            if start == end
                || self.global_weights[start..end]
                    .iter()
                    .any(|weight| weight.upload_group != group)
            {
                return Err(DeltafinError::new(format!(
                    "native global transfer group {group} is empty or non-contiguous"
                )));
            }
            let LooseWeightReadPlan {
                descriptors,
                buffer_lengths,
                read_plan,
            } = loose_weight_read_plan(
                &self.global_weights[start..end],
                layout,
                chunk_bytes,
                CachePolicy::Streaming,
                false,
            )?;
            plans.push(GlobalSpinePlan {
                group,
                descriptors,
                buffer_lengths,
                read_plan,
            });
            start = end;
        }
        if start != self.global_weights.len() {
            return Err(DeltafinError::new(
                "native global roster contains an unknown transfer group",
            ));
        }
        Ok(plans.into_boxed_slice())
    }

    pub fn global_loose_read_plans_default(
        &self,
        layout: &SourceLayout,
    ) -> Result<Box<[GlobalSpinePlan]>> {
        self.global_loose_read_plans(layout, DEFAULT_SPINE_CHUNK_BYTES)
    }

    /// Compile the complete loose-file spine into immutable deferred plans.
    ///
    /// This does not open any tensor component. Consequently all 93 plans can
    /// remain resident as a small native manifest even under Darwin's default
    /// descriptor limit; workers validate/open/read/close each exact source on
    /// demand. The returned order is the model's authoritative layer order.
    pub fn loose_spine_read_plans_with_cache_policies(
        &self,
        layout: &SourceLayout,
        chunk_bytes: usize,
        cache_policies: &[CachePolicy],
    ) -> Result<Box<[LayerSpinePlan]>> {
        self.loose_spine_read_plans_with_cache_policies_and_descriptors(
            layout,
            chunk_bytes,
            cache_policies,
            false,
        )
    }

    pub(crate) fn loose_spine_read_plans_with_cache_policies_and_descriptors(
        &self,
        layout: &SourceLayout,
        chunk_bytes: usize,
        cache_policies: &[CachePolicy],
        persistent_descriptors: bool,
    ) -> Result<Box<[LayerSpinePlan]>> {
        if cache_policies.len() != self.layers.len() {
            return Err(DeltafinError::new(format!(
                "spine cache-policy roster has {} layers, expected {}",
                cache_policies.len(),
                self.layers.len(),
            )));
        }
        self.layers
            .iter()
            .zip(cache_policies.iter().copied())
            .map(|(layer, cache_policy)| {
                if persistent_descriptors {
                    layer.loose_read_plan_with_cache_policy_and_descriptors(
                        layout,
                        chunk_bytes,
                        cache_policy,
                        true,
                    )
                } else {
                    layer.loose_read_plan_with_cache_policy(layout, chunk_bytes, cache_policy)
                }
            })
            .collect::<Result<Vec<_>>>()
            .map(Vec::into_boxed_slice)
    }

    #[cfg(test)]
    pub fn loose_spine_read_plans_default(
        &self,
        layout: &SourceLayout,
    ) -> Result<Box<[LayerSpinePlan]>> {
        let cache_policies = vec![CachePolicy::Streaming; self.layers.len()];
        self.loose_spine_read_plans_with_cache_policies(
            layout,
            DEFAULT_SPINE_CHUNK_BYTES,
            &cache_policies,
        )
    }

    /// Admit the complete packed spine before inference starts. Publication is
    /// all-or-nothing: a missing, stale, malformed, or descriptor-budget-
    /// exceeding layer drops every already-opened candidate and returns an
    /// error. Payload chunks remain lazy and are authenticated by their first
    /// scheduled read.
    pub fn open_packed_spine_with_cache_policies(
        &self,
        model_root: &Path,
        pack_directory: Option<&Path>,
        cache_policies: &[CachePolicy],
    ) -> Result<PackedSpineCatalog> {
        if cache_policies.len() != self.layers.len() {
            return Err(DeltafinError::new(format!(
                "spine cache-policy roster has {} layers, expected {}",
                cache_policies.len(),
                self.layers.len(),
            )));
        }
        let identity = self.pack_identity(model_root)?;
        let directory = pack_directory
            .map(Path::to_path_buf)
            .unwrap_or_else(|| model_root.join(self.representation.pack_directory_name()));
        if !directory.is_dir() {
            return Err(DeltafinError::new(format!(
                "resident-spine pack directory does not exist: {}",
                directory.display()
            )));
        }
        let mut layers = Vec::with_capacity(self.layers.len());
        for (layer, cache_policy) in self.layers.iter().zip(cache_policies.iter().copied()) {
            let path = directory.join(format!("layer-{:03}.dfsp", layer.index));
            layers.push(
                layer
                    .packed_read_plan_with_cache_policy(&path, identity, cache_policy)
                    .map_err(|error| {
                        DeltafinError::new(format!(
                            "admit resident-spine layer {} transactionally: {error}",
                            layer.index
                        ))
                    })?,
            );
        }
        if layers.len() != K3_LAYER_COUNT {
            return Err(DeltafinError::new(format!(
                "resident-spine catalog has {} layers, expected {K3_LAYER_COUNT}",
                layers.len()
            )));
        }
        Ok(PackedSpineCatalog {
            layers: layers.into_boxed_slice(),
        })
    }

    pub fn pack_identity(&self, root: &Path) -> Result<PackIdentity> {
        let model_path = root.join("k3-meta/config.json");
        let inventory_path = root.join("k3-meta/tensor_inventory_offsets.json");
        let model = digest_file(&model_path).map_err(|error| {
            DeltafinError::new(format!(
                "hash model contract {}: {error}",
                model_path.display()
            ))
        })?;
        let source_inventory = digest_file(&inventory_path).map_err(|error| {
            DeltafinError::new(format!(
                "hash source inventory {}: {error}",
                inventory_path.display()
            ))
        })?;
        Ok(PackIdentity::new(
            model,
            source_inventory,
            self.layout_digest(),
        ))
    }

    pub fn layout_digest(&self) -> [u8; 32] {
        let mut canonical = Vec::with_capacity(512 * 1024);
        canonical.extend_from_slice(b"deltafin-k3-layer-spine-layout-v1\0");
        for layer in &self.layers {
            canonical.extend_from_slice(&layer.index.to_le_bytes());
            canonical.push(match layer.kind {
                LayerKind::Kda => 1,
                LayerKind::Mla => 2,
            });
            canonical.extend_from_slice(&(layer.weights.len() as u32).to_le_bytes());
            for weight in &layer.weights {
                canonical.extend_from_slice(&(weight.name.len() as u32).to_le_bytes());
                canonical.extend_from_slice(weight.name.as_bytes());
                canonical.push(match weight.storage {
                    WeightStorage::Raw(WeightDType::Bf16) => 1,
                    WeightStorage::Raw(WeightDType::F32) => 2,
                    WeightStorage::RowI8F16Scale {
                        logical: WeightDType::Bf16,
                    } => 3,
                    WeightStorage::RowI8F16Scale {
                        logical: WeightDType::F32,
                    } => 4,
                });
                canonical.push(weight.shape.len() as u8);
                for &dimension in weight.shape.iter() {
                    canonical.extend_from_slice(&dimension.to_le_bytes());
                }
                canonical.extend_from_slice(&weight.upload_group.to_le_bytes());
                canonical.extend_from_slice(&weight.upload_order.to_le_bytes());
            }
        }
        digest_bytes(&canonical)
    }

    fn validate_totals(&self) -> Result<()> {
        let actual = (
            self.logical_tensors(),
            self.quantized_tensors(),
            self.raw_tensors(),
            self.source_components(),
        );
        let expected = match self.representation {
            SpineRepresentation::OriginalBf16 => (
                K3_LOGICAL_SPINE_TENSORS,
                0,
                K3_BF16_RAW_SPINE_TENSORS,
                K3_BF16_SPINE_SOURCE_COMPONENTS,
            ),
            SpineRepresentation::QuantizedInt8 => (
                K3_LOGICAL_SPINE_TENSORS,
                K3_Q8_SPINE_TENSORS,
                K3_RAW_SPINE_TENSORS,
                K3_SPINE_SOURCE_COMPONENTS,
            ),
        };
        if actual != expected {
            return Err(DeltafinError::new(format!(
                "native target roster mismatch: logical/q8/raw/components={actual:?}, expected {expected:?}"
            )));
        }
        if self.embedding_weight.slot != WeightSlot::TokenEmbedding
            || self.embedding_weight.name != "language_model.model.embed_tokens.weight"
            || self.embedding_weight.shape.as_ref() != [163_840, 7_168]
            || self.embedding_weight.storage != WeightStorage::Raw(WeightDType::Bf16)
            || self.embedding_weight.upload_group != 0
            || self.embedding_weight.expected_data_bytes()? != 163_840 * 7_168 * 2
        {
            return Err(DeltafinError::new(
                "native exact BF16 embedding-row roster is incomplete",
            ));
        }
        let expected_global_components = match self.representation {
            SpineRepresentation::OriginalBf16 => K3_BF16_GLOBAL_SOURCE_COMPONENTS,
            SpineRepresentation::QuantizedInt8 => K3_GLOBAL_SOURCE_COMPONENTS,
        };
        if self.global_weights.len() != K3_GLOBAL_LOGICAL_TENSORS
            || self
                .global_weights
                .iter()
                .map(|weight| {
                    if matches!(weight.storage, WeightStorage::RowI8F16Scale { .. }) {
                        2
                    } else {
                        1
                    }
                })
                .sum::<usize>()
                != expected_global_components
        {
            return Err(DeltafinError::new(
                "native final/head global roster is incomplete",
            ));
        }
        Ok(())
    }
}

fn pack_dtype(dtype: WeightDType) -> DType {
    match dtype {
        WeightDType::Bf16 => DType::Bf16,
        WeightDType::F32 => DType::F32,
    }
}

fn aligned_cursor(value: usize) -> Result<usize> {
    let remainder = value % SPINE_COMPONENT_ALIGNMENT;
    if remainder == 0 {
        return Ok(value);
    }
    value
        .checked_add(SPINE_COMPONENT_ALIGNMENT - remainder)
        .ok_or_else(|| DeltafinError::new("spine component alignment overflows usize"))
}

fn push_padding(
    extents: &mut Vec<Extent>,
    destination: BufferKind,
    current: usize,
    aligned: usize,
) {
    if aligned > current {
        extents.push(Extent::zero(destination, current, aligned - current));
    }
}

fn require_source_length(path: &Path, expected: u64) -> Result<()> {
    let metadata = path.metadata().map_err(|error| {
        DeltafinError::new(format!("inspect spine source {}: {error}", path.display()))
    })?;
    if !metadata.is_file() {
        return Err(DeltafinError::new(format!(
            "spine source is not a regular file: {}",
            path.display()
        )));
    }
    if metadata.len() != expected {
        return Err(DeltafinError::new(format!(
            "spine source {} is {} bytes, expected {expected}",
            path.display(),
            metadata.len()
        )));
    }
    Ok(())
}

fn compile_embedding_weight() -> WeightSpec {
    WeightSpec {
        slot: WeightSlot::TokenEmbedding,
        name: "language_model.model.embed_tokens.weight".into(),
        shape: [163_840, 7_168].into(),
        storage: WeightStorage::Raw(WeightDType::Bf16),
        // Embedding is not a provider upload group. Zero is reserved here so
        // it cannot be mistaken for either immutable global bind group.
        upload_group: 0,
        upload_order: 0,
    }
}

fn compile_global_weights(representation: SpineRepresentation) -> Box<[WeightSpec]> {
    let bf16 = WeightStorage::Raw(WeightDType::Bf16);
    let q8 = WeightStorage::RowI8F16Scale {
        logical: WeightDType::Bf16,
    };
    let linear = match representation {
        SpineRepresentation::OriginalBf16 => bf16,
        SpineRepresentation::QuantizedInt8 => q8,
    };
    [
        WeightSpec {
            slot: WeightSlot::FinalNorm,
            name: "language_model.model.norm.weight".into(),
            shape: [7_168].into(),
            storage: bf16,
            upload_group: 1,
            upload_order: 1,
        },
        WeightSpec {
            slot: WeightSlot::OutputAttentionResidualNorm,
            name: "language_model.model.output_attn_res_norm.weight".into(),
            shape: [7_168].into(),
            storage: bf16,
            upload_group: 1,
            upload_order: 2,
        },
        WeightSpec {
            slot: WeightSlot::OutputAttentionResidualProjection,
            name: "language_model.model.output_attn_res_proj.weight".into(),
            shape: [1, 7_168].into(),
            storage: bf16,
            upload_group: 1,
            upload_order: 3,
        },
        WeightSpec {
            slot: WeightSlot::LanguageModelHead,
            name: "language_model.lm_head.weight".into(),
            shape: [163_840, 7_168].into(),
            storage: linear,
            upload_group: 2,
            upload_order: 4,
        },
    ]
    .into()
}

fn compile_layer(
    index: usize,
    kind: LayerKind,
    representation: SpineRepresentation,
) -> Result<LayerProgram> {
    let layer =
        u32::try_from(index).map_err(|_| DeltafinError::new("layer index does not fit in u32"))?;
    let prefix = format!("language_model.model.layers.{index}");
    let mut weights = Vec::with_capacity(match (index, kind) {
        (0, LayerKind::Kda) => 23,
        (_, LayerKind::Kda) => 28,
        (_, LayerKind::Mla) => 22,
    });

    let mut push = |slot, suffix: &str, shape: &[u64], storage, group| {
        let order = weights.len() as u16;
        weights.push(WeightSpec {
            slot,
            name: format!("{prefix}.{suffix}"),
            shape: shape.into(),
            storage,
            upload_group: group,
            upload_order: order,
        });
    };
    let bf16 = WeightStorage::Raw(WeightDType::Bf16);
    let f32 = WeightStorage::Raw(WeightDType::F32);
    let q8 = WeightStorage::RowI8F16Scale {
        logical: WeightDType::Bf16,
    };
    let linear = match representation {
        SpineRepresentation::OriginalBf16 => bf16,
        SpineRepresentation::QuantizedInt8 => q8,
    };

    push(
        WeightSlot::InputLayerNorm,
        "input_layernorm.weight",
        &[7168],
        bf16,
        0,
    );
    push(
        WeightSlot::SelfAttentionResidualNorm,
        "self_attention_res_norm.weight",
        &[7168],
        bf16,
        0,
    );
    push(
        WeightSlot::SelfAttentionResidualProjection,
        "self_attention_res_proj.weight",
        &[1, 7168],
        bf16,
        0,
    );
    push(
        WeightSlot::PostAttentionLayerNorm,
        "post_attention_layernorm.weight",
        &[7168],
        bf16,
        0,
    );
    push(
        WeightSlot::MlpResidualNorm,
        "mlp_res_norm.weight",
        &[7168],
        bf16,
        0,
    );
    push(
        WeightSlot::MlpResidualProjection,
        "mlp_res_proj.weight",
        &[1, 7168],
        bf16,
        0,
    );

    match kind {
        LayerKind::Kda => {
            push(WeightSlot::KdaALog, "self_attn.A_log", &[128], f32, 10);
            push(
                WeightSlot::KdaDtBias,
                "self_attn.dt_bias",
                &[12_288],
                f32,
                10,
            );
            for (slot, name) in [
                (WeightSlot::KdaQueryConvolution, "q_conv1d"),
                (WeightSlot::KdaKeyConvolution, "k_conv1d"),
                (WeightSlot::KdaValueConvolution, "v_conv1d"),
            ] {
                push(
                    slot,
                    &format!("self_attn.{name}.weight"),
                    &[12_288, 1, 4],
                    f32,
                    10,
                );
            }
            push(
                WeightSlot::KdaOutputNorm,
                "self_attn.o_norm.weight",
                &[128],
                f32,
                10,
            );
            for (slot, name, shape) in [
                (WeightSlot::KdaQueryProjection, "q_proj", [12_288, 7_168]),
                (WeightSlot::KdaKeyProjection, "k_proj", [12_288, 7_168]),
                (WeightSlot::KdaValueProjection, "v_proj", [12_288, 7_168]),
                (WeightSlot::KdaGateProjection, "g_proj", [12_288, 7_168]),
                (WeightSlot::KdaFeatureAProjection, "f_a_proj", [128, 7_168]),
                (WeightSlot::KdaFeatureBProjection, "f_b_proj", [12_288, 128]),
                (WeightSlot::KdaBetaProjection, "b_proj", [96, 7_168]),
                (WeightSlot::KdaOutputProjection, "o_proj", [7_168, 12_288]),
            ] {
                push(
                    slot,
                    &format!("self_attn.{name}.weight"),
                    &shape,
                    linear,
                    11,
                );
            }
        }
        LayerKind::Mla => {
            for (slot, name, shape) in [
                (WeightSlot::MlaQueryAProjection, "q_a_proj", [1_536, 7_168]),
                (WeightSlot::MlaQueryBProjection, "q_b_proj", [18_432, 1_536]),
                (
                    WeightSlot::MlaKeyValueAProjection,
                    "kv_a_proj_with_mqa",
                    [576, 7_168],
                ),
                (
                    WeightSlot::MlaKeyValueBProjection,
                    "kv_b_proj",
                    [24_576, 512],
                ),
                (WeightSlot::MlaGateProjection, "g_proj", [12_288, 7_168]),
                (WeightSlot::MlaOutputProjection, "o_proj", [7_168, 12_288]),
            ] {
                push(
                    slot,
                    &format!("self_attn.{name}.weight"),
                    &shape,
                    linear,
                    11,
                );
            }
            push(
                WeightSlot::MlaQueryANorm,
                "self_attn.q_a_layernorm.weight",
                &[1_536],
                bf16,
                10,
            );
            push(
                WeightSlot::MlaKeyValueANorm,
                "self_attn.kv_a_layernorm.weight",
                &[512],
                bf16,
                10,
            );
        }
    }

    if index == 0 {
        for (slot, name, shape) in [
            (
                WeightSlot::DenseGateProjection,
                "gate_proj",
                [33_792, 7_168],
            ),
            (WeightSlot::DenseUpProjection, "up_proj", [33_792, 7_168]),
            (
                WeightSlot::DenseDownProjection,
                "down_proj",
                [7_168, 33_792],
            ),
        ] {
            push(slot, &format!("mlp.{name}.weight"), &shape, linear, 20);
        }
    } else {
        for (slot, suffix, shape, storage) in [
            (
                WeightSlot::MoeGateWeight,
                "block_sparse_moe.gate.weight",
                [896, 7_168],
                linear,
            ),
            (
                WeightSlot::MoeRoutedDownProjection,
                "block_sparse_moe.routed_expert_down_proj.weight",
                [3_584, 7_168],
                linear,
            ),
            (
                WeightSlot::MoeRoutedUpProjection,
                "block_sparse_moe.routed_expert_up_proj.weight",
                [7_168, 3_584],
                linear,
            ),
            (
                WeightSlot::MoeSharedGateProjection,
                "block_sparse_moe.shared_experts.gate_proj.weight",
                [6_144, 7_168],
                linear,
            ),
            (
                WeightSlot::MoeSharedUpProjection,
                "block_sparse_moe.shared_experts.up_proj.weight",
                [6_144, 7_168],
                linear,
            ),
            (
                WeightSlot::MoeSharedDownProjection,
                "block_sparse_moe.shared_experts.down_proj.weight",
                [7_168, 6_144],
                linear,
            ),
        ] {
            push(slot, suffix, &shape, storage, 20);
        }
        push(
            WeightSlot::MoeGateCorrectionBias,
            "block_sparse_moe.gate.e_score_correction_bias",
            &[896],
            f32,
            20,
        );
        push(
            WeightSlot::MoeRoutedNorm,
            "block_sparse_moe.routed_expert_norm.weight",
            &[3_584],
            bf16,
            20,
        );
    }

    let expected = match (index, kind) {
        (0, LayerKind::Kda) => 23,
        (_, LayerKind::Kda) => 28,
        (_, LayerKind::Mla) => 22,
    };
    if weights.len() != expected {
        return Err(DeltafinError::new(format!(
            "layer {index} compiled {} weight slots, expected {expected}",
            weights.len()
        )));
    }
    Ok(LayerProgram {
        index: layer,
        kind,
        weights: weights.into_boxed_slice(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inventory::K3Inventory;
    use crate::platform::Device;
    use crate::provider::{
        NativeProviderSession, SpineComponent, SpineLayerRetention, SpineStoredScalar,
    };
    use crate::storage::Reader;
    use std::collections::BTreeMap;
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_PACK_TEST: AtomicU64 = AtomicU64::new(1);

    fn repository_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    #[test]
    fn exact_model_compiles_to_the_audited_fixed_roster() {
        let model = ModelSpec::load_from_root(&repository_root()).unwrap();
        let program = TargetProgram::compile(&model).unwrap();
        assert_eq!(program.layers.len(), K3_LAYER_COUNT);
        assert_eq!(program.logical_tensors(), K3_LOGICAL_SPINE_TENSORS);
        assert_eq!(program.representation(), SpineRepresentation::OriginalBf16);
        assert_eq!(program.quantized_tensors(), 0);
        assert_eq!(program.raw_tensors(), K3_BF16_RAW_SPINE_TENSORS);
        assert_eq!(program.source_components(), K3_BF16_SPINE_SOURCE_COMPONENTS);
        assert_eq!(program.global_weights.len(), K3_GLOBAL_LOGICAL_TENSORS);
        assert_eq!(program.embedding_weight.slot, WeightSlot::TokenEmbedding);
        assert_eq!(
            program.embedding_weight.storage,
            WeightStorage::Raw(WeightDType::Bf16)
        );
        assert_eq!(program.layers[0].weights.len(), 23);
        assert_eq!(program.layers[1].weights.len(), 28);
        assert_eq!(program.layers[3].weights.len(), 22);
    }

    #[test]
    fn exact_program_is_a_bijection_with_the_authenticated_language_inventory() {
        const K3_LANGUAGE_RESIDENT_TENSORS: usize =
            K3_LOGICAL_SPINE_TENSORS + K3_GLOBAL_LOGICAL_TENSORS + 1;
        const K3_EXPERT_INVENTORY_RECORDS: usize = 92 * 896 * 6;
        const K3_VISION_TENSORS: usize = 165;
        const K3_MM_PROJECTOR_TENSORS: usize = 3;

        let root = repository_root();
        let model = ModelSpec::load_from_root(&root).unwrap();
        let program = TargetProgram::compile(&model).unwrap();
        let inventory = K3Inventory::load_from_root(&root).unwrap();

        let mut expected = BTreeMap::new();
        for weight in std::iter::once(&program.embedding_weight).chain(
            program
                .global_weights
                .iter()
                .chain(program.layers.iter().flat_map(|layer| layer.weights.iter())),
        ) {
            let dtype = match weight.storage {
                WeightStorage::Raw(WeightDType::Bf16) => "BF16",
                WeightStorage::Raw(WeightDType::F32) => "F32",
                WeightStorage::RowI8F16Scale { .. } => {
                    panic!("the exact inventory gate received a quantized weight")
                }
            };
            assert!(
                expected
                    .insert(weight.name.as_str(), (dtype, weight.shape.as_ref()))
                    .is_none(),
                "duplicate tensor in native language roster: {}",
                weight.name,
            );
        }
        assert_eq!(expected.len(), K3_LANGUAGE_RESIDENT_TENSORS);

        let mut actual_language = BTreeMap::new();
        let mut expert_records = 0usize;
        let mut vision_tensors = 0usize;
        let mut mm_projector_tensors = 0usize;
        for (name, record) in inventory.iter() {
            if name.contains(".block_sparse_moe.experts.") {
                assert!(name.starts_with("language_model."));
                expert_records += 1;
            } else if name.starts_with("language_model.") {
                assert!(actual_language.insert(name, record).is_none());
            } else if name.starts_with("vision_tower.") {
                vision_tensors += 1;
            } else if name.starts_with("mm_projector.") {
                mm_projector_tensors += 1;
            } else {
                panic!("unclassified tensor in authenticated K3 inventory: {name}");
            }
        }

        assert_eq!(expert_records, K3_EXPERT_INVENTORY_RECORDS);
        assert_eq!(vision_tensors, K3_VISION_TENSORS);
        assert_eq!(mm_projector_tensors, K3_MM_PROJECTOR_TENSORS);
        assert_eq!(actual_language.len(), K3_LANGUAGE_RESIDENT_TENSORS);

        for (name, (dtype, shape)) in expected {
            let record = actual_language.remove(name).unwrap_or_else(|| {
                panic!("native language tensor is absent from inventory: {name}")
            });
            assert_eq!(record.dtype, dtype, "dtype mismatch for {name}");
            assert_eq!(record.shape.as_slice(), shape, "shape mismatch for {name}");
        }
        assert!(
            actual_language.is_empty(),
            "authenticated inventory has language tensors absent from the native program: {:?}",
            actual_language.keys().collect::<Vec<_>>(),
        );
    }

    #[test]
    fn explicit_int8_compiles_to_the_audited_non_weight_exact_roster() {
        let model = ModelSpec::load_from_root(&repository_root()).unwrap();
        let program =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        assert_eq!(program.quantized_tensors(), K3_Q8_SPINE_TENSORS);
        assert_eq!(program.raw_tensors(), K3_RAW_SPINE_TENSORS);
        assert_eq!(program.source_components(), K3_SPINE_SOURCE_COMPONENTS);
        assert!(!program.representation().is_weight_exact());
    }

    #[test]
    fn fp32_execution_arena_matches_the_three_exact_k3_layer_classes() {
        let model = ModelSpec::load_from_root(&repository_root()).unwrap();
        let program =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        assert_eq!(
            program.layers[0].fp32_projection_arena_bytes().unwrap(),
            4_680_974_336,
        );
        assert_eq!(
            program.layers[1].fp32_projection_arena_bytes().unwrap(),
            2_534_014_976,
        );
        assert_eq!(
            program.layers[3].fp32_projection_arena_bytes().unwrap(),
            1_688_469_504,
        );
        assert_eq!(
            program.fp32_spine_execution_arena_bytes().unwrap(),
            4_680_974_336,
        );

        // The arena is defined by logical K3 matrices, not by the source
        // codec. Exact original-BF16 can reuse these same aligned views when
        // its dense MPS execution path is admitted.
        let exact = TargetProgram::compile(&model).unwrap();
        assert_eq!(
            exact.fp32_spine_execution_arena_bytes().unwrap(),
            program.fp32_spine_execution_arena_bytes().unwrap(),
        );
    }

    #[test]
    fn original_and_quantized_packs_have_distinct_names_and_identities() {
        assert_ne!(
            SpineRepresentation::OriginalBf16.pack_directory_name(),
            SpineRepresentation::QuantizedInt8.pack_directory_name(),
        );
        let model = ModelSpec::load_from_root(&repository_root()).unwrap();
        let original = TargetProgram::compile(&model).unwrap();
        let quantized =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        assert_ne!(original.layout_digest(), quantized.layout_digest());
    }

    #[test]
    fn provider_residency_keeps_large_exact_matrices_bf16_and_promotes_small_values() {
        let model = ModelSpec::load_from_root(&repository_root()).unwrap();
        let program = TargetProgram::compile(&model).unwrap();
        let layer_bytes = program.provider_layer_bytes().unwrap();
        assert_eq!(layer_bytes.len(), K3_LAYER_COUNT);
        assert!(layer_bytes.iter().all(|bytes| *bytes > 0));

        let large_matrix = 163_840_u64 * 7_168;
        let promoted_tail = 3_u64 * 7_168 * 4;
        assert_eq!(
            program.provider_global_bytes().unwrap(),
            large_matrix * 2 + promoted_tail,
        );

        let int8 =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        let promoted_row_scales = 163_840_u64 * 4;
        assert_eq!(
            int8.provider_global_bytes().unwrap(),
            large_matrix + promoted_row_scales + promoted_tail,
        );

        let embedding = &program.embedding_weight;
        assert_eq!(embedding.expected_data_bytes().unwrap(), large_matrix * 2,);
        let final_norm = &program.global_weights[0];
        assert_eq!(final_norm.provider_resident_bytes().unwrap(), 7_168 * 4);
    }

    #[test]
    fn int8_source_layer_accounting_matches_the_public_stream_policy() {
        let model = ModelSpec::load_from_root(&repository_root()).unwrap();
        let program =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        let bytes = program.int8_stream_layer_bytes().unwrap();
        assert_eq!(bytes.len(), K3_LAYER_COUNT);
        assert_eq!(bytes.iter().sum::<u64>(), 54_397_786_304);
    }

    #[test]
    fn loose_spine_preserves_each_layer_cache_admission_policy() {
        let root = repository_root();
        let model = ModelSpec::load_from_root(&root).unwrap();
        let program =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        let layout = SourceLayout::under(&root);
        let mut policies = vec![CachePolicy::Streaming; K3_LAYER_COUNT];
        policies[0] = CachePolicy::Resident;
        policies[1] = CachePolicy::Resident;
        let plans = program
            .loose_spine_read_plans_with_cache_policies(
                &layout,
                DEFAULT_SPINE_CHUNK_BYTES,
                &policies,
            )
            .unwrap();
        assert_eq!(
            plans[0].read_plan().cache_policy(),
            Some(CachePolicy::Resident)
        );
        assert_eq!(
            plans[1].read_plan().cache_policy(),
            Some(CachePolicy::Resident)
        );
        assert_eq!(
            plans[2].read_plan().cache_policy(),
            Some(CachePolicy::Streaming)
        );

        assert!(
            program
                .loose_spine_read_plans_with_cache_policies(
                    &layout,
                    DEFAULT_SPINE_CHUNK_BYTES,
                    &policies[..K3_LAYER_COUNT - 1],
                )
                .is_err()
        );
    }

    #[test]
    fn complete_int8_loose_spine_can_reserve_one_lazy_descriptor_per_component() {
        let root = repository_root();
        let model = ModelSpec::load_from_root(&root).unwrap();
        let program =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        crate::storage::prepare_persistent_descriptor_capacity(
            program.source_components(),
            crate::storage::LOOSE_SPINE_DESCRIPTOR_RESERVE,
        )
        .unwrap();
        let policies = vec![CachePolicy::Streaming; K3_LAYER_COUNT];
        let plans = program
            .loose_spine_read_plans_with_cache_policies_and_descriptors(
                &SourceLayout::under(&root),
                DEFAULT_SPINE_CHUNK_BYTES,
                &policies,
                true,
            )
            .unwrap();
        assert_eq!(plans.len(), K3_LAYER_COUNT);
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.read_plan().persistent_source_count())
                .sum::<usize>(),
            program.source_components()
        );
        assert_eq!(
            plans
                .iter()
                .map(|plan| plan.read_plan().opened_persistent_source_count())
                .sum::<usize>(),
            0
        );
    }

    #[test]
    fn exact_raw_bf16_projection_roster_never_assumes_fp32_residency() {
        let model = ModelSpec::load_from_root(&repository_root()).unwrap();
        let program = TargetProgram::compile(&model).unwrap();
        let mut projections = 0_usize;
        for weight in program
            .layers
            .iter()
            .flat_map(|layer| layer.weights.iter())
            .chain(program.global_weights.iter())
        {
            if weight.storage == WeightStorage::Raw(WeightDType::Bf16)
                && weight.is_large_projection_matrix()
            {
                projections += 1;
                assert_eq!(
                    weight.provider_resident_bytes().unwrap(),
                    weight.expected_data_bytes().unwrap(),
                    "{} was budgeted as an expanded fp32 matrix",
                    weight.name,
                );
                assert_eq!(
                    weight.provider_resident_bytes().unwrap(),
                    weight.element_count().unwrap() * WeightDType::Bf16.byte_width(),
                );
            }
        }
        assert!(
            projections > 1_000,
            "exact K3 roster lost its BF16 projections"
        );

        let head = program
            .global_weights
            .iter()
            .find(|weight| weight.slot == WeightSlot::LanguageModelHead)
            .unwrap();
        assert!(head.is_large_projection_matrix());
        assert_eq!(
            head.provider_resident_bytes().unwrap(),
            163_840_u64 * 7_168 * 2
        );
    }

    #[test]
    fn compile_revalidates_a_changed_model_spec_before_building_a_roster() {
        let mut model = ModelSpec::load_from_root(&repository_root()).unwrap();
        model.hidden_size = 1;
        let error = TargetProgram::compile(&model).unwrap_err();
        assert!(error.to_string().contains("hidden_size=1"));
    }

    #[test]
    fn representative_source_paths_and_lengths_are_exact() {
        let model = ModelSpec::load_from_root(&repository_root()).unwrap();
        let program = TargetProgram::compile(&model).unwrap();
        let layout = SourceLayout::under(&repository_root());

        let exact = program.layers[0]
            .weights
            .iter()
            .find(|weight| weight.slot == WeightSlot::KdaQueryProjection)
            .unwrap();
        assert_eq!(exact.expected_data_bytes().unwrap(), 12_288 * 7_168 * 2);
        assert_eq!(exact.expected_scale_bytes().unwrap(), None);
        assert!(exact.source_paths(&layout).data.ends_with(
            "k3-resident/tensors/language_model.model.layers.0.self_attn.q_proj.weight"
        ));

        let int8 =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        let q = int8.layers[0]
            .weights
            .iter()
            .find(|weight| weight.slot == WeightSlot::KdaQueryProjection)
            .unwrap();
        assert_eq!(q.expected_data_bytes().unwrap(), 12_288 * 7_168);
        assert_eq!(q.expected_scale_bytes().unwrap(), Some(12_288 * 2));
        assert!(q.source_paths(&layout).data.ends_with(
            "k3-resident-int8/tensors/language_model.model.layers.0.self_attn.q_proj.weight.i8"
        ));

        let raw = program.layers[0]
            .weights
            .iter()
            .find(|weight| weight.slot == WeightSlot::KdaQueryConvolution)
            .unwrap();
        assert_eq!(raw.expected_data_bytes().unwrap(), 12_288 * 4 * 4);
        assert_eq!(raw.expected_scale_bytes().unwrap(), None);
        assert!(raw.source_paths(&layout).data.ends_with(
            "k3-resident/tensors/language_model.model.layers.0.self_attn.q_conv1d.weight"
        ));
    }

    #[test]
    fn installed_source_components_match_the_compiled_roster_when_present() {
        let root = repository_root();
        let layout = SourceLayout::under(&root);
        if !layout.resident_tensors.is_dir() {
            return;
        }
        let model = ModelSpec::load_from_root(&root).unwrap();
        let validate_sources = |program: &TargetProgram| {
            for weight in std::iter::once(&program.embedding_weight).chain(
                program
                    .global_weights
                    .iter()
                    .chain(program.layers.iter().flat_map(|layer| layer.weights.iter())),
            ) {
                let paths = weight.source_paths(&layout);
                let data = std::fs::metadata(&paths.data).unwrap_or_else(|error| {
                    panic!("missing source {}: {error}", paths.data.display())
                });
                assert!(data.is_file(), "not a file: {}", paths.data.display());
                assert_eq!(
                    data.len(),
                    weight.expected_data_bytes().unwrap(),
                    "wrong data length for {}",
                    weight.name
                );
                match (paths.scales, weight.expected_scale_bytes().unwrap()) {
                    (Some(path), Some(expected)) => {
                        let scales = std::fs::metadata(&path).unwrap_or_else(|error| {
                            panic!("missing scale source {}: {error}", path.display())
                        });
                        assert!(scales.is_file(), "not a file: {}", path.display());
                        assert_eq!(
                            scales.len(),
                            expected,
                            "wrong scale length for {}",
                            weight.name
                        );
                    }
                    (None, None) => {}
                    _ => panic!("source/scale contract mismatch for {}", weight.name),
                }
            }
        };

        let exact = TargetProgram::compile(&model).unwrap();
        validate_sources(&exact);
        let exact_loose = exact.layers[0].loose_read_plan_default(&layout).unwrap();
        let exact_query = exact_loose
            .descriptors()
            .iter()
            .find(|descriptor| descriptor.slot == WeightSlot::KdaQueryProjection as u32)
            .unwrap();
        assert_eq!(exact_query.encoding, SPINE_ENCODING_RAW_BF16);
        assert_eq!(exact_query.data_buffer, SPINE_BUFFER_OTHER);
        assert_eq!(exact_query.auxiliary_buffer, SPINE_BUFFER_NONE);
        assert_eq!(exact_loose.buffer_lengths().quantized, 0);
        assert_eq!(exact_loose.buffer_lengths().scales, 0);
        assert!(exact_loose.buffer_lengths().other > 0);
        let exact_globals = exact.global_loose_read_plans_default(&layout).unwrap();
        assert_eq!(
            exact_globals
                .iter()
                .map(|plan| plan.read_plan().source_count())
                .sum::<usize>(),
            K3_BF16_GLOBAL_SOURCE_COMPONENTS,
        );
        let exact_layers = exact.loose_spine_read_plans_default(&layout).unwrap();
        assert_eq!(exact_layers.len(), K3_LAYER_COUNT);
        assert_eq!(
            exact_layers
                .iter()
                .map(|plan| plan.read_plan().source_count())
                .sum::<usize>(),
            K3_BF16_SPINE_SOURCE_COMPONENTS,
        );

        if !layout.int8_tensors.is_dir() {
            return;
        }

        let program =
            TargetProgram::compile_with_representation(&model, SpineRepresentation::QuantizedInt8)
                .unwrap();
        validate_sources(&program);

        let identity = program.pack_identity(&root).unwrap();
        program.layers[0].pack_builder(&layout, identity).unwrap();
        let loose = program.layers[0].loose_read_plan_default(&layout).unwrap();
        assert_eq!(loose.layer(), 0);
        assert_eq!(loose.descriptors().len(), 23);
        let query = loose
            .descriptors()
            .iter()
            .find(|descriptor| descriptor.slot == WeightSlot::KdaQueryProjection as u32)
            .unwrap();
        assert_eq!(query.encoding, SPINE_ENCODING_ROW_I8_F16_SCALE);
        assert_eq!(&query.shape[..2], &[12_288, 7_168]);
        assert_eq!(query.data_buffer, SPINE_BUFFER_QUANTIZED);
        assert_eq!(query.auxiliary_buffer, SPINE_BUFFER_SCALES);
        assert!(loose.buffer_lengths().quantized > 0);
        assert!(loose.buffer_lengths().scales > 0);
        assert!(loose.buffer_lengths().other > 0);

        let globals = program.global_loose_read_plans_default(&layout).unwrap();
        assert_eq!(globals.len(), K3_GLOBAL_TRANSFER_GROUPS);
        assert_eq!(globals[0].group(), 1);
        assert_eq!(globals[0].descriptors().len(), 3);
        assert_eq!(globals[1].group(), 2);
        assert_eq!(globals[1].descriptors().len(), 1);
        assert_eq!(
            globals[1].descriptors()[0].slot,
            WeightSlot::LanguageModelHead as u32
        );
        assert_eq!(
            globals
                .iter()
                .map(|plan| plan.read_plan().source_count())
                .sum::<usize>(),
            K3_GLOBAL_SOURCE_COMPONENTS
        );
        assert_eq!(
            globals
                .iter()
                .map(|plan| plan.read_plan().persistent_source_count())
                .sum::<usize>(),
            0
        );

        let loose_layers = program.loose_spine_read_plans_default(&layout).unwrap();
        assert_eq!(loose_layers.len(), K3_LAYER_COUNT);
        assert_eq!(
            loose_layers
                .iter()
                .map(|plan| plan.read_plan().source_count())
                .sum::<usize>(),
            K3_SPINE_SOURCE_COMPONENTS
        );
        assert_eq!(
            loose_layers
                .iter()
                .map(|plan| plan.read_plan().persistent_source_count())
                .sum::<usize>(),
            0
        );
        assert_ne!(program.layout_digest(), [0; 32]);
    }

    #[test]
    fn authenticated_pack_becomes_one_descriptor_and_one_provider_slab() {
        let serial = NEXT_PACK_TEST.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "deltafin-program-pack-{}-{serial}.dfsp",
            std::process::id()
        ));
        let identity = PackIdentity::new([1; 32], [2; 32], [3; 32]);
        let q_name = "language_model.model.layers.7.q.weight";
        let raw_name = "language_model.model.layers.7.norm.weight";
        let mut builder = PackBuilder::new(7, identity).unwrap();
        builder.push(BuildTensor::row_i8_f16_scale(
            q_name,
            DType::Bf16,
            [2, 3],
            ComponentSource::bytes(vec![1, 2, 3, 4, 5, 6]),
            ComponentSource::bytes(vec![0, 60, 0, 64]),
            4,
            0,
        ));
        builder.push(BuildTensor::raw(
            raw_name,
            DType::Bf16,
            [2],
            ComponentSource::bytes(vec![7, 8, 9, 10]),
            5,
            1,
        ));
        builder.write_atomic(&path).unwrap();

        let layer = LayerProgram {
            index: 7,
            kind: LayerKind::Kda,
            weights: vec![
                WeightSpec {
                    slot: WeightSlot::KdaQueryProjection,
                    name: q_name.into(),
                    shape: [2, 3].into(),
                    storage: WeightStorage::RowI8F16Scale {
                        logical: WeightDType::Bf16,
                    },
                    upload_group: 4,
                    upload_order: 0,
                },
                WeightSpec {
                    slot: WeightSlot::InputLayerNorm,
                    name: raw_name.into(),
                    shape: [2].into(),
                    storage: WeightStorage::Raw(WeightDType::Bf16),
                    upload_group: 5,
                    upload_order: 1,
                },
            ]
            .into_boxed_slice(),
        };
        let plan = layer
            .packed_read_plan_with_cache_policy(&path, identity, CachePolicy::Streaming)
            .unwrap();
        assert_eq!(plan.read_plan().source_count(), 1);
        assert_eq!(plan.buffer_lengths().quantized, 0);
        assert_eq!(plan.buffer_lengths().scales, 0);
        assert!(plan.buffer_lengths().other >= 3 * 64 * 1024);
        assert!(plan.descriptors().iter().all(|descriptor| {
            descriptor.data_buffer == SPINE_BUFFER_OTHER
                && (descriptor.auxiliary_buffer == SPINE_BUFFER_NONE
                    || descriptor.auxiliary_buffer == SPINE_BUFFER_OTHER)
        }));

        let reader = Reader::new(2).unwrap();
        let (buffers, _) = reader.read(plan.read_plan()).unwrap();
        for descriptor in plan.descriptors() {
            let start = descriptor.data_offset as usize;
            let end = start + descriptor.data_length as usize;
            let expected: &[u8] = if descriptor.slot == WeightSlot::KdaQueryProjection as u32 {
                &[1, 2, 3, 4, 5, 6]
            } else {
                &[7, 8, 9, 10]
            };
            assert_eq!(&buffers.other()[start..end], expected);
        }

        // Regression: packed components intentionally share the authenticated
        // Other slab. The provider must honor each descriptor's buffer ID
        // instead of requiring the loose layout's Quantized/Scales labels.
        let provider = NativeProviderSession::target(Device::Cpu).unwrap();
        let report = provider
            .bind_spine_layer(
                7,
                1,
                plan.descriptors(),
                &buffers,
                SpineLayerRetention::Transient,
            )
            .unwrap();
        assert_eq!(report.tensor_count, 2);
        let q = provider
            .read_spine_tensor_f32(
                7,
                1,
                WeightSlot::KdaQueryProjection as u32,
                SpineComponent::Data,
                6,
            )
            .unwrap();
        assert_eq!(q.stored_scalar, SpineStoredScalar::I8);
        assert_eq!(&*q.values, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let scales = provider
            .read_spine_tensor_f32(
                7,
                1,
                WeightSlot::KdaQueryProjection as u32,
                SpineComponent::Auxiliary,
                2,
            )
            .unwrap();
        assert_eq!(scales.stored_scalar, SpineStoredScalar::F32);
        assert_eq!(&*scales.values, &[1.0, 2.0]);
        drop(buffers);
        fs::remove_file(path).unwrap();
    }
}
