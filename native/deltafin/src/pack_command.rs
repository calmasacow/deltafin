//! Resumable DFSP construction inside the single `deltafin` executable.

use std::fs;
use std::path::{Path, PathBuf};

use crate::cli::PackSpineArgs;
use crate::config::SpineRequest;
use crate::error::{DeltafinError, Result};
use crate::model::ModelSpec;
use crate::packfile::PackFile;
use crate::program::{K3_LAYER_COUNT, SourceLayout, SpineRepresentation, TargetProgram};

pub fn run(arguments: PackSpineArgs) -> Result<()> {
    let model = ModelSpec::load_from_root(&arguments.model_root)?;
    let representation = match SpineRequest::parse(arguments.spine.as_deref())? {
        SpineRequest::Auto | SpineRequest::Bf16 => SpineRepresentation::OriginalBf16,
        SpineRequest::Int8 => SpineRepresentation::QuantizedInt8,
    };
    let program = TargetProgram::compile_with_representation(&model, representation)?;
    let identity = program.pack_identity(&arguments.model_root)?;
    let sources = SourceLayout::under(&arguments.model_root);
    let output = arguments.output.unwrap_or_else(|| {
        arguments
            .model_root
            .join(program.representation().pack_directory_name())
    });

    if arguments.verify_only {
        if !output.is_dir() {
            return Err(DeltafinError::new(format!(
                "pack directory does not exist: {}",
                output.display()
            )));
        }
    } else {
        fs::create_dir_all(&output).map_err(|error| {
            DeltafinError::new(format!(
                "create pack directory {}: {error}",
                output.display()
            ))
        })?;
    }
    if !output.is_dir() {
        return Err(DeltafinError::new(format!(
            "pack output is not a directory: {}",
            output.display()
        )));
    }
    eprintln!(
        "[pack] representation={} weight_exact={} output={}",
        program.representation(),
        program.representation().is_weight_exact(),
        output.display(),
    );

    let selected: Box<dyn Iterator<Item = usize>> = match arguments.layer {
        Some(layer) if (layer as usize) < K3_LAYER_COUNT => {
            Box::new(std::iter::once(layer as usize))
        }
        Some(layer) => {
            return Err(DeltafinError::new(format!(
                "--layer {layer} is outside the full K3 range 0..{}",
                K3_LAYER_COUNT - 1
            )));
        }
        None => Box::new(0..K3_LAYER_COUNT),
    };

    for index in selected {
        let layer = &program.layers[index];
        let destination = layer_path(&output, index);
        if destination.exists() {
            let pack = PackFile::open_for(&destination, layer.index, identity)
                .map_err(|error| pack_error("open", &destination, error))?;
            pack.verify_all()
                .map_err(|error| pack_error("verify", &destination, error))?;
            println!("verified {}", destination.display());
            continue;
        }
        if arguments.verify_only {
            return Err(DeltafinError::new(format!(
                "missing layer pack: {}",
                destination.display()
            )));
        }
        eprintln!(
            "[pack] layer {index}/{}: {} tensors from {} immutable source components",
            K3_LAYER_COUNT - 1,
            layer.weights.len(),
            layer.source_components()
        );
        let builder = layer.pack_builder(&sources, identity)?;
        let pack = builder
            .write_atomic(&destination)
            .map_err(|error| pack_error("build", &destination, error))?;
        println!(
            "built {} ({:.2} GiB, {} tensors)",
            destination.display(),
            pack.header().file_bytes as f64 / 1024_f64.powi(3),
            pack.tensors().len()
        );
    }
    Ok(())
}

fn layer_path(output: &Path, index: usize) -> PathBuf {
    output.join(format!("layer-{index:03}.dfsp"))
}

fn pack_error(operation: &str, path: &Path, error: crate::packfile::PackError) -> DeltafinError {
    DeltafinError::new(format!(
        "{operation} layer pack {}: {error}",
        path.display()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_names_sort_in_execution_order() {
        let root = Path::new("/packs");
        assert_eq!(layer_path(root, 0), root.join("layer-000.dfsp"));
        assert_eq!(layer_path(root, 92), root.join("layer-092.dfsp"));
    }
}
