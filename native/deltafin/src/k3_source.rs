//! Immutable Kimi K3 source identity used by native setup and fetch commands.
//!
//! Native Deltafin needs only inert model metadata and tokenizer data.  It
//! deliberately does not download Moonshot's executable Python modeling or
//! tokenization modules into its production installation path.

use crate::error::{DeltafinError, Result};
use crate::packfile::Digest;

pub const MODEL_REPOSITORY: &str = "moonshotai/Kimi-K3";
pub const MODEL_REVISION: &str = "c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721";
pub const DEFAULT_HF_HOST: &str = "huggingface.co";
pub const DEFAULT_HF_PATH: &str =
    "/moonshotai/Kimi-K3/resolve/c5d1dd4c428bd1ce8b88c5044f3b6ccde9e3b721/";

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct PinnedMetadataFile {
    pub name: &'static str,
    pub size: u64,
    pub sha256: Digest,
}

pub const NATIVE_METADATA_FILES: [PinnedMetadataFile; 5] = [
    PinnedMetadataFile {
        name: "LICENSE",
        size: 3_065,
        sha256: digest([
            0x20, 0xc7, 0x97, 0xce, 0x19, 0xaf, 0x0c, 0x17, 0xde, 0x52, 0xc6, 0xaf, 0xb1, 0x44,
            0x64, 0x47, 0x68, 0xa5, 0x91, 0xc5, 0x21, 0x65, 0x5f, 0x5e, 0xbf, 0x57, 0x12, 0xc9,
            0x85, 0x0f, 0x28, 0x87,
        ]),
    },
    PinnedMetadataFile {
        name: "config.json",
        size: 7_006,
        sha256: digest([
            0x97, 0x10, 0xe1, 0x21, 0xa5, 0x8d, 0x03, 0xac, 0x92, 0xc8, 0xd6, 0xda, 0x28, 0x7a,
            0x19, 0x54, 0x19, 0x94, 0x31, 0x9a, 0xfb, 0xbe, 0x6d, 0x62, 0x02, 0xaf, 0x00, 0x1f,
            0xfd, 0x37, 0x92, 0x13,
        ]),
    },
    PinnedMetadataFile {
        name: "generation_config.json",
        size: 53,
        sha256: digest([
            0xc6, 0x64, 0x8c, 0x25, 0xe9, 0x70, 0x5a, 0xf7, 0xfb, 0xa8, 0x84, 0x7e, 0x24, 0x38,
            0x40, 0xd2, 0x1b, 0x5c, 0xc6, 0x3d, 0xde, 0xb6, 0x29, 0x7f, 0x75, 0x0a, 0x7d, 0xdb,
            0xb6, 0xa0, 0x28, 0x36,
        ]),
    },
    PinnedMetadataFile {
        name: "tokenizer_config.json",
        size: 3_478,
        sha256: digest([
            0x5d, 0x08, 0x03, 0xc9, 0x4d, 0xb9, 0xcd, 0x78, 0x76, 0x34, 0x99, 0xe0, 0x95, 0x6c,
            0x95, 0xfd, 0x5a, 0x22, 0x5c, 0x14, 0xa7, 0x27, 0xe5, 0xa6, 0xcf, 0x5d, 0xb3, 0xf9,
            0x6f, 0x01, 0x0a, 0x6e,
        ]),
    },
    PinnedMetadataFile {
        name: "tiktoken.model",
        size: 2_795_286,
        sha256: digest([
            0xb6, 0xc4, 0x97, 0xa7, 0x46, 0x9b, 0x33, 0xce, 0xd9, 0xc3, 0x8a, 0xfb, 0x1a, 0xd6,
            0xe4, 0x7f, 0x03, 0xf5, 0xe5, 0xdc, 0x05, 0xf1, 0x59, 0x30, 0x79, 0x92, 0x10, 0xec,
            0x05, 0x0c, 0x51, 0x03,
        ]),
    },
];

const fn digest(bytes: [u8; 32]) -> Digest {
    bytes
}

pub fn base_url() -> Result<String> {
    let host = std::env::var("K3_HF_HOST").unwrap_or_else(|_| DEFAULT_HF_HOST.into());
    let path = std::env::var("K3_HF_PATH").unwrap_or_else(|_| DEFAULT_HF_PATH.into());
    base_url_from(&host, &path)
}

pub fn base_url_from(host: &str, path: &str) -> Result<String> {
    let host = host.trim();
    if host.is_empty()
        || host.bytes().any(|byte| {
            byte.is_ascii_whitespace()
                || byte.is_ascii_control()
                || matches!(byte, b'/' | b'\\' | b'?' | b'#' | b'@')
        })
    {
        return Err(DeltafinError::new(
            "K3_HF_HOST must be one HTTPS host with no scheme, path, userinfo, or control characters",
        ));
    }
    let mut path = path.trim().to_owned();
    if path
        .bytes()
        .any(|byte| byte.is_ascii_control() || matches!(byte, b'\\' | b'?' | b'#'))
    {
        return Err(DeltafinError::new(
            "K3_HF_PATH must be an absolute URL path without query, fragment, backslash, or control characters",
        ));
    }
    if !path.starts_with('/') {
        path.insert(0, '/');
    }
    if !path.ends_with('/') {
        path.push('/');
    }
    if path
        .split('/')
        .any(|component| matches!(component, "." | ".."))
    {
        return Err(DeltafinError::new(
            "K3_HF_PATH cannot contain dot path components",
        ));
    }
    Ok(format!("https://{host}{path}"))
}

pub fn metadata_url(base: &str, name: &str) -> Result<String> {
    if !NATIVE_METADATA_FILES.iter().any(|spec| spec.name == name) {
        return Err(DeltafinError::new(format!(
            "metadata file {name:?} is outside the native K3 allowlist"
        )));
    }
    Ok(format!("{base}{name}?download=true"))
}

pub fn shard_name(one_based: usize) -> Result<String> {
    if !(1..=crate::inventory::SHARD_COUNT).contains(&one_based) {
        return Err(DeltafinError::new(format!(
            "K3 shard number {one_based} is outside 1..={} ",
            crate::inventory::SHARD_COUNT
        )));
    }
    Ok(format!("model-{one_based:05}-of-000096.safetensors"))
}

pub fn shard_url(base: &str, one_based: usize) -> Result<String> {
    Ok(format!("{base}{}?download=true", shard_name(one_based)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn immutable_default_and_explicit_mirror_are_https_only() {
        assert_eq!(
            base_url_from(DEFAULT_HF_HOST, DEFAULT_HF_PATH).unwrap(),
            format!("https://{DEFAULT_HF_HOST}{DEFAULT_HF_PATH}")
        );
        assert_eq!(
            base_url_from("mirror.example:8443", "private/k3").unwrap(),
            "https://mirror.example:8443/private/k3/"
        );
        for (host, path) in [
            ("https://evil.example", "/x"),
            ("user@evil.example", "/x"),
            ("evil.example/path", "/x"),
            ("good.example", "/../x"),
            ("good.example", "/x?token=secret"),
            ("good.example\n--output", "/x"),
        ] {
            assert!(
                base_url_from(host, path).is_err(),
                "accepted {host:?} {path:?}"
            );
        }
    }

    #[test]
    fn metadata_and_shard_names_are_fixed_allowlists() {
        let base = base_url_from(DEFAULT_HF_HOST, DEFAULT_HF_PATH).unwrap();
        assert!(
            metadata_url(&base, "config.json")
                .unwrap()
                .contains(MODEL_REVISION)
        );
        assert!(metadata_url(&base, "modeling_kimi_k3.py").is_err());
        assert_eq!(shard_name(96).unwrap(), "model-00096-of-000096.safetensors");
        assert!(shard_name(0).is_err());
        assert!(shard_name(97).is_err());
    }

    #[test]
    #[ignore = "reads the installed native K3 metadata files"]
    fn installed_native_metadata_matches_every_pin() {
        let metadata = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../k3-meta");
        for spec in NATIVE_METADATA_FILES {
            let raw = std::fs::read(metadata.join(spec.name)).unwrap();
            assert_eq!(raw.len() as u64, spec.size, "{} size", spec.name);
            assert_eq!(
                crate::packfile::digest_bytes(&raw),
                spec.sha256,
                "{} SHA",
                spec.name
            );
        }
    }
}
