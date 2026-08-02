//! Release policy for Deltafin's local, process-free `curl-sys` build fork.
//!
//! Keep this separate from the broader release-policy test because the native
//! build graph is frequently edited in parallel.  These checks are static: a
//! release audit must not need to run a helper program to decide which
//! `libcurl` build machinery entered the locked product graph.

use std::fs;
use std::path::{Path, PathBuf};

const FORK: &str = "native/deltafin-curl-sys-direct";
const UPSTREAM_BINDINGS_SHA256: &str =
    "39a88d4ba414b16656681707c5f1dc418c6327d9e81e61126ff4944e22ae2230";
const UPSTREAM_LICENSE_SHA256: &str =
    "f96def8cba2793fb8582fd12ca6d4dc0ef4ee239e8c3f80e809ec43648da6199";

fn repository_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("native/deltafin must remain two levels below the repository")
        .to_path_buf()
}

fn read_text(path: &Path) -> String {
    fs::read_to_string(path).unwrap_or_else(|error| {
        panic!("read curl release-policy input {}: {error}", path.display())
    })
}

fn read_bytes(path: &Path) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        panic!("read curl release-policy input {}: {error}", path.display())
    })
}

/// Remove TOML comments without treating a `#` inside a quoted string as one.
fn toml_without_comments(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    for line in source.lines() {
        let mut escaped = false;
        let mut quoted = false;
        for character in line.chars() {
            if character == '#' && !quoted {
                break;
            }
            output.push(character);
            if character == '"' && !escaped {
                quoted = !quoted;
            }
            escaped = character == '\\' && !escaped;
            if character != '\\' {
                escaped = false;
            }
        }
        output.push('\n');
    }
    output
}

fn toml_table<'a>(source: &'a str, name: &str) -> Option<&'a str> {
    let header = format!("[{name}]");
    let start = source.find(&header)? + header.len();
    let tail = &source[start..];
    let end = tail
        .match_indices('\n')
        .find_map(|(offset, _)| {
            tail[offset + 1..]
                .trim_start()
                .starts_with('[')
                .then_some(offset)
        })
        .unwrap_or(tail.len());
    Some(&tail[..end])
}

fn toml_assignment<'a>(table: &'a str, wanted: &str) -> Option<&'a str> {
    let mut offset = 0;
    while offset < table.len() {
        let line_end = table[offset..]
            .find('\n')
            .map_or(table.len(), |relative| offset + relative);
        let line = table[offset..line_end].trim();
        let Some((key, initial)) = line.split_once('=') else {
            offset = (line_end + 1).min(table.len());
            continue;
        };
        if key.trim() != wanted {
            offset = (line_end + 1).min(table.len());
            continue;
        }

        let value_start = table[offset..line_end]
            .find('=')
            .expect("assignment line contains equals")
            + offset
            + 1;
        let mut square_depth = 0_i32;
        let mut brace_depth = 0_i32;
        let mut quoted = false;
        let mut escaped = false;
        let mut end = value_start;
        for (relative, character) in table[value_start..].char_indices() {
            if character == '"' && !escaped {
                quoted = !quoted;
            }
            if !quoted {
                match character {
                    '[' => square_depth += 1,
                    ']' => square_depth -= 1,
                    '{' => brace_depth += 1,
                    '}' => brace_depth -= 1,
                    '\n' if square_depth == 0 && brace_depth == 0 => {
                        end = value_start + relative;
                        break;
                    }
                    _ => {}
                }
            }
            escaped = character == '\\' && !escaped;
            if character != '\\' {
                escaped = false;
            }
            end = value_start + relative + character.len_utf8();
        }
        let value = table[value_start..end].trim();
        return (!value.is_empty() || !initial.trim().is_empty()).then_some(value);
    }
    None
}

fn toml_strings(value: &str) -> Vec<String> {
    let mut strings = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut escaped = false;
    for character in value.chars() {
        if quoted {
            if character == '"' && !escaped {
                strings.push(std::mem::take(&mut current));
                quoted = false;
            } else {
                current.push(character);
            }
            escaped = character == '\\' && !escaped;
            if character != '\\' {
                escaped = false;
            }
        } else if character == '"' {
            quoted = true;
            escaped = false;
        }
    }
    strings
}

fn inline_string_field(value: &str, field: &str) -> Option<String> {
    let body = value.trim().strip_prefix('{')?.strip_suffix('}')?;
    for part in body.split(',') {
        let (key, value) = part.split_once('=')?;
        if key.trim() == field {
            return toml_strings(value).into_iter().next();
        }
    }
    None
}

#[derive(Debug)]
struct LockedPackage {
    name: String,
    version: String,
    source: Option<String>,
    checksum: Option<String>,
    dependencies: Vec<String>,
}

fn locked_packages(lock: &str) -> Vec<LockedPackage> {
    lock.split("[[package]]")
        .skip(1)
        .map(|block| {
            let clean = toml_without_comments(block);
            let scalar = |name: &str| {
                toml_assignment(&clean, name)
                    .and_then(|value| toml_strings(value).into_iter().next())
            };
            LockedPackage {
                name: scalar("name").expect("locked package has a name"),
                version: scalar("version").expect("locked package has a version"),
                source: scalar("source"),
                checksum: scalar("checksum"),
                dependencies: toml_assignment(&clean, "dependencies")
                    .map(toml_strings)
                    .unwrap_or_default(),
            }
        })
        .collect()
}

/// Strip Rust comments and literals before looking for executable build edges.
/// This deliberately keeps punctuation and identifiers, then callers remove
/// whitespace so ordinary formatting changes do not weaken the check.
fn rust_code_only(source: &str) -> String {
    #[derive(Clone, Copy)]
    enum State {
        Code,
        LineComment,
        BlockComment(usize),
        String,
        RawString(usize),
    }

    let bytes = source.as_bytes();
    let mut output = String::with_capacity(source.len());
    let mut state = State::Code;
    let mut index = 0;
    while index < bytes.len() {
        match state {
            State::Code if bytes[index..].starts_with(b"//") => {
                state = State::LineComment;
                output.push(' ');
                index += 2;
            }
            State::Code if bytes[index..].starts_with(b"/*") => {
                state = State::BlockComment(1);
                output.push(' ');
                index += 2;
            }
            State::Code if bytes[index] == b'"' => {
                state = State::String;
                output.push(' ');
                index += 1;
            }
            State::Code if bytes[index] == b'r' => {
                let mut cursor = index + 1;
                while cursor < bytes.len() && bytes[cursor] == b'#' {
                    cursor += 1;
                }
                if cursor < bytes.len() && bytes[cursor] == b'"' {
                    state = State::RawString(cursor - index - 1);
                    output.push(' ');
                    index = cursor + 1;
                } else {
                    output.push('r');
                    index += 1;
                }
            }
            State::Code => {
                output.push(bytes[index] as char);
                index += 1;
            }
            State::LineComment if bytes[index] == b'\n' => {
                state = State::Code;
                output.push('\n');
                index += 1;
            }
            State::LineComment => index += 1,
            State::BlockComment(depth) if bytes[index..].starts_with(b"/*") => {
                state = State::BlockComment(depth + 1);
                index += 2;
            }
            State::BlockComment(depth) if bytes[index..].starts_with(b"*/") => {
                state = if depth == 1 {
                    State::Code
                } else {
                    State::BlockComment(depth - 1)
                };
                index += 2;
            }
            State::BlockComment(_) => index += 1,
            State::String if bytes[index] == b'\\' => index = (index + 2).min(bytes.len()),
            State::String if bytes[index] == b'"' => {
                state = State::Code;
                index += 1;
            }
            State::String => index += 1,
            State::RawString(hashes) if bytes[index] == b'"' => {
                let end = index + 1 + hashes;
                if end <= bytes.len() && bytes[index + 1..end].iter().all(|byte| *byte == b'#') {
                    state = State::Code;
                    index = end;
                } else {
                    index += 1;
                }
            }
            State::RawString(_) => index += 1,
        }
    }
    output
}

fn compact(source: &str) -> String {
    source
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect()
}

fn rust_u32_const(source: &str, name: &str) -> Option<u32> {
    let code = compact(&rust_code_only(source));
    let marker = format!("const{name}:u32=");
    let value = code
        .split_once(&marker)?
        .1
        .split_once(';')?
        .0
        .replace('_', "");
    value
        .strip_prefix("0x")
        .and_then(|hex| u32::from_str_radix(hex, 16).ok())
        .or_else(|| value.parse().ok())
}

fn sha256(input: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    const ROUND: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let bit_length = (input.len() as u64).wrapping_mul(8);
    let padded_length = (input.len() + 9).div_ceil(64) * 64;
    let mut message = Vec::with_capacity(padded_length);
    message.extend_from_slice(input);
    message.push(0x80);
    message.resize(padded_length - 8, 0);
    message.extend_from_slice(&bit_length.to_be_bytes());

    let mut hash = INITIAL;
    for block in message.chunks_exact(64) {
        let mut words = [0_u32; 64];
        for (index, word) in words[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes(
                block[index * 4..index * 4 + 4]
                    .try_into()
                    .expect("SHA-256 word is four bytes"),
            );
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = hash;
        for index in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choice = (e & f) ^ ((!e) & g);
            let first = h
                .wrapping_add(s1)
                .wrapping_add(choice)
                .wrapping_add(ROUND[index])
                .wrapping_add(words[index]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let second = s0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(first);
            d = c;
            c = b;
            b = a;
            a = first.wrapping_add(second);
        }
        for (slot, value) in hash.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }

    let mut bytes = [0_u8; 32];
    for (destination, word) in bytes.chunks_exact_mut(4).zip(hash) {
        destination.copy_from_slice(&word.to_be_bytes());
    }
    bytes
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write;
        write!(&mut output, "{byte:02x}").expect("write into String");
    }
    output
}

#[test]
fn workspace_selects_only_the_reviewed_nonpublishable_local_fork() {
    let root = repository_root();
    let workspace = toml_without_comments(&read_text(&root.join("Cargo.toml")));
    let workspace_table = toml_table(&workspace, "workspace").expect("root workspace table");
    let members = toml_strings(
        toml_assignment(workspace_table, "members").expect("workspace members assignment"),
    );
    assert!(
        members.iter().any(|member| member == FORK),
        "the reviewed curl-sys fork is not a workspace member"
    );

    let patch = toml_table(&workspace, "patch.crates-io").expect("crates.io patch table");
    let curl_patch = toml_assignment(patch, "curl-sys").expect("curl-sys crates.io patch");
    assert_eq!(
        inline_string_field(curl_patch, "path").as_deref(),
        Some(FORK)
    );

    let manifest = toml_without_comments(&read_text(&root.join(FORK).join("Cargo.toml")));
    let package = toml_table(&manifest, "package").expect("fork package table");
    assert_eq!(
        toml_assignment(package, "publish").map(str::trim),
        Some("false"),
        "the local ABI/build fork must never be published as upstream curl-sys"
    );
}

#[test]
fn fork_build_has_no_process_or_helper_dependency_edge() {
    let root = repository_root().join(FORK);
    let manifest = toml_without_comments(&read_text(&root.join("Cargo.toml")));
    assert!(
        toml_table(&manifest, "build-dependencies")
            .is_none_or(|table| !table.lines().any(|line| line.contains('='))),
        "the direct curl-sys fork must retain an empty build-dependency closure"
    );

    for relative in ["build.rs", "build_support.rs"] {
        let code = compact(&rust_code_only(&read_text(&root.join(relative))));
        for forbidden in [
            "std::process",
            "process::Command",
            "Command::new(",
            "pkg_config",
            "vcpkg",
            "cmake",
            "cc::",
        ] {
            assert!(
                !code.contains(forbidden),
                "{relative} contains a forbidden build-helper edge {forbidden:?}"
            );
        }
    }
}

#[test]
fn upstream_ffi_and_license_bytes_remain_exact() {
    assert_eq!(
        hex(&sha256(b"abc")),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "release-policy SHA-256 implementation failed its standard vector"
    );
    let fork = repository_root().join(FORK);
    assert_eq!(
        hex(&sha256(&read_bytes(&fork.join("lib.rs")))),
        UPSTREAM_BINDINGS_SHA256,
        "curl-sys 0.4.90+curl-8.21.0 FFI bindings diverged from upstream"
    );
    assert_eq!(
        hex(&sha256(&read_bytes(&fork.join("LICENSE")))),
        UPSTREAM_LICENSE_SHA256,
        "curl-sys 0.4.90+curl-8.21.0 license diverged from upstream"
    );
}

#[test]
fn every_native_download_caller_retains_runtime_version_tls_and_https_gates() {
    let root = repository_root();
    for relative in [
        "native/deltafin/src/trusted_download.rs",
        "native/deltafin-bootstrap/src/lib.rs",
    ] {
        let source = read_text(&root.join(relative));
        assert_eq!(
            rust_u32_const(&source, "MINIMUM_LIBCURL_VERSION"),
            Some(0x07_1c_00),
            "{relative} no longer requires libcurl 7.28.0 (curl_multi_wait)"
        );
        let normalized = compact(&source);
        for required in [
            "version.version_num()<MINIMUM_LIBCURL_VERSION",
            "!version.feature_ssl()",
            ".protocols()",
            ".eq_ignore_ascii_case(\"https\")",
            "require_https_capable_libcurl()?;",
        ] {
            assert!(
                normalized.contains(required),
                "{relative} lost runtime libcurl capability gate {required:?}"
            );
        }
    }
}

#[test]
fn lockfile_closes_curl_sys_over_only_libc() {
    let packages = locked_packages(&read_text(&repository_root().join("Cargo.lock")));
    let curl_sys: Vec<_> = packages
        .iter()
        .filter(|package| package.name == "curl-sys")
        .collect();
    assert_eq!(curl_sys.len(), 1, "locked graph must contain one curl-sys");
    let curl_sys = curl_sys[0];
    assert_eq!(curl_sys.version, "0.4.90+curl-8.21.0");
    assert_eq!(
        curl_sys.source, None,
        "curl-sys must resolve to the local patch"
    );
    assert_eq!(
        curl_sys.checksum, None,
        "a local path package has no registry checksum"
    );
    assert_eq!(curl_sys.dependencies, ["libc"]);

    for forbidden in ["libz-sys", "vcpkg"] {
        assert!(
            packages.iter().all(|package| package.name != forbidden),
            "generic curl-sys helper dependency {forbidden:?} re-entered Cargo.lock"
        );
    }
}
