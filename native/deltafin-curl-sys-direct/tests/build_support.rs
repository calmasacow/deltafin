#[path = "../build_support.rs"]
#[allow(dead_code)]
mod build_support;

use std::ffi::CStr;
use std::fs;
use std::path::PathBuf;

#[test]
fn build_path_has_no_child_process_or_build_dependency_escape_hatch() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for relative in ["build.rs", "build_support.rs"] {
        let source = fs::read_to_string(root.join(relative)).expect("read build source");
        for forbidden in [
            "std::process::Command",
            "process::Command",
            "Command::new(",
            "Command::new (",
            "command(\"sh\")",
            "command(\"bash\")",
        ] {
            assert!(
                !source.contains(forbidden),
                "{} contains forbidden child-process edge {:?}",
                relative,
                forbidden
            );
        }
    }
    let manifest = fs::read_to_string(root.join("Cargo.toml")).expect("read Cargo manifest");
    assert!(
        !manifest.contains("build-dependencies"),
        "direct curl-sys fork must retain an empty build-dependency closure"
    );
    for forbidden in ["pkg-config", "vcpkg", "cmake", "cc =", "cc = {"] {
        assert!(
            !manifest.contains(forbidden),
            "direct curl-sys manifest contains forbidden helper dependency {:?}",
            forbidden
        );
    }
}

#[test]
fn exact_feature_vocabulary_remains_compatible_with_curl_0_4_50() {
    let manifest = fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .expect("read Cargo manifest");
    for feature in [
        "force-system-lib-on-osx",
        "http2",
        "mesalink",
        "ntlm",
        "poll_7_68_0",
        "protocol-ftp",
        "rustls",
        "spnego",
        "ssl",
        "static-curl",
        "static-ssl",
        "upkeep_7_62_0",
        "windows-static-ssl",
        "zlib-ng-compat",
    ] {
        assert!(
            manifest.lines().any(|line| {
                line.trim_start()
                    .strip_prefix(feature)
                    .is_some_and(|tail| tail.trim_start().starts_with('='))
            }),
            "missing curl 0.4.50 forwarded feature {:?}",
            feature
        );
    }
}

#[test]
fn linked_host_libcurl_supports_the_runtime_download_contract() {
    unsafe {
        let version = curl_sys::curl_version();
        assert!(!version.is_null(), "curl_version returned null");
        assert!(
            !CStr::from_ptr(version).to_bytes().is_empty(),
            "curl_version returned an empty version"
        );

        let info = curl_sys::curl_version_info(curl_sys::CURLVERSION_NOW);
        assert!(!info.is_null(), "curl_version_info returned null");
        assert_ne!(
            (*info).features & curl_sys::CURL_VERSION_SSL,
            0,
            "system libcurl has no TLS backend"
        );
        let protocols = (*info).protocols;
        assert!(!protocols.is_null(), "system libcurl protocol list is null");
        let mut has_https = false;
        for index in 0..256 {
            let protocol = *protocols.add(index);
            if protocol.is_null() {
                break;
            }
            if CStr::from_ptr(protocol).to_bytes() == b"https" {
                has_https = true;
                break;
            }
        }
        assert!(has_https, "system libcurl has no HTTPS protocol support");

        assert_eq!(
            curl_sys::curl_global_init(curl_sys::CURL_GLOBAL_DEFAULT),
            curl_sys::CURLE_OK,
            "curl_global_init failed"
        );
        let easy = curl_sys::curl_easy_init();
        assert!(!easy.is_null(), "curl_easy_init returned null");
        curl_sys::curl_easy_cleanup(easy);
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
#[test]
fn real_linux_x86_64_library_passes_bounded_discovery() {
    let (library, discovery_path) =
        build_support::find_validated_linux_lib(build_support::LinuxMachine::X86_64)
            .expect("find and validate host x86-64 libcurl");
    assert!(library.is_absolute());
    assert!(discovery_path.is_absolute());
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
#[test]
fn real_linux_aarch64_library_passes_bounded_discovery() {
    let (library, discovery_path) =
        build_support::find_validated_linux_lib(build_support::LinuxMachine::Aarch64)
            .expect("find and validate host aarch64 libcurl");
    assert!(library.is_absolute());
    assert!(discovery_path.is_absolute());
}
