mod build_support;

use build_support::{
    ensure_native_build, ensure_supported_features, find_validated_linux_lib, install_linker_name,
    parse_supported_target, Target,
};
use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build_support.rs");
    println!(
        "cargo:rustc-check-cfg=cfg(\
            libcurl_vendored,\
            link_libnghttp2,\
            link_libz,\
            link_openssl,\
        )"
    );

    ensure_supported_features(&[
        ("http2", cfg!(feature = "http2")),
        ("mesalink", cfg!(feature = "mesalink")),
        ("ntlm", cfg!(feature = "ntlm")),
        ("poll_7_68_0", cfg!(feature = "poll_7_68_0")),
        ("protocol-ftp", cfg!(feature = "protocol-ftp")),
        ("rustls", cfg!(feature = "rustls")),
        ("spnego", cfg!(feature = "spnego")),
        ("static-curl", cfg!(feature = "static-curl")),
        ("static-ssl", cfg!(feature = "static-ssl")),
        ("upkeep_7_62_0", cfg!(feature = "upkeep_7_62_0")),
        ("windows-static-ssl", cfg!(feature = "windows-static-ssl")),
        ("zlib-ng-compat", cfg!(feature = "zlib-ng-compat")),
    ])
    .unwrap_or_else(|message| panic!("{}", message));

    let target_text = env::var("TARGET").expect("Cargo did not provide TARGET");
    let host_text = env::var("HOST").expect("Cargo did not provide HOST");
    ensure_native_build(&host_text, &target_text).unwrap_or_else(|message| panic!("{}", message));
    match parse_supported_target(&target_text).unwrap_or_else(|message| panic!("{}", message)) {
        Target::MacArm64 => {
            // Apple supplies libcurl in the platform SDK. Let rustc's selected
            // Apple linker resolve that SDK library; do not invoke xcrun,
            // curl-config, pkg-config, a shell, or any other helper here.
            println!("cargo:rustc-link-lib=dylib=curl");
        }
        Target::Linux(machine) => {
            let (library, discovery_path) =
                find_validated_linux_lib(machine).unwrap_or_else(|message| panic!("{}", message));
            println!("cargo:rerun-if-changed={}", discovery_path.display());
            println!("cargo:rerun-if-changed={}", library.display());

            let out_dir = PathBuf::from(
                env::var_os("OUT_DIR").expect("Cargo did not provide a valid OUT_DIR"),
            );
            let link_dir = install_linker_name(&out_dir, &library)
                .unwrap_or_else(|message| panic!("{}", message));
            let printable = link_dir.to_str().unwrap_or_else(|| {
                panic!("Cargo OUT_DIR is not valid UTF-8 and cannot be emitted safely")
            });
            if printable.contains(['\n', '\r']) {
                panic!("Cargo OUT_DIR contains a metadata line break");
            }
            println!("cargo:rustc-link-search=native={printable}");
            println!("cargo:rustc-link-lib=dylib=curl");
        }
    }
}
