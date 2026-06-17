use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

const SHERPA_ONNX_VERSION: &str = "1.13.3";
const RELEASE_BASE_URL: &str = "https://github.com/k2-fsa/sherpa-onnx/releases/download";

pub fn stage_sherpa_dlls_for_bundle() {
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os != "windows" { return; }
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let dest_dir = manifest_dir.join("sherpa-dlls");
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=SHERPA_ONNX_ARCHIVE_DIR");
    println!("cargo:rerun-if-changed={}", dest_dir.display());        Ok(v)
    }
}

fn find_lib_dir(root: &Path) -> Option<PathBuf> {
    let direct = root.join("lib");
    if direct.is_dir() { return Some(direct); }
    for entry in fs::read_dir(root).ok()? {
        let entry = entry.ok()?;
        let path = entry.path();
        if path.is_dir() {
            let lib = path.join("lib");
            if lib.is_dir() { return Some(lib); }
        }
    }
    None
}
