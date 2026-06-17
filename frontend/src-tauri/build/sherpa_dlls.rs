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
    println!("cargo:rerun-if-changed={}", dest_dir.display());
    if let Ok(target) = std::env::var("TARGET") {
        let profile = std::env::var("PROFILE").unwrap_or_else(|_| "release".into());
        if let Some(release_dir) = target_dir_from_manifest(&manifest_dir, &target, &profile) {
            println!("cargo:rerun-if-changed={}", release_dir.display());
        }
    }
    if let Err(err) = try_stage(&manifest_dir, &dest_dir) {
        println!("cargo:warning=Sherpa DLL staging failed: {}", err);
    }
}

fn target_dir_from_manifest(manifest_dir: &Path, target: &str, profile: &str) -> Option<PathBuf> {
    let workspace_target = manifest_dir.join("..").join("..").join("target").join(target).join(profile);
    if workspace_target.is_dir() { return Some(workspace_target); }
    let crate_target = manifest_dir.join("target").join(target).join(profile);
    if crate_target.is_dir() { Some(crate_target) } else { None }
}


fn try_stage(manifest_dir: &Path, dest_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(dest_dir).map_err(|e| e.to_string())?;
    for dll in collect_dll_sources(manifest_dir)? {
        let name = dll.file_name().ok_or_else(|| format!("invalid DLL path: {}", dll.display()))?;
        let dest = dest_dir.join(name);
        fs::copy(&dll, &dest).map_err(|e| format!("copy failed: {}", e))?;
    }
    println!("cargo:warning=Staged runtime DLL(s)");
    Ok(())
}

fn collect_dll_sources(manifest_dir: &Path) -> Result<Vec<PathBuf>, String> {
    if let Some(dir) = env_lib_dir() {
        match dlls_in_dir(&dir) {
            Ok(v) if !v.is_empty() => return Ok(v),
            _ => {}
        }
    }
    if let Ok(target) = std::env::var("TARGET") {
        let profile = std::env::var("PROFILE").unwrap_or_else(|_| "release".into());
        if let Some(release_dir) = target_dir_from_manifest(manifest_dir, &target, &profile) {
            match sherpa_dlls_in_dir(&release_dir) {
                Ok(v) if !v.is_empty() => return Ok(v),
                _ => {}
            }
        }
        if let Some(prebuilt_lib) = sherpa_prebuilt_lib_dir(manifest_dir, &target) {
            match dlls_in_dir(&prebuilt_lib) {
                Ok(v) if !v.is_empty() => return Ok(v),
                _ => {}
            }
        }
    }
    download_prebuilt_dlls(manifest_dir)
}
fn env_lib_dir() -> Option<PathBuf> {
    let path = std::env::var_os("SHERPA_ONNX_LIB_DIR")?;
    let path = PathBuf::from(path);
    if path.is_dir() { Some(path) } else { None }
}

fn sherpa_prebuilt_lib_dir(manifest_dir: &Path, target: &str) -> Option<PathBuf> {
    let target_root = target_dir_from_manifest(manifest_dir, target, "release")
        .or_else(|| target_dir_from_manifest(manifest_dir, target, "debug"))?;
    let cache_root = target_root.join("sherpa-onnx-prebuilt");
    if !cache_root.is_dir() { return None; }
    let archive_stem = format!("sherpa-onnx-v{}-win-x64-shared-MT-Release-lib", SHERPA_ONNX_VERSION);
    let lib_dir = cache_root.join(&archive_stem).join("lib");
    if lib_dir.is_dir() { return Some(lib_dir); }
    let mut found = None;
    if let Ok(entries) = fs::read_dir(&cache_root) {
        for entry in entries.flatten() {
            let candidate = entry.path().join("lib");
            if candidate.is_dir() && dlls_in_dir(&candidate).map(|d| !d.is_empty()).unwrap_or(false) {
                found = Some(candidate);
                break;
            }
        }
    }
    found
}
fn sherpa_dlls_in_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    Ok(dlls_in_dir(dir)?.into_iter().filter(|p| is_sherpa_runtime_dll(p)).collect())
}

fn is_sherpa_runtime_dll(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else { return false; };
    let lower = name.to_ascii_lowercase();
    if !lower.ends_with(".dll") { return false; }
    lower.contains("sherpa") || lower.contains("onnxruntime") || lower.starts_with("kaldi")
        || lower.contains("kaldifst") || lower.contains("fst") || lower.contains("fbank")
        || lower.contains("kissfft") || lower.contains("piper") || lower.contains("espeak")
        || lower.contains("ssentencepiece") || lower.contains("ucd")
}

fn dlls_in_dir(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_file() && path.extension().map(|e| e.eq_ignore_ascii_case("dll")).unwrap_or(false) {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}

fn download_prebuilt_dlls(manifest_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let archive_name = format!(
        "sherpa-onnx-v{}-win-x64-shared-MT-Release-lib.tar.bz2",
        SHERPA_ONNX_VERSION
    );
    if let Some(local_dir) = std::env::var_os("SHERPA_ONNX_ARCHIVE_DIR") {
        let local_archive = PathBuf::from(local_dir).join(&archive_name);
        if local_archive.is_file() {
            return extract_dlls_from_archive(&local_archive, manifest_dir);
        }
    }
    let url = format!("{RELEASE_BASE_URL}/v{SHERPA_ONNX_VERSION}/{archive_name}");
    println!("cargo:warning=Downloading runtime DLL archive from {url}");
    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
        .map_err(|e| e.to_string())?;
    let response = client.get(&url).send().map_err(|e| format!("download failed: {e}"))?;
    if !response.status().is_success() {
        return Err(format!("HTTP {} for {url}", response.status()));
    }
    let temp_dir = std::env::temp_dir().join(format!("meetily-sherpa-dlls-{}", std::process::id()));
    fs::create_dir_all(&temp_dir).map_err(|e| e.to_string())?;
    let archive_path = temp_dir.join(&archive_name);
    {
        let mut file = fs::File::create(&archive_path).map_err(|e| e.to_string())?;
        let bytes = response.bytes().map_err(|e| e.to_string())?;
        file.write_all(&bytes).map_err(|e| e.to_string())?;
    }
    extract_dlls_from_archive(&archive_path, manifest_dir)
}

fn extract_dlls_from_archive(archive_path: &Path, manifest_dir: &Path) -> Result<Vec<PathBuf>, String> {
    use bzip2::read::BzDecoder;
    let extract_root = std::env::temp_dir().join(format!("meetily-sherpa-extract-{}", std::process::id()));
    if extract_root.exists() { let _ = fs::remove_dir_all(&extract_root); }
    fs::create_dir_all(&extract_root).map_err(|e| e.to_string())?;
    let tar_file = fs::File::open(archive_path).map_err(|e| e.to_string())?;
    let decoder = BzDecoder::new(tar_file);
    let mut archive = tar::Archive::new(decoder);
    archive.unpack(&extract_root).map_err(|e| format!("unpack failed: {e}"))?;
    let lib_dir = find_lib_dir(&extract_root).ok_or_else(|| format!("no lib directory in archive {}", archive_path.display()))?;
    match dlls_in_dir(&lib_dir) {
        Ok(v) if v.is_empty() => return Err("archive contained no DLL files".into()),
        Ok(v) => {
            if let Ok(target) = std::env::var("TARGET") {
                if let Some(target_root) = target_dir_from_manifest(manifest_dir, &target, "release") {
                    let cache_lib = target_root.join("sherpa-onnx-prebuilt").join(format!("sherpa-onnx-v{}-win-x64-shared-MT-Release-lib", SHERPA_ONNX_VERSION)).join("lib");
                    let _ = copy_dll_tree(&lib_dir, &cache_lib);
                }
            }
            Ok(v)
        }
        Err(e) => Err(e),
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
fn copy_dll_tree(src_lib: &Path, dest_lib: &Path) -> Result<(), String> {
    fs::create_dir_all(dest_lib).map_err(|e| e.to_string())?;
    for dll in dlls_in_dir(src_lib)? {
        let name = dll.file_name().ok_or_else(|| "bad dll path".to_string())?;
        fs::copy(&dll, dest_lib.join(name)).map_err(|e| e.to_string())?;
    }
    Ok(())
}

// staged for tauri resources and nsis hooks
