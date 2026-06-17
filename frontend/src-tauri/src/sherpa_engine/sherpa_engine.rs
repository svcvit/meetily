use anyhow::{anyhow, Result};
use sherpa_onnx::{OfflineRecognizer, OfflineRecognizerConfig, OfflineSenseVoiceModelConfig};
use std::path::Path;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tokio::io::{AsyncWriteExt, BufWriter};
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tokio::time::timeout;

/// Quantization type for Sherpa models
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QuantizationType {
    FP32,   // Full precision
    Int8,   // 8-bit integer quantization (faster)
}

impl Default for QuantizationType {
    fn default() -> Self {
        QuantizationType::Int8 // Default to int8 for best performance
    }
}

/// Model status for Sherpa models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Available,
    Missing,
    Downloading { progress: u8 },
    Error(String),
    Corrupted { file_size: u64, expected_min_size: u64 },
}

/// Detailed download progress info (MB-based with speed)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgress {
    /// Bytes downloaded so far
    pub downloaded_bytes: u64,
    /// Total file size in bytes
    pub total_bytes: u64,
    /// Downloaded in MB (for display)
    pub downloaded_mb: f64,
    /// Total size in MB (for display)
    pub total_mb: f64,
    /// Download speed in MB/s
    pub speed_mbps: f64,
    /// Percentage complete (0-100)
    pub percent: u8,
}

impl DownloadProgress {
    pub fn new(downloaded: u64, total: u64, speed_mbps: f64) -> Self {
        let percent = if total > 0 {
            ((downloaded as f64 / total as f64) * 100.0).min(100.0) as u8
        } else {
            0
        };
        Self {
            downloaded_bytes: downloaded,
            total_bytes: total,
            downloaded_mb: downloaded as f64 / (1024.0 * 1024.0),
            total_mb: total as f64 / (1024.0 * 1024.0),
            speed_mbps,
            percent,
        }
    }
}

/// Information about a Sherpa model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub name: String,
    pub path: PathBuf,
    pub size_mb: u32,
    pub quantization: QuantizationType,
    pub speed: String,     // Performance description
    pub status: ModelStatus,
    pub description: String,
}

#[derive(Debug)]
pub enum SherpaEngineError {
    ModelNotLoaded,
    ModelNotFound(String),
    TranscriptionFailed(String),
    DownloadFailed(String),
    IoError(std::io::Error),
    Other(String),
}

impl std::fmt::Display for SherpaEngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SherpaEngineError::ModelNotLoaded => write!(f, "No Sherpa model loaded"),
            SherpaEngineError::ModelNotFound(name) => write!(f, "Model '{}' not found", name),
            SherpaEngineError::TranscriptionFailed(err) => write!(f, "Transcription failed: {}", err),
            SherpaEngineError::DownloadFailed(err) => write!(f, "Download failed: {}", err),
            SherpaEngineError::IoError(err) => write!(f, "IO error: {}", err),
            SherpaEngineError::Other(err) => write!(f, "Error: {}", err),
        }
    }
}

impl std::error::Error for SherpaEngineError {}

impl From<std::io::Error> for SherpaEngineError {
    fn from(err: std::io::Error) -> Self {
        SherpaEngineError::IoError(err)
    }
}

struct LoadedSherpaModel {
    model_dir: PathBuf,
}

pub struct SherpaEngine {
    models_dir: PathBuf,
    current_model: Arc<RwLock<Option<LoadedSherpaModel>>>,
    current_model_name: Arc<RwLock<Option<String>>>,
    pub(crate) available_models: Arc<RwLock<HashMap<String, ModelInfo>>>,
    cancel_download_flag: Arc<RwLock<Option<String>>>, // Model name being cancelled
    // Active downloads tracking to prevent concurrent downloads
    pub(crate) active_downloads: Arc<RwLock<HashSet<String>>>, // Set of models currently being downloaded
}

pub fn map_sensevoice_language(language: Option<String>) -> String {
    match language.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
        None | Some("auto") | Some("auto-translate") | Some("auto-detect") | Some("detect") => {
            "auto".to_string()
        }
        Some("zh") | Some("chinese") | Some("cmn") | Some("mandarin") => "zh".to_string(),
        Some("en") | Some("english") => "en".to_string(),
        Some("ja") | Some("japanese") => "ja".to_string(),
        Some("ko") | Some("korean") => "ko".to_string(),
        Some("yue") | Some("cantonese") => "yue".to_string(),
        Some(other) => {
            let lower = other.to_lowercase();
            match lower.as_str() {
                "auto" | "zh" | "en" | "ja" | "ko" | "yue" => lower,
                _ => "auto".to_string(),
            }
        }
    }
}

fn transcribe_blocking(model_dir: &Path, language: &str, samples: Vec<f32>) -> Result<String> {
    let model_path = model_dir.join("model.int8.onnx");
    let tokens_path = model_dir.join("tokens.txt");

    let mut config = OfflineRecognizerConfig::default();
    config.model_config.sense_voice = OfflineSenseVoiceModelConfig {
        model: Some(model_path.to_string_lossy().into_owned()),
        language: Some(language.to_string()),
        use_itn: true,
    };
    config.model_config.tokens = Some(tokens_path.to_string_lossy().into_owned());

    let recognizer = OfflineRecognizer::create(&config)
        .ok_or_else(|| anyhow!("Failed to create Sherpa recognizer"))?;
    let stream = recognizer.create_stream();
    stream.accept_waveform(16_000, &samples);
    recognizer.decode(&stream);

    let result = stream
        .get_result()
        .ok_or_else(|| anyhow!("Sherpa returned empty transcription result"))?;
    if result.text.is_empty() {
        return Err(anyhow!("Sherpa returned empty transcription result"));
    }
    Ok(result.text)
}


impl SherpaEngine {
    /// Create a new Sherpa engine with optional custom models directory
    pub fn new_with_models_dir(models_dir: Option<PathBuf>) -> Result<Self> {
        let models_dir = if let Some(dir) = models_dir {
            dir.join("sherpa-onnx") // Sherpa models in subdirectory
        } else {
            // Fallback to default location
            let current_dir = std::env::current_dir()
                .map_err(|e| anyhow!("Failed to get current directory: {}", e))?;

            if cfg!(debug_assertions) {
                // Development mode
                current_dir.join("models").join("sherpa-onnx")
            } else {
                // Production mode
                dirs::data_dir()
                    .or_else(|| dirs::home_dir())
                    .ok_or_else(|| anyhow!("Could not find system data directory"))?
                    .join("Meetily")
                    .join("models")
                    .join("sherpa-onnx")
            }
        };

        log::info!("SherpaEngine using models directory: {}", models_dir.display());

        // Create directory if it doesn't exist
        if !models_dir.exists() {
            std::fs::create_dir_all(&models_dir)?;
        }

        Ok(Self {
            models_dir,
            current_model: Arc::new(RwLock::new(None)),
            current_model_name: Arc::new(RwLock::new(None)),
            available_models: Arc::new(RwLock::new(HashMap::new())),
            cancel_download_flag: Arc::new(RwLock::new(None)),
            // Initialize active downloads tracking
            active_downloads: Arc::new(RwLock::new(HashSet::new())),
        })
    }

    /// Discover available Sherpa models
    pub async fn discover_models(&self) -> Result<Vec<ModelInfo>> {
        let models_dir = &self.models_dir;
        let mut models = Vec::new();

        // Sherpa model configurations
        // SenseVoice Sherpa-ONNX model catalog
        // Sizes match actual download sizes (encoder + decoder + preprocessor + vocab)
        let model_configs = [
            (
                crate::config::DEFAULT_SHERPA_MODEL,
                239_u32,
                QuantizationType::Int8,
                "Fast multilingual",
                "SenseVoice int8 multilingual model",
            ),
        ];

        // Get active downloads to override status
        let active_downloads = self.active_downloads.read().await;

        for (name, size_mb, quantization, speed, description) in model_configs {
            let model_path = models_dir.join(name);

            // Check if model is currently downloading
            let status = if active_downloads.contains(name) {
                // If downloading, preserve that status regardless of file system
                // We don't know the exact progress here without more state, but 0 is safe fallback
                // The progress events will update the UI
                ModelStatus::Downloading { progress: 0 }
            } else if model_path.exists() {
                let required_files = vec!["model.int8.onnx", "tokens.txt"];
                let all_files_exist = required_files.iter().all(|file| {
                    model_path.join(file).exists()
                });

                if all_files_exist {
                    // Validate model by checking file sizes
                    match self.validate_model_directory(&model_path).await {
                        Ok(_) => ModelStatus::Available,
                        Err(_) => {
                            log::warn!("Model directory {} appears corrupted", name);
                            // Calculate total size of existing files
                            let mut total_size = 0u64;
                            for file in required_files {
                                if let Ok(metadata) = std::fs::metadata(model_path.join(file)) {
                                    total_size += metadata.len();
                                }
                            }
                            ModelStatus::Corrupted {
                                file_size: total_size,
                                expected_min_size: (size_mb as u64) * 1024 * 1024,
                            }
                        }
                    }
                } else {
                    ModelStatus::Missing
                }
            } else {
                ModelStatus::Missing
            };

            let model_info = ModelInfo {
                name: name.to_string(),
                path: model_path,
                size_mb: size_mb as u32,
                quantization: quantization.clone(),
                speed: speed.to_string(),
                status,
                description: description.to_string(),
            };

            models.push(model_info);
        }

        // Update internal cache
        let mut available_models = self.available_models.write().await;
        available_models.clear();
        for model in &models {
            available_models.insert(model.name.clone(), model.clone());
        }

        Ok(models)
    }

    /// Validate model directory by checking if all required files exist AND have valid sizes
    async fn validate_model_directory(&self, model_dir: &PathBuf) -> Result<()> {
        let expected_sizes: [(&str, u64); 2] = [
            ("model.int8.onnx", 200_000_000),
            ("tokens.txt", 1_000),
        ];

        for (filename, min_size) in expected_sizes {
            let file_path = model_dir.join(filename);
            if !file_path.exists() {
                return Err(anyhow!("{} not found", filename));
            }

            let metadata = std::fs::metadata(&file_path)?;
            if metadata.len() < min_size {
                return Err(anyhow!(
                    "{} is incomplete: {} bytes (expected at least {} bytes)",
                    filename,
                    metadata.len(),
                    min_size
                ));
            }
        }

        Ok(())
    }

    /// Clean incomplete model directory before download
    /// Removes all files if directory exists but model is not Available
    async fn clean_incomplete_model_directory(&self, model_dir: &PathBuf) -> Result<()> {
        if !model_dir.exists() {
            return Ok(()); // Nothing to clean
        }

        // Validate the directory
        match self.validate_model_directory(model_dir).await {
            Ok(_) => {
                log::info!("Model directory is valid, no cleanup needed");
                return Ok(());
            }
            Err(validation_error) => {
                log::warn!(
                    "Model directory exists but is invalid: {}. Cleaning up...",
                    validation_error
                );

                // List and remove all files in the directory
                let mut entries = fs::read_dir(model_dir).await
                    .map_err(|e| anyhow!("Failed to read model directory: {}", e))?;

                let mut removed_count = 0;
                while let Some(entry) = entries.next_entry().await
                    .map_err(|e| anyhow!("Failed to read directory entry: {}", e))?
                {
                    let path = entry.path();
                    if path.is_file() {
                        match fs::remove_file(&path).await {
                            Ok(_) => {
                                log::info!("Removed incomplete file: {:?}", path.file_name());
                                removed_count += 1;
                            }
                            Err(e) => {
                                log::warn!("Failed to remove file {:?}: {}", path, e);
                            }
                        }
                    }
                }

                log::info!("Cleaned {} incomplete files from model directory", removed_count);
                Ok(())
            }
        }
    }

    /// Load a Sherpa model
    pub async fn load_model(&self, model_name: &str) -> Result<()> {
        let models = self.available_models.read().await;
        let model_info = models
            .get(model_name)
            .ok_or_else(|| anyhow!("Model {} not found", model_name))?;

        match model_info.status {
            ModelStatus::Available => {
                // Check if this model is already loaded
                if let Some(current_model) = self.current_model_name.read().await.as_ref() {
                    if current_model == model_name {
                        log::info!("Sherpa model {} is already loaded, skipping reload", model_name);
                        return Ok(());
                    }

                    // Unload current model before loading new one
                    log::info!("Unloading current Sherpa model '{}' before loading '{}'", current_model, model_name);
                    self.unload_model().await;
                }

                log::info!("Loading Sherpa model: {}", model_name);

                self.validate_model_directory(&model_info.path).await?;

                *self.current_model.write().await = Some(LoadedSherpaModel {
                    model_dir: model_info.path.clone(),
                });
                *self.current_model_name.write().await = Some(model_name.to_string());

                log::info!("Successfully loaded Sherpa model: {}", model_name);
                Ok(())
            }
            ModelStatus::Missing => {
                Err(anyhow!("Sherpa model {} is not downloaded", model_name))
            }
            ModelStatus::Downloading { .. } => {
                Err(anyhow!("Sherpa model {} is currently downloading", model_name))
            }
            ModelStatus::Error(ref err) => {
                Err(anyhow!("Sherpa model {} has error: {}", model_name, err))
            }
            ModelStatus::Corrupted { .. } => {
                Err(anyhow!("Sherpa model {} is corrupted and cannot be loaded", model_name))
            }
        }
    }

    /// Unload the current model
    pub async fn unload_model(&self) -> bool {
        let mut model_guard = self.current_model.write().await;
        let unloaded = model_guard.take().is_some();
        if unloaded {
            log::info!("Sherpa model unloaded");
        }

        let mut model_name_guard = self.current_model_name.write().await;
        model_name_guard.take();

        unloaded
    }

    /// Get the currently loaded model name
    pub async fn get_current_model(&self) -> Option<String> {
        self.current_model_name.read().await.clone()
    }

    /// Check if a model is loaded
    pub async fn is_model_loaded(&self) -> bool {
        self.current_model.read().await.is_some()
    }

    pub async fn transcribe_audio(
        &self,
        audio_data: Vec<f32>,
        language: Option<String>,
    ) -> Result<String> {
        let model_dir = {
            let guard = self.current_model.read().await;
            guard
                .as_ref()
                .ok_or_else(|| anyhow!("No Sherpa model loaded. Please load a model first."))?
                .model_dir
                .clone()
        };

        let sense_language = map_sensevoice_language(language);
        let duration_seconds = audio_data.len() as f64 / 16_000.0;
        log::debug!(
            "Sherpa transcribing {} samples ({:.1}s), language={}",
            audio_data.len(),
            duration_seconds,
            sense_language
        );

        let text = tokio::task::spawn_blocking(move || {
            transcribe_blocking(&model_dir, &sense_language, audio_data)
        })
        .await
        .map_err(|e| anyhow!("Sherpa transcription task failed: {}", e))??;

        log::debug!("Sherpa transcription result: '{}'", text);
        Ok(text)
    }

    /// Get the models directory path
    pub async fn get_models_directory(&self) -> PathBuf {
        self.models_dir.clone()
    }

    /// Delete a corrupted model
    pub async fn delete_model(&self, model_name: &str) -> Result<String> {
        log::info!("Attempting to delete Sherpa model: {}", model_name);

        // Get model info to find the directory path
        let model_info = {
            let models = self.available_models.read().await;
            models.get(model_name).cloned()
        };

        let model_info = model_info.ok_or_else(|| anyhow!("Sherpa model '{}' not found", model_name))?;

        log::info!("Sherpa model '{}' has status: {:?}", model_name, model_info.status);

        // Allow deletion of corrupted or available models
        match &model_info.status {
            ModelStatus::Corrupted { .. } | ModelStatus::Available => {
                // Delete the entire model directory
                if model_info.path.exists() {
                    fs::remove_dir_all(&model_info.path).await
                        .map_err(|e| anyhow!("Failed to delete directory '{}': {}", model_info.path.display(), e))?;
                    log::info!("Successfully deleted Sherpa model directory: {}", model_info.path.display());
                } else {
                    log::warn!("Directory '{}' does not exist, nothing to delete", model_info.path.display());
                }

                // Update model status to Missing
                {
                    let mut models = self.available_models.write().await;
                    if let Some(model) = models.get_mut(model_name) {
                        model.status = ModelStatus::Missing;
                    }
                }

                Ok(format!("Successfully deleted Sherpa model '{}'", model_name))
            }
            _ => {
                Err(anyhow!(
                    "Can only delete corrupted or available Sherpa models. Model '{}' has status: {:?}",
                    model_name,
                    model_info.status
                ))
            }
        }
    }

    /// Download a Sherpa model from HuggingFace (backward-compatible wrapper)
    pub async fn download_model(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(u8) + Send>>,
    ) -> Result<()> {
        // Wrap simple callback to use detailed version
        let detailed_callback: Option<Box<dyn Fn(DownloadProgress) + Send>> =
            progress_callback.map(|cb| {
                Box::new(move |p: DownloadProgress| cb(p.percent)) as Box<dyn Fn(DownloadProgress) + Send>
            });
        self.download_model_detailed(model_name, detailed_callback).await
    }

    /// Download a Sherpa model with detailed progress (MB/speed/resume support)
    pub async fn download_model_detailed(
        &self,
        model_name: &str,
        progress_callback: Option<Box<dyn Fn(DownloadProgress) + Send>>,
    ) -> Result<()> {
        log::info!("Starting download for Sherpa model: {}", model_name);

        // Check if download is already in progress for this model
        {
            let active = self.active_downloads.read().await;
            if active.contains(model_name) {
                log::warn!("Download already in progress for Sherpa model: {}", model_name);
                return Err(anyhow!("Download already in progress for model: {}", model_name));
            }
        }

        // Add to active downloads
        {
            let mut active = self.active_downloads.write().await;
            active.insert(model_name.to_string());
        }

        // Clear any previous cancellation flag for this model
        {
            let mut cancel_flag = self.cancel_download_flag.write().await;
            *cancel_flag = None;
        }

        // Get model info
        let model_info = {
            let models = self.available_models.read().await;
            match models.get(model_name).cloned() {
                Some(info) => info,
                None => {
                    // Remove from active downloads on error
                    let mut active = self.active_downloads.write().await;
                    active.remove(model_name);
                    return Err(anyhow!("Model {} not found", model_name));
                }
            }
        };

        // Update model status to downloading
        {
            let mut models = self.available_models.write().await;
            if let Some(model) = models.get_mut(model_name) {
                model.status = ModelStatus::Downloading { progress: 0 };
            }
        }

        // HuggingFace base URL for Sherpa models (version-specific)
        let base_url = match model_name {
            name if name == crate::config::DEFAULT_SHERPA_MODEL => {
                "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17-int8/resolve/main"
            }
            _ => {
                self.active_downloads.write().await.remove(model_name);
                return Err(anyhow!("No download URL configured for model {}", model_name));
            }
        };
        // Determine which files to download based on quantization
        let files_to_download = vec!["model.int8.onnx", "tokens.txt"];

        // Create model directory
        let model_dir = &model_info.path;
        if !model_dir.exists() {
            if let Err(e) = fs::create_dir_all(model_dir).await {
                // Remove from active downloads on error
                let mut active = self.active_downloads.write().await;
                active.remove(model_name);
                return Err(anyhow!("Failed to create model directory: {}", e));
            }
        }

        // Clean up incomplete downloads before starting
        log::info!("Checking for incomplete model files to clean up...");
        if let Err(e) = self.clean_incomplete_model_directory(model_dir).await {
            log::warn!("Failed to clean incomplete model directory: {}", e);
            // Continue anyway - we'll handle errors during download
        }

        // Optimized HTTP client for large file downloads
        let client = reqwest::Client::builder()
            .tcp_nodelay(true)              // Disable Nagle's algorithm for better streaming
            .pool_max_idle_per_host(1)      // Keep connection alive
            .timeout(Duration::from_secs(3600))  // 1 hour timeout for large files
            .connect_timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| anyhow!("Failed to create HTTP client: {}", e))?;

        let total_files = files_to_download.len();

        // Calculate total download size for weighted progress
        // Note: These are approximate sizes based on HuggingFace repo inspection
        let file_sizes: std::collections::HashMap<&str, u64> = [
            ("model.int8.onnx", 239_000_000u64),
            ("tokens.txt", 50_000u64),
        ]
        .iter()
        .cloned()
        .collect();

        // Calculate total expected download size
        let total_size_bytes: u64 = files_to_download.iter()
            .filter_map(|f| file_sizes.get(*f))
            .copied()
            .sum();

        // Check for existing downloads (complete or partial) to calculate resume offset
        let mut already_downloaded: u64 = 0;
        for filename in &files_to_download {
            let file_path = model_dir.join(filename);
            if file_path.exists() {
                if let Ok(metadata) = fs::metadata(&file_path).await {
                    let file_size = metadata.len();
                    let expected_size = file_sizes.get(*filename).copied().unwrap_or(0);
                    // Count all existing bytes (complete files capped at expected size, partial as-is)
                    // This ensures progress starts from where we left off
                    already_downloaded += file_size.min(expected_size);
                }
            }
        }

        let mut total_downloaded: u64 = already_downloaded;

        // Timing for speed calculation
        let download_start_time = Instant::now();
        let mut last_report_time = Instant::now();
        let mut bytes_since_last_report: u64 = 0;
        let mut last_reported_progress: u8 = 0;

        log::info!(
            "Starting weighted download for {} files, total size: {:.2} MB (already downloaded: {:.2} MB)",
            total_files,
            total_size_bytes as f64 / 1_048_576.0,
            already_downloaded as f64 / 1_048_576.0
        );

        for (index, filename) in files_to_download.iter().enumerate() {
            let file_url = format!("{}/{}", base_url, filename);
            let file_path = model_dir.join(filename);

            // Check for existing partial file to resume
            let existing_size: u64 = if file_path.exists() {
                fs::metadata(&file_path).await.map(|m| m.len()).unwrap_or(0)
            } else {
                0
            };

            let expected_size = file_sizes.get(*filename).copied().unwrap_or(0);

            // Skip if file is already complete (with 1% tolerance for size variations)
            let size_tolerance = (expected_size as f64 * 0.99) as u64;
            if existing_size >= size_tolerance && expected_size > 0 {
                log::info!(
                    "Skipping complete file: {} ({:.2} MB, expected: {:.2} MB)",
                    filename,
                    existing_size as f64 / 1_048_576.0,
                    expected_size as f64 / 1_048_576.0
                );
                continue;
            }

            log::info!("Downloading file {}/{}: {} (resuming from {} bytes)", index + 1, total_files, filename, existing_size);

            // Build request with optional Range header for resume
            let mut request = client.get(&file_url);
            if existing_size > 0 {
                request = request.header("Range", format!("bytes={}-", existing_size));
                log::info!("Resuming download from byte {}", existing_size);
            }

            let mut response = request.send().await
                .map_err(|e| {
                    anyhow!("Failed to start download for {}: {}", filename, e)
                })?;

            // Handle response status
            let (file_total_size, resuming) = if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
                // Server supports resume, get remaining size
                let remaining = response.content_length().unwrap_or(0);
                log::info!("Server supports resume, remaining: {} bytes", remaining);
                (existing_size + remaining, true)
            } else if response.status().is_success() {
                // Fresh download or server doesn't support resume
                if existing_size > 0 {
                    log::warn!("Server doesn't support resume for {}, starting fresh download", filename);
                }
                (response.content_length().unwrap_or(0), false)
            } else if response.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
                // 416: Range not satisfiable - file complete or invalid range
                log::warn!("Server returned 416 Range Not Satisfiable for {}", filename);

                let size_tolerance = (expected_size as f64 * 0.99) as u64;
                if existing_size >= size_tolerance && expected_size > 0 {
                    // File is complete - skip it
                    log::info!("File {} complete ({} bytes). Skipping.", filename, existing_size);
                    continue;
                } else {
                    // File incomplete but server won't accept range - delete and retry
                    log::warn!(
                        "File {} incomplete ({}/{} bytes). Deleting and retrying.",
                        filename, existing_size, expected_size
                    );

                    if let Err(e) = fs::remove_file(&file_path).await {
                        let mut active = self.active_downloads.write().await;
                        active.remove(model_name);
                        return Err(anyhow!("Failed to delete incomplete file {}: {}", filename, e));
                    }

                    // Retry without Range header
                    log::info!("Retrying {} without resume", filename);
                    response = client.get(&file_url).send().await
                        .map_err(|e| anyhow!("Retry failed for {}: {}", filename, e))?;

                    if !response.status().is_success() {
                        let mut active = self.active_downloads.write().await;
                        active.remove(model_name);
                        return Err(anyhow!("Retry failed for {} with status: {}", filename, response.status()));
                    }

                    (response.content_length().unwrap_or(0), false)
                }
            } else {
                // Other errors
                let mut active = self.active_downloads.write().await;
                active.remove(model_name);
                return Err(anyhow!("Download failed for {} with status: {}", filename, response.status()));
            };

            // Open file for writing (append if resuming, create new if not)
            let file = if resuming {
                fs::OpenOptions::new()
                    .append(true)
                    .open(&file_path)
                    .await
                    .map_err(|e| anyhow!("Failed to open file for resume {}: {}", filename, e))?
            } else {
                fs::File::create(&file_path)
                    .await
                    .map_err(|e| anyhow!("Failed to create file {}: {}", filename, e))?
            };

            // Use buffered writer for better I/O performance (8MB buffer)
            let mut writer = BufWriter::with_capacity(8 * 1024 * 1024, file);

            // Stream download
            use futures_util::StreamExt;
            let mut stream = response.bytes_stream();
            let mut file_downloaded = if resuming { existing_size } else { 0u64 };

            loop {
                // Check for cancellation before processing chunk
                {
                    let cancel_flag = self.cancel_download_flag.read().await;
                    if cancel_flag.as_ref() == Some(&model_name.to_string()) {
                        log::info!("Download cancelled for {}", model_name);
                        // Flush and keep partial file for resume on next attempt
                        let _ = writer.flush().await;
                        drop(writer);
                        // Remove from active downloads on cancellation
                        let mut active = self.active_downloads.write().await;
                        active.remove(model_name);
                        return Err(anyhow!("Download cancelled by user"));
                    }
                }

                // Add per-chunk timeout (30 seconds) to detect stalled connections
                let next_result = timeout(Duration::from_secs(30), stream.next()).await;

                let chunk = match next_result {
                    // Timeout - no data received for 30 seconds
                    Err(_) => {
                        log::warn!("Download timeout for {}: no data received for 30 seconds", model_name);
                        let _ = writer.flush().await;

                        // Remove from active downloads
                        {
                            let mut active = self.active_downloads.write().await;
                            active.remove(model_name);
                        }

                        // Update model status to Missing so retry can work
                        {
                            let mut models = self.available_models.write().await;
                            if let Some(model) = models.get_mut(model_name) {
                                model.status = ModelStatus::Missing;
                            }
                        }

                        return Err(anyhow!("Download timeout - No data received for 30 seconds"));
                    },
                    // Stream ended
                    Ok(None) => break,
                    // Got chunk result
                    Ok(Some(chunk_result)) => {
                        match chunk_result {
                            Ok(c) => c,
                            // Detect error type for better user feedback
                            Err(e) => {
                                log::error!("Download error for {}: {:?}", model_name, e);
                                let _ = writer.flush().await;

                                // Remove from active downloads
                                {
                                    let mut active = self.active_downloads.write().await;
                                    active.remove(model_name);
                                }

                                // Update model status to Missing so retry can work
                                {
                                    let mut models = self.available_models.write().await;
                                    if let Some(model) = models.get_mut(model_name) {
                                        model.status = ModelStatus::Missing;
                                    }
                                }

                                let error_msg = if e.is_timeout() {
                                    "Connection timeout - Check your internet"
                                } else if e.is_connect() {
                                    "Connection failed - Check your internet"
                                } else if e.is_body() {
                                    "Stream interrupted - Network unstable"
                                } else {
                                    "Download error"
                                };

                                return Err(anyhow!("{}: {}", error_msg, e));
                            }
                        }
                    }
                };

                if let Err(e) = writer.write_all(&chunk).await {
                    // Remove from active downloads on error
                    {
                        let mut active = self.active_downloads.write().await;
                        active.remove(model_name);
                    }

                    // Update model status to Missing so retry can work
                    {
                        let mut models = self.available_models.write().await;
                        if let Some(model) = models.get_mut(model_name) {
                            model.status = ModelStatus::Missing;
                        }
                    }

                    return Err(anyhow!("Failed to write chunk to file: {}", e));
                }

                let chunk_len = chunk.len() as u64;
                file_downloaded += chunk_len;
                total_downloaded += chunk_len;
                bytes_since_last_report += chunk_len;

                // Calculate weighted overall progress based on total bytes downloaded
                let overall_progress = if total_size_bytes > 0 {
                    ((total_downloaded as f64 / total_size_bytes as f64) * 100.0).min(99.0) as u8
                } else {
                    // Fallback to per-file progress if total size unknown
                    ((index as f64 + (file_downloaded as f64 / file_total_size.max(1) as f64)) / total_files as f64 * 100.0) as u8
                };

                // Report every 1% progress change OR every 500ms for smooth UI updates
                let elapsed_since_report = last_report_time.elapsed();
                let progress_changed = overall_progress > last_reported_progress;
                let time_threshold = elapsed_since_report >= Duration::from_millis(500);
                let is_complete = file_downloaded >= file_total_size;

                let should_report = progress_changed || time_threshold || is_complete;

                if should_report {
                    // Calculate download speed
                    let speed_mbps = if elapsed_since_report.as_secs_f64() >= 0.1 {
                        (bytes_since_last_report as f64 / (1024.0 * 1024.0)) / elapsed_since_report.as_secs_f64()
                    } else {
                        // Fallback to overall average speed
                        let total_elapsed = download_start_time.elapsed().as_secs_f64();
                        if total_elapsed > 0.0 {
                            ((total_downloaded - already_downloaded) as f64 / (1024.0 * 1024.0)) / total_elapsed
                        } else {
                            0.0
                        }
                    };

                    last_reported_progress = overall_progress;
                    last_report_time = Instant::now();
                    bytes_since_last_report = 0;

                    // Create detailed progress and report
                    let progress = DownloadProgress::new(total_downloaded, total_size_bytes, speed_mbps);
                    if let Some(ref callback) = progress_callback {
                        callback(progress);
                    }

                    // Update model status
                    {
                        let mut models = self.available_models.write().await;
                        if let Some(model) = models.get_mut(model_name) {
                            model.status = ModelStatus::Downloading { progress: overall_progress };
                        }
                    }
                }
            }

            // Flush the buffered writer
            if let Err(e) = writer.flush().await {
                // Remove from active downloads on error
                {
                    let mut active = self.active_downloads.write().await;
                    active.remove(model_name);
                }

                // Update model status to Missing so retry can work
                {
                    let mut models = self.available_models.write().await;
                    if let Some(model) = models.get_mut(model_name) {
                        model.status = ModelStatus::Missing;
                    }
                }

                return Err(anyhow!("Failed to flush file {}: {}", filename, e));
            }

            log::info!(
                "Completed download: {} ({:.2} MB, overall progress: {:.1}%)",
                filename,
                file_downloaded as f64 / 1_048_576.0,
                (total_downloaded as f64 / total_size_bytes as f64) * 100.0
            );
        }

        // Report 100% progress with final speed
        let total_elapsed = download_start_time.elapsed().as_secs_f64();
        let final_speed = if total_elapsed > 0.0 {
            ((total_downloaded - already_downloaded) as f64 / (1024.0 * 1024.0)) / total_elapsed
        } else {
            0.0
        };
        let final_progress = DownloadProgress::new(total_size_bytes, total_size_bytes, final_speed);
        if let Some(ref callback) = progress_callback {
            callback(final_progress);
        }

        // Update model status to available
        {
            let mut models = self.available_models.write().await;
            if let Some(model) = models.get_mut(model_name) {
                model.status = ModelStatus::Available;
                model.path = model_dir.clone();
            }
        }

        // Remove from active downloads on completion
        {
            let mut active = self.active_downloads.write().await;
            active.remove(model_name);
        }

        // Clear cancellation flag on successful completion
        {
            let mut cancel_flag = self.cancel_download_flag.write().await;
            if cancel_flag.as_ref() == Some(&model_name.to_string()) {
                *cancel_flag = None;
            }
        }

        log::info!("Download completed for Sherpa model: {}", model_name);
        Ok(())
    }

    /// Cancel an ongoing model download
    pub async fn cancel_download(&self, model_name: &str) -> Result<()> {
        log::info!("Cancelling download for Sherpa model: {}", model_name);

        // Set cancellation flag to interrupt the download loop
        {
            let mut cancel_flag = self.cancel_download_flag.write().await;
            *cancel_flag = Some(model_name.to_string());
        }

        // Remove from active downloads
        {
            let mut active = self.active_downloads.write().await;
            active.remove(model_name);
        }

        // Update model status to Missing (so it can be retried)
        {
            let mut models = self.available_models.write().await;
            if let Some(model) = models.get_mut(model_name) {
                model.status = ModelStatus::Missing;
            }
        }

        // Clean up partially downloaded files
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await; // Brief delay to let download loop exit

        let model_path = self.models_dir.join(model_name);
        if model_path.exists() {
            if let Err(e) = fs::remove_dir_all(&model_path).await {
                log::warn!("Failed to clean up cancelled download directory: {}", e);
            } else {
                log::info!("Cleaned up cancelled download directory: {}", model_path.display());
            }
        }

        Ok(())
    }
}
