import { DEFAULT_SHERPA_MODEL } from '@/constants/modelDefaults';
import { invoke } from '@tauri-apps/api/core';

// Types for Sherpa (Sherpa-ONNX SenseVoice) integration
export interface SherpaModelInfo {
  name: string;
  path: string;
  size_mb: number;
  accuracy: ModelAccuracy;
  speed: ProcessingSpeed;
  status: ModelStatus;
  description?: string;
  quantization: QuantizationType;
}

export type QuantizationType = 'FP32' | 'Int8';
export type ModelAccuracy = 'High' | 'Good' | 'Decent';
export type ProcessingSpeed = 'Slow' | 'Medium' | 'Fast' | 'Very Fast' | 'Ultra Fast';

export type ModelStatus =
  | 'Available'
  | 'Missing'
  | { Downloading: number }
  | { Error: string }
  | { Corrupted: { file_size: number; expected_min_size: number } };

export interface SherpaEngineState {
  currentModel: string | null;
  availableModels: SherpaModelInfo[];
  isLoading: boolean;
  error: string | null;
}

// User-friendly model display configuration
export interface ModelDisplayInfo {
  friendlyName: string;
  icon: string;
  tagline: string;
  recommended?: boolean;
  tier: 'fastest' | 'balanced' | 'precise';
}

export const MODEL_DISPLAY_CONFIG: Record<string, ModelDisplayInfo> = {
  [DEFAULT_SHERPA_MODEL]: {
    friendlyName: 'SenseVoice',
    icon: '🀄',
    tagline: 'Real time • Best for Chinese and multilingual speech',
    recommended: true,
    tier: 'fastest'
  }
};

// Model configuration for Sherpa models (matching Rust implementation)
// Supported models: sherpa-tdt-0.6b in v2 and v3 variants
// Source: https://huggingface.co/istupakov/sherpa-tdt-0.6b-v3-onnx
export const SHERPA_MODEL_CONFIGS: Record<string, Partial<SherpaModelInfo>> = {
  [DEFAULT_SHERPA_MODEL]: {
    description: 'SenseVoice multilingual model optimized for Chinese',
    size_mb: 239,
    accuracy: 'High',
    speed: 'Very Fast',
    quantization: 'Int8'
  }
};

// Helper functions
export function getModelIcon(accuracy: ModelAccuracy): string {
  switch (accuracy) {
    case 'High': return '🔥';
    case 'Good': return '⚡';
    case 'Decent': return '🚀';
    default: return '📊';
  }
}

// Get user-friendly display name for a model
export function getModelDisplayName(modelName: string): string {
  const displayInfo = MODEL_DISPLAY_CONFIG[modelName];
  return displayInfo?.friendlyName || modelName;
}

// Get model display info (icon, tagline, etc.)
export function getModelDisplayInfo(modelName: string): ModelDisplayInfo | null {
  return MODEL_DISPLAY_CONFIG[modelName] || null;
}

export function getStatusColor(status: ModelStatus): string {
  if (status === 'Available') return 'green';
  if (status === 'Missing') return 'gray';
  if (typeof status === 'object' && 'Downloading' in status) return 'blue';
  if (typeof status === 'object' && 'Error' in status) return 'red';
  return 'gray';
}

export function formatFileSize(sizeMb: number): string {
  if (sizeMb >= 1000) {
    return `${(sizeMb / 1000).toFixed(1)}GB`;
  }
  return `${sizeMb}MB`;
}

// Helper function to check if model is quantized
export function isQuantizedModel(modelName: string): boolean {
  return modelName.includes('int8');
}

// Helper function to get model performance badge
export function getModelPerformanceBadge(quantization: QuantizationType): { label: string; color: string } {
  switch (quantization) {
    case 'FP32':
      return { label: 'Full Precision', color: 'blue' };
    case 'Int8':
      return { label: 'Int8 Quantized', color: 'green' };
    default:
      return { label: 'Standard', color: 'gray' };
  }
}

export function getRecommendedModel(_systemSpecs?: { ram: number; cores: number }): string {
  // Default to Int8 quantized model (fastest)
  return DEFAULT_SHERPA_MODEL;

  // For any system, prefer Int8 for speed
  // FP32 can be used if user explicitly wants higher precision
  return DEFAULT_SHERPA_MODEL;
}

// Tauri command wrappers for Sherpa backend
export class SherpaAPI {
  static async init(): Promise<void> {
    await invoke('sherpa_init');
  }

  static async getAvailableModels(): Promise<SherpaModelInfo[]> {
    return await invoke('sherpa_get_available_models');
  }

  static async loadModel(modelName: string): Promise<void> {
    await invoke('sherpa_load_model', { modelName });
  }

  static async getCurrentModel(): Promise<string | null> {
    return await invoke('sherpa_get_current_model');
  }

  static async isModelLoaded(): Promise<boolean> {
    return await invoke('sherpa_is_model_loaded');
  }

  static async transcribeAudio(audioData: number[]): Promise<string> {
    return await invoke('sherpa_transcribe_audio', { audioData });
  }

  static async getModelsDirectory(): Promise<string> {
    return await invoke('sherpa_get_models_directory');
  }

  static async downloadModel(modelName: string): Promise<void> {
    await invoke('sherpa_download_model', { modelName });
  }

  static async cancelDownload(modelName: string): Promise<void> {
    await invoke('sherpa_cancel_download', { modelName });
  }

  static async deleteCorruptedModel(modelName: string): Promise<string> {
    return await invoke('sherpa_delete_corrupted_model', { modelName });
  }

  static async hasAvailableModels(): Promise<boolean> {
    return await invoke('sherpa_has_available_models');
  }

  static async validateModelReady(): Promise<string> {
    return await invoke('sherpa_validate_model_ready');
  }

  static async openModelsFolder(): Promise<void> {
    await invoke('open_sherpa_models_folder');
  }
}
