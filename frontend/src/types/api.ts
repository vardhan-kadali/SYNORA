/**
 * SYNORA — Frontend TypeScript Contracts
 * Exact structural mirrors of synora/api/schemas.py
 */

export interface RootResponse {
  name: string;
  tagline: string;
  status: string;
  api_version: string;
}

export interface HealthResponse {
  status: 'healthy' | 'unhealthy' | string;
  model_loaded: boolean;
  model_name: string;
  feature_dimensions: number;
  version: string;
}

export interface AudioMetadata {
  duration_seconds: number;
  sample_rate: number;
  channels: number;
}

export interface ModelInfo {
  name: string;
  feature_dimensions: number;
}

export interface ProbabilityDistribution {
  fake: number;
  real: number;
}

export interface AnalysisResponse {
  success: boolean;
  prediction: 'Fake' | 'Real';
  probabilities: ProbabilityDistribution;
  decision_threshold: number;
  audio: AudioMetadata;
  model: ModelInfo;
  disclaimer: string;
}

export interface ApiErrorResponse {
  success: false;
  error: string;
  detail: string;
}

