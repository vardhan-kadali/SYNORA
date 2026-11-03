/**
 * SYNORA — API Client Service
 * Encapsulates all HTTP communication with the Stage 2 FastAPI backend.
 */

import type { HealthResponse, AnalysisResponse, ApiErrorResponse } from '../types/api';

const DEFAULT_API_BASE_URL = 'http://127.0.0.1:8000';

export function getApiBaseUrl(): string {
  const envUrl = import.meta.env.VITE_API_BASE_URL;
  if (envUrl && typeof envUrl === 'string' && envUrl.trim().length > 0) {
    return envUrl.trim().replace(/\/+$/, '');
  }
  return DEFAULT_API_BASE_URL;
}

export class ApiError extends Error {
  statusCode: number;
  errorCode?: string;

  constructor(message: string, statusCode: number, errorCode?: string) {
    super(message);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.errorCode = errorCode;
  }
}

/**
 * Checks system readiness and model loading status via GET /health.
 */
export async function checkHealth(): Promise<HealthResponse> {
  const baseUrl = getApiBaseUrl();
  try {
    const res = await fetch(`${baseUrl}/health`, {
      method: 'GET',
      headers: {
        Accept: 'application/json',
      },
    });

    if (!res.ok) {
      if (res.status === 503) {
        const data = (await res.json().catch(() => null)) as HealthResponse | null;
        return (
          data || {
            status: 'unhealthy',
            model_loaded: false,
            model_name: 'SYNORA A-Combined',
            feature_dimensions: 40,
            version: '1.0.0',
          }
        );
      }
      throw new ApiError(`Health check returned HTTP ${res.status}`, res.status);
    }

    const data: HealthResponse = await res.json();
    return data;
  } catch (err: unknown) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(
      'Unable to connect to SYNORA backend service. Verify that the API server is running.',
      0
    );
  }
}

/**
 * Uploads an audio recording to POST /api/v1/analyze for forensic inference.
 */
export async function analyzeAudio(file: File): Promise<AnalysisResponse> {
  const baseUrl = getApiBaseUrl();
  const formData = new FormData();
  // Exact field name expected by synora/api/routes.py: 'file'
  formData.append('file', file, file.name);

  let response: Response;
  try {
    response = await fetch(`${baseUrl}/api/v1/analyze`, {
      method: 'POST',
      body: formData,
      headers: {
        Accept: 'application/json',
      },
    });
  } catch (err: unknown) {
    throw new ApiError(
      'Network connection failed. The SYNORA inference service is unreachable.',
      0
    );
  }

  if (!response.ok) {
    let errorMessage = `Analysis request failed with HTTP ${response.status}.`;
    let errorCode: string | undefined;

    try {
      const errorJson = (await response.json()) as ApiErrorResponse | { detail?: string };
      if ('detail' in errorJson && typeof errorJson.detail === 'string') {
        errorMessage = errorJson.detail;
      }
      if ('error' in errorJson && typeof errorJson.error === 'string') {
        errorCode = errorJson.error;
      }
    } catch {
      // Fallback status descriptions if response body isn't JSON
      if (response.status === 413) {
        errorMessage = 'Upload exceeded maximum allowed file size (25 MB) or duration (60 seconds).';
      } else if (response.status === 415) {
        errorMessage = 'Unsupported audio format. Supported formats include WAV, FLAC, OGG, and MP3.';
      } else if (response.status === 503) {
        errorMessage = 'Inference engine is currently unavailable or failed to initialize.';
      } else if (response.status === 500) {
        errorMessage = 'An internal error occurred during forensic acoustic analysis.';
      }
    }

    throw new ApiError(errorMessage, response.status, errorCode);
  }

  const result: AnalysisResponse = await response.json();
  return result;
}
