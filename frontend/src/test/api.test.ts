import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { getApiBaseUrl, checkHealth, analyzeAudio, ApiError } from '../services/api';

describe('API Client Service', () => {
  const originalFetch = global.fetch;

  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    global.fetch = originalFetch;
  });

  it('1. getApiBaseUrl defaults to http://127.0.0.1:8000 when unset', () => {
    const url = getApiBaseUrl();
    expect(url).toBe('http://127.0.0.1:8000');
  });

  it('2. checkHealth fetches GET /health and returns health payload', async () => {
    const mockHealth = {
      status: 'healthy',
      model_loaded: true,
      model_name: 'SYNORA-A-Combined-LogisticRegression',
      feature_dimensions: 40,
      version: '1.0.0',
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => mockHealth,
    } as Response);

    const result = await checkHealth();
    expect(result).toEqual(mockHealth);
    expect(global.fetch).toHaveBeenCalledWith('http://127.0.0.1:8000/health', expect.any(Object));
  });

  it('3. analyzeAudio sends FormData with field "file" to POST /api/v1/analyze', async () => {
    const mockResponse = {
      success: true,
      prediction: 'Fake',
      probabilities: { fake: 0.999863, real: 0.000137 },
      decision_threshold: 0.50,
      audio: { duration_seconds: 3.0, sample_rate: 16000, channels: 1 },
      model: { name: 'SYNORA A-Combined', feature_dimensions: 40 },
      disclaimer: 'These probabilities are statistical estimates based on acoustic features.',
    };

    let sentFormData: FormData | null = null;
    global.fetch = vi.fn().mockImplementation(async (_url, options) => {
      sentFormData = options.body;
      return {
        ok: true,
        status: 200,
        json: async () => mockResponse,
      } as Response;
    });

    const dummyFile = new File(['fake-audio-bytes'], 'test.wav', { type: 'audio/wav' });
    const result = await analyzeAudio(dummyFile);

    expect(result.prediction).toBe('Fake');
    expect(result.probabilities.fake).toBeCloseTo(0.999863);
    expect(sentFormData).toBeInstanceOf(FormData);
    expect((sentFormData as unknown as FormData).get('file')).toBeTruthy();
  });

  it('4. analyzeAudio throws ApiError with HTTP detail on 400 Bad Request', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 400,
      json: async () => ({
        success: false,
        error: 'HTTP_400',
        detail: 'Silent audio detected: zero frames exceed speech activity threshold.',
      }),
    } as Response);

    const dummyFile = new File(['silence'], 'silent.wav', { type: 'audio/wav' });
    await expect(analyzeAudio(dummyFile)).rejects.toThrow(
      'Silent audio detected: zero frames exceed speech activity threshold.'
    );
  });

  it('5. analyzeAudio throws ApiError on 503 Service Unavailable', async () => {
    global.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 503,
      json: async () => ({
        success: false,
        error: 'HTTP_503',
        detail: 'Inference service is currently unavailable or model failed to initialize.',
      }),
    } as Response);

    const dummyFile = new File(['bytes'], 'test.wav', { type: 'audio/wav' });
    await expect(analyzeAudio(dummyFile)).rejects.toThrow('Inference service is currently unavailable');
  });

  it('6. analyzeAudio handles network failure gracefully', async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error('Failed to fetch'));

    const dummyFile = new File(['bytes'], 'test.wav', { type: 'audio/wav' });
    await expect(analyzeAudio(dummyFile)).rejects.toThrow('Network connection failed');
  });
});

