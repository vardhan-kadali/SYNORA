/// <reference types="@testing-library/jest-dom/vitest" />
import React from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import App from '../App';
import * as api from '../services/api';

describe('App Full User Journey Integration', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it('1. Mounts and fetches health status', async () => {
    vi.spyOn(api, 'checkHealth').mockResolvedValue({
      status: 'healthy',
      model_loaded: true,
      model_name: 'SYNORA A-Combined',
      feature_dimensions: 40,
      version: '1.0.0',
    });

    render(<App />);
    expect(screen.getByRole('heading', { name: 'SYNORA', level: 1 })).toBeInTheDocument();
    expect(screen.getByText('Acoustic Forensic Detection')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Engine Online (40D)')).toBeInTheDocument();
    });
  });

  it('2. Completes end-to-end analysis journey from file upload to results and reset', async () => {
    vi.spyOn(api, 'checkHealth').mockResolvedValue({
      status: 'healthy',
      model_loaded: true,
      model_name: 'SYNORA A-Combined',
      feature_dimensions: 40,
      version: '1.0.0',
    });

    vi.spyOn(api, 'analyzeAudio').mockResolvedValue({
      success: true,
      prediction: 'Fake',
      probabilities: { fake: 0.999863, real: 0.000137 },
      decision_threshold: 0.50,
      audio: { duration_seconds: 3.0, sample_rate: 16000, channels: 1 },
      model: { name: 'SYNORA A-Combined', feature_dimensions: 40 },
      disclaimer:
        'These probabilities are statistical estimates based on acoustic features. They do not establish definitive proof that an audio recording is genuine or AI-generated.',
    });

    render(<App />);

    // 1. Select audio file
    const wavFile = new File(['audio-bytes'], 'syn_sample.wav', { type: 'audio/wav' });
    const input = document.getElementById('audio-file-input') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [wavFile] } });

    expect(screen.getByText('syn_sample.wav')).toBeInTheDocument();

    // 2. Click Analyze
    const analyzeBtn = screen.getByRole('button', { name: /Analyze Audio/i });
    fireEvent.click(analyzeBtn);

    // 3. Verify Results are displayed
    await waitFor(() => {
      expect(screen.getByText('Predicted Fake (Synthetic Artifacts Detected)')).toBeInTheDocument();
      expect(screen.getByText('99.99%')).toBeInTheDocument();
    });

    // 4. Click Analyze Another Recording
    const resetBtn = screen.getByRole('button', { name: /Analyze Another Recording/i });
    fireEvent.click(resetBtn);

    // 5. Verify back to upload section
    expect(screen.getByText('Upload Speech Recording')).toBeInTheDocument();
  });

  it('3. Displays error alert when analysis API fails without exposing paths', async () => {
    vi.spyOn(api, 'checkHealth').mockResolvedValue({
      status: 'healthy',
      model_loaded: true,
      model_name: 'SYNORA A-Combined',
      feature_dimensions: 40,
      version: '1.0.0',
    });

    vi.spyOn(api, 'analyzeAudio').mockRejectedValue(
      new api.ApiError('Silent audio detected: zero frames exceed speech activity threshold.', 400)
    );

    render(<App />);

    const wavFile = new File(['silent-bytes'], 'silent.wav', { type: 'audio/wav' });
    const input = document.getElementById('audio-file-input') as HTMLInputElement;
    fireEvent.change(input, { target: { files: [wavFile] } });

    const analyzeBtn = screen.getByRole('button', { name: /Analyze Audio/i });
    fireEvent.click(analyzeBtn);

    await waitFor(() => {
      expect(screen.getByText('Invalid Audio Stream')).toBeInTheDocument();
      expect(
        screen.getByText('Silent audio detected: zero frames exceed speech activity threshold.')
      ).toBeInTheDocument();
    });
  });
});
