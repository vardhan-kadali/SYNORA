/// <reference types="@testing-library/jest-dom/vitest" />
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { Header } from '../components/Header';
import { AudioUpload } from '../components/AudioUpload';
import { AnalysisProgress } from '../components/AnalysisProgress';
import { AnalysisResults } from '../components/AnalysisResults';
import { ErrorAlert } from '../components/ErrorAlert';
import type { AnalysisResponse, HealthResponse } from '../types/api';

describe('UI Components Unit Tests', () => {
  it('1. Header displays branding, tagline, and online health status', () => {
    const health: HealthResponse = {
      status: 'healthy',
      model_loaded: true,
      model_name: 'SYNORA A-Combined',
      feature_dimensions: 40,
      version: '1.0.0',
    };

    render(<Header health={health} loadingHealth={false} />);
    expect(screen.getByText('SYNORA')).toBeInTheDocument();
    expect(
      screen.getByText('Detecting the Synthetic. Protecting the Real.')
    ).toBeInTheDocument();
    expect(screen.getByText('Engine Online (40D)')).toBeInTheDocument();
  });

  it('2. AudioUpload rejects empty 0-byte file', () => {
    const onAnalyze = vi.fn();
    render(<AudioUpload onAnalyze={onAnalyze} isAnalyzing={false} />);

    const emptyFile = new File([], 'empty.wav', { type: 'audio/wav' });
    const input = document.getElementById('audio-file-input') as HTMLInputElement;

    fireEvent.change(input, { target: { files: [emptyFile] } });
    expect(screen.getByText(/The selected file is empty/i)).toBeInTheDocument();
    expect(onAnalyze).not.toHaveBeenCalled();
  });

  it('3. AudioUpload rejects unsupported file extensions', () => {
    const onAnalyze = vi.fn();
    render(<AudioUpload onAnalyze={onAnalyze} isAnalyzing={false} />);

    const pdfFile = new File(['%PDF...'], 'document.pdf', { type: 'application/pdf' });
    const input = document.getElementById('audio-file-input') as HTMLInputElement;

    fireEvent.change(input, { target: { files: [pdfFile] } });
    expect(screen.getByText(/Unsupported format '.pdf'/i)).toBeInTheDocument();
  });

  it('4. AudioUpload accepts valid WAV file and triggers onAnalyze on button click', () => {
    const onAnalyze = vi.fn();
    render(<AudioUpload onAnalyze={onAnalyze} isAnalyzing={false} />);

    const wavFile = new File(['RIFF...fake audio bytes'], 'sample.wav', { type: 'audio/wav' });
    const input = document.getElementById('audio-file-input') as HTMLInputElement;

    fireEvent.change(input, { target: { files: [wavFile] } });
    expect(screen.getByText('sample.wav')).toBeInTheDocument();

    const analyzeBtn = screen.getByRole('button', { name: /Analyze Audio/i });
    expect(analyzeBtn).toBeInTheDocument();
    fireEvent.click(analyzeBtn);

    expect(onAnalyze).toHaveBeenCalledWith(wavFile);
  });

  it('5. AnalysisProgress displays forensic pipeline stages', () => {
    render(<AnalysisProgress />);
    expect(
      screen.getByText('Forensic Audio Analysis in Progress')
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Extracting 40-dim acoustic features/i)
    ).toBeInTheDocument();
  });

  it('6. AnalysisResults displays Fake prediction, probabilities, metadata, and scientific disclaimer', () => {
    const mockResult: AnalysisResponse = {
      success: true,
      prediction: 'Fake',
      probabilities: { fake: 0.941234, real: 0.058766 },
      decision_threshold: 0.50,
      audio: { duration_seconds: 3.5, sample_rate: 16000, channels: 1 },
      model: { name: 'SYNORA A-Combined', feature_dimensions: 40 },
      disclaimer:
        'These probabilities are statistical estimates based on acoustic features. They do not establish definitive proof that an audio recording is genuine or AI-generated.',
    };

    const onReset = vi.fn();
    render(<AnalysisResults result={mockResult} onReset={onReset} />);

    // Prediction
    expect(screen.getByText('Predicted Fake (Synthetic Artifacts Detected)')).toBeInTheDocument();
    expect(screen.getByText('Prediction: Fake')).toBeInTheDocument();

    // Probabilities
    expect(screen.getByText('94.12%')).toBeInTheDocument();
    expect(screen.getByText('5.88%')).toBeInTheDocument();
    expect(screen.getByText(/P\(fake \| x\) = 0.941234/)).toBeInTheDocument();

    // Metadata
    expect(screen.getByText('3.50 seconds')).toBeInTheDocument();
    expect(screen.getByText('16,000 Hz')).toBeInTheDocument();
    expect(screen.getByText('SYNORA A-Combined')).toBeInTheDocument();

    // Mandatory Scientific Humility Disclaimer
    expect(
      screen.getByText(
        'These probabilities are statistical estimates based on acoustic features. They do not establish definitive proof that an audio recording is genuine or AI-generated.'
      )
    ).toBeInTheDocument();

    // Reset button
    const resetBtn = screen.getByRole('button', { name: /Analyze Another Recording/i });
    fireEvent.click(resetBtn);
    expect(onReset).toHaveBeenCalled();
  });

  it('7. AnalysisResults displays Real prediction correctly', () => {
    const mockResult: AnalysisResponse = {
      success: true,
      prediction: 'Real',
      probabilities: { fake: 0.021000, real: 0.979000 },
      decision_threshold: 0.50,
      audio: { duration_seconds: 2.1, sample_rate: 16000, channels: 1 },
      model: { name: 'SYNORA A-Combined', feature_dimensions: 40 },
      disclaimer: 'These probabilities are statistical estimates based on acoustic features.',
    };

    render(<AnalysisResults result={mockResult} onReset={vi.fn()} />);
    expect(screen.getByText('Predicted Real (Likely Human Speech)')).toBeInTheDocument();
    expect(screen.getByText('Prediction: Real')).toBeInTheDocument();
    expect(screen.getByText('97.90%')).toBeInTheDocument();
  });

  it('8. ErrorAlert renders message and triggers dismiss callback', () => {
    const onDismiss = vi.fn();
    render(<ErrorAlert title="Upload Limit Exceeded" message="Audio file is too large." onDismiss={onDismiss} />);

    expect(screen.getByText('Upload Limit Exceeded')).toBeInTheDocument();
    expect(screen.getByText('Audio file is too large.')).toBeInTheDocument();

    const dismissBtn = screen.getByRole('button', { name: /Dismiss error message/i });
    fireEvent.click(dismissBtn);
    expect(onDismiss).toHaveBeenCalled();
  });
});

