import React from 'react';
import type { AnalysisResponse } from '../types/api';
import {
  ShieldIcon,
  AlertTriangleIcon,
  WaveformIcon,
  CpuIcon,
  RefreshIcon,
  InfoIcon,
} from './Icons';

interface AnalysisResultsProps {
  result: AnalysisResponse;
  onReset: () => void;
}

export const AnalysisResults: React.FC<AnalysisResultsProps> = ({ result, onReset }) => {
  const isFake = result.prediction === 'Fake';
  const fakeProb = result.probabilities.fake;
  const realProb = result.probabilities.real;
  const fakePct = (fakeProb * 100).toFixed(2);
  const realPct = (realProb * 100).toFixed(2);

  return (
    <div className="results-container" role="region" aria-label="Forensic Analysis Results">
      {/* 1. Primary Verdict Banner */}
      <div className={`verdict-banner ${isFake ? 'verdict-fake' : 'verdict-real'}`}>
        <div className="verdict-info">
          <div className="verdict-icon-box">
            {isFake ? <AlertTriangleIcon size={32} /> : <ShieldIcon size={32} />}
          </div>
          <div>
            <div className="verdict-label-meta">Forensic Classification</div>
            <div className="verdict-heading">
              {isFake ? 'Predicted Fake (Synthetic Artifacts Detected)' : 'Predicted Real (Likely Human Speech)'}
            </div>
            <div className="verdict-summary">
              {isFake
                ? 'Acoustic patterns exhibit spectral and cepstral characteristics consistent with speech synthesis.'
                : 'Acoustic patterns exhibit spectral and cepstral characteristics consistent with genuine human vocalization.'}
            </div>
          </div>
        </div>
        <div className="verdict-badge">
          Prediction: {result.prediction}
        </div>
      </div>

      {/* 2. Probability Distribution Card */}
      <div className="probability-card">
        <div className="card-title">
          <WaveformIcon size={20} />
          <span>Estimated Probability Distribution</span>
        </div>
        <div className="card-subtitle">
          These are exploratory temperature-scaled model probability estimates (T = 1.123). Their empirical calibration has not been established.
        </div>

        <div className="probability-scores-grid">
          <div className="score-box fake-box">
            <div className="score-label">Synthetic (Fake) Probability</div>
            <div className="score-value">{fakePct}%</div>
            <div className="score-fraction">P(fake | x) = {fakeProb.toFixed(6)}</div>
          </div>

          <div className="score-box real-box">
            <div className="score-label">Genuine (Real) Probability</div>
            <div className="score-value">{realPct}%</div>
            <div className="score-fraction">P(real | x) = {realProb.toFixed(6)}</div>
          </div>
        </div>

        {/* Dual Distribution Bar Meter */}
        <div className="meter-container">
          <div className="meter-track" aria-label="Probability meter">
            <div
              className="meter-fill-fake"
              style={{ width: `${fakeProb * 100}%` }}
              title={`Fake: ${fakePct}%`}
            />
            <div
              className="meter-fill-real"
              style={{ width: `${realProb * 100}%` }}
              title={`Real: ${realPct}%`}
            />
            <div className="meter-threshold-marker" title="Decision Threshold (0.50)" />
          </div>

          <div className="meter-legend">
            <span>Fake ({fakePct}%)</span>
            <span>Threshold = {result.decision_threshold.toFixed(2)}</span>
            <span>Real ({realPct}%)</span>
          </div>

          <div className="threshold-explainer">
            <strong>Operating Decision Rule:</strong> Binary classification assigns <em>Fake</em> if P(fake | x) &ge; {result.decision_threshold.toFixed(2)}, otherwise <em>Real</em>.
          </div>
        </div>
      </div>

      {/* 3. Telemetry & Metadata Details Grid */}
      <div className="telemetry-grid">
        {/* Audio Recording Metadata */}
        <div className="telemetry-card">
          <div className="card-title">
            <WaveformIcon size={18} />
            <span>Audio Stream Metadata</span>
          </div>
          <div className="telemetry-rows">
            <div className="telemetry-row">
              <span className="telemetry-key">Analyzed Duration</span>
              <span className="telemetry-val">{result.audio.duration_seconds.toFixed(2)} seconds</span>
            </div>
            <div className="telemetry-row">
              <span className="telemetry-key">Processing Sample Rate</span>
              <span className="telemetry-val">{result.audio.sample_rate.toLocaleString()} Hz</span>
            </div>
            <div className="telemetry-row">
              <span className="telemetry-key">Channel Downmix</span>
              <span className="telemetry-val">{result.audio.channels > 1 ? `${result.audio.channels} &rarr; Mono (1 ch)` : 'Mono (1 ch)'}</span>
            </div>
          </div>
        </div>

        {/* Inference Model Specifications */}
        <div className="telemetry-card">
          <div className="card-title">
            <CpuIcon size={18} />
            <span>Model Specifications</span>
          </div>
          <div className="telemetry-rows">
            <div className="telemetry-row">
              <span className="telemetry-key">Active Model</span>
              <span className="telemetry-val">{result.model.name}</span>
            </div>
            <div className="telemetry-row">
              <span className="telemetry-key">Feature Dimensionality</span>
              <span className="telemetry-val">{result.model.feature_dimensions} dimensions</span>
            </div>
            <div className="telemetry-row">
              <span className="telemetry-key">Decision Threshold</span>
              <span className="telemetry-val">{result.decision_threshold.toFixed(2)}</span>
            </div>
          </div>
        </div>
      </div>

      {/* 4. Scientific Humility Disclaimer Card */}
      <div className="disclaimer-card">
        <div className="disclaimer-icon">
          <InfoIcon size={24} />
        </div>
        <div>
          <div className="disclaimer-heading">Scientific Disclaimer & Limitations</div>
          <div className="disclaimer-body">
            {result.disclaimer}
          </div>
        </div>
      </div>

      {/* 5. Analyze Another File Action */}
      <div className="results-actions">
        <button type="button" className="btn-secondary" onClick={onReset}>
          <RefreshIcon size={18} />
          Analyze Another Recording
        </button>
      </div>
    </div>
  );
};
