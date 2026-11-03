import React from 'react';

export const AnalysisProgress: React.FC = () => {
  return (
    <div className="analysis-progress-card" role="status" aria-live="polite">
      <div className="spinner-ring" />
      <h2 className="progress-title">Forensic Audio Analysis in Progress</h2>
      <p className="progress-subtitle">
        The following stages describe the analysis pipeline. Progress is illustrative and does not represent live backend updates.
      </p>

      <div className="progress-steps-list">
        <div className="progress-step-item active">
          <div className="step-indicator" />
          <span>Decoding audio stream & resampling to 16 kHz mono</span>
        </div>
        <div className="progress-step-item active">
          <div className="step-indicator" />
          <span>Extracting 40-dim acoustic features (LFCC, Wiener Flatness, Flux, Deltas)</span>
        </div>
        <div className="progress-step-item active">
          <div className="step-indicator" />
          <span>Applying StandardScaler feature normalization</span>
        </div>
        <div className="progress-step-item active">
          <div className="step-indicator" />
          <span>Computing model decision scores via temperature scaling (T = 1.123)</span>
        </div>
      </div>
    </div>
  );
};

