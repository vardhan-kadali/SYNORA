import React from 'react';
import { WaveformIcon } from './Icons';
import type { HealthResponse } from '../types/api';

interface HeaderProps {
  health: HealthResponse | null;
  loadingHealth: boolean;
}

export const Header: React.FC<HeaderProps> = ({ health, loadingHealth }) => {
  const isHealthy = health?.status === 'healthy' && health?.model_loaded;

  return (
    <header className="header">
      <div className="header-inner">
        <div className="brand-section">
          <div className="brand-logo-icon">
            <WaveformIcon size={22} />
          </div>
          <div className="brand-text">
            <h1>SYNORA</h1>
            <span className="tagline">Detecting the Synthetic. Protecting the Real.</span>
          </div>
        </div>

        <div className="header-status" title={health ? `Model: ${health.model_name}` : 'Checking connection...'}>
          <span className={`status-dot ${loadingHealth ? '' : isHealthy ? 'online' : 'offline'}`} />
          <span>
            {loadingHealth
              ? 'Connecting...'
              : isHealthy
              ? `Engine Online (${health.feature_dimensions}D)`
              : 'Service Unavailable'}
          </span>
        </div>
      </div>
    </header>
  );
};
