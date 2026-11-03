import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { AudioUpload } from './components/AudioUpload';
import { AnalysisProgress } from './components/AnalysisProgress';
import { AnalysisResults } from './components/AnalysisResults';
import { ErrorAlert } from './components/ErrorAlert';
import { Footer } from './components/Footer';
import { checkHealth, analyzeAudio, ApiError } from './services/api';
import type { HealthResponse, AnalysisResponse } from './types/api';
import './App.css';

export const App: React.FC = () => {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState<boolean>(true);
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<{ title?: string; message: string } | null>(null);

  // Poll or check health on initial mount
  useEffect(() => {
    let mounted = true;

    const fetchHealth = async () => {
      try {
        const data = await checkHealth();
        if (mounted) {
          setHealth(data);
          setLoadingHealth(false);
        }
      } catch {
        if (mounted) {
          setHealth(null);
          setLoadingHealth(false);
        }
      }
    };

    fetchHealth();
    return () => {
      mounted = false;
    };
  }, []);

  const handleAnalyze = async (file: File) => {
    setError(null);
    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      const result = await analyzeAudio(file);
      setAnalysisResult(result);
    } catch (err: unknown) {
      if (err instanceof ApiError) {
        let title = 'Analysis Error';
        if (err.statusCode === 503) {
          title = 'Inference Engine Unavailable';
        } else if (err.statusCode === 413) {
          title = 'Payload Limit Exceeded';
        } else if (err.statusCode === 415) {
          title = 'Unsupported Audio Format';
        } else if (err.statusCode === 400) {
          title = 'Invalid Audio Stream';
        } else if (err.statusCode === 0) {
          title = 'Service Unreachable';
        }
        setError({ title, message: err.message });
      } else {
        setError({
          title: 'Unexpected Error',
          message: 'An unexpected client error occurred while processing the request.',
        });
      }
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleReset = () => {
    setAnalysisResult(null);
    setError(null);
  };

  return (
    <div className="app-container">
      <Header health={health} loadingHealth={loadingHealth} />

      <main className="main-content">
        {/* Hero Section */}
        <section className="hero-section">
          <div className="hero-badge">Acoustic Forensic Detection</div>
          <h2 className="hero-title">
            Acoustic Forensic Analysis for <span>Speech Verification</span>
          </h2>
          <p className="hero-subtitle">
            SYNORA analyzes subtle spectral and cepstral anomalies across 40 acoustic dimensions
            to evaluate synthetic voice generation in Indian-language and telephony audio.
          </p>
        </section>

        {/* Global Error Alert */}
        {error && (
          <ErrorAlert
            title={error.title}
            message={error.message}
            onDismiss={() => setError(null)}
          />
        )}

        {/* Dynamic View: Upload vs Progress vs Results */}
        {isAnalyzing ? (
          <AnalysisProgress />
        ) : analysisResult ? (
          <AnalysisResults result={analysisResult} onReset={handleReset} />
        ) : (
          <AudioUpload
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
            disabled={!health?.model_loaded && !loadingHealth}
          />
        )}
      </main>

      <Footer />
    </div>
  );
};

export default App;
