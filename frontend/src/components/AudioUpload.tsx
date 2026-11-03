import React, { useState, useRef } from 'react';
import type { ChangeEvent, DragEvent } from 'react';
import { UploadCloudIcon, FileAudioIcon, XIcon, WaveformIcon, InfoIcon } from './Icons';

interface AudioUploadProps {
  onAnalyze: (file: File) => void;
  isAnalyzing: boolean;
  disabled?: boolean;
}

const MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024; // 25 MB
const SUPPORTED_EXTENSIONS = ['.wav', '.flac', '.ogg', '.mp3'];

export const AudioUpload: React.FC<AudioUploadProps> = ({
  onAnalyze,
  isAnalyzing,
  disabled = false,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState<boolean>(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const validateAndSetFile = (file: File) => {
    setValidationError(null);

    // 1. Empty check
    if (file.size === 0) {
      setValidationError('The selected file is empty (0 bytes). Please choose a valid audio file.');
      return;
    }

    // 2. Size check (25 MB)
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setValidationError(
        `File size (${formatFileSize(file.size)}) exceeds the maximum allowed limit of 25 MB.`
      );
      return;
    }

    // 3. Extension check
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      setValidationError(
        `Unsupported format '${ext}'. Allowed formats are WAV, FLAC, OGG, and MP3.`
      );
      return;
    }

    // Passed validation
    setSelectedFile(file);
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
    }
    const objectUrl = URL.createObjectURL(file);
    setAudioPreviewUrl(objectUrl);
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSetFile(e.target.files[0]);
    }
  };

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSetFile(e.dataTransfer.files[0]);
    }
  };

  const handleRemoveFile = () => {
    setSelectedFile(null);
    setValidationError(null);
    if (audioPreviewUrl) {
      URL.revokeObjectURL(audioPreviewUrl);
      setAudioPreviewUrl(null);
    }
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleAnalyzeClick = () => {
    if (!selectedFile || isAnalyzing || disabled) return;
    onAnalyze(selectedFile);
  };

  return (
    <div className="upload-card">
      <input
        ref={fileInputRef}
        type="file"
        id="audio-file-input"
        accept=".wav,.flac,.ogg,.mp3,audio/wav,audio/flac,audio/ogg,audio/mpeg"
        style={{ display: 'none' }}
        onChange={handleFileChange}
        disabled={isAnalyzing || disabled}
      />

      {!selectedFile ? (
        <div
          className={`dropzone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragOver={handleDrag}
          onDragLeave={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              fileInputRef.current?.click();
            }
          }}
        >
          <div className="dropzone-icon">
            <UploadCloudIcon size={32} />
          </div>
          <div className="dropzone-title">Upload Speech Recording</div>
          <p className="dropzone-prompt">
            Drag and drop an audio file here, or <span style={{ color: 'var(--accent-cyan)' }}>browse local files</span>
          </p>
          <div className="dropzone-meta">
            <span>
              <WaveformIcon size={14} /> Formats: WAV, FLAC, OGG, MP3
            </span>
            <span>
              <InfoIcon size={14} /> Max Size: 25 MB
            </span>
            <span>
              <InfoIcon size={14} /> Max Duration: 60s
            </span>
          </div>
        </div>
      ) : (
        <div className="selected-file-panel">
          <div className="file-header-row">
            <div className="file-identity">
              <div className="file-icon">
                <FileAudioIcon size={24} />
              </div>
              <div className="file-details">
                <div className="file-name">{selectedFile.name}</div>
                <div className="file-meta-tags">
                  <span>{formatFileSize(selectedFile.size)}</span>
                  <span>•</span>
                  <span>{selectedFile.type || 'audio/' + selectedFile.name.split('.').pop()}</span>
                </div>
              </div>
            </div>
            {!isAnalyzing && (
              <button
                type="button"
                className="btn-remove-file"
                onClick={handleRemoveFile}
                title="Remove selected file"
                aria-label="Remove audio file"
              >
                <XIcon size={20} />
              </button>
            )}
          </div>

          {audioPreviewUrl && (
            <div className="audio-preview-bar">
              <audio controls src={audioPreviewUrl}>
                Your browser does not support the audio element.
              </audio>
            </div>
          )}
        </div>
      )}

      {validationError && (
        <div style={{ marginTop: '1rem', color: 'var(--status-fake)', fontSize: '0.8125rem' }}>
          {validationError}
        </div>
      )}

      {selectedFile && (
        <div className="action-row">
          <button
            type="button"
            className="btn-analyze"
            onClick={handleAnalyzeClick}
            disabled={isAnalyzing || disabled || !!validationError}
          >
            <WaveformIcon size={18} />
            {isAnalyzing ? 'Analyzing Audio...' : 'Analyze Audio'}
          </button>
        </div>
      )}
    </div>
  );
};
