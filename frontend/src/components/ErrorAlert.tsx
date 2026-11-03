import React from 'react';
import { AlertTriangleIcon, XIcon } from './Icons';

interface ErrorAlertProps {
  title?: string;
  message: string;
  onDismiss?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({
  title = 'Analysis Failed',
  message,
  onDismiss,
}) => {
  return (
    <div className="alert-card" role="alert">
      <div className="alert-content">
        <div className="alert-icon">
          <AlertTriangleIcon size={22} />
        </div>
        <div>
          <div className="alert-title">{title}</div>
          <div className="alert-message">{message}</div>
        </div>
      </div>
      {onDismiss && (
        <button
          type="button"
          className="alert-close"
          onClick={onDismiss}
          aria-label="Dismiss error message"
        >
          <XIcon size={18} />
        </button>
      )}
    </div>
  );
};

