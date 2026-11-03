import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <div>
          <span className="footer-brand">SYNORA</span> &mdash; Audio Deepfake Detection System
        </div>
        <div>
          &copy; {new Date().getFullYear()} SYNORA Research. Detecting the Synthetic. Protecting the Real.
        </div>
      </div>
    </footer>
  );
};

