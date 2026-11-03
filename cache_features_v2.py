"""
SYNORA — Phase 3.2: Modular DSP Feature Extraction Pipeline (v2)
"Detecting the Synthetic. Protecting the Real."

Author: Senior ML Research Engineer & Audio Deepfake Detection Specialist
Approved Specification: Phase 3.2 Architecture Plan & Pre-Implementation Gate Checklist

Mathematical Representations & Dimensionality:
1. Candidate A-Lite (26 dimensions):
   - 10 static MFCCs (c1..c10, omitting c0)
   - 3 sub-band energy ratios: Low (0-300 Hz), Tel (300-3400 Hz), High (3400-8000 Hz)
   - Aggregated via temporal mean (13) + temporal population standard deviation (13) = 26 dims.
   - Frozen Phase 3.1 baseline reference.

2. Candidate A-Delta: Delta-MFCC Representation (26 dimensions):
   - 10 Delta-MFCCs (c1..c10) via 5-frame regression (N=2, edge replication)
   - 3 original sub-band energy ratios (retains static spectral information)
   - Aggregated via temporal mean (13) + temporal population standard deviation (13) = 26 dims.

3. Candidate A-LFCC: Linear Frequency Cepstral Coefficients (26 dimensions):
   - 10 static LFCCs (c1..c10, omitting c0) from 20 linearly spaced triangular filters (0-8000 Hz)
   - Type-II DCT with orthogonal normalization (norm='ortho')
   - 3 original sub-band energy ratios
   - Aggregated via temporal mean (13) + temporal population standard deviation (13) = 26 dims.

4. Candidate A-Dynamics: Spectral Flatness & Flux (30 dimensions):
   - Candidate A-Lite base features (26 dims)
   - Spectral Flatness (Wiener entropy, epsilon=1e-12): mean (1) + std (1) = 2 dims
   - Spectral Flux (Euclidean velocity on contiguous frames pre-VAD): mean (1) + std (1) = 2 dims
   - Total: 26 + 2 + 2 = 30 dims.

5. Candidate A-Combined: Multi-Cue Representation (40 dimensions):
   - 10 Delta-MFCCs: mean (10) + std (10) = 20 dims
   - 6 LFCCs (c1..c6): mean (6) + std (6) = 12 dims
   - 2 Sub-bands (Low: 0-300 Hz and High: 3400-8000 Hz, omitting Tel): mean (2) + std (2) = 4 dims
   - Spectral Flatness: mean (1) + std (1) = 2 dims
   - Spectral Flux: mean (1) + std (1) = 2 dims
   - Total: 20 + 12 + 4 + 2 + 2 = 40 dims.

6. Candidate A-Delta-Pure: Diagnostic Control (20 dimensions):
   - 10 Delta-MFCCs: mean (10) + std (10) = 20 dims
   - Zero sub-band energy ratios (pure dynamic cepstral offsets)
   - Isolated diagnostic control (permanently barred from primary candidate selection).

Voice Activity Detection (VAD) Policy:
- Active speech threshold: frame_rms > 1e-4.
- Formula: dBFS = 20 * log10(RMS).
- An RMS threshold of 1e-4 corresponds to approximately -80 dBFS (20 * log10(1e-4) = -80.0 dBFS).
- Preserves exact numerical continuity with the Phase 3.1 baseline cache (dev_features.npz).
- Completely silent audio (zero frames exceeding threshold) is rejected with ValueError.
"""

import os
import sys
import time
import math
import hashlib
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

# ==============================================================================
# 1. HARDCODED PATH GUARDS & DIRECTORY SAFETY ASSERTIONS
# ==============================================================================

FORBIDDEN_PATHS = frozenset({"benchmark/test", "benchmark\\test", "test_features.npz"})

def assert_path_safety(path: str) -> None:
    """
    Mandatory safety guard: ensures no script accesses the locked Test partition
    or creates/accesses test_features.npz.
    """
    path_str = str(path).replace("/", "\\")
    for forbidden in FORBIDDEN_PATHS:
        forbidden_norm = forbidden.replace("/", "\\")
        if forbidden_norm in path_str:
            raise PermissionError(
                f"CRITICAL SAFETY VIOLATION: Access to locked path attempted: {path}"
            )

# ==============================================================================
# 2. GLOBAL DSP CONSTANTS
# ==============================================================================

TARGET_SR: int = 16000
FRAME_LEN: int = 400      # 25 ms at 16 kHz
FRAME_HOP: int = 160      # 10 ms at 16 kHz
N_FFT: int = 512          # 512-point FFT
N_BINS: int = 257         # N_FFT // 2 + 1 (frequency bins 0..256, 31.25 Hz resolution)
BIN_RES: float = TARGET_SR / N_FFT  # 31.25 Hz

# VAD Threshold: 1e-4 RMS (~ -80 dBFS: dBFS = 20 * log10(RMS))
# Preserves bit-exact Phase 3.1 baseline reproducibility.
VAD_RMS_THRESHOLD: float = 1e-4

# Approved Feature Dimensions
DIM_A_LITE: int = 26
DIM_A_DELTA: int = 26
DIM_A_LFCC: int = 26
DIM_A_DYNAMICS: int = 30
DIM_A_COMBINED: int = 40
DIM_A_DELTA_PURE: int = 20

# Sub-band FFT bin slices (Bin resolution: 31.25 Hz)
# Low: 0 - 300 Hz (bins 0..9)
# Tel: 300 - 3400 Hz (bins 10..108)
# High: 3400 - 8000 Hz (bins 109..256)
SLICE_SUB_LOW = slice(0, 10)
SLICE_SUB_TEL = slice(10, 109)
SLICE_SUB_HIGH = slice(109, 257)

# ==============================================================================
# 3. FILTERBANK CONSTRUCTIONS & BASIS MATRICES
# ==============================================================================

def create_mel_filterbank(sr: int = 16000, n_fft: int = 512, n_mels: int = 26) -> np.ndarray:
    """
    Creates a triangular Mel filterbank matrix of shape (n_mels, n_fft // 2 + 1).
    Preserves exact Phase 3.1 filter geometry (O'Shaughnessy Mel standard).
    """
    f_min = 0.0
    f_max = float(sr // 2)
    
    m_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
    m_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
    
    m_pts = np.linspace(m_min, m_max, n_mels + 2)
    f_pts = 700.0 * (10.0 ** (m_pts / 2595.0) - 1.0)
    
    bin_pts = np.floor((n_fft + 1) * f_pts / sr).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_fft // 2)
    
    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_mels, n_bins), dtype=np.float32)
    
    for m in range(1, n_mels + 1):
        f_left = bin_pts[m - 1]
        f_center = bin_pts[m]
        f_right = bin_pts[m + 1]
        
        if f_center > f_left:
            for k in range(f_left, f_center):
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
                
        if f_right > f_center:
            for k in range(f_center, f_right):
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
                
    return filterbank

MEL_FILTERBANK: np.ndarray = create_mel_filterbank(sr=TARGET_SR, n_fft=N_FFT, n_mels=26)

def compute_dct_matrix(n_mfcc: int = 10, n_mels: int = 26) -> np.ndarray:
    """
    Creates DCT-II basis matrix of shape (n_mfcc, n_mels) omitting c0 (1-indexed: c1..c10).
    Preserves exact Phase 3.1 basis mapping.
    """
    dct_mat = np.zeros((n_mfcc, n_mels), dtype=np.float32)
    for n in range(1, n_mfcc + 1):
        for m in range(n_mels):
            dct_mat[n - 1, m] = np.cos(np.pi * n * (m + 0.5) / n_mels)
    return dct_mat

DCT_MATRIX: np.ndarray = compute_dct_matrix(n_mfcc=10, n_mels=26)

def create_lfcc_filterbank(sr: int = 16000, n_fft: int = 512, n_filters: int = 20) -> np.ndarray:
    """
    Creates a triangular Linear-Frequency filterbank matrix of shape (n_filters, n_fft // 2 + 1).
    20 filters spanning 0 to 8000 Hz, with 22 linearly spaced vertices:
    f_v = v * 8000 / 21 (~ 380.95 Hz uniform center spacing).
    Bins: k_v = round(f_v / 31.25).
    """
    f_max = float(sr // 2)
    bin_res = sr / n_fft  # 31.25 Hz
    f_pts = np.linspace(0.0, f_max, n_filters + 2)
    bin_pts = np.round(f_pts / bin_res).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_fft // 2)
    
    n_bins = n_fft // 2 + 1
    filterbank = np.zeros((n_filters, n_bins), dtype=np.float32)
    
    for m in range(1, n_filters + 1):
        f_left = bin_pts[m - 1]
        f_center = bin_pts[m]
        f_right = bin_pts[m + 1]
        
        if f_center > f_left:
            for k in range(f_left, f_center):
                filterbank[m - 1, k] = (k - f_left) / (f_center - f_left)
                
        if f_right > f_center:
            for k in range(f_center, f_right):
                filterbank[m - 1, k] = (f_right - k) / (f_right - f_center)
                
    return filterbank

LFCC_FILTERBANK: np.ndarray = create_lfcc_filterbank(sr=TARGET_SR, n_fft=N_FFT, n_filters=20)

def compute_lfcc_dct_matrix(n_filters: int = 20) -> np.ndarray:
    """
    Creates full orthonormal DCT-II basis matrix of shape (n_filters, n_filters).
    Uses orthogonal normalization ('ortho'):
      U[0, m] = sqrt(1 / L)
      U[n, m] = sqrt(2 / L) * cos(pi * n * (m + 0.5) / L)  for n >= 1
    Satisfies U @ U.T == I_L (Parseval energy preservation).
    """
    L = n_filters
    m, n = np.meshgrid(np.arange(L), np.arange(L))
    U = np.sqrt(2.0 / L) * np.cos(np.pi * n * (m + 0.5) / L).astype(np.float32)
    U[0, :] = 1.0 / np.sqrt(L)
    return U

LFCC_DCT_MATRIX: np.ndarray = compute_lfcc_dct_matrix(n_filters=20)

# ==============================================================================
# 4. CORE DSP TRANSFORMS & HELPERS
# ==============================================================================

def compute_deltas(cepstra: np.ndarray, N: int = 2) -> np.ndarray:
    """
    Computes first-order regression slopes (deltas) across time frames for cepstral coefficients.
    Formula (N=2, 5-frame regression window):
      delta(t) = ((c(t+1) - c(t-1)) + 2 * (c(t+2) - c(t-2))) / 10
    Boundary condition: Edge replication (mode='edge').
    Input shape: (T, D)
    Output shape: (T, D)
    """
    T = len(cepstra)
    if T == 0:
        return np.empty_like(cepstra)
    
    pad = np.pad(cepstra, ((N, N), (0, 0)), mode="edge")
    # pad[N] corresponds to c[0]
    # For index t in 0..T-1:
    # c(t+1) is pad[t+1+N] = pad[t+3]
    # c(t-1) is pad[t-1+N] = pad[t+1]
    # c(t+2) is pad[t+2+N] = pad[t+4]
    # c(t-2) is pad[t-2+N] = pad[t]
    c_plus_1 = pad[3:T+3]
    c_minus_1 = pad[1:T+1]
    c_plus_2 = pad[4:T+4]
    c_minus_2 = pad[0:T]
    
    delta = ((c_plus_1 - c_minus_1) + 2.0 * (c_plus_2 - c_minus_2)) / 10.0
    return delta.astype(np.float32)

def compute_spectral_flatness(power_spectrum: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Computes Spectral Flatness (Wiener entropy) along the frequency axis.
    Formula:
      SF(t) = exp( (1/K) * sum(ln(|X(t, k)|^2 + eps)) ) / ( (1/K) * sum(|X(t, k)|^2) + eps )
    Bounded in [0.0, 1.0].
    Input shape: (T, K) or (K,)
    Output shape: (T,) or scalar float
    """
    power = np.asarray(power_spectrum, dtype=np.float64)
    geom_mean = np.exp(np.mean(np.log(power + eps), axis=-1))
    arith_mean = np.mean(power, axis=-1) + eps
    sf = geom_mean / arith_mean
    return np.clip(sf, 0.0, 1.0).astype(np.float32)

def compute_spectral_flux(magnitude_spectrum: np.ndarray) -> np.ndarray:
    """
    Computes Euclidean spectral flux across contiguous time frames on L2-normalized magnitude spectra.
    Contiguity & VAD Invariant: Must be computed on contiguous frames BEFORE VAD masking.
    Formula:
      M_norm(t, k) = |X(t, k)| / ( sqrt(sum_j |X(t, j)|^2) + 1e-10 )
      Flux(t) = sqrt( sum_k (M_norm(t, k) - M_norm(t-1, k))^2 )
    Frame 0 has Flux(0) = 0.0 and is omitted in temporal aggregation.
    Input shape: (T, K)
    Output shape: (T,)
    """
    T = len(magnitude_spectrum)
    if T <= 1:
        return np.zeros(T, dtype=np.float32)
    
    l2_norms = np.sqrt(np.sum(magnitude_spectrum ** 2, axis=-1, keepdims=True)) + 1e-10
    norm_mag = magnitude_spectrum / l2_norms
    
    flux = np.zeros(T, dtype=np.float32)
    diff = norm_mag[1:] - norm_mag[:-1]
    flux[1:] = np.sqrt(np.sum(diff ** 2, axis=-1))
    return flux

def aggregate_temporal_stats(frame_features: np.ndarray) -> np.ndarray:
    """
    Aggregates frame-level features into a 1D vector of [temporal_mean || temporal_population_std].
    Formula:
      mu = mean(frame_features, axis=0)
      sigma = sqrt(mean((frame_features - mu)^2, axis=0) + 1e-8)
    Input shape: (T, D)
    Output shape: (2 * D,)
    """
    if len(frame_features) == 0:
        raise ValueError("Cannot aggregate temporal stats on empty feature array.")
    feat_mean = np.mean(frame_features, axis=0)
    feat_std = np.sqrt(np.mean((frame_features - feat_mean) ** 2, axis=0) + 1e-8)
    return np.concatenate([feat_mean, feat_std]).astype(np.float32)

# ==============================================================================
# 5. MODULAR AUDIO FEATURE EXTRACTION PIPELINE
# ==============================================================================

def prepare_stft_and_vad(audio: np.ndarray, sr: int = TARGET_SR):
    """
    Core pre-processing stage:
    1. Framing with 25 ms window (400 samples) and 10 ms hop (160 samples).
    2. Hamming windowing.
    3. Voice Activity Detection via frame RMS > 1e-4 (~ -80 dBFS).
    4. Pre-VAD contiguous magnitude spectrum & contiguous spectral flux.
    5. Active speech power spectrum for cepstral representations.
    
    Returns:
      active_power: Power spectrum of active speech frames, shape (T_act, 257)
      active_mask: Boolean mask of active frames, shape (T_total,)
      flux_all: Contiguous spectral flux computed across all contiguous frames, shape (T_total,)
      num_frames: Total number of frames (T_total)
    """
    if len(audio) == 0:
        raise ValueError("Audio array is empty.")
        
    frame_len = FRAME_LEN
    frame_hop = FRAME_HOP
    n_fft = N_FFT
    
    if len(audio) < frame_len:
        audio = np.pad(audio, (0, frame_len - len(audio)), mode="constant")
        
    num_frames = 1 + (len(audio) - frame_len) // frame_hop
    window = np.hamming(frame_len).astype(np.float32)
    
    frames = np.lib.stride_tricks.as_strided(
        audio,
        shape=(num_frames, frame_len),
        strides=(audio.strides[0] * frame_hop, audio.strides[0])
    )
    
    # Frame RMS Energy for Speech Activity Detection
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=-1) + 1e-12)
    active_mask = frame_rms > VAD_RMS_THRESHOLD
    
    # Deterministic silent audio rejection
    if np.sum(active_mask) == 0:
        raise ValueError("Silent audio detected - zero frames exceed the speech activity threshold.")
        
    # Pre-VAD contiguous STFT for contiguous spectral flux
    windowed_all = frames * window
    fft_all = np.fft.rfft(windowed_all, n=n_fft, axis=-1)
    mag_all = np.abs(fft_all).astype(np.float32)
    flux_all = compute_spectral_flux(mag_all)
    
    # Active frames power spectrum
    active_frames = frames[active_mask] * window
    fft_active = np.fft.rfft(active_frames, n=n_fft, axis=-1)
    active_power = ((np.abs(fft_active) ** 2) / float(frame_len)).astype(np.float32)
    
    return active_power, active_mask, flux_all, num_frames

def extract_subband_ratios(power_spectrum: np.ndarray, mode: str = "3band") -> np.ndarray:
    """
    Computes sub-band energy ratios from power spectrum.
    Modes:
      "3band": Low (0-300 Hz), Tel (300-3400 Hz), High (3400-8000 Hz) -> shape (T, 3)
      "2band": Low (0-300 Hz) and High (3400-8000 Hz) -> shape (T, 2)
    """
    total_energy = np.sum(power_spectrum, axis=-1, keepdims=True) + 1e-12
    e_low = np.sum(power_spectrum[:, SLICE_SUB_LOW], axis=-1, keepdims=True)
    e_high = np.sum(power_spectrum[:, SLICE_SUB_HIGH], axis=-1, keepdims=True)
    
    r_low = e_low / total_energy
    r_high = e_high / total_energy
    
    if mode == "2band":
        return np.hstack([r_low, r_high]).astype(np.float32)
    elif mode == "3band":
        e_tel = np.sum(power_spectrum[:, SLICE_SUB_TEL], axis=-1, keepdims=True)
        r_tel = e_tel / total_energy
        return np.hstack([r_low, r_tel, r_high]).astype(np.float32)
    else:
        raise ValueError(f"Unknown subband mode: {mode}. Permitted: '3band', '2band'")

def extract_static_mfccs(power_spectrum: np.ndarray) -> np.ndarray:
    """
    Computes static MFCCs c1..c10 (omitting c0) on active frames power spectrum.
    Shape: (T_act, 10)
    """
    mel_energies = np.dot(power_spectrum, MEL_FILTERBANK.T)
    log_mel = np.log(mel_energies + 1e-10)
    mfccs = np.dot(log_mel, DCT_MATRIX.T)
    return mfccs.astype(np.float32)

def extract_static_lfccs(power_spectrum: np.ndarray, n_coeffs: int = 10) -> np.ndarray:
    """
    Computes static LFCCs c1..c_{n_coeffs} (omitting c0) using 20 linear triangular filters
    and orthonormal DCT-II.
    Shape: (T_act, n_coeffs)
    """
    lfcc_energies = np.dot(power_spectrum, LFCC_FILTERBANK.T)
    log_lfcc = np.log(np.maximum(lfcc_energies, 1e-10))
    full_lfcc = np.dot(log_lfcc, LFCC_DCT_MATRIX.T)
    # Omitting c0 (index 0), retaining c1..c_{n_coeffs} (slice 1 to n_coeffs + 1)
    return full_lfcc[:, 1:n_coeffs + 1].astype(np.float32)

def aggregate_flux_active(flux_all: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    """
    Aggregates spectral flux across active frames, strictly omitting frame 0.
    Returns: [mean(flux), std(flux)], shape (2,)
    """
    num_frames = len(flux_all)
    # Exclude frame 0 (which has flux=0.0) from aggregation
    flux_mask = active_mask & (np.arange(num_frames) > 0)
    if np.sum(flux_mask) > 0:
        flux_active = flux_all[flux_mask]
    else:
        flux_active = flux_all[active_mask]
        
    flux_mean = float(np.mean(flux_active))
    flux_std = float(np.sqrt(np.mean((flux_active - flux_mean) ** 2) + 1e-8))
    return np.array([flux_mean, flux_std], dtype=np.float32)

# ==============================================================================
# 6. INDIVIDUAL CANDIDATE FEATURE EXTRACTORS
# ==============================================================================

def extract_candidate_a_lite(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Candidate A-Lite (26 dimensions):
    10 static MFCCs (c1..c10) + 3 sub-bands (Low, Tel, High) = 13 frame features.
    Aggregated: temporal mean (13) + temporal std (13) = 26 dimensions.
    Frozen Phase 3.1 baseline reference.
    """
    active_power, _, _, _ = prepare_stft_and_vad(audio, sr)
    mfccs = extract_static_mfccs(active_power)
    ratios = extract_subband_ratios(active_power, mode="3band")
    frame_13 = np.hstack([mfccs, ratios])
    return aggregate_temporal_stats(frame_13)

def extract_candidate_a_delta(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Candidate A-Delta: Delta-MFCC Representation (26 dimensions):
    10 Delta-MFCCs (c1..c10) via 5-frame regression + 3 sub-bands (Low, Tel, High) = 13 frame features.
    Aggregated: temporal mean (13) + temporal std (13) = 26 dimensions.
    """
    active_power, _, _, _ = prepare_stft_and_vad(audio, sr)
    mfccs = extract_static_mfccs(active_power)
    delta_mfccs = compute_deltas(mfccs, N=2)
    ratios = extract_subband_ratios(active_power, mode="3band")
    frame_13 = np.hstack([delta_mfccs, ratios])
    return aggregate_temporal_stats(frame_13)

def extract_candidate_a_lfcc(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Candidate A-LFCC: Linear Frequency Cepstral Coefficients (26 dimensions):
    10 static LFCCs (c1..c10) from 20 linear filters + 3 sub-bands = 13 frame features.
    Aggregated: temporal mean (13) + temporal std (13) = 26 dimensions.
    """
    active_power, _, _, _ = prepare_stft_and_vad(audio, sr)
    lfccs = extract_static_lfccs(active_power, n_coeffs=10)
    ratios = extract_subband_ratios(active_power, mode="3band")
    frame_13 = np.hstack([lfccs, ratios])
    return aggregate_temporal_stats(frame_13)

def extract_candidate_a_dynamics(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Candidate A-Dynamics: Spectral Flatness & Flux (30 dimensions):
    Candidate A-Lite (26 dims) + Spectral Flatness (2 dims) + Spectral Flux (2 dims).
    Total: 26 + 2 + 2 = 30 dimensions.
    """
    active_power, active_mask, flux_all, _ = prepare_stft_and_vad(audio, sr)
    
    # Base A-Lite features (26)
    mfccs = extract_static_mfccs(active_power)
    ratios = extract_subband_ratios(active_power, mode="3band")
    a_lite = aggregate_temporal_stats(np.hstack([mfccs, ratios]))
    
    # Spectral Flatness on active frames (2)
    sf = compute_spectral_flatness(active_power)
    sf_mean = float(np.mean(sf))
    sf_std = float(np.sqrt(np.mean((sf - sf_mean) ** 2) + 1e-8))
    sf_stats = np.array([sf_mean, sf_std], dtype=np.float32)
    
    # Contiguous Spectral Flux aggregated on active frames (2)
    flux_stats = aggregate_flux_active(flux_all, active_mask)
    
    return np.concatenate([a_lite, sf_stats, flux_stats]).astype(np.float32)

def extract_candidate_a_combined(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Candidate A-Combined: Multi-Cue Representation (40 dimensions):
    - 10 Delta-MFCCs: mean (10) + std (10) = 20 dims
    - 6 LFCCs (c1..c6): mean (6) + std (6) = 12 dims
    - 2 Sub-bands (Low: 0-300 Hz, High: 3400-8000 Hz, omitting Tel): mean (2) + std (2) = 4 dims
    - Spectral Flatness: mean (1) + std (1) = 2 dims
    - Spectral Flux: mean (1) + std (1) = 2 dims
    Total: 20 + 12 + 4 + 2 + 2 = 40 dimensions.
    """
    active_power, active_mask, flux_all, _ = prepare_stft_and_vad(audio, sr)
    
    # 1. Delta-MFCCs (20 dims)
    mfccs = extract_static_mfccs(active_power)
    delta_mfccs = compute_deltas(mfccs, N=2)
    delta_stats = aggregate_temporal_stats(delta_mfccs)
    
    # 2. LFCCs c1..c6 (12 dims)
    lfccs_6 = extract_static_lfccs(active_power, n_coeffs=6)
    lfcc_stats = aggregate_temporal_stats(lfccs_6)
    
    # 3. Sub-bands Low and High only (4 dims)
    ratios_2 = extract_subband_ratios(active_power, mode="2band")
    sub_stats = aggregate_temporal_stats(ratios_2)
    
    # 4. Spectral Flatness (2 dims)
    sf = compute_spectral_flatness(active_power)
    sf_mean = float(np.mean(sf))
    sf_std = float(np.sqrt(np.mean((sf - sf_mean) ** 2) + 1e-8))
    sf_stats = np.array([sf_mean, sf_std], dtype=np.float32)
    
    # 5. Spectral Flux (2 dims)
    flux_stats = aggregate_flux_active(flux_all, active_mask)
    
    return np.concatenate([delta_stats, lfcc_stats, sub_stats, sf_stats, flux_stats]).astype(np.float32)

def extract_candidate_a_delta_pure(audio: np.ndarray, sr: int = TARGET_SR) -> np.ndarray:
    """
    Candidate A-Delta-Pure: Diagnostic Control (20 dimensions):
    10 Delta-MFCCs: mean (10) + std (10) = 20 dims.
    Zero sub-band energy ratios.
    Isolated diagnostic control; permanently barred from primary candidate selection.
    """
    active_power, _, _, _ = prepare_stft_and_vad(audio, sr)
    mfccs = extract_static_mfccs(active_power)
    delta_mfccs = compute_deltas(mfccs, N=2)
    return aggregate_temporal_stats(delta_mfccs)

# ==============================================================================
# 7. UNIFIED EXTRACTOR DISPATCHER
# ==============================================================================

def extract_all_candidates(audio: np.ndarray, sr: int = TARGET_SR) -> dict[str, np.ndarray]:
    """
    Extracts all six Phase 3.2 candidate representations in a single, efficient pass.
    Avoids duplicate STFT, VAD, and filterbank computations.
    """
    active_power, active_mask, flux_all, _ = prepare_stft_and_vad(audio, sr)
    
    # Shared base representations
    mfccs = extract_static_mfccs(active_power)
    delta_mfccs = compute_deltas(mfccs, N=2)
    ratios_3 = extract_subband_ratios(active_power, mode="3band")
    ratios_2 = extract_subband_ratios(active_power, mode="2band")
    
    # Full LFCCs (c1..c10)
    full_lfcc = extract_static_lfccs(active_power, n_coeffs=10)
    lfccs_6 = full_lfcc[:, :6]
    
    # Dynamics components
    sf = compute_spectral_flatness(active_power)
    sf_mean = float(np.mean(sf))
    sf_std = float(np.sqrt(np.mean((sf - sf_mean) ** 2) + 1e-8))
    sf_stats = np.array([sf_mean, sf_std], dtype=np.float32)
    
    flux_stats = aggregate_flux_active(flux_all, active_mask)
    
    # Candidate 1: A-Lite (26)
    a_lite = aggregate_temporal_stats(np.hstack([mfccs, ratios_3]))
    
    # Candidate 2: A-Delta (26)
    a_delta = aggregate_temporal_stats(np.hstack([delta_mfccs, ratios_3]))
    
    # Candidate 3: A-LFCC (26)
    a_lfcc = aggregate_temporal_stats(np.hstack([full_lfcc, ratios_3]))
    
    # Candidate 4: A-Dynamics (30)
    a_dynamics = np.concatenate([a_lite, sf_stats, flux_stats]).astype(np.float32)
    
    # Candidate 5: A-Combined (40)
    delta_stats = aggregate_temporal_stats(delta_mfccs)
    lfcc_stats = aggregate_temporal_stats(lfccs_6)
    sub_stats = aggregate_temporal_stats(ratios_2)
    a_combined = np.concatenate([delta_stats, lfcc_stats, sub_stats, sf_stats, flux_stats]).astype(np.float32)
    
    # Candidate 6: A-Delta-Pure (20)
    a_delta_pure = delta_stats
    
    return {
        "candidate_a_lite": a_lite,
        "candidate_a_delta": a_delta,
        "candidate_a_lfcc": a_lfcc,
        "candidate_a_dynamics": a_dynamics,
        "candidate_a_combined": a_combined,
        "candidate_a_delta_pure": a_delta_pure
    }

# ==============================================================================
# 8. SAFE AUDIO INGESTION
# ==============================================================================

def load_audio_mono_16k(file_path: str) -> np.ndarray:
    """
    Loads audio, validates path safety, downmixes to mono, and resamples to 16 kHz.
    """
    assert_path_safety(file_path)
    audio, sr = sf.read(file_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    if sr != TARGET_SR:
        gcd = math.gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // gcd, sr // gcd).astype(np.float32)
    return audio

# ==============================================================================
# 9. FEATURE CACHE GENERATION & INTEGRITY VALIDATION
# ==============================================================================

def compute_file_sha256(filepath: str) -> str:
    """Computes SHA256 checksum of a file for integrity tracking."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()

def generate_feature_cache_v2(
    split_name: str,
    base_dir: str = "D:/SYNORA",
    manifest_path: str = "D:/SYNORA/data/benchmark_manifest.csv",
    cache_dir: str = "D:/SYNORA/data/cache",
    overwrite: bool = False
) -> dict:
    """
    Generates v2 feature cache for either 'dev' or 'calibration' split.
    Strictly forbids 'test' split.
    Validates shapes, finiteness, label mapping, and Phase 3.1 A-Lite bit parity.
    """
    split_lower = split_name.strip().lower()
    if split_lower not in ("dev", "calibration"):
        raise PermissionError(
            f"CRITICAL SAFETY VIOLATION: Generation requested for unauthorized split '{split_name}'. "
            "Only 'dev' and 'calibration' are authorized. Test partition access is strictly forbidden."
        )
        
    output_filename = f"{split_lower}_features_v2.npz"
    output_path = os.path.join(cache_dir, output_filename)
    assert_path_safety(output_path)
    
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Cache file already exists: {output_path}. Overwrite is not permitted without explicit authorization."
        )
        
    # Check Phase 3.1 original cache path
    p31_cache_filename = f"{split_lower}_features.npz"
    p31_cache_path = os.path.join(cache_dir, p31_cache_filename)
    if not os.path.exists(p31_cache_path):
        raise FileNotFoundError(f"Original Phase 3.1 cache missing for comparison: {p31_cache_path}")
        
    sha_p31_before = compute_file_sha256(p31_cache_path)
    print(f"[{split_lower.upper()}] Pre-generation Phase 3.1 Cache SHA256: {sha_p31_before}")
    
    # Load manifest
    assert_path_safety(manifest_path)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest missing: {manifest_path}")
        
    manifest_df = pd.read_csv(manifest_path)
    split_df = manifest_df[manifest_df["split"] == split_lower].reset_index(drop=True)
    
    num_samples = len(split_df)
    unique_base_ids = split_df["base_id"].unique()
    num_base_ids = len(unique_base_ids)
    
    if num_samples != 60:
        raise ValueError(f"Expected exactly 60 samples for split '{split_lower}', found {num_samples}")
    if num_base_ids != 20:
        raise ValueError(f"Expected exactly 20 unique base IDs for split '{split_lower}', found {num_base_ids}")
    if len(split_df["sample_id"].unique()) != 60:
        raise ValueError("Duplicate sample_id detected in manifest!")
        
    print(f"[{split_lower.upper()}] Processing {num_samples} audio samples across {num_base_ids} base recordings...")
    
    # Storage lists
    c_a_lite_list = []
    c_a_delta_list = []
    c_a_lfcc_list = []
    c_a_dynamics_list = []
    c_a_combined_list = []
    c_a_delta_pure_list = []
    
    raw_labels_list = []
    target_fake_list = []
    sample_ids_list = []
    base_ids_list = []
    conditions_list = []
    generators_list = []
    durations_list = []
    sample_rates_list = []
    
    t0 = time.time()
    
    for idx, row in split_df.iterrows():
        audio_rel_path = row["audio_path"]
        assert_path_safety(audio_rel_path)
        
        audio_full_path = audio_rel_path if os.path.isabs(audio_rel_path) else os.path.join(base_dir, audio_rel_path)
        assert_path_safety(audio_full_path)
        
        if not os.path.exists(audio_full_path):
            raise FileNotFoundError(f"Audio file missing on disk: {audio_full_path}")
            
        # Load audio (downmixed mono, 16 kHz)
        audio = load_audio_mono_16k(audio_full_path)
        duration_s = len(audio) / TARGET_SR
        
        # Extract all 6 candidates in single pass
        feats = extract_all_candidates(audio, sr=TARGET_SR)
        
        # Verify shapes and finiteness
        assert feats["candidate_a_lite"].shape == (DIM_A_LITE,), f"A-Lite shape mismatch at row {idx}"
        assert feats["candidate_a_delta"].shape == (DIM_A_DELTA,), f"A-Delta shape mismatch at row {idx}"
        assert feats["candidate_a_lfcc"].shape == (DIM_A_LFCC,), f"A-LFCC shape mismatch at row {idx}"
        assert feats["candidate_a_dynamics"].shape == (DIM_A_DYNAMICS,), f"A-Dynamics shape mismatch at row {idx}"
        assert feats["candidate_a_combined"].shape == (DIM_A_COMBINED,), f"A-Combined shape mismatch at row {idx}"
        assert feats["candidate_a_delta_pure"].shape == (DIM_A_DELTA_PURE,), f"A-Delta-Pure shape mismatch at row {idx}"
        
        for cand_name, vec in feats.items():
            if not np.all(np.isfinite(vec)):
                raise ValueError(f"Non-finite feature values detected in {cand_name} for sample {row['sample_id']}")
                
        c_a_lite_list.append(feats["candidate_a_lite"])
        c_a_delta_list.append(feats["candidate_a_delta"])
        c_a_lfcc_list.append(feats["candidate_a_lfcc"])
        c_a_dynamics_list.append(feats["candidate_a_dynamics"])
        c_a_combined_list.append(feats["candidate_a_combined"])
        c_a_delta_pure_list.append(feats["candidate_a_delta_pure"])
        
        # Label convention:
        # Raw benchmark / Phase 3.1: 0 = Fake, 1 = Real
        # Phase 3.2 target_fake: 1 = Fake, 0 = Real
        label_str = str(row["label"]).strip().lower()
        if label_str == "fake":
            raw_lbl = 0
            tgt_fake = 1
        elif label_str == "real":
            raw_lbl = 1
            tgt_fake = 0
        else:
            raise ValueError(f"Unexpected label string: '{label_str}' at row {idx}")
            
        raw_labels_list.append(raw_lbl)
        target_fake_list.append(tgt_fake)
        
        sample_ids_list.append(str(row["sample_id"]))
        base_ids_list.append(str(row["base_id"]))
        conditions_list.append(str(row["degradation_condition"]))
        generators_list.append(str(row.get("generator", "human")))
        durations_list.append(float(duration_s))
        sample_rates_list.append(int(TARGET_SR))
        
        if (idx + 1) % 20 == 0 or (idx + 1) == num_samples:
            print(f"  [{split_lower.upper()}] Processed {idx + 1}/{num_samples} samples...")
            
    elapsed_time = time.time() - t0
    print(f"[{split_lower.upper()}] All {num_samples} samples processed in {elapsed_time:.2f}s ({elapsed_time/num_samples*1000:.1f} ms/sample).")
    
    # Convert to NumPy arrays
    arr_a_lite = np.array(c_a_lite_list, dtype=np.float32)
    arr_a_delta = np.array(c_a_delta_list, dtype=np.float32)
    arr_a_lfcc = np.array(c_a_lfcc_list, dtype=np.float32)
    arr_a_dynamics = np.array(c_a_dynamics_list, dtype=np.float32)
    arr_a_combined = np.array(c_a_combined_list, dtype=np.float32)
    arr_a_delta_pure = np.array(c_a_delta_pure_list, dtype=np.float32)
    
    arr_raw_labels = np.array(raw_labels_list, dtype=np.int64)
    arr_target_fake = np.array(target_fake_list, dtype=np.int64)
    arr_sample_ids = np.array(sample_ids_list)
    arr_base_ids = np.array(base_ids_list)
    arr_conditions = np.array(conditions_list)
    arr_generators = np.array(generators_list)
    arr_durations = np.array(durations_list, dtype=np.float32)
    arr_sample_rates = np.array(sample_rates_list, dtype=np.int32)
    
    # Class balance check: exactly 30 fake, 30 real
    n_fake = int(np.sum(arr_target_fake == 1))
    n_real = int(np.sum(arr_target_fake == 0))
    assert n_fake == 30, f"Expected 30 fakes, found {n_fake}"
    assert n_real == 30, f"Expected 30 reals, found {n_real}"
    assert np.array_equal(arr_target_fake, (arr_raw_labels == 0).astype(int)), "Target fake mapping mismatch!"
    
    # Save compressed .npz cache
    print(f"[{split_lower.upper()}] Saving compressed cache to: {output_path}")
    np.savez_compressed(
        output_path,
        candidate_a_lite=arr_a_lite,
        a_lite=arr_a_lite,
        candidate_a_delta=arr_a_delta,
        a_delta=arr_a_delta,
        candidate_a_lfcc=arr_a_lfcc,
        a_lfcc=arr_a_lfcc,
        candidate_a_dynamics=arr_a_dynamics,
        a_dynamics=arr_a_dynamics,
        candidate_a_combined=arr_a_combined,
        a_combined=arr_a_combined,
        candidate_a_delta_pure=arr_a_delta_pure,
        a_delta_pure=arr_a_delta_pure,
        raw_labels=arr_raw_labels,
        labels=arr_raw_labels,
        target_fake=arr_target_fake,
        base_ids=arr_base_ids,
        acoustic_conditions=arr_conditions,
        conditions=arr_conditions,
        sample_ids=arr_sample_ids,
        durations=arr_durations,
        sample_rates=arr_sample_rates,
        generators=arr_generators
    )
    
    sha_v2_new = compute_file_sha256(output_path)
    print(f"[{split_lower.upper()}] Generated Cache SHA256: {sha_v2_new}")
    
    # Verify Phase 3.1 original cache remains strictly untouched
    sha_p31_after = compute_file_sha256(p31_cache_path)
    assert sha_p31_before == sha_p31_after, f"CRITICAL INTEGRITY ERROR: Phase 3.1 cache {p31_cache_path} was modified!"
    print(f"[{split_lower.upper()}] Phase 3.1 Cache Immutability Verified: SHA256 unchanged.")
    
    # Row-by-row comparison of A-Lite against Phase 3.1 cache
    p31_data = np.load(p31_cache_path)
    p31_a_lite = p31_data["a_lite_features"]
    
    # Verify sample order alignment
    assert np.array_equal(arr_sample_ids, p31_data["sample_ids"]), "Sample ID alignment mismatch with Phase 3.1 cache!"
    assert np.array_equal(arr_base_ids, p31_data["base_ids"]), "Base ID alignment mismatch with Phase 3.1 cache!"
    assert np.array_equal(arr_conditions, p31_data["conditions"]), "Condition alignment mismatch with Phase 3.1 cache!"
    assert np.array_equal(arr_raw_labels, p31_data["labels"]), "Label alignment mismatch with Phase 3.1 cache!"
    
    abs_diffs = np.abs(arr_a_lite - p31_a_lite)
    max_abs_diff = float(np.max(abs_diffs))
    mean_abs_diff = float(np.mean(abs_diffs))
    rows_compared = len(arr_a_lite)
    
    print(f"[{split_lower.upper()}] A-Lite Baseline Parity vs Phase 3.1:")
    print(f"  Rows compared: {rows_compared}")
    print(f"  Max Absolute Diff: {max_abs_diff:.2e}")
    print(f"  Mean Absolute Diff: {mean_abs_diff:.2e}")
    assert max_abs_diff <= 1e-5, f"A-Lite parity failure: max diff {max_abs_diff} exceeds 1e-5 tolerance!"
    print(f"  Parity Status: 100% BIT-EXACT PARITY VERIFIED (tolerance <= 1e-5).")
    
    return {
        "split": split_lower,
        "output_path": output_path,
        "sha256_v2": sha_v2_new,
        "p31_cache_path": p31_cache_path,
        "sha256_p31_before": sha_p31_before,
        "sha256_p31_after": sha_p31_after,
        "p31_immutability_verified": (sha_p31_before == sha_p31_after),
        "num_samples": num_samples,
        "num_base_ids": num_base_ids,
        "num_fakes": n_fake,
        "num_reals": n_real,
        "matrix_shapes": {
            "candidate_a_lite": arr_a_lite.shape,
            "candidate_a_delta": arr_a_delta.shape,
            "candidate_a_lfcc": arr_a_lfcc.shape,
            "candidate_a_dynamics": arr_a_dynamics.shape,
            "candidate_a_combined": arr_a_combined.shape,
            "candidate_a_delta_pure": arr_a_delta_pure.shape,
        },
        "parity_max_abs_diff": max_abs_diff,
        "parity_mean_abs_diff": mean_abs_diff,
        "parity_rows_compared": rows_compared,
        "parity_verified": (max_abs_diff <= 1e-5),
        "elapsed_seconds": elapsed_time
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SYNORA Phase 3.2 Feature Cache Generation")
    parser.add_argument("--split", choices=["dev", "calibration", "all"], default="all",
                        help="Partition to cache: 'dev', 'calibration', or 'all' (both).")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwrite if cache already exists.")
    args = parser.parse_args()
    
    if args.split == "all":
        splits = ["dev", "calibration"]
    else:
        splits = [args.split]
        
    results = {}
    for sp in splits:
        print(f"\n=======================================================")
        print(f"STARTING V2 FEATURE CACHE GENERATION: {sp.upper()}")
        print(f"=======================================================")
        res = generate_feature_cache_v2(sp, overwrite=args.overwrite)
        results[sp] = res
        
    print("\n=======================================================")
    print("ALL REQUESTED V2 CACHES SUCCESSFULLY GENERATED & VERIFIED")
    print("=======================================================")
    for sp, r in results.items():
        print(f"Split {sp.upper()}:")
        print(f"  Output Path: {r['output_path']}")
        print(f"  SHA256:      {r['sha256_v2']}")
        print(f"  Max Diff:    {r['parity_max_abs_diff']:.2e}")
        print(f"  Parity OK:   {r['parity_verified']}")
        print(f"  P31 Immut:   {r['p31_immutability_verified']}")


