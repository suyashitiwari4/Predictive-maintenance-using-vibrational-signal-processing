"""
features.py
------------
Shared Digital Signal Processing (DSP) and feature extraction module.
Used by both the CWRU (fault-type classification) and IMS
(degradation-stage classification) pipelines, so all signals are
processed the same way regardless of source dataset.
"""

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew


# ---------------------------------------------------------------
# 1. Preprocessing / filtering
# ---------------------------------------------------------------

def bandpass_filter(x, fs, low=10, high=None, order=4):
    """
    Butterworth bandpass (or lowpass if high is None) filter to remove
    DC offset / very low-frequency drift and high-frequency sensor noise.

    x    : 1D raw vibration signal
    fs   : sampling frequency (Hz)
    low  : lower cutoff (Hz)
    high : upper cutoff (Hz). Defaults to just under Nyquist.
    """
    nyq = fs / 2.0
    if high is None:
        high = nyq * 0.98
    low_n = low / nyq
    high_n = high / nyq
    b, a = sp_signal.butter(order, [low_n, high_n], btype="band")
    return sp_signal.filtfilt(b, a, x)


def segment_signal(x, window_size, step=None):
    """
    Split a long signal into fixed-length overlapping/non-overlapping
    windows. Returns a 2D array [n_windows, window_size].
    """
    if step is None:
        step = window_size  # non-overlapping by default
    n = len(x)
    starts = range(0, n - window_size + 1, step)
    return np.array([x[s:s + window_size] for s in starts])


# ---------------------------------------------------------------
# 2. Time-domain features
# ---------------------------------------------------------------

def time_domain_features(window):
    x = np.asarray(window, dtype=float)
    rms = np.sqrt(np.mean(x ** 2))
    peak = np.max(np.abs(x))
    mean_abs = np.mean(np.abs(x))
    sqrt_abs_mean = np.mean(np.sqrt(np.abs(x))) ** 2

    return {
        "mean": np.mean(x),
        "variance": np.var(x),
        "std": np.std(x),
        "rms": rms,
        "energy": np.sum(x ** 2),
        "peak": peak,
        "peak_to_peak": np.max(x) - np.min(x),
        "crest_factor": peak / rms if rms > 0 else 0.0,
        "kurtosis": kurtosis(x),
        "skewness": skew(x),
        "shape_factor": rms / (mean_abs + 1e-12),
        "impulse_factor": peak / (mean_abs + 1e-12),
        "margin_factor": peak / (sqrt_abs_mean + 1e-12),
        "clearance_factor": peak / (sqrt_abs_mean + 1e-12),
    }


# ---------------------------------------------------------------
# 3. Frequency-domain features
# ---------------------------------------------------------------

def frequency_domain_features(window, fs):
    x = np.asarray(window, dtype=float)
    n = len(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mag = np.abs(np.fft.rfft(x)) / n

    # Skip the DC component (index 0) when finding dominant frequency
    dom_idx = np.argmax(mag[1:]) + 1
    dominant_freq = freqs[dom_idx]

    spectral_energy = np.sum(mag ** 2)
    spectral_centroid = (
        np.sum(freqs * mag) / (np.sum(mag) + 1e-12)
    )
    spectral_spread = np.sqrt(
        np.sum(((freqs - spectral_centroid) ** 2) * mag) / (np.sum(mag) + 1e-12)
    )

    # Spectral entropy: how "spread out" vs "peaky" the spectrum is
    psd_norm = mag ** 2
    psd_norm = psd_norm / (np.sum(psd_norm) + 1e-12)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
    spectral_entropy /= np.log2(len(psd_norm))  # normalize to [0, 1]

    spectral_kurt = kurtosis(mag)

    # Multi-band power: split spectrum into low/mid/high thirds of Nyquist
    nyq = fs / 2.0
    bands = {
        "band_power_low": band_power(freqs, mag, nyq / 6, bandwidth=nyq / 6),
        "band_power_mid": band_power(freqs, mag, nyq / 2, bandwidth=nyq / 6),
        "band_power_high": band_power(freqs, mag, 5 * nyq / 6, bandwidth=nyq / 6),
    }

    feats = {
        "dominant_freq": dominant_freq,
        "spectral_energy": spectral_energy,
        "spectral_centroid": spectral_centroid,
        "spectral_spread": spectral_spread,
        "spectral_std": np.std(mag),
        "spectral_entropy": spectral_entropy,
        "spectral_kurtosis": spectral_kurt,
    }
    feats.update(bands)
    return feats, freqs, mag


def compute_spectrogram(x, fs, nperseg=256):
    """
    Short-Time Fourier Transform based spectrogram, for visualizing how
    the frequency content of the signal changes over time (a single FFT
    only shows the average spectrum across the whole window).
    """
    f, t, Sxx = sp_signal.spectrogram(np.asarray(x, dtype=float), fs=fs, nperseg=nperseg)
    return f, t, Sxx


def band_power(freqs, mag, f_center, bandwidth=5.0):
    """
    Power around a specific characteristic fault frequency
    (e.g. BPFO, BPFI, BSF, FTF for bearings). Useful when you know
    the bearing geometry and shaft speed.
    """
    mask = (freqs >= f_center - bandwidth) & (freqs <= f_center + bandwidth)
    return float(np.sum(mag[mask] ** 2))


# ---------------------------------------------------------------
# 4. Bearing characteristic fault frequencies (optional, geometry-based)
# ---------------------------------------------------------------

def bearing_fault_frequencies(rpm, n_balls, ball_diameter, pitch_diameter, contact_angle_deg=0):
    """
    Compute BPFO, BPFI, BSF, FTF given bearing geometry.
    Angles in degrees, diameters in consistent units (mm or inches).
    """
    fr = rpm / 60.0  # shaft rotation frequency (Hz)
    theta = np.radians(contact_angle_deg)
    ratio = (ball_diameter / pitch_diameter) * np.cos(theta)

    bpfo = (n_balls / 2.0) * fr * (1 - ratio)
    bpfi = (n_balls / 2.0) * fr * (1 + ratio)
    bsf = (pitch_diameter / (2 * ball_diameter)) * fr * (1 - ratio ** 2)
    ftf = (fr / 2.0) * (1 - ratio)

    return {"BPFO": bpfo, "BPFI": bpfi, "BSF": bsf, "FTF": ftf}


# ---------------------------------------------------------------
# 5. Full feature vector for one window
# ---------------------------------------------------------------

def extract_features(window, fs):
    """
    Combines time-domain + frequency-domain features into a single
    flat dict for one signal window. This is the function both
    train_cwru.py and train_ims.py call per window.
    """
    feats = {}
    feats.update(time_domain_features(window))
    freq_feats, freqs, mag = frequency_domain_features(window, fs)
    feats.update(freq_feats)
    return feats


def extract_features_dataframe(windows, fs):
    """
    Apply extract_features to every window and return a pandas DataFrame.
    """
    import pandas as pd
    rows = [extract_features(w, fs) for w in windows]
    return pd.DataFrame(rows)
