"""
Module: feature_extraction.py
Description: Extracts 29 acoustic features (13 MFCC + 16 TF-NMF) for individual vowel sets.
Healthy (Class 0: 29 samples) | PD (Class 1: 30 samples) -> Shape: (59, 29)
"""

import os
import numpy as np
import librosa
from sklearn.decomposition import NMF


# ---------------------------------------------------------
# Section 1: Statistical & Temporal Descriptors for NMF
# ---------------------------------------------------------

def calculate_hoyer_sparsity(vector: np.ndarray) -> float:
    """
    Computes Hoyer Sparsity for a 1D activation vector.
    Value ranges from 0 (completely dense) to 1 (maximally sparse).
    """
    n = len(vector)
    if n <= 1:
        return 0.0

    l1_norm = np.sum(np.abs(vector))
    l2_norm = np.sqrt(np.sum(vector ** 2))

    if l2_norm == 0:
        return 0.0

    sparsity = (np.sqrt(n) - (l1_norm / l2_norm)) / (np.sqrt(n) - 1)
    return float(sparsity)


# ---------------------------------------------------------
# Section 2: TF-NMF Feature Extractor (16 Features)
# ---------------------------------------------------------

def extract_tfnmf_features(y: np.ndarray, sr: int, n_components: int = 4) -> np.ndarray:
    """
    Computes magnitude STFT and Multiplicative-Update NMF decomposition.
    Extracts 4 temporal descriptors per component (Total: 16 features).
    """
    # 1. Compute STFT magnitude spectrogram
    stft_matrix = np.abs(librosa.stft(y, n_fft=1024, hop_length=512))
    stft_matrix = np.maximum(stft_matrix, 1e-8)

    # 2. Factorize spectrogram V ~ W * H
    nmf_model = NMF(
        n_components=n_components,
        init="nndsvda",
        solver="mu",
        random_state=42,
        max_iter=500,
        tol=1e-3
    )
    nmf_model.fit_transform(stft_matrix)
    h_matrix = nmf_model.components_  # Shape: (n_components, time_frames)

    descriptors = []
    for k in range(n_components):
        h_row = h_matrix[k, :]

        # Descriptor 1: Mean Energy
        mean_val = float(np.mean(h_row))
        # Descriptor 2: Standard Deviation
        std_val = float(np.std(h_row))
        # Descriptor 3: Discontinuity (Frame-to-frame jump)
        disc_val = float(np.mean(np.abs(np.diff(h_row)))) if len(h_row) > 1 else 0.0
        # Descriptor 4: Hoyer Sparsity
        spar_val = calculate_hoyer_sparsity(h_row)

        descriptors.extend([mean_val, std_val, disc_val, spar_val])

    return np.array(descriptors)


# ---------------------------------------------------------
# Section 3: MFCC Feature Extractor (13 Features)
# ---------------------------------------------------------

def extract_mfcc_features(y: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """
    Extracts static 13 MFCCs averaged across time frames.
    """
    mfcc_frames = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        n_mels=40,
        n_fft=1024,
        hop_length=512
    )
    return np.mean(mfcc_frames, axis=1)


# ---------------------------------------------------------
# Section 4: Single File & Per-Vowel Dataset Loader
# ---------------------------------------------------------

def extract_file_features(file_path: str) -> np.ndarray:
    """
    Loads, normalizes, and extracts the 29-feature vector from a single audio file.
    """
    y, sr = librosa.load(file_path, sr=22050)
    y = librosa.util.normalize(y)

    mfcc_feats = extract_mfcc_features(y, sr, n_mfcc=13)
    nmf_feats = extract_tfnmf_features(y, sr, n_components=4)

    return np.concatenate([mfcc_feats, nmf_feats])


def get_available_vowels(healthy_root: str, pd_root: str):
    """
    Returns sorted list of common vowel subfolders present in both roots.
    """
    h_vowels = set([d for d in os.listdir(healthy_root) if os.path.isdir(os.path.join(healthy_root, d))])
    pd_vowels = set([d for d in os.listdir(pd_root) if os.path.isdir(os.path.join(pd_root, d))])
    return sorted(list(h_vowels.intersection(pd_vowels)))


def load_vowel_dataset(healthy_root: str, pd_root: str, vowel_name: str):
    """
    Loads audio files for a single vowel type.
    Returns:
        X: Feature matrix of shape (59, 29)
        y: Target label vector of shape (59,) [0 for Healthy, 1 for PD]
    """
    h_dir = os.path.join(healthy_root, vowel_name)
    pd_dir = os.path.join(pd_root, vowel_name)

    h_files = sorted([os.path.join(h_dir, f) for f in os.listdir(h_dir) if f.lower().endswith(".wav")])
    pd_files = sorted([os.path.join(pd_dir, f) for f in os.listdir(pd_dir) if f.lower().endswith(".wav")])

    features = []
    labels = []

    for f in h_files:
        features.append(extract_file_features(f))
        labels.append(0)

    for f in pd_files:
        features.append(extract_file_features(f))
        labels.append(1)

    return np.array(features, dtype=np.float64), np.array(labels, dtype=np.int32)