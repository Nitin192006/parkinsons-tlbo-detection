"""
Module: live_inference.py
Description: Robust live voice testing with audio trimming (VAD)
and energy normalization to prevent false-positive Parkinson's predictions.
"""

import os
import time
import numpy as np
import librosa
import sounddevice as sd
from scipy.io.wavfile import write
from src.feature_extraction import extract_file_features


def preprocess_and_save_cleaned(raw_file_path: str, target_sr: int = 22050) -> str:
    """
    Trims silence/breath tails, normalizes peak amplitude, and saves cleaned audio.
    """
    y, sr = librosa.load(raw_file_path, sr=target_sr)

    # 1. Trim leading and trailing silence (top_db=25 threshold)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)

    # Fallback if audio was mostly silence
    if len(y_trimmed) < int(0.5 * target_sr):
        y_trimmed = y

    # 2. Peak normalization
    max_val = np.max(np.abs(y_trimmed))
    if max_val > 1e-6:
        y_trimmed = y_trimmed / max_val

    # Save cleaned file
    cleaned_path = raw_file_path.replace(".wav", "_clean.wav")
    write(cleaned_path, target_sr, (y_trimmed * 32767).astype(np.int16))
    return cleaned_path


def predict_sample(file_path: str, vowel: str, trained_models: dict):
    """
    Runs inference on an audio file using pre-trained models.
    """
    vowel = vowel.upper()
    if vowel not in trained_models:
        raise ValueError(f"Model for vowel /{vowel}/ is not trained.")

    model_bundle = trained_models[vowel]
    optimizer = model_bundle["optimizer"]
    scaler = model_bundle["scaler"]
    classifier = model_bundle["classifier"]

    # 1. Preprocess and clean audio
    cleaned_path = preprocess_and_save_cleaned(file_path)

    # 2. Extract 29 features from cleaned audio
    feats_29 = extract_file_features(cleaned_path).reshape(1, -1)

    # 3. Map to optimal 5 features using TLBO mask
    feats_5 = optimizer.transform(feats_29)

    # 4. Scale and predict
    feats_scaled = scaler.transform(feats_5)
    prob_pd = float(classifier.predict_proba(feats_scaled)[0, 1])

    # Calibrated decision threshold
    prediction = "Parkinson's Disease" if prob_pd >= 0.55 else "Healthy"

    return {
        "vowel": vowel,
        "prediction": prediction,
        "pd_probability": prob_pd,
        "healthy_probability": 1.0 - prob_pd,
    }


def record_and_predict(vowel: str, trained_models: dict, duration: float = 3.5, sample_rate: int = 22050):
    """
    Records live audio from the microphone with clean trimming and runs inference.
    """
    os.makedirs(os.path.join("outputs", "recordings"), exist_ok=True)
    raw_path = os.path.join("outputs", "recordings", f"live_{vowel.upper()}_{int(time.time())}.wav")

    print(f"\n[*] Target Vowel: /{vowel.upper()}/")
    print(f"[*] Recording starts in 2 seconds. Take a breath and sustain the sound steadily for {duration}s...")
    time.sleep(2)

    print("[>>>] RECORDING NOW... [>>>]")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("[*] Recording complete. Analyzing voice stability...")

    # Save raw audio
    write(raw_path, sample_rate, (audio * 32767).astype(np.int16))

    res = predict_sample(raw_path, vowel, trained_models)
    res["saved_audio_path"] = raw_path
    return res