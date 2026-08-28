"""
Module: live_inference.py
Description: Performs live inference via microphone recording or audio file
using models and feature selectors trained during pipeline execution.
"""

import os
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from src.feature_extraction import extract_file_features


def predict_sample(file_path: str, vowel: str, trained_models: dict):
    """
    Runs inference on a single .wav file using the pre-trained model for that vowel.
    """
    vowel = vowel.upper()
    if vowel not in trained_models:
        raise ValueError(f"Model for vowel /{vowel}/ is not trained.")

    model_bundle = trained_models[vowel]
    optimizer = model_bundle["optimizer"]
    scaler = model_bundle["scaler"]
    classifier = model_bundle["classifier"]

    # 1. Extract 29 features
    feats_29 = extract_file_features(file_path).reshape(1, -1)

    # 2. Select optimal 5 features using TLBO mask
    feats_5 = optimizer.transform(feats_29)

    # 3. Standard scale and predict
    feats_scaled = scaler.transform(feats_5)
    prob_pd = float(classifier.predict_proba(feats_scaled)[0, 1])
    prediction = "Parkinson's Disease" if prob_pd >= 0.5 else "Healthy"

    return {
        "vowel": vowel,
        "prediction": prediction,
        "pd_probability": prob_pd,
        "healthy_probability": 1.0 - prob_pd
    }


def record_and_predict(vowel: str, trained_models: dict, duration: float = 3.0, sample_rate: int = 22050):
    """
    Records live audio from the microphone and runs inference.
    """
    os.makedirs(os.path.join("outputs", "recordings"), exist_ok=True)
    save_path = os.path.join("outputs", "recordings", f"live_{vowel.upper()}_{int(time.time())}.wav")

    print(f"\n[*] Target Vowel: /{vowel.upper()}/")
    print(f"[*] Recording starts in 2 seconds (sustain sound steadily for {duration}s)...")
    time.sleep(2)

    print("[>>>] RECORDING NOW... [>>>]")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("[*] Recording complete.")

    write(save_path, sample_rate, (audio * 32767).astype(np.int16))
    res = predict_sample(save_path, vowel, trained_models)
    res["saved_audio_path"] = save_path
    return res