"""
Module: live_inference.py
Description: Performs live inference on unseen microphone recordings or single .wav files
using the trained TLBO-selected features and calibrated SVM models.
"""

import os
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from src.feature_extraction import extract_file_features, load_vowel_dataset
from src.tlbo_optimizer import TLBOFeatureSelector


class LiveParkinsonsPredictor:
    """
    Trains and stores calibrated per-vowel classifiers for instant inference.
    """
    def __init__(self, healthy_root: str = "data/Vowel_Healthy", pd_root: str = "data/Vowel_PD"):
        self.healthy_root = healthy_root
        self.pd_root = pd_root
        self.models = {}
        self.scalers = {}
        self.optimizers = {}
        self._train_models()

    def _train_models(self):
        vowels = ["A", "E", "I", "O", "U"]
        print("[+] Pre-training production models across all vowels...")

        for v in vowels:
            h_dir = os.path.join(self.healthy_root, v)
            p_dir = os.path.join(self.pd_root, v)
            if not (os.path.exists(h_dir) and os.path.exists(p_dir)):
                continue

            X, y = load_vowel_dataset(self.healthy_root, self.pd_root, v)

            # 1. Feature selection via TLBO (29 -> 5)
            opt = TLBOFeatureSelector(n_learners=40, max_iter=100, target_features=5, random_state=42)
            opt.fit(X, y)
            X_opt = opt.transform(X)

            # 2. Standard scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_opt)

            # 3. Calibrated SVM Classifier
            base_clf = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=42)
            calibrated_clf = CalibratedClassifierCV(estimator=base_clf, cv=3, method="sigmoid")
            calibrated_clf.fit(X_scaled, y)

            self.optimizers[v] = opt
            self.scalers[v] = scaler
            self.models[v] = calibrated_clf

        print("[+] All vowel inference models ready.\n")

    def predict_file(self, file_path: str, vowel: str):
        """
        Runs inference on an individual audio file for a given vowel.
        """
        vowel = vowel.upper()
        if vowel not in self.models:
            raise ValueError(f"Model for vowel /{vowel}/ is not available.")

        # 1. Extract 29 features
        feats_29 = extract_file_features(file_path).reshape(1, -1)

        # 2. Transform to optimal 5 features
        feats_5 = self.optimizers[vowel].transform(feats_29)

        # 3. Scale and predict
        feats_scaled = self.scalers[vowel].transform(feats_5)
        prob_pd = float(self.models[vowel].predict_proba(feats_scaled)[0, 1])
        prediction = "Parkinson's Disease" if prob_pd >= 0.5 else "Healthy"

        return {
            "vowel": vowel,
            "prediction": prediction,
            "pd_probability": prob_pd,
            "healthy_probability": 1.0 - prob_pd
        }

    def record_and_predict(self, vowel: str, duration: float = 3.0, sample_rate: int = 22050):
        """
        Records live audio from microphone and returns diagnosis.
        """
        os.makedirs("outputs/recordings", exist_ok=True)
        temp_path = os.path.join("outputs", "recordings", f"live_{vowel.upper()}_{int(time.time())}.wav")

        print(f"\n[*] Prepare to phonate sustained vowel: /{vowel.upper()}/")
        print(f"[*] Recording starting in 2 seconds (hold steady for {duration} seconds)...")
        time.sleep(2)

        print("[>>>] RECORDING NOW... [>>>]")
        audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
        print("[*] Recording completed.")

        # Save to wav
        write(temp_path, sample_rate, (audio_data * 32767).astype(np.int16))

        # Run diagnosis
        result = self.predict_file(temp_path, vowel)
        result["saved_audio_path"] = temp_path
        return result