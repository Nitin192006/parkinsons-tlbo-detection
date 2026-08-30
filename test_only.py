"""
Module: test_only.py
Description: Instant inference runner.
Loads models from outputs/trained_models.pkl and runs mic or file tests without retraining.
"""

import os
import pickle
from src.live_inference import predict_sample, record_and_predict


def main():
    model_path = os.path.join("outputs", "trained_models.pkl")
    if not os.path.exists(model_path):
        print(f"[!] Model file '{model_path}' not found.")
        print("[!] Please run 'python main.py' once to train and save the models.")
        return

    print("[*] Loading pre-trained models from disk...")
    with open(model_path, "rb") as f:
        trained_models = pickle.load(f)
    print(f"[+] Ready. Available vowels: {list(trained_models.keys())}")

    while True:
        print("\n" + "=" * 50)
        print("INSTANT TESTING INTERFACE (NO RETRAINING)")
        print("=" * 50)
        print("1. Record live audio from Microphone")
        print("2. Test a custom .wav audio file")
        print("3. Exit")
        choice = input("Enter choice (1/2/3): ").strip()

        if choice == "1":
            vowel = input("Enter vowel (A, E, I, O, U): ").strip().upper()
            if vowel not in trained_models:
                print(f"[!] Invalid vowel: {vowel}")
                continue
            res = record_and_predict(vowel, trained_models)
            print("\n" + "-" * 40)
            print(f"RESULT FOR /{vowel}/")
            print(f"Prediction:         {res['prediction']}")
            print(f"PD Confidence:      {res['pd_probability']*100:.2f}%")
            print(f"Healthy Confidence: {res['healthy_probability']*100:.2f}%")
            print(f"Audio Saved At:     {res['saved_audio_path']}")
            print("-" * 40)

        elif choice == "2":
            file_path = input("Enter path to .wav file: ").strip().strip('"')
            vowel = input("Enter vowel (A, E, I, O, U): ").strip().upper()
            if vowel not in trained_models:
                print(f"[!] Invalid vowel: {vowel}")
                continue
            if not os.path.exists(file_path):
                print(f"[!] File not found: {file_path}")
                continue
            try:
                res = predict_sample(file_path, vowel, trained_models)
                print("\n" + "-" * 40)
                print(f"RESULT FOR /{vowel}/")
                print(f"Prediction:         {res['prediction']}")
                print(f"PD Confidence:      {res['pd_probability']*100:.2f}%")
                print(f"Healthy Confidence: {res['healthy_probability']*100:.2f}%")
                print("-" * 40)
            except Exception as e:
                print(f"[!] Error: {e}")

        elif choice == "3":
            print("[*] Exiting test interface.")
            break
        else:
            print("[!] Invalid option selected.")


if __name__ == "__main__":
    main()