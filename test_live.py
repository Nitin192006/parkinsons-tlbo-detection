"""
Module: test_live.py
Description: CLI interface for interactive live voice diagnosis.
"""

from src.live_inference import LiveParkinsonsPredictor


def main():
    predictor = LiveParkinsonsPredictor()

    while True:
        print("\n" + "=" * 50)
        print("PARKINSON'S DISEASE LIVE VOICE TESTING")
        print("=" * 50)
        print("1. Record live audio via Microphone")
        print("2. Test an existing .wav audio file")
        print("3. Exit")
        choice = input("Select option (1/2/3): ").strip()

        if choice == "1":
            vowel = input("Enter vowel target (A, E, I, O, U): ").strip().upper()
            if vowel not in ["A", "E", "I", "O", "U"]:
                print("[!] Invalid vowel selected.")
                continue

            res = predictor.record_and_predict(vowel=vowel, duration=3.0)
            print("\n" + "-" * 40)
            print(f"DIAGNOSTIC RESULT FOR /{vowel}/")
            print("-" * 40)
            print(f"Prediction:         {res['prediction']}")
            print(f"PD Confidence:      {res['pd_probability']*100:.2f}%")
            print(f"Healthy Confidence: {res['healthy_probability']*100:.2f}%")
            print(f"Audio Saved At:     {res['saved_audio_path']}")
            print("-" * 40)

        elif choice == "2":
            file_path = input("Enter path to .wav file: ").strip().strip('"')
            vowel = input("Enter corresponding vowel (A, E, I, O, U): ").strip().upper()
            if vowel not in ["A", "E", "I", "O", "U"]:
                print("[!] Invalid vowel selected.")
                continue

            try:
                res = predictor.predict_file(file_path=file_path, vowel=vowel)
                print("\n" + "-" * 40)
                print(f"DIAGNOSTIC RESULT FOR /{vowel}/")
                print("-" * 40)
                print(f"Prediction:         {res['prediction']}")
                print(f"PD Confidence:      {res['pd_probability']*100:.2f}%")
                print(f"Healthy Confidence: {res['healthy_probability']*100:.2f}%")
                print("-" * 40)
            except Exception as e:
                print(f"[!] Error processing file: {e}")

        elif choice == "3":
            print("[*] Exiting.")
            break
        else:
            print("[!] Invalid selection.")


if __name__ == "__main__":
    main()