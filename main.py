"""
Module: main.py
Description: End-to-end multi-vowel evaluation pipeline followed by an interactive
testing menu for microphone recording or custom audio file evaluation.
Also saves trained models to outputs/trained_models.pkl for instant re-testing.
"""

import os
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from src.feature_extraction import get_available_vowels, load_vowel_dataset
from src.tlbo_optimizer import TLBOFeatureSelector
from src.evaluate import evaluate_svm_classifier
from src.visualize import plot_convergence, plot_confusion_matrices, plot_metrics_comparison
from src.live_inference import predict_sample, record_and_predict


def run_pipeline():
    healthy_root = os.path.join("data", "Vowel_Healthy")
    pd_root = os.path.join("data", "Vowel_PD")

    vowels = get_available_vowels(healthy_root, pd_root)
    if not vowels:
        raise FileNotFoundError(f"No matching vowel folders found in {healthy_root} and {pd_root}.")

    print(f"[+] Found Vowels to process: {vowels}")
    
    results = {}
    convergence_histories = {}
    confusion_matrices = {}
    vowel_probabilities = []
    ground_truth_labels = None
    trained_models = {}

    for vowel in vowels:
        print("\n" + "=" * 65)
        print(f"TRAINING & EVALUATING VOWEL: /{vowel}/")
        print("=" * 65)

        X, y = load_vowel_dataset(healthy_root, pd_root, vowel)
        ground_truth_labels = y

        # 1. TLBO Feature Selection (29 -> 5 features)
        optimizer = TLBOFeatureSelector(n_learners=50, max_iter=200, target_features=5, random_state=42)
        optimizer.fit(X, y)
        selected_idx = optimizer.best_indices
        X_opt = optimizer.transform(X)

        convergence_histories[vowel] = optimizer.history
        print(f"[+] TLBO Selected 5 Features: {selected_idx}")
        print(f"[+] Scatter Ratio (Sw/Sb): {optimizer.best_fitness:.6f}")

        # 2. Cross-Validation Evaluation
        opt_metrics = evaluate_svm_classifier(X_opt, y, n_splits=5)
        print(f"[+] 5-Fold SVM -> Acc: {opt_metrics['Accuracy']*100:.2f}%, F1: {opt_metrics['F1_Score']:.4f}, MCC: {opt_metrics['MCC']:.4f}")

        vowel_probabilities.append(opt_metrics["Probabilities"])
        confusion_matrices[f"Vowel /{vowel}/"] = np.array([
            [opt_metrics["TN"], opt_metrics["FP"]],
            [opt_metrics["FN"], opt_metrics["TP"]]
        ])
        results[vowel] = {
            "indices": selected_idx,
            "scatter_ratio": optimizer.best_fitness,
            "metrics": opt_metrics
        }

        # 3. Train final model for live inference
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_opt)
        base_clf = SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced", random_state=42)
        final_clf = CalibratedClassifierCV(estimator=base_clf, cv=3, method="sigmoid")
        final_clf.fit(X_scaled, y)

        trained_models[vowel] = {
            "optimizer": optimizer,
            "scaler": scaler,
            "classifier": final_clf
        }

    # Comparative Summary Table
    print("\n\n" + "=" * 90)
    print(f"{'VOWEL':<8}{'ACCURACY':<12}{'SENSITIVITY':<14}{'SPECIFICITY':<14}{'PRECISION':<12}{'F1-SCORE':<12}{'MCC':<10}")
    print("=" * 90)
    for v, data in results.items():
        m = data["metrics"]
        print(f"/{v}/     {m['Accuracy']*100:>6.2f}%     {m['Sensitivity']*100:>8.2f}%     {m['Specificity']*100:>8.2f}%     {m['Precision']*100:>8.2f}%     {m['F1_Score']:>8.4f}    {m['MCC']:>8.4f}")
    print("=" * 90)

    # Subject-Level Ensemble
    ensemble_metrics = {}
    if len(vowels) > 1 and ground_truth_labels is not None:
        avg_probs = np.mean(vowel_probabilities, axis=0)
        ensemble_preds = (avg_probs >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(ground_truth_labels, ensemble_preds, labels=[0, 1]).ravel()
        ens_acc = (tp + tn) / len(ground_truth_labels)
        ens_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ens_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ens_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        ens_f1 = 2 * (ens_prec * ens_sens) / (ens_prec + ens_sens) if (ens_prec + ens_sens) > 0 else 0.0
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        ens_mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        ensemble_metrics = {
            "Accuracy": ens_acc,
            "Sensitivity": ens_sens,
            "Specificity": ens_spec,
            "Precision": ens_prec,
            "F1_Score": ens_f1,
            "MCC": ens_mcc
        }
        confusion_matrices["Ensemble (Soft Voting)"] = np.array([[tn, fp], [fn, tp]])

        print("\n" + "=" * 90)
        print("SUBJECT-LEVEL MULTI-VOWEL ENSEMBLE")
        print("=" * 90)
        print(f"Accuracy:    {ens_acc * 100:.2f}%")
        print(f"Sensitivity: {ens_sens * 100:.2f}%")
        print(f"Specificity: {ens_spec * 100:.2f}%")
        print(f"Precision:   {ens_prec * 100:.2f}%")
        print(f"F1-Score:    {ens_f1:.4f}")
        print(f"MCC:         {ens_mcc:.4f}")
        print("=" * 90)

    # Save visualization plots
    os.makedirs("outputs", exist_ok=True)
    plot_convergence(convergence_histories, "outputs/tlbo_convergence.png")
    plot_confusion_matrices(confusion_matrices, "outputs/confusion_matrices.png")
    if ensemble_metrics:
        plot_metrics_comparison(results, ensemble_metrics, "outputs/vowel_comparison.png")

    # Save trained models for instant inference in test_only.py
    model_save_path = os.path.join("outputs", "trained_models.pkl")
    with open(model_save_path, "wb") as f:
        pickle.dump(trained_models, f)
    print(f"\n[+] Trained models saved to '{model_save_path}'.")

    # Interactive Testing Interface
    while True:
        print("\n" + "=" * 50)
        print("LIVE VOICE TESTING INTERFACE")
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
            res = predict_sample(file_path, vowel, trained_models)
            print("\n" + "-" * 40)
            print(f"RESULT FOR /{vowel}/")
            print(f"Prediction:         {res['prediction']}")
            print(f"PD Confidence:      {res['pd_probability']*100:.2f}%")
            print(f"Healthy Confidence: {res['healthy_probability']*100:.2f}%")
            print("-" * 40)

        elif choice == "3":
            print("[*] Exiting pipeline.")
            break
        else:
            print("[!] Invalid option selected.")


if __name__ == "__main__":
    run_pipeline()