"""
Module: main.py
Description: Full pipeline with tuned SVM, deep TLBO optimization (29 -> 5 features),
individual vowel evaluation, and Subject-Level Multi-Vowel Soft Voting Ensemble.
"""

import os
import numpy as np
from src.feature_extraction import get_available_vowels, load_vowel_dataset
from src.tlbo_optimizer import TLBOFeatureSelector
from src.evaluate import evaluate_svm_classifier
from sklearn.metrics import confusion_matrix


def run_pipeline():
    healthy_root = os.path.join("data", "Vowel_Healthy")
    pd_root = os.path.join("data", "Vowel_PD")

    vowels = get_available_vowels(healthy_root, pd_root)
    if not vowels:
        raise FileNotFoundError(f"No matching vowel folders found in {healthy_root} and {pd_root}.")

    print(f"[+] Found Vowels to evaluate: {vowels}")
    
    results = {}
    vowel_probabilities = []
    ground_truth_labels = None

    for vowel in vowels:
        print("\n" + "=" * 65)
        print(f"PROCESSING VOWEL: /{vowel}/")
        print("=" * 65)

        X, y = load_vowel_dataset(healthy_root, pd_root, vowel)
        ground_truth_labels = y

        # TLBO Feature Selection (29 -> exactly 5 features, deeper search)
        optimizer = TLBOFeatureSelector(n_learners=50, max_iter=200, target_features=5, random_state=42)
        optimizer.fit(X, y)
        selected_idx = optimizer.best_indices
        X_opt = optimizer.transform(X)

        print(f"[+] TLBO Selected 5 Features: {selected_idx}")
        print(f"[+] Scatter Ratio (Sw/Sb): {optimizer.best_fitness:.6f}")

        # Tuned SVM Evaluation on selected 5 features
        opt_metrics = evaluate_svm_classifier(X_opt, y, n_splits=5)
        print(f"[+] Tuned SVM (5 Features) -> Acc: {opt_metrics['Accuracy']*100:.2f}%, Sens: {opt_metrics['Sensitivity']*100:.2f}%, Spec: {opt_metrics['Specificity']*100:.2f}%, F1: {opt_metrics['F1_Score']:.4f}, MCC: {opt_metrics['MCC']:.4f}")

        vowel_probabilities.append(opt_metrics["Probabilities"])
        results[vowel] = {
            "indices": selected_idx,
            "scatter_ratio": optimizer.best_fitness,
            "metrics": opt_metrics
        }

    # Summary Table Across All Vowels
    print("\n\n" + "=" * 90)
    print(f"{'VOWEL':<8}{'ACCURACY':<12}{'SENSITIVITY':<14}{'SPECIFICITY':<14}{'PRECISION':<12}{'F1-SCORE':<12}{'MCC':<10}")
    print("=" * 90)
    for v, data in results.items():
        m = data["metrics"]
        print(f"/{v}/     {m['Accuracy']*100:>6.2f}%     {m['Sensitivity']*100:>8.2f}%     {m['Specificity']*100:>8.2f}%     {m['Precision']*100:>8.2f}%     {m['F1_Score']:>8.4f}    {m['MCC']:>8.4f}")
    print("=" * 90)

    # Subject-Level Ensemble: Soft Voting across all 5 vowels
    if len(vowels) > 1 and ground_truth_labels is not None:
        avg_probabilities = np.mean(vowel_probabilities, axis=0)
        ensemble_preds = (avg_probabilities >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(ground_truth_labels, ensemble_preds, labels=[0, 1]).ravel()
        ens_acc = (tp + tn) / len(ground_truth_labels)
        ens_sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        ens_spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        ens_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        ens_f1 = 2 * (ens_prec * ens_sens) / (ens_prec + ens_sens) if (ens_prec + ens_sens) > 0 else 0.0
        denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        ens_mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

        print("\n" + "=" * 90)
        print("SUBJECT-LEVEL MULTI-VOWEL ENSEMBLE (SOFT VOTING ACROSS ALL 5 VOWELS)")
        print("=" * 90)
        print(f"Confusion Matrix -> TP: {tp}, FP: {fp}, FN: {fn}, TN: {tn}")
        print(f"Accuracy:    {ens_acc * 100:.2f}%")
        print(f"Sensitivity: {ens_sens * 100:.2f}%")
        print(f"Specificity: {ens_spec * 100:.2f}%")
        print(f"Precision:   {ens_prec * 100:.2f}%")
        print(f"F1-Score:    {ens_f1:.4f}")
        print(f"MCC:         {ens_mcc:.4f}")
        print("=" * 90)


if __name__ == "__main__":
    run_pipeline()