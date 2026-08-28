"""
Module: main.py
Description: End-to-end execution pipeline.
Processes all vowel datasets independently, performs TLBO feature selection (29 -> 5),
evaluates with Stratified 5-Fold SVM, and outputs a formatted comparative table.
"""

import os
import numpy as np
from src.feature_extraction import get_available_vowels, load_vowel_dataset
from src.tlbo_optimizer import TLBOFeatureSelector
from src.evaluate import evaluate_svm_classifier


def run_pipeline():
    healthy_root = os.path.join("data", "Vowel_Healthy")
    pd_root = os.path.join("data", "Vowel_PD")

    vowels = get_available_vowels(healthy_root, pd_root)
    if not vowels:
        raise FileNotFoundError(f"No matching vowel folders found in {healthy_root} and {pd_root}.")

    print(f"[+] Found Vowels to evaluate: {vowels}")
    
    results = {}

    for vowel in vowels:
        print("\n" + "=" * 65)
        print(f"PROCESSING VOWEL: /{vowel}/")
        print("=" * 65)

        # 1. Load dataset (59 x 29)
        X, y = load_vowel_dataset(healthy_root, pd_root, vowel)
        print(f"[+] Dataset Shape: {X.shape} (Healthy: {np.sum(y == 0)}, PD: {np.sum(y == 1)})")

        # 2. Baseline Evaluation (All 29 features)
        base_metrics = evaluate_svm_classifier(X, y, n_splits=5)
        print(f"[-] Baseline (All 29 Features) -> Acc: {base_metrics['Accuracy']*100:.2f}%, F1: {base_metrics['F1_Score']:.4f}, MCC: {base_metrics['MCC']:.4f}")

        # 3. TLBO Feature Selection (29 -> 5 features)
        optimizer = TLBOFeatureSelector(n_learners=30, max_iter=100, target_features=5, random_state=42)
        optimizer.fit(X, y)
        selected_idx = optimizer.best_indices
        X_opt = optimizer.transform(X)

        print(f"[+] TLBO Selected Top 5 Features: {selected_idx}")
        print(f"[+] Minimized Scatter Ratio (Sw/Sb): {optimizer.best_fitness:.6f}")

        # 4. Optimized SVM Evaluation (5 features)
        opt_metrics = evaluate_svm_classifier(X_opt, y, n_splits=5)
        print(f"[+] Optimized (Selected 5 Features) -> Acc: {opt_metrics['Accuracy']*100:.2f}%, F1: {opt_metrics['F1_Score']:.4f}, MCC: {opt_metrics['MCC']:.4f}")

        results[vowel] = {
            "indices": selected_idx,
            "scatter_ratio": optimizer.best_fitness,
            "metrics": opt_metrics
        }

    # Comparative Summary Across All Vowels
    print("\n\n" + "=" * 90)
    print(f"{'VOWEL':<8}{'ACCURACY':<12}{'SENSITIVITY':<14}{'SPECIFICITY':<14}{'PRECISION':<12}{'F1-SCORE':<12}{'MCC':<10}")
    print("=" * 90)
    for v, data in results.items():
        m = data["metrics"]
        print(f"/{v}/     {m['Accuracy']*100:>6.2f}%     {m['Sensitivity']*100:>8.2f}%     {m['Specificity']*100:>8.2f}%     {m['Precision']*100:>8.2f}%     {m['F1_Score']:>8.4f}    {m['MCC']:>8.4f}")
    print("=" * 90)


if __name__ == "__main__":
    run_pipeline()