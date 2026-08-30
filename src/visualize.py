"""
Module: visualize.py
Description: Generates and saves visual analytics figures:
1. TLBO Optimization Convergence Curves
2. Confusion Matrix Heatmaps
3. Performance Metrics Comparison Bar Chart
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_convergence(history_dict: dict, save_path: str = "outputs/tlbo_convergence.png"):
    """
    Plots the TLBO fitness (Scatter Ratio Sw/Sb) convergence curves.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    
    for vowel, history in history_dict.items():
        plt.plot(history, label=f"Vowel /{vowel}/", linewidth=2)

    plt.title("TLBO Feature Selection Convergence (Minimizing Sw / Sb)", fontsize=14, fontweight="bold")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Scatter Ratio (Sw / Sb)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Convergence Plot to: {save_path}")


def plot_confusion_matrices(cm_dict: dict, save_path: str = "outputs/confusion_matrices.png"):
    """
    Plots confusion matrix heatmaps for all vowels and ensemble.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    n_plots = len(cm_dict)
    cols = 3
    rows = int(np.ceil(n_plots / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten()

    for idx, (title, cm) in enumerate(cm_dict.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Healthy", "PD"], yticklabels=["Healthy", "PD"],
                    annot_kws={"size": 14, "weight": "bold"})
        ax.set_title(f"Confusion Matrix: {title}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=10)
        ax.set_ylabel("True Label", fontsize=10)

    # Hide unused subplots
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Confusion Matrices to: {save_path}")


def plot_metrics_comparison(results: dict, ensemble_metrics: dict, save_path: str = "outputs/vowel_comparison.png"):
    """
    Plots grouped bar chart comparing clinical metrics across vowels and ensemble.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    categories = list(results.keys()) + ["Ensemble"]
    
    acc = [results[v]["metrics"]["Accuracy"] * 100 for v in results] + [ensemble_metrics["Accuracy"] * 100]
    sens = [results[v]["metrics"]["Sensitivity"] * 100 for v in results] + [ensemble_metrics["Sensitivity"] * 100]
    spec = [results[v]["metrics"]["Specificity"] * 100 for v in results] + [ensemble_metrics["Specificity"] * 100]
    f1 = [results[v]["metrics"]["F1_Score"] * 100 for v in results] + [ensemble_metrics["F1_Score"] * 100]

    x = np.arange(len(categories))
    width = 0.2

    plt.figure(figsize=(12, 6))
    plt.bar(x - 1.5 * width, acc, width, label="Accuracy (%)")
    plt.bar(x - 0.5 * width, sens, width, label="Sensitivity (%)")
    plt.bar(x + 0.5 * width, spec, width, label="Specificity (%)")
    plt.bar(x + 1.5 * width, f1, width, label="F1-Score (x100)")

    plt.xlabel("Phonation Target / Model", fontsize=12, fontweight="bold")
    plt.ylabel("Score (%)", fontsize=12, fontweight="bold")
    plt.title("Diagnostic Performance Comparison Across Vowels & Ensemble", fontsize=14, fontweight="bold")
    plt.xticks(x, [f"/{c}/" if len(c) == 1 else c for c in categories], fontsize=11)
    plt.ylim(0, 105)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved Metrics Comparison Plot to: {save_path}")