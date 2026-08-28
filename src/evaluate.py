"""
Module: evaluate.py
Description: Evaluates classification performance using Stratified K-Fold SVM
and computes Sensitivity, Specificity, Precision, F1-Score, and MCC.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Section 1: Stratified K-Fold Cross-Validation Evaluator
# ---------------------------------------------------------

def evaluate_svm_classifier(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Performs Stratified 5-Fold Cross Validation with Z-score feature scaling.
    Computes all standard clinical performance metrics.
    
    Classes:
        0 = Healthy (Negative)
        1 = Parkinson's Disease (Positive)
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Standardize features within each fold to avoid data leakage
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Support Vector Classifier with RBF kernel
        clf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
        clf.fit(X_train_scaled, y_train)

        y_pred = clf.predict(X_test_scaled)

        # Compute confusion matrix entries: labels=[0, 1] -> TN, FP, FN, TP
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    # Cumulative metric calculations
    total_samples = total_tp + total_tn + total_fp + total_fn
    accuracy = (total_tp + total_tn) / total_samples if total_samples > 0 else 0.0

    # Sensitivity (Recall) = TP / (TP + FN)
    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

    # Specificity = TN / (TN + FP)
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0

    # Precision = TP / (TP + FP)
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    # F1-Score = 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    if (precision + sensitivity) > 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0.0

    # Matthews Correlation Coefficient (MCC)
    denom = np.sqrt(
        float((total_tp + total_fp) * (total_tp + total_fn) * (total_tn + total_fp) * (total_tn + total_fn))
    )
    mcc = ((total_tp * total_tn) - (total_fp * total_fn)) / denom if denom > 0 else 0.0

    return {
        "TP": int(total_tp),
        "FP": int(total_fp),
        "FN": int(total_fn),
        "TN": int(total_tn),
        "Accuracy": float(accuracy),
        "Sensitivity": float(sensitivity),
        "Specificity": float(specificity),
        "Precision": float(precision),
        "F1_Score": float(f1_score),
        "MCC": float(mcc)
    }