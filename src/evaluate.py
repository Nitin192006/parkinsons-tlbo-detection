"""
Module: evaluate.py
Description: Evaluates classification performance using Stratified K-Fold SVM
with hyperparameter tuning and modern CalibratedClassifierCV probability estimation.
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


def evaluate_svm_classifier(X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """
    Performs Stratified 5-Fold Cross Validation with Z-score feature scaling,
    grid search for optimal SVM hyperparameters, and calibrated probabilities.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    # Probability storage for multi-vowel ensemble aggregation
    fold_probabilities = np.zeros(len(y))

    # Parameter grid for RBF SVM
    param_grid = {
        'C': [0.1, 1.0, 5.0, 10.0, 50.0, 100.0],
        'gamma': ['scale', 'auto', 0.01, 0.05, 0.1, 0.5, 1.0]
    }

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Prevent data leakage by scaling inside each fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Inner cross-validation to select best C and gamma
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(
            SVC(kernel="rbf", class_weight="balanced", random_state=42),
            param_grid,
            cv=inner_cv,
            scoring="f1"
        )
        grid.fit(X_train_scaled, y_train)
        
        best_base_clf = grid.best_estimator_

        # Calibrate classifier probabilities without deprecation warnings
        calibrated_clf = CalibratedClassifierCV(
            estimator=best_base_clf,
            cv=3,
            method="sigmoid"
        )
        calibrated_clf.fit(X_train_scaled, y_train)

        y_pred = calibrated_clf.predict(X_test_scaled)
        fold_probabilities[test_idx] = calibrated_clf.predict_proba(X_test_scaled)[:, 1]

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

    total_samples = total_tp + total_tn + total_fp + total_fn
    accuracy = (total_tp + total_tn) / total_samples if total_samples > 0 else 0.0
    sensitivity = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0

    if (precision + sensitivity) > 0:
        f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    else:
        f1_score = 0.0

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
        "MCC": float(mcc),
        "Probabilities": fold_probabilities
    }