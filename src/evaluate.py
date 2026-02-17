"""
Compute metrics (accuracy, precision, recall, F1, confusion matrix) and append to results CSV.
Works for classical, LSTM, and BERT outputs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import AVERAGING, METRICS, RESULTS_CSV


def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    """
    Returns dict with model_name, accuracy, precision, recall, f1, and confusion_matrix (as list of lists).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=AVERAGING, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=AVERAGING, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=AVERAGING, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def append_results(metrics_dict: dict, results_path: Path = None):
    """Append one row to the results CSV (create file if missing)."""
    results_path = results_path or RESULTS_CSV
    row = {
        "model": metrics_dict["model"],
        "accuracy": metrics_dict["accuracy"],
        "precision": metrics_dict["precision"],
        "recall": metrics_dict["recall"],
        "f1": metrics_dict["f1"],
        "confusion_matrix": str(metrics_dict["confusion_matrix"]),
    }
    df = pd.DataFrame([row])
    if results_path.exists():
        existing = pd.read_csv(results_path)
        df = pd.concat([existing, df], ignore_index=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_path, index=False)


def evaluate_and_save(y_true, y_pred, model_name: str, results_path: Path = None):
    """Compute metrics, optionally save to CSV, return metrics dict."""
    metrics = compute_metrics(y_true, y_pred, model_name)
    append_results(metrics, results_path)
    return metrics


def print_metrics(metrics: dict):
    """Pretty-print metrics and confusion matrix."""
    print(f"\n--- {metrics['model']} ---")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:       {metrics['f1']:.4f}")
    print("Confusion matrix:")
    print(np.array(metrics["confusion_matrix"]))


if __name__ == "__main__":
    # Example: run after classical models; will be called from train_* scripts or run_all
    import joblib
    from config import MODELS_DIR, RESULTS_CSV
    from data_loading import get_splits
    from preprocessing import get_corpus, preprocess_df

    train_df, val_df, test_df = get_splits()
    train_df = preprocess_df(train_df)
    val_df = preprocess_df(val_df)
    test_df = preprocess_df(test_df)
    test_texts = get_corpus(test_df)
    y_test = test_df["label"].values

    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    X_test = vectorizer.transform(test_texts)

    for name in ["logistic_regression", "svm", "random_forest"]:
        path = MODELS_DIR / f"classical_{name}.joblib"
        if not path.exists():
            continue
        clf = joblib.load(path)
        y_pred = clf.predict(X_test)
        m = evaluate_and_save(y_test, y_pred, name, RESULTS_CSV)
        print_metrics(m)
