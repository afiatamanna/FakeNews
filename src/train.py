"""Train TF-IDF + Logistic Regression for fake news detection."""
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from . import config
from .load_data import load_dataset


def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("Loading data...")
    train_df, test_df = load_dataset()
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=config.MAX_FEATURES, ngram_range=(1, 2), stop_words="english")),
        ("clf", LogisticRegression(max_iter=500, random_state=config.RANDOM_STATE)),
    ])

    print("Training...")
    pipe.fit(train_df["text"], train_df["label"])

    preds = pipe.predict(test_df["text"])
    acc = accuracy_score(test_df["label"], preds)
    p, r, f1, _ = precision_recall_fscore_support(test_df["label"], preds, average="binary")
    cm = confusion_matrix(test_df["label"], preds)

    # Save metrics as CSV
    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.csv")
    metrics_df = pd.DataFrame([
        ["accuracy", round(acc, 4)],
        ["precision", round(p, 4)],
        ["recall", round(r, 4)],
        ["f1", round(f1, 4)],
        ["tn", int(cm[0, 0])],
        ["fp", int(cm[0, 1])],
        ["fn", int(cm[1, 0])],
        ["tp", int(cm[1, 1])],
    ], columns=["metric", "value"])
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Metrics saved to {metrics_path}")
    print(metrics_df.to_string(index=False))

    joblib.dump(pipe, os.path.join(config.MODEL_DIR, "pipeline.joblib"))
    return pipe, train_df, test_df
