"""
Train classical ML models: Logistic Regression, SVM, Random Forest.
Uses TF-IDF features on cleaned text.
"""
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from config import (
    CLASSICAL_MODELS,
    LABEL_COLUMN,
    MODELS_DIR,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RANDOM_STATE,
    SVM_C,
    SVM_KERNEL,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from data_loading import get_splits
from preprocessing import get_corpus, preprocess_df


def build_tfidf_and_fit(train_texts):
    """Build TF-IDF vectorizer on train texts. Returns vectorizer and X_train matrix."""
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    return vectorizer, X_train


def train_classical_models():
    """
    Load data, preprocess, train LR, SVM, RF. Save vectorizer and each model.
    Returns (vectorizer, dict of model_name -> fitted classifier, train/val/test data).
    """
    train_df, val_df, test_df = get_splits()
    train_df = preprocess_df(train_df)
    val_df = preprocess_df(val_df)
    test_df = preprocess_df(test_df)

    train_texts = get_corpus(train_df)
    val_texts = get_corpus(val_df)
    test_texts = get_corpus(test_df)

    y_train = train_df[LABEL_COLUMN].values
    y_val = val_df[LABEL_COLUMN].values
    y_test = test_df[LABEL_COLUMN].values

    vectorizer, X_train = build_tfidf_and_fit(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    models = {}

    if "logistic_regression" in CLASSICAL_MODELS:
        clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        models["logistic_regression"] = clf

    if "svm" in CLASSICAL_MODELS:
        clf = LinearSVC(C=SVM_C, random_state=RANDOM_STATE, max_iter=2000)
        clf.fit(X_train, y_train)
        models["svm"] = clf

    if "random_forest" in CLASSICAL_MODELS:
        clf = RandomForestClassifier(
            n_estimators=RF_N_ESTIMATORS,
            max_depth=RF_MAX_DEPTH,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_train, y_train)
        models["random_forest"] = clf

    joblib.dump(vectorizer, MODELS_DIR / "tfidf_vectorizer.joblib")
    for name, clf in models.items():
        joblib.dump(clf, MODELS_DIR / f"classical_{name}.joblib")

    data = {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
    }
    return vectorizer, models, data


if __name__ == "__main__":
    train_classical_models()
    print("Classical models and TF-IDF vectorizer saved to", MODELS_DIR)
