"""
Explainable AI: SHAP and LIME for fake news classifier (classical model).
Generates figures for a few test examples and saves to outputs/figures/.
"""
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES_DIR, MODELS_DIR
from data_loading import get_splits
from preprocessing import get_corpus, preprocess_df


def load_classical_pipeline():
    """Load TF-IDF vectorizer and a classical model (e.g. logistic regression)."""
    vectorizer = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    model = joblib.load(MODELS_DIR / "classical_logistic_regression.joblib")
    return vectorizer, model


def explain_with_shap(num_samples=5, max_display=15, background_size=100):
    """
    Use SHAP LinearExplainer on logistic regression (TF-IDF).
    Saves a bar plot of feature importance for test examples.
    """
    try:
        import shap
    except ImportError:
        print("SHAP not installed. pip install shap")
        return

    vectorizer, model = load_classical_pipeline()
    _, _, test_df = get_splits()
    test_df = preprocess_df(test_df)
    texts = get_corpus(test_df)
    X = vectorizer.transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    n = X.shape[0]
    idx = np.random.RandomState(42).choice(n, size=min(background_size, n), replace=False)
    X_background = X[idx]
    X_small = X[:num_samples]

    if hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, X_background)
        shap_values = explainer.shap_values(X_small)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        X_dense = X.toarray()
        def pred_fn(x):
            return model.predict_proba(x)[:, 1]
        bg = X_dense[idx]
        explainer = shap.KernelExplainer(pred_fn, bg)
        shap_values = explainer.shap_values(X_small.toarray(), nsamples=50)

    shap_values = np.atleast_2d(shap_values)
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:max_display]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top_idx)), mean_abs[top_idx], color="steelblue")
    ax.set_yticks(range(len(top_idx)))
    ax.set_yticklabels([feature_names[i] for i in top_idx], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP feature importance (Logistic Regression)")
    ax.invert_yaxis()
    plt.tight_layout()
    out_path = FIGURES_DIR / "shap_summary.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved", out_path)


def explain_with_lime(num_samples=3, num_features=15):
    """
    Use LIME for text: explain a few test examples with the classical pipeline.
    Saves one figure per example.
    """
    try:
        from lime import lime_text
        from lime.lime_text import LimeTextExplainer
    except ImportError:
        print("LIME not installed. pip install lime")
        return

    vectorizer, model = load_classical_pipeline()
    _, _, test_df = get_splits()
    test_df = preprocess_df(test_df)
    texts = get_corpus(test_df)
    y_test = test_df["label"].values

    def pred_fn(texts_list):
        X = vectorizer.transform(texts_list)
        return model.predict_proba(X)

    explainer = LimeTextExplainer(class_names=["Fake", "Real"])
    for i in range(min(num_samples, len(texts))):
        text = texts[i]
        true_label = y_test[i]
        exp = explainer.explain_instance(
            text,
            pred_fn,
            num_features=num_features,
            num_samples=500,
        )
        fig = exp.as_pyplot_figure()
        fig.suptitle(f"LIME explanation (true: {'Real' if true_label == 1 else 'Fake'})")
        out_path = FIGURES_DIR / f"lime_example_{i+1}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved", out_path)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    print("Running SHAP...")
    explain_with_shap(num_samples=5, max_display=15)
    print("Running LIME...")
    explain_with_lime(num_samples=3, num_features=15)


if __name__ == "__main__":
    main()
