"""Explain predictions with SHAP (optional)."""
import os
from . import config


def explain_sample(pipe, texts, background_texts, max_display=10, save_path=None):
    """Explain a few samples with SHAP and save summary plot.
    background_texts: sample of training texts for SHAP background (e.g. train_df['text'].iloc[:100]).
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Install: pip install shap matplotlib")
        return None

    vec = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    X = vec.transform(texts)
    X_bg = vec.transform(background_texts)
    if hasattr(X_bg, "toarray"):
        X_bg = X_bg.toarray()
    if hasattr(X, "toarray"):
        X = X.toarray()
    feature_names = vec.get_feature_names_out().tolist()

    explainer = shap.LinearExplainer(clf, X_bg)
    shap_vals = explainer.shap_values(X)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    preds = pipe.predict(texts)
    print("Sample predictions:", ["FAKE" if p == 1 else "REAL" for p in preds])

    save_path = save_path or os.path.join(config.OUTPUT_DIR, "shap_summary.png")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_vals, X, feature_names=feature_names, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"SHAP summary saved to {save_path}")
    return shap_vals
