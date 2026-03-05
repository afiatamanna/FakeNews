
"""
Run fake news detection pipeline: train TF-IDF + Logistic Regression, evaluate, optional SHAP.
Usage: python run_fake_news.py [--explain]
"""
import argparse
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import train
from src import config


def main():
    parser = argparse.ArgumentParser(description="Fake news detection pipeline")
    parser.add_argument("--explain", action="store_true", help="Run SHAP on a few test samples")
    args = parser.parse_args()

    print("=== Fake News Detection (TF-IDF + Logistic Regression) ===\n")
    pipe, train_df, test_df = train()

    if args.explain:
        try:
            from src.explain import explain_sample
            bg = train_df["text"].iloc[:50].tolist()
            samples = test_df["text"].iloc[:5].tolist()
            explain_sample(pipe, samples, bg, max_display=12)
        except Exception as e:
            print("SHAP explain skipped:", e)


if __name__ == "__main__":
    main()
