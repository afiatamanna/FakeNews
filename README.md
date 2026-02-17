# Fake News Detection with Explainable AI

## Problem Statement

Automatically detecting fake news is a real-world NLP problem with implications for trust and safety. This project compares several approaches—from TF-IDF + logistic regression to transformer-based classifiers—and adds interpretability so that predictions can be explained.

## Dataset

- **Default:** Loaded automatically from [HuggingFace (Pulk17/Fake-News-Detection-dataset)](https://huggingface.co/datasets/Pulk17/Fake-News-Detection-dataset) (~30k rows). No manual download; set `DATA_SOURCE = "huggingface"` in `src/config.py`.
- **Optional:** Use local CSVs from [Kaggle Fake and Real News](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset): set `DATA_SOURCE = "local"` and place `Fake.csv` and `True.csv` in `data/` or `data/raw/`.
- See [data/README.md](data/README.md) for details.

## Methods

| Model | Description |
|-------|-------------|
| Logistic Regression | TF-IDF features + linear classifier |
| SVM | TF-IDF + LinearSVC |
| Random Forest | TF-IDF + ensemble of trees |
| LSTM | Word-level embedding + LSTM (PyTorch) |
| BERT | Fine-tuned DistilBERT (HuggingFace) |
| XAI | SHAP and LIME on the logistic regression pipeline |

## Results

After running the pipeline, see:

- **Comparison table:** `outputs/results.csv` (accuracy, precision, recall, F1 per model).
- **Confusion matrices:** stored in the same CSV (as text) and can be plotted from the evaluation script.
- **Explainability:** `outputs/figures/shap_summary.png` and `outputs/figures/lime_example_*.png`.

Example results (placeholder; run the pipeline to fill):

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| logistic_regression | — | — | — | — |
| svm | — | — | — | — |
| random_forest | — | — | — | — |
| lstm | — | — | — | — |
| bert | — | — | — | — |

## How to Run

### 1. Environment

```bash
cd /path/to/python
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Data

Download the Kaggle dataset and place `Fake.csv` and `True.csv` in `data/` or `data/raw/`.

### 3. Train and evaluate

From the project root (so that `src` is importable):

```bash
# Option A: run steps manually (from project root)
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python src/train_classical.py
python src/train_lstm.py
python src/train_bert.py
python src/evaluate.py    # re-run evaluation from saved models
python src/explain.py     # SHAP + LIME figures
```

Or use the script:

```bash
chmod +x scripts/run_all.sh
./scripts/run_all.sh
```

### 4. Research report

See [report/report.md](report/report.md) for the short research report (abstract, methodology, results, discussion, limitations).

## Repository Structure

```
├── README.md
├── requirements.txt
├── data/
│   ├── README.md
│   └── (Fake.csv, True.csv)
├── report/
│   ├── report.md
│   └── report.pdf (export from report.md)
├── src/
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── train_classical.py
│   ├── train_lstm.py
│   ├── train_bert.py
│   ├── evaluate.py
│   └── explain.py
├── notebooks/
│   └── 01_eda.ipynb
├── outputs/
│   ├── models/
│   ├── figures/
│   └── results.csv
└── scripts/
    └── run_all.sh
```

## Evaluation Metrics

- **Accuracy**, **Precision**, **Recall**, **F1** (macro-averaged)
- **Confusion matrix** per model
- Results are appended to `outputs/results.csv` for a single comparison table.

## Future Work

- Bias analysis (e.g. by topic or source)
- Domain adaptation (train on one source, test on another)
- Multilingual or cross-lingual fake news detection
- Fine-grained veracity (e.g. LIAR dataset) instead of binary

## License

Code in this repo is for portfolio/educational use. Dataset usage must comply with the original dataset licenses (Kaggle, LIAR).
