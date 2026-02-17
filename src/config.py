"""
Central configuration: paths, hyperparameters, model names.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_FAKE = DATA_DIR / "Fake.csv"
RAW_TRUE = DATA_DIR / "True.csv"
# Allow raw files in data/raw/ as alternative
RAW_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUTS_DIR / "models"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_CSV = OUTPUTS_DIR / "results.csv"

for d in (DATA_DIR, OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
# "huggingface" = load from HuggingFace (no local CSV needed)
# "local" = load from data/Fake.csv and data/True.csv only
# "auto" = try local first; if not found, use HuggingFace
DATA_SOURCE = "huggingface"
HUGGINGFACE_DATASET = "Pulk17/Fake-News-Detection-dataset"  # ~30k rows, title/text/subject/date/label
RANDOM_STATE = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"  # 0 = fake, 1 = real
MAX_SEQ_LENGTH = 512  # for BERT
MAX_LSTM_LENGTH = 200  # for LSTM

# ---------------------------------------------------------------------------
# Classical ML (sklearn)
# ---------------------------------------------------------------------------
CLASSICAL_MODELS = ["logistic_regression", "svm", "random_forest"]
TFIDF_MAX_FEATURES = 50_000
TFIDF_NGRAM_RANGE = (1, 2)
SVM_C = 1.0
SVM_KERNEL = "linear"
RF_N_ESTIMATORS = 200
RF_MAX_DEPTH = 20

# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
LSTM_EMBED_DIM = 128
LSTM_HIDDEN_DIM = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_EPOCHS = 5
LSTM_BATCH_SIZE = 32
LSTM_LR = 1e-3
VOCAB_SIZE = 50_000  # from tokenizer built on train

# ---------------------------------------------------------------------------
# BERT / Transformers
# ---------------------------------------------------------------------------
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_EPOCHS = 3
BERT_BATCH_SIZE = 16
BERT_LR = 2e-5

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
METRICS = ["accuracy", "precision", "recall", "f1"]
AVERAGING = "macro"  # for precision, recall, f1
