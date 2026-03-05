"""Config for fake news detection."""
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# data CSV (must have a text column and a label column)
DATA_CSV = os.path.join(DATA_DIR, "fake_news.csv")
CSV_TEXT_COL = "text"
CSV_LABEL_COL = "label"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
