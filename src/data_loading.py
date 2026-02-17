"""
Load fake/real news data: from HuggingFace (default) or from local CSVs.
"""
from pathlib import Path

import pandas as pd

from config import (
    DATA_DIR,
    DATA_SOURCE,
    HUGGINGFACE_DATASET,
    LABEL_COLUMN,
    RANDOM_STATE,
    RAW_DIR,
    TEXT_COLUMN,
    TRAIN_RATIO,
    VAL_RATIO,
)


def _find_fake_true_paths():
    """Resolve paths for Fake.csv and True.csv (data/ or data/raw/)."""
    for base in (DATA_DIR, RAW_DIR):
        fake_path = base / "Fake.csv"
        true_path = base / "True.csv"
        if fake_path.exists() and true_path.exists():
            return fake_path, true_path
    return None, None


def load_from_huggingface() -> pd.DataFrame:
    """
    Load fake news dataset from HuggingFace. Returns DataFrame with text, label (0=fake, 1=real).
    Uses config HUGGINGFACE_DATASET (e.g. Pulk17/Fake-News-Detection-dataset, ~30k rows).
    """
    from datasets import load_dataset

    ds = load_dataset(HUGGINGFACE_DATASET, split="train")
    df = ds.to_pandas()
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    # Ensure text: use 'text' or fallback to 'title'
    if "text" not in df.columns and "title" in df.columns:
        df[TEXT_COLUMN] = df["title"].astype(str)
    elif "text" in df.columns:
        df[TEXT_COLUMN] = df["text"].fillna("").astype(str)
    else:
        df[TEXT_COLUMN] = ""
    # Label: dataset uses 0=fake, 1=real (same as ours)
    if "label" not in df.columns:
        raise ValueError(f"Dataset {HUGGINGFACE_DATASET} has no 'label' column. Columns: {list(df.columns)}")
    df[LABEL_COLUMN] = df["label"].astype(int)
    # Drop empty text
    df = df[df[TEXT_COLUMN].str.strip().str.len() > 0].copy()
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def load_raw_data() -> pd.DataFrame:
    """
    Load data from HuggingFace (default) or from local Fake.csv/True.csv.
    Returns DataFrame with text, label (0=fake, 1=real).
    """
    use_local = DATA_SOURCE == "local"
    if DATA_SOURCE == "auto":
        fake_path, true_path = _find_fake_true_paths()
        use_local = fake_path is not None and true_path is not None
    if not use_local:
        return load_from_huggingface()

    fake_path, true_path = _find_fake_true_paths()
    if fake_path is None or true_path is None:
        raise FileNotFoundError(
            "Fake.csv and True.csv not found. Place them in data/ or data/raw/, "
            "or set DATA_SOURCE='huggingface' in config.py to use the HuggingFace dataset."
        )

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Normalize column names (some versions use 'title' / 'text')
    for df in (fake_df, true_df):
        cols = [c.strip().lower() for c in df.columns]
        df.columns = cols

    # Ensure text column exists (sometimes 'title' + 'text' or just one)
    def get_text(row):
        if "text" in row and pd.notna(row["text"]) and str(row["text"]).strip():
            return str(row["text"]).strip()
        if "title" in row and pd.notna(row["title"]):
            return str(row["title"]).strip()
        return ""

    fake_df[TEXT_COLUMN] = fake_df.apply(get_text, axis=1)
    true_df[TEXT_COLUMN] = true_df.apply(get_text, axis=1)

    fake_df[LABEL_COLUMN] = 0
    true_df[LABEL_COLUMN] = 1

    # Drop rows with empty text
    fake_df = fake_df[fake_df[TEXT_COLUMN].str.len() > 0]
    true_df = true_df[true_df[TEXT_COLUMN].str.len() > 0]

    combined = pd.concat([fake_df, true_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    return combined


def train_val_test_split(df: pd.DataFrame):
    """
    Stratified train/val/test split.
    Returns (train_df, val_df, test_df).
    """
    n = len(df)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val :]
    return train_df, val_df, test_df


def get_splits():
    """
    Load raw data and return (train_df, val_df, test_df).
    """
    df = load_raw_data()
    return train_val_test_split(df)


if __name__ == "__main__":
    train, val, test = get_splits()
    print("Train:", len(train), "Val:", len(val), "Test:", len(test))
    print("Label distribution (train):", train[LABEL_COLUMN].value_counts().to_dict())
