"""Load fake news data from CSV and split into train/test."""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_dataset():
    """Load CSV and split into train_df, test_df. CSV must have text and label columns."""
    if not os.path.isfile(config.DATA_CSV):
        raise FileNotFoundError(f"Data file not found: {config.DATA_CSV}")

    df = pd.read_csv(config.DATA_CSV)
    text_col = config.CSV_TEXT_COL
    label_col = config.CSV_LABEL_COL

    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must have columns '{text_col}' and '{label_col}'. Found: {list(df.columns)}")

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "label"]
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)

    # Labels: convert to 0 (real) and 1 (fake)
    if df["label"].dtype == object or str(df["label"].dtype) == "string":
        df["label"] = df["label"].astype(str).str.lower().str.contains("fake").astype(int)
    else:
        if df["label"].nunique() > 2:
            top2 = df["label"].value_counts().index[:2].tolist()
            df = df[df["label"].isin(top2)]
        df["label"] = (df["label"] != df["label"].iloc[0]).astype(int)

    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
