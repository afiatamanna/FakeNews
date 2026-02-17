"""
Text preprocessing: cleaning and tokenization for classical ML and LSTM.
"""
import re
from typing import List, Optional

import pandas as pd

from config import TEXT_COLUMN


def clean_text(text: str) -> str:
    """
    Basic cleaning: lowercase, remove URLs, non-alphanumeric (keep spaces),
    collapse whitespace.
    """
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_df(df: pd.DataFrame, text_column: str = TEXT_COLUMN) -> pd.DataFrame:
    """Add a cleaned text column (cleaned_text) for classical ML / TF-IDF."""
    out = df.copy()
    out["cleaned_text"] = out[text_column].fillna("").astype(str).map(clean_text)
    return out


def get_corpus(df: pd.DataFrame, use_cleaned: bool = True) -> List[str]:
    """Return list of text for vectorizer (cleaned or raw)."""
    col = "cleaned_text" if (use_cleaned and "cleaned_text" in df.columns) else TEXT_COLUMN
    return df[col].fillna("").astype(str).tolist()
