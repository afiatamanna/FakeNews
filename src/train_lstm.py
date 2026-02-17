"""
Train LSTM for fake news classification. Uses PyTorch and a simple word-level vocab.
"""
import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import (
    LABEL_COLUMN,
    LSTM_BATCH_SIZE,
    LSTM_DROPOUT,
    LSTM_EMBED_DIM,
    LSTM_EPOCHS,
    LSTM_HIDDEN_DIM,
    LSTM_LR,
    LSTM_NUM_LAYERS,
    MAX_LSTM_LENGTH,
    MODELS_DIR,
    RANDOM_STATE,
    VOCAB_SIZE,
)
from data_loading import get_splits
from evaluate import evaluate_and_save, print_metrics
from preprocessing import preprocess_df


def build_vocab(texts, max_size=VOCAB_SIZE):
    """Build word -> idx from list of texts. 0 = PAD, 1 = UNK."""
    from collections import Counter
    counter = Counter()
    for t in texts:
        counter.update(t.split())
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for w, _ in counter.most_common(max_size - 2):
        if w not in vocab:
            vocab[w] = len(vocab)
            if len(vocab) >= max_size:
                break
    return vocab


def text_to_ids(texts, vocab, max_len):
    """Convert list of strings to padded array of ids (numpy)."""
    out = []
    for t in texts:
        tokens = t.split()[:max_len]
        ids = [vocab.get(w, vocab["<UNK>"]) for w in tokens]
        pad = max_len - len(ids)
        ids = ids + [vocab["<PAD>"]] * pad
        out.append(ids)
    return np.array(out, dtype=np.int64)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, num_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False,
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, L)
        emb = self.embed(x)
        out, (h_n, _) = self.lstm(emb)
        last = h_n[-1]
        return self.fc(self.drop(last))


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def predict(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def train_lstm():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = get_splits()
    train_df = preprocess_df(train_df)
    val_df = preprocess_df(val_df)
    test_df = preprocess_df(test_df)

    train_texts = train_df["cleaned_text"].fillna("").astype(str).tolist()
    val_texts = val_df["cleaned_text"].fillna("").astype(str).tolist()
    test_texts = test_df["cleaned_text"].fillna("").astype(str).tolist()

    vocab = build_vocab(train_texts, VOCAB_SIZE)
    max_len = MAX_LSTM_LENGTH

    X_train = text_to_ids(train_texts, vocab, max_len)
    X_val = text_to_ids(val_texts, vocab, max_len)
    X_test = text_to_ids(test_texts, vocab, max_len)
    y_train = train_df[LABEL_COLUMN].values.astype(np.int64)
    y_val = val_df[LABEL_COLUMN].values.astype(np.int64)
    y_test = test_df[LABEL_COLUMN].values.astype(np.int64)

    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )
    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=LSTM_BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=LSTM_BATCH_SIZE)

    model = LSTMClassifier(
        vocab_size=len(vocab),
        embed_dim=LSTM_EMBED_DIM,
        hidden_dim=LSTM_HIDDEN_DIM,
        num_layers=LSTM_NUM_LAYERS,
        dropout=LSTM_DROPOUT,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_LR)

    for epoch in range(LSTM_EPOCHS):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"LSTM Epoch {epoch+1}/{LSTM_EPOCHS} train loss: {loss:.4f}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "lstm_model.pt")
    joblib.dump({"vocab": vocab, "max_len": max_len}, MODELS_DIR / "lstm_vocab.joblib")

    y_pred = predict(model, test_loader, device)
    metrics = evaluate_and_save(y_test, y_pred, "lstm")
    print_metrics(metrics)
    return model, vocab, metrics


if __name__ == "__main__":
    train_lstm()
