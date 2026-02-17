"""
Fine-tune DistilBERT (HuggingFace) for fake news classification.
"""
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from config import (
    BERT_BATCH_SIZE,
    BERT_EPOCHS,
    BERT_LR,
    BERT_MODEL_NAME,
    LABEL_COLUMN,
    MAX_SEQ_LENGTH,
    MODELS_DIR,
    RANDOM_STATE,
    TEXT_COLUMN,
)
from data_loading import get_splits
from evaluate import evaluate_and_save, print_metrics
from preprocessing import preprocess_df


def train_bert():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df, val_df, test_df = get_splits()
    train_df = preprocess_df(train_df)
    val_df = preprocess_df(val_df)
    test_df = preprocess_df(test_df)

    # Use cleaned_text for BERT too (or TEXT_COLUMN for raw)
    text_col = "cleaned_text" if "cleaned_text" in train_df.columns else TEXT_COLUMN
    train_texts = train_df[text_col].fillna("").astype(str).tolist()
    val_texts = val_df[text_col].fillna("").astype(str).tolist()
    test_texts = test_df[text_col].fillna("").astype(str).tolist()
    y_train = train_df[LABEL_COLUMN].values
    y_val = val_df[LABEL_COLUMN].values
    y_test = test_df[LABEL_COLUMN].values

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    def tokenize(examples):
        return tokenizer(
            examples[text_col],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            return_tensors=None,
        )

    train_ds = Dataset.from_dict({text_col: train_texts, "labels": y_train.tolist()})
    val_ds = Dataset.from_dict({text_col: val_texts, "labels": y_val.tolist()})
    test_ds = Dataset.from_dict({text_col: test_texts, "labels": y_test.tolist()})
    train_ds = train_ds.map(tokenize, batched=True, remove_columns=[text_col])
    val_ds = val_ds.map(tokenize, batched=True, remove_columns=[text_col])
    test_ds = test_ds.map(tokenize, batched=True, remove_columns=[text_col])

    # Rename to "labels" for Trainer
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME,
        num_labels=2,
    )

    training_args = TrainingArguments(
        output_dir=str(MODELS_DIR / "bert_checkpoints"),
        num_train_epochs=BERT_EPOCHS,
        per_device_train_batch_size=BERT_BATCH_SIZE,
        per_device_eval_batch_size=BERT_BATCH_SIZE,
        learning_rate=BERT_LR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=RANDOM_STATE,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )
    trainer.train()

    # Save final model and tokenizer
    save_dir = MODELS_DIR / "bert_final"
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    # Predict on test set
    pred_output = trainer.predict(test_ds)
    y_pred = np.argmax(pred_output.predictions, axis=1)
    metrics = evaluate_and_save(y_test, y_pred, "bert")
    print_metrics(metrics)
    return model, tokenizer, metrics


if __name__ == "__main__":
    train_bert()
