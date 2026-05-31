"""
src/transformer_models.py
--------------------------
Fine-tune and evaluate XLM-RoBERTa and AfriBERTa for Swahili sentiment analysis.

Usage:
    python src/transformer_models.py --model xlm-roberta
    python src/transformer_models.py --model afriberta
    python src/transformer_models.py --model both

    # Reads:  data/processed/cleaned_data.csv
    # Writes: results/transformer_results.csv
    #         results/figures/confusion_<model>.png
    #         models/<model_name>/  (saved HuggingFace model)
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

try:
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        Trainer, TrainingArguments, EarlyStoppingCallback
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    Dataset = object
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    Trainer = None
    TrainingArguments = None
    EarlyStoppingCallback = None
    print("[ERROR] Missing required packages for transformers.")
    print("Install with: pip install -r requirements.txt")
    print("Required: torch, transformers, sentencepiece")



# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_data.csv")
RESULTS_PATH        = os.path.join("results", "transformer_results.csv")
FIGURES_DIR         = os.path.join("results", "figures")
MODELS_DIR          = "models"
LABEL_NAMES         = ["negative", "neutral", "positive"]
RANDOM_SEED         = 42

MODEL_CONFIGS = {
    "xlm-roberta": {
        "checkpoint": "xlm-roberta-base",
        "save_name":  "xlm_roberta_swahili",
        "display":    "XLM-RoBERTa",
    },
    "afriberta": {
        "checkpoint": "castorini/afriberta_large",
        "save_name":  "afriberta_swahili",
        "display":    "AfriBERTa",
    },
}


# ─────────────────────────────────────────────
# Dataset class
# ─────────────────────────────────────────────

class SwahiliSentimentDataset(Dataset):
    """PyTorch Dataset for tokenised Swahili tweets."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item


# ─────────────────────────────────────────────
# Metrics for Trainer
# ─────────────────────────────────────────────

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted", zero_division=0),
    }


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_data(path: str = PROCESSED_DATA_PATH):
    df = pd.read_csv(path)
    X  = df["clean_text"].fillna("").tolist()
    y  = df["label_encoded"].astype(int).tolist()
    print(f"[INFO] Loaded {len(df):,} rows from '{path}'")
    return X, y


def save_confusion_matrix(name: str, y_true, y_pred, save_dir: str = FIGURES_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f"{name} – Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fname = name.lower().replace(" ", "_").replace("-", "_")
    out = os.path.join(save_dir, f"confusion_{fname}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to '{out}'")


# ─────────────────────────────────────────────
# Core fine-tuning function
# ─────────────────────────────────────────────

def fine_tune(
    config_key:   str,
    X_train:      list,
    y_train:      list,
    X_test:       list,
    y_test:       list,
    num_epochs:   int = 3,
    batch_size:   int = 16,
    max_len:      int = 128,
    learning_rate: float = 2e-5,
) -> dict:
    """Fine-tune a single transformer model and return metrics dict."""

    if not TRANSFORMERS_AVAILABLE:
        raise RuntimeError("transformers / torch not available.")

    cfg       = MODEL_CONFIGS[config_key]
    ckpt      = cfg["checkpoint"]
    name      = cfg["display"]
    save_path = os.path.join(MODELS_DIR, cfg["save_name"])

    print(f"\n[INFO] Fine-tuning {name} ({ckpt})")
    print(f"       Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"       Epochs: {num_epochs}  |  Batch: {batch_size}  |  LR: {learning_rate}")

    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=False)
    model     = AutoModelForSequenceClassification.from_pretrained(ckpt, num_labels=3)

    # Tokenise
    train_enc = tokenizer(X_train, truncation=True, padding=True, max_length=max_len)
    test_enc  = tokenizer(X_test,  truncation=True, padding=True, max_length=max_len)

    train_ds = SwahiliSentimentDataset(train_enc, y_train)
    test_ds  = SwahiliSentimentDataset(test_enc,  y_test)

    training_args = TrainingArguments(
        output_dir                  = os.path.join("results", "checkpoints", cfg["save_name"]),
        num_train_epochs            = num_epochs,
        per_device_train_batch_size = batch_size,
        per_device_eval_batch_size  = batch_size,
        learning_rate               = learning_rate,
        weight_decay                = 0.01,
        warmup_ratio                = 0.1,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        logging_steps               = 20,
        report_to                   = "none",
        seed                        = RANDOM_SEED,
        fp16                        = torch.cuda.is_available(),
    )

    trainer = Trainer(
        model           = model,
        args            = training_args,
        train_dataset   = train_ds,
        eval_dataset    = test_ds,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Evaluate
    test_enc_pt = tokenizer(X_test, truncation=True, padding=True,
                             max_length=max_len, return_tensors="pt")
    model.eval()
    with torch.no_grad():
        outputs = model(**test_enc_pt)
    preds = torch.argmax(outputs.logits, dim=1).numpy()

    print(f"\n── {name} ──────────────────────────────────────")
    print(classification_report(y_test, preds, target_names=LABEL_NAMES, zero_division=0))

    save_confusion_matrix(name, y_test, preds)

    # Save model & tokenizer
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"[INFO] Model saved to '{save_path}'")

    return {
        "Model":          name,
        "Accuracy":       accuracy_score(y_test, preds),
        "F1 Score (W)":   f1_score(y_test, preds, average="weighted", zero_division=0),
        "Precision (W)":  precision_score(y_test, preds, average="weighted", zero_division=0),
        "Recall (W)":     recall_score(y_test, preds, average="weighted", zero_division=0),
    }


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune transformer models for Swahili sentiment.")
    parser.add_argument("--model",    choices=["xlm-roberta", "afriberta", "both"],
                        default="both", help="Which model to train")
    parser.add_argument("--input",    default=PROCESSED_DATA_PATH)
    parser.add_argument("--epochs",   type=int,   default=3)
    parser.add_argument("--batch",    type=int,   default=16)
    parser.add_argument("--lr",       type=float, default=2e-5)
    parser.add_argument("--max-len",  type=int,   default=128)
    return parser.parse_args()


def main():
    if not TRANSFORMERS_AVAILABLE:
        print("[ERROR] Required transformer dependencies are missing.")
        print("Install them with: pip install -r requirements.txt")
        sys.exit(1)

    args = parse_args()

    X, y = load_data(args.input)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    keys = list(MODEL_CONFIGS.keys()) if args.model == "both" else [args.model]
    results = []

    for key in keys:
        row = fine_tune(
            config_key    = key,
            X_train       = X_train,
            y_train       = y_train,
            X_test        = X_test,
            y_test        = y_test,
            num_epochs    = args.epochs,
            batch_size    = args.batch,
            max_len       = args.max_len,
            learning_rate = args.lr,
        )
        results.append(row)

    results_df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\n[INFO] Results saved to '{RESULTS_PATH}'")
    print("\n── Transformer Summary ─────────────────────────")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))
    print("\n[INFO] Transformer training complete.")


if __name__ == "__main__":
    main()
