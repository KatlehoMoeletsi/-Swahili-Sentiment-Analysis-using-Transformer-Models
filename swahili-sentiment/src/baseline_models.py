"""
src/baseline_models.py
----------------------
Train and evaluate TF-IDF + classical ML baselines for Swahili sentiment.

Models
------
  - Logistic Regression
  - Naive Bayes (Multinomial)
  - Linear SVM

Usage:
    python src/baseline_models.py
    # Reads:  data/processed/cleaned_data.csv
    # Writes: results/baseline_results.csv
    #         results/figures/confusion_<model>.png
    #         models/tfidf_vectorizer.pkl
    #         models/logistic_regression.pkl
    #         models/naive_bayes.pkl
    #         models/svm.pkl
"""

import os
import argparse
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_data.csv")
RESULTS_PATH        = os.path.join("results", "baseline_results.csv")
FIGURES_DIR         = os.path.join("results", "figures")
MODELS_DIR          = "models"

LABEL_NAMES = ["negative", "neutral", "positive"]
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def load_data(path: str = PROCESSED_DATA_PATH):
    """Load processed data and return X, y arrays."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df):,} rows from '{path}'")
    X = df["clean_text"].fillna("").tolist()
    y = df["label_encoded"].tolist()
    return X, y


def make_vectorizer(ngram_range=(1, 2), max_features=5000) -> TfidfVectorizer:
    return TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        sublinear_tf=True,          # apply log(tf) scaling
        strip_accents="unicode",
        min_df=2,
    )


def evaluate(name: str, y_true, y_pred) -> dict:
    """Return a metrics dict and print a classification report."""
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n── {name} ──────────────────────────────────────")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES, zero_division=0))

    return {"Model": name, "Accuracy": acc, "F1 Score (W)": f1,
            "Precision (W)": prec, "Recall (W)": rec}


def save_confusion_matrix(name: str, y_true, y_pred, save_dir: str = FIGURES_DIR) -> None:
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f"{name} – Confusion Matrix", fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fname = name.lower().replace(" ", "_")
    out = os.path.join(save_dir, f"confusion_{fname}.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved to '{out}'")


def save_comparison_chart(results: list, save_dir: str = FIGURES_DIR) -> None:
    """Bar chart comparing F1 scores across models."""
    os.makedirs(save_dir, exist_ok=True)
    df = pd.DataFrame(results)
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    bars = ax.bar(df["Model"], df["F1 Score (W)"], color=colors, edgecolor="white")
    for bar, val in zip(bars, df["F1 Score (W)"]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f"{val:.4f}",
                ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_title("Baseline Model Comparison – Weighted F1 Score", fontsize=13, fontweight="bold")
    ax.set_ylabel("Weighted F1 Score")
    ax.spines[["top", "right"]].set_visible(False)
    out = os.path.join(save_dir, "baseline_comparison.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] Comparison chart saved to '{out}'")


# ─────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────

def train_and_evaluate(processed_path: str = PROCESSED_DATA_PATH) -> pd.DataFrame:
    X, y = load_data(processed_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"[INFO] Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Fit vectorizer
    vectorizer = make_vectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf  = vectorizer.transform(X_test)

    results = []
    preds   = {}

    # ── Logistic Regression ───────────────────
    lr = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
    random_state=RANDOM_SEED
)
    lr.fit(X_train_tfidf, y_train)
    lr_pred = lr.predict(X_test_tfidf)
    results.append(evaluate("Logistic Regression", y_test, lr_pred))
    preds["Logistic Regression"] = lr_pred

    # ── Naive Bayes ───────────────────────────
    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    results.append(evaluate("Naive Bayes", y_test, nb_pred))
    preds["Naive Bayes"] = nb_pred

    # ── Linear SVM ───────────────────────────
    svm = LinearSVC(C=1.0, random_state=RANDOM_SEED, max_iter=2000)
    svm.fit(X_train_tfidf, y_train)
    svm_pred = svm.predict(X_test_tfidf)
    results.append(evaluate("Linear SVM", y_test, svm_pred))
    preds["Linear SVM"] = svm_pred

    # ── Save artifacts ────────────────────────
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(lr,         os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    joblib.dump(nb,         os.path.join(MODELS_DIR, "naive_bayes.pkl"))
    joblib.dump(svm,        os.path.join(MODELS_DIR, "svm.pkl"))
    print(f"\n[INFO] Models saved to '{MODELS_DIR}/'")

    for name, pred in preds.items():
        save_confusion_matrix(name, y_test, pred)

    results_df = pd.DataFrame(results)
    save_comparison_chart(results)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"[INFO] Results saved to '{RESULTS_PATH}'")

    print("\n── Summary ─────────────────────────────────────")
    print(results_df.to_string(index=False, float_format="{:.4f}".format))

    return results_df


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train TF-IDF baseline classifiers.")
    parser.add_argument("--input", default=PROCESSED_DATA_PATH, help="Processed CSV path")
    return parser.parse_args()


def main():
    args = parse_args()
    train_and_evaluate(args.input)
    print("\n[INFO] Baseline training complete.")


if __name__ == "__main__":
    main()
