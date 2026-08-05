"""
src/preprocess.py
-----------------
Text preprocessing pipeline for Swahili sentiment analysis.

Usage:
    python src/preprocess.py
    # Reads:  data/raw/dev.tsv
    # Writes: data/processed/cleaned_data.csv
"""

import re
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False
    print("[WARN] 'emoji' package not installed. Emoji characters will be removed via regex fallback.")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

LABEL_MAP = {"negative": 0, "neutral": 1, "positive": 2}

RAW_DATA_PATH      = os.path.join("data", "raw", "dev.tsv")
PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_data.csv")
FIGURES_DIR         = os.path.join("results", "figures")


# ─────────────────────────────────────────────
# Text Cleaning
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Clean a single Swahili tweet.

    Steps
    -----
    1. Lowercase
    2. Remove URLs
    3. Remove @mentions
    4. Remove #hashtag symbols (preserve word)
    5. Remove emoji
    6. Remove digits
    7. Remove non-alphabetic characters
    8. Collapse whitespace

    Parameters
    ----------
    text : str
        Raw tweet text.

    Returns
    -------
    str
        Cleaned text.
    """
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)   # keep the word, drop the #

    if EMOJI_AVAILABLE:
        text = emoji.replace_emoji(text, replace="")
    else:
        # Unicode emoji ranges fallback
        text = re.sub(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
            r"\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+",
            "", text
        )

    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────
# Dataset loading & processing
# ─────────────────────────────────────────────

def load_raw(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw TSV file into a DataFrame."""
    df = pd.read_csv(path, sep="\t")
    print(f"[INFO] Loaded {len(df):,} rows from '{path}'")
    print(f"[INFO] Columns: {df.columns.tolist()}")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to raw DataFrame.

    Returns a new DataFrame with extra columns:
      - clean_text    : cleaned tweet text
      - label_encoded : integer label (0=negative, 1=neutral, 2=positive)
    """
    df = df.copy()

    # Clean text
    df["clean_text"] = df["tweet"].apply(clean_text)

    # Encode labels
    df["label_encoded"] = df["label"].map(LABEL_MAP)

    missing = df["label_encoded"].isna().sum()
    if missing > 0:
        print(f"[WARN] {missing} rows had unrecognised labels and will be dropped.")
        df = df.dropna(subset=["label_encoded"])

    df["label_encoded"] = df["label_encoded"].astype(int)
    return df


def print_stats(df: pd.DataFrame) -> None:
    """Print dataset statistics."""
    print("\n── Dataset Statistics ──────────────────────────")
    print(f"  Total samples : {len(df):,}")
    counts = df["label"].value_counts()
    for label, n in counts.items():
        pct = 100 * n / len(df)
        print(f"  {label:<12}: {n:>5,}  ({pct:.1f}%)")
    print("────────────────────────────────────────────────\n")


def plot_distribution(df: pd.DataFrame, save_dir: str = FIGURES_DIR) -> None:
    """Save a bar chart of label distribution."""
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["negative", "neutral", "positive"]
    colors = ["#d9534f", "#f0ad4e", "#5cb85c"]
    counts = df["label"].value_counts().reindex(order)
    bars = ax.bar(order, counts.values, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(val), ha="center", va="bottom", fontsize=11)
    ax.set_title("AfriSenti Swahili – Sentiment Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sentiment Class")
    ax.set_ylabel("Number of Tweets")
    ax.spines[["top", "right"]].set_visible(False)
    out = os.path.join(save_dir, "label_distribution.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"[INFO] Distribution chart saved to '{out}'")


def save_processed(df: pd.DataFrame, path: str = PROCESSED_DATA_PATH) -> None:
    """Save cleaned DataFrame to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[INFO] Cleaned data saved to '{path}'")


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess AfriSenti Swahili data.")
    parser.add_argument("--input",  default=RAW_DATA_PATH,       help="Path to raw TSV file")
    parser.add_argument("--output", default=PROCESSED_DATA_PATH, help="Path for processed CSV")
    parser.add_argument("--no-plot", action="store_true",         help="Skip distribution chart")
    return parser.parse_args()


def main():
    args = parse_args()
    df = load_raw(args.input)
    df = preprocess(df)
    print_stats(df)
    if not args.no_plot:
        plot_distribution(df)
    save_processed(df, args.output)
    print("[INFO] Preprocessing complete.")


if __name__ == "__main__":
    main()
