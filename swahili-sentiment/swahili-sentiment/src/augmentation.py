"""
src/augmentation.py
-------------------
Simple back-translation data augmentation for Swahili training data.

Translates a fraction of training tweets Swahili → English → Swahili
and appends them as additional training samples.

Usage:
    python src/augmentation.py
    # Reads:  data/processed/cleaned_data.csv
    # Writes: data/processed/augmented_data.csv
"""

import os
import argparse
import random
import pandas as pd

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("[WARN] transformers not installed. Augmentation will not run.")


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

PROCESSED_DATA_PATH = os.path.join("data", "processed", "cleaned_data.csv")
AUGMENTED_DATA_PATH = os.path.join("data", "processed", "augmented_data.csv")

# Helsinki-NLP models for Swahili <-> English
SW_TO_EN_MODEL = "Helsinki-NLP/opus-mt-sw-en"
EN_TO_SW_MODEL = "Helsinki-NLP/opus-mt-en-sw"

RANDOM_SEED       = 42
AUGMENT_FRACTION  = 0.20   # fraction of training data to augment


# ─────────────────────────────────────────────
# Back-translation
# ─────────────────────────────────────────────

def back_translate(texts: list, sw_to_en, en_to_sw, batch_size: int = 16) -> list:
    """
    Translate a list of Swahili texts to English then back to Swahili.

    Returns a list of back-translated strings (same length as input).
    Falls back to original text on any translation error.
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i: i + batch_size]
        try:
            en_batch = [r["translation_text"] for r in sw_to_en(batch)]
            sw_batch = [r["translation_text"] for r in en_to_sw(en_batch)]
            results.extend(sw_batch)
        except Exception as e:
            print(f"[WARN] Translation error at batch {i}: {e}")
            results.extend(batch)   # fall back to originals
    return results


def quality_filter(original: list, augmented: list, min_len: int = 5) -> list:
    """
    Return only augmented texts that:
      - are not identical to the original (augmentation added something)
      - are at least min_len characters long
    """
    filtered = []
    for orig, aug in zip(original, augmented):
        if aug != orig and len(aug) >= min_len:
            filtered.append(aug)
        else:
            filtered.append(None)
    return filtered


# ─────────────────────────────────────────────
# Main augmentation routine
# ─────────────────────────────────────────────

def augment(
    processed_path: str = PROCESSED_DATA_PATH,
    augmented_path: str = AUGMENTED_DATA_PATH,
    fraction:       float = AUGMENT_FRACTION,
):
    if not TRANSFORMERS_AVAILABLE:
        print("[ERROR] transformers required. Install with: pip install transformers sentencepiece")
        return None

    random.seed(RANDOM_SEED)
    df = pd.read_csv(processed_path)
    print(f"[INFO] Loaded {len(df):,} rows.")

    n_aug = int(len(df) * fraction)
    aug_indices = random.sample(range(len(df)), n_aug)
    aug_df      = df.iloc[aug_indices].copy().reset_index(drop=True)

    print(f"[INFO] Augmenting {n_aug:,} samples ({fraction*100:.0f}% of dataset).")
    print("[INFO] Loading translation models (this may take a moment)…")

    sw_to_en = pipeline("translation", model=SW_TO_EN_MODEL, device=-1)
    en_to_sw = pipeline("translation", model=EN_TO_SW_MODEL, device=-1)

    original_texts  = aug_df["clean_text"].tolist()
    translated_back = back_translate(original_texts, sw_to_en, en_to_sw)
    filtered        = quality_filter(original_texts, translated_back)

    # Build augmented rows (skip failed translations)
    aug_rows = []
    for i, text in enumerate(filtered):
        if text is not None:
            row = aug_df.iloc[i].copy()
            row["clean_text"] = text
            row["augmented"]  = True
            aug_rows.append(row)

    n_kept = len(aug_rows)
    print(f"[INFO] {n_kept:,} / {n_aug:,} augmented texts passed quality filter "
          f"({100*n_kept/n_aug:.1f}% kept).")

    df["augmented"] = False
    augmented_df = pd.DataFrame(aug_rows)
    combined     = pd.concat([df, augmented_df], ignore_index=True)

    os.makedirs(os.path.dirname(augmented_path), exist_ok=True)
    combined.to_csv(augmented_path, index=False)
    print(f"[INFO] Augmented dataset ({len(combined):,} rows) saved to '{augmented_path}'")
    return combined


# ─────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Back-translation augmentation for Swahili.")
    parser.add_argument("--input",    default=PROCESSED_DATA_PATH)
    parser.add_argument("--output",   default=AUGMENTED_DATA_PATH)
    parser.add_argument("--fraction", type=float, default=AUGMENT_FRACTION,
                        help="Fraction of data to augment (default 0.20)")
    return parser.parse_args()


def main():
    args = parse_args()
    augment(args.input, args.output, args.fraction)
    print("[INFO] Augmentation complete.")


if __name__ == "__main__":
    main()
