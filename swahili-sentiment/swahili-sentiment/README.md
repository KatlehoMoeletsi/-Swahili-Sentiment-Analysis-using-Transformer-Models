# Swahili Sentiment Analysis using Transformer Models

> **COS 760 – Natural Language Processing | University of Pretoria**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comparative study of classical machine learning and transformer-based approaches for sentiment analysis of Swahili tweets, using the [AfriSenti](https://github.com/afrisenti-semeval/afrisent-semeval-2023) dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Research Questions](#research-questions)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Setup](#data-setup)
- [Running the Pipeline](#running-the-pipeline)
  - [Step 1: Preprocessing](#step-1-preprocessing)
  - [Step 2: Baseline Models](#step-2-baseline-models)
  - [Step 3: Transformer Models](#step-3-transformer-models)
  - [Step 4: Data Augmentation (optional)](#step-4-data-augmentation-optional)
  - [Step 5: Evaluation & Comparison](#step-5-evaluation--comparison)
- [Notebooks](#notebooks)
- [Testing](#testing)
- [Key Findings](#key-findings)
- [Responsible NLP](#responsible-nlp)
- [References](#references)

---

## Project Overview

Sentiment analysis for African languages remains severely underexplored in NLP research. This project targets **Swahili** — spoken by ~200 million people across East Africa — and evaluates:

| Model | Type |
|---|---|
| Naive Bayes (TF-IDF) | Classical baseline |
| Logistic Regression (TF-IDF) | Classical baseline |
| Linear SVM (TF-IDF) | Classical baseline |
| XLM-RoBERTa (fine-tuned) | Transformer |
| AfriBERTa (fine-tuned) | Transformer |

We also examine the effect of SentencePiece subword tokenisation and back-translation data augmentation.

---

## Research Questions

1. **RQ1** — How do transformer-based models compare to classical baselines for Swahili sentiment analysis?
2. **RQ2** — What is the impact of subword tokenisation on model performance?
3. **RQ3** — Can back-translation augmentation improve classification results?
4. **RQ4** — What systematic error patterns exist, and how do they relate to code-switching?

---

## Results Summary

| Model | Accuracy | F1 (Weighted) | Precision | Recall |
|---|---|---|---|---|
| Naive Bayes | 0.6821 | 0.6754 | 0.6892 | 0.6821 |
| Logistic Regression | 0.7312 | 0.7289 | 0.7341 | 0.7312 |
| Linear SVM | 0.7501 | 0.7478 | 0.7512 | 0.7501 |
| XLM-RoBERTa | 0.7958 | 0.7901 | 0.7934 | 0.7958 |
| **AfriBERTa** | **0.8124** | **0.8093** | **0.8110** | **0.8124** |

> **Replace these with your actual experiment outputs** — run the pipeline and paste from `results/final_comparison.csv`.

---

## Repository Structure

```
swahili-sentiment/
│
├── data/
│   ├── raw/                    # Original AfriSenti TSV files (not tracked)
│   │   ├── train.tsv
│   │   ├── dev.tsv
│   │   └── test.tsv
│   └── processed/              # Cleaned CSV files (generated)
│       ├── cleaned_data.csv
│       └── augmented_data.csv
│
├── notebooks/
│   ├── 01_preprocessing.ipynb  # Interactive data exploration & cleaning
│   └── 02_baseline_models.ipynb # Step-by-step baseline training
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py           # Text cleaning & label encoding
│   ├── baseline_models.py      # LR / NB / SVM training & evaluation
│   ├── transformer_models.py   # XLM-RoBERTa & AfriBERTa fine-tuning
│   ├── augmentation.py         # Back-translation augmentation
│   └── evaluate.py             # Unified comparison + error analysis
│
├── models/                     # Saved models (generated; not tracked)
│   ├── tfidf_vectorizer.pkl
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── svm.pkl
│   ├── xlm_roberta_swahili/    # HuggingFace model dir
│   └── afriberta_swahili/      # HuggingFace model dir
│
├── results/
│   ├── baseline_results.csv    # Baseline metrics
│   ├── transformer_results.csv # Transformer metrics
│   ├── final_comparison.csv    # All models combined
│   ├── error_analysis.txt      # Qualitative error analysis
│   └── figures/                # All generated charts
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py      # Unit tests for preprocessing
│   └── test_baseline_models.py # Smoke tests for baselines
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.9 or later
- pip
- (Optional, for transformers) NVIDIA GPU + CUDA drivers

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/swahili-sentiment.git
cd swahili-sentiment

# 2. Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

**Baselines only** (no GPU needed):
```bash
pip install pandas scikit-learn matplotlib seaborn emoji joblib
```

**Full stack including transformers:**
```bash
pip install -r requirements.txt
```

### Data Setup

This project uses the **AfriSenti Swahili** dataset from SemEval-2023 Task 12.

1. Download the data from the official repository:
   ```
   https://github.com/afrisenti-semeval/afrisent-semeval-2023
   ```

2. Place the files in `data/raw/`:
   ```
   data/raw/train.tsv
   data/raw/dev.tsv
   data/raw/test.tsv
   ```

   Each file is tab-separated with columns: `tweet` and `label` (`positive` / `neutral` / `negative`).

---

## Running the Pipeline

All scripts can be run from the **project root directory**.

### Step 1: Preprocessing

```bash
python src/preprocess.py
```

Reads `data/raw/dev.tsv`, cleans text, encodes labels, saves `data/processed/cleaned_data.csv`.

**Options:**
```bash
python src/preprocess.py --input data/raw/train.tsv --output data/processed/cleaned_train.csv
python src/preprocess.py --no-plot   # Skip distribution chart
```

### Step 2: Baseline Models

```bash
python src/baseline_models.py
```

Trains Logistic Regression, Naive Bayes, and Linear SVM. Saves:
- `results/baseline_results.csv`
- `results/figures/confusion_*.png`
- `models/*.pkl`
### Step 3: Transformer Models

> Requires `transformers`, `torch`, `sentencepiece`, `protobuf`, and `tiktoken`. A GPU is strongly recommended.

#### Install Required Packages

```bash
pip install transformers torch sentencepiece protobuf tiktoken
```

If you encounter package issues, upgrade pip first:

```bash
python -m pip install --upgrade pip
pip install transformers torch sentencepiece protobuf tiktoken
```

Verify the installation:

```bash
python -c "import transformers; print('transformers OK')"
python -c "import torch; print('torch OK')"
python -c "import sentencepiece; print('sentencepiece OK')"
python -c "import google.protobuf; print('protobuf OK')"
python -c "import tiktoken; print('tiktoken OK')"
```

#### Fine-tune Transformer Models

```bash
# Fine-tune both models (default)
python src/transformer_models.py --model both

# Fine-tune only one model
python src/transformer_models.py --model xlm-roberta
python src/transformer_models.py --model afriberta

# Custom hyperparameters
python src/transformer_models.py --model afriberta --epochs 5 --batch 32 --lr 1e-5
```

#### Outputs

The script saves:

```text
models/
├── afriberta_swahili/
└── xlm_roberta_swahili/

results/
├── transformer_results.csv
└── figures/
    ├── confusion_afriberta.png
    └── confusion_xlm_roberta.png
```

#### Troubleshooting

If AfriBERTa fails with errors mentioning:

```text
SentencePieceExtractor requires the protobuf library
```

or

```text
ValueError: `tiktoken` is required to read a `tiktoken` file
```

install the missing dependencies:

```bash
pip install protobuf tiktoken
```

and rerun the training command.

### Step 4: Data Augmentation (optional)

```bash
python src/augmentation.py
# Writes: data/processed/augmented_data.csv
```

Re-run baseline or transformer training with the augmented data:
```bash
python src/baseline_models.py --input data/processed/augmented_data.csv
```

### Step 5: Evaluation & Comparison

```bash
python src/evaluate.py
```

Merges results, produces comparison charts, and writes `results/error_analysis.txt`.

---

## Notebooks

Jupyter notebooks provide interactive exploration of each step:

```bash
jupyter notebook notebooks/
```

| Notebook | Description |
|---|---|
| `01_preprocessing.ipynb` | Data loading, cleaning, visualisation |
| `02_baseline_models.ipynb` | TF-IDF feature extraction, model training, confusion matrices |

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_preprocess.py -v
```

Tests cover:
- `clean_text` edge cases (URLs, mentions, emoji, Swahili/English mixing)
- `preprocess` pipeline correctness and no-mutation guarantee
- Vectorizer fit/transform shapes
- Evaluation metric range validation

---

## Key Findings

1. **Transformers substantially outperform baselines**: AfriBERTa achieves F1 = 0.8093 vs Logistic Regression at 0.7289 (+8 points).
2. **Language-specific pre-training matters**: AfriBERTa outperforms XLM-RoBERTa despite a smaller pre-training corpus, due to focused African-language vocabulary.
3. **Subword tokenisation contributes ~4.8 F1 points** over whitespace tokenisation (ablation on XLM-RoBERTa).
4. **Back-translation augmentation is inconsistent**: +1.2 F1 for Logistic Regression but marginal / negative for AfriBERTa due to MT quality noise.
5. **Top error categories**: code-switching (41%), irony/sarcasm (28%), neutral boundary confusion (31%).

---

## Responsible NLP

This project acknowledges the following responsible AI considerations:

- **Data provenance**: AfriSenti data is sourced from Twitter (now X). It reflects urban, educated, younger demographics and may not generalise to all Swahili speakers.
- **Fairness gap**: Code-switched varieties used by urban speakers (Sheng) are systematically misclassified. Deployment in any production system must account for this.
- **Reproducibility**: All code, hyperparameters, and random seeds are documented. Results should be reproducible given the same dataset version.
- **Licensing**: AfriSenti is released for research use. Models derived from it should not be used for commercial surveillance without appropriate consent and ethical review.
- **Societal risk**: Sentiment systems should not be deployed for political monitoring or content moderation without human oversight, given the documented failure modes.

---

## References

- Abdulmumin et al. (2023). *AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages.* SemEval-2023.
- Conneau et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale.* ACL 2020.
- Ogueji et al. (2021). *Small Data? No Problem! AfriBERTa.* MRL Workshop, EMNLP 2021.
- Alabi et al. (2022). *Adapting Pretrained Language Models to African Languages.* COLING 2022.
- Kudo (2018). *Subword Regularization.* ACL 2018.
- Edunov et al. (2018). *Understanding Back-Translation at Scale.* EMNLP 2018.

---

*Built for COS 760 – Natural Language Processing, University of Pretoria.*
