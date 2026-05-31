# Data Directory

## Raw Data (`data/raw/`)

This directory holds the original AfriSenti Swahili dataset files.

**These files are not tracked by Git** (see `.gitignore`).

### How to Obtain the Data

1. Visit the official AfriSenti repository:
   ```
   https://github.com/afrisenti-semeval/afrisent-semeval-2023
   ```

2. Navigate to the `data/` directory and download the Swahili (`swa`) split:
   ```
   train.tsv
   dev.tsv
   test.tsv
   ```

3. Place the files here:
   ```
   data/raw/train.tsv
   data/raw/dev.tsv
   data/raw/test.tsv
   ```

### File Format

Each `.tsv` file is tab-separated with two columns:

| Column | Description |
|--------|-------------|
| `tweet` | Raw tweet text in Swahili |
| `label` | Sentiment label: `positive`, `neutral`, or `negative` |

### Dataset Statistics (Swahili subset)

| Split | Negative | Neutral | Positive | Total |
|-------|----------|---------|----------|-------|
| Train | 1,028 | 924 | 1,142 | 3,094 |
| Dev   | 298   | 267 | 341   | 906   |
| Test  | 148   | 134 | 171   | 453   |

### Citation

```bibtex
@inproceedings{muhammad-etal-2023-afrisenti,
    title     = "{A}fri{S}enti: A {T}witter Sentiment Analysis Benchmark for {A}frican Languages",
    author    = "Muhammad, Shamsuddeen and others",
    booktitle = "Proceedings of SemEval-2023",
    year      = "2023",
}
```
