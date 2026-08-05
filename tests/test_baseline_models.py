"""
tests/test_baseline_models.py
------------------------------
Smoke tests for src/baseline_models.py

Run:  pytest tests/
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.baseline_models import make_vectorizer, evaluate, LABEL_NAMES


class TestVectorizer:
    def test_creates_tfidf_vectorizer(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        v = make_vectorizer()
        assert isinstance(v, TfidfVectorizer)

    def test_fits_and_transforms(self):
        v = make_vectorizer(max_features=100)
        texts = ["hello world", "swahili text here", "another sentence"]
        X = v.fit_transform(texts)
        assert X.shape[0] == 3
        assert X.shape[1] <= 100

    def test_transform_test_set(self):
        v = make_vectorizer(max_features=50)
        train = ["hello world habari", "swahili leo ni siku"]
        test  = ["hello leo unknown_word"]
        v.fit_transform(train)
        X_test = v.transform(test)
        assert X_test.shape[0] == 1


class TestEvaluate:
    def test_returns_dict_with_required_keys(self):
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 2, 0, 2, 1]
        result = evaluate("Test Model", y_true, y_pred)
        for key in ["Model", "Accuracy", "F1 Score (W)", "Precision (W)", "Recall (W)"]:
            assert key in result

    def test_perfect_predictions(self):
        y = [0, 1, 2, 0, 1, 2]
        result = evaluate("Perfect", y, y)
        assert result["Accuracy"] == 1.0
        assert result["F1 Score (W)"] == 1.0

    def test_model_name_preserved(self):
        y = [0, 1, 2]
        result = evaluate("My Classifier", y, y)
        assert result["Model"] == "My Classifier"

    def test_metrics_in_valid_range(self):
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 1, 1, 2, 2, 0]
        result = evaluate("Range Test", y_true, y_pred)
        for key in ["Accuracy", "F1 Score (W)", "Precision (W)", "Recall (W)"]:
            assert 0.0 <= result[key] <= 1.0
