"""
Model loading and inference utilities.

Kept separate from app.py so the Dash file stays UI-focused, and so
these functions are independently unit-testable (see tests/).
"""

from __future__ import annotations

import json
import os
import pickle

from src import config
from src.text_utils import clean_tweet


def load_logreg():
    """Load the TF-IDF vectorizer + Logistic Regression model, if present."""
    import joblib

    if os.path.exists(config.LOGREG_MODEL_PATH) and os.path.exists(
        config.TFIDF_VECTORIZER_PATH
    ):
        return joblib.load(config.LOGREG_MODEL_PATH), joblib.load(
            config.TFIDF_VECTORIZER_PATH
        )
    return None, None


def load_lstm():
    """Load the Keras LSTM model + tokenizer + config, if present."""
    if not (
        os.path.exists(config.LSTM_MODEL_PATH)
        and os.path.exists(config.TOKENIZER_PATH)
        and os.path.exists(config.LSTM_CONFIG_PATH)
    ):
        return None, None, None

    from tensorflow.keras.models import load_model

    model = load_model(config.LSTM_MODEL_PATH)
    with open(config.TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    with open(config.LSTM_CONFIG_PATH) as f:
        lstm_config = json.load(f)
    return model, tokenizer, lstm_config


def load_model_metadata():
    """Load models/model_metadata.json, if present. Returns {} otherwise."""
    if os.path.exists(config.MODEL_METADATA_PATH):
        with open(config.MODEL_METADATA_PATH) as f:
            return json.load(f)
    return {}


def load_error_analysis():
    """Load models/error_analysis.json (misclassified examples, confusion
    matrices, confusion pairs), if present. Returns {} otherwise."""
    if os.path.exists(config.ERROR_ANALYSIS_PATH):
        with open(config.ERROR_ANALYSIS_PATH) as f:
            return json.load(f)
    return {}


def predict_logreg(texts, model, vectorizer):
    """Return (predicted_labels, top_confidence, full_probabilities).

    full_probabilities is an (n_samples, n_classes) array in the order of
    model.classes_ (also returned) so callers can render per-class bars.
    """
    cleaned = [clean_tweet(t) for t in texts]
    X = vectorizer.transform(cleaned)
    preds = model.predict(X)
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)
    return preds, conf, proba, list(model.classes_)


def predict_lstm(texts, model, tokenizer, lstm_config):
    """Return (predicted_labels, top_confidence, full_probabilities, classes)."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    cleaned = [clean_tweet(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(cleaned)
    X = pad_sequences(
        seqs, maxlen=lstm_config["max_len"], padding="post", truncating="post"
    )
    probs = model.predict(X, verbose=0)
    idx = probs.argmax(axis=1)
    classes = lstm_config["classes"]
    preds = [classes[i] for i in idx]
    conf = probs.max(axis=1)
    return preds, conf, probs, classes
