"""
Central configuration for the Twitter Sentiment Analysis project.

Keeping these values in one place (instead of scattered across notebooks
and app.py) means changing e.g. the vocab size or random seed only
requires editing this file, and the notebooks / dashboard / tests all
stay in sync.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

DATA_PATH = os.path.join(DATA_DIR, "tweets.csv")

LOGREG_MODEL_PATH = os.path.join(MODELS_DIR, "logreg_model.joblib")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.keras")
TOKENIZER_PATH = os.path.join(MODELS_DIR, "tokenizer.pickle")
LSTM_CONFIG_PATH = os.path.join(MODELS_DIR, "lstm_config.json")
MODEL_METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.json")
ERROR_ANALYSIS_PATH = os.path.join(MODELS_DIR, "error_analysis.json")

# ---------------------------------------------------------------------------
# Data / split
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
SENTIMENT_CLASSES = ["negative", "neutral", "positive"]

# ---------------------------------------------------------------------------
# TF-IDF / Logistic Regression
# ---------------------------------------------------------------------------
TFIDF_MAX_FEATURES = 20000
TFIDF_NGRAM_RANGE = (1, 2)
LOGREG_MAX_ITER = 1000
LOGREG_C = 1.0

# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------
MAX_VOCAB_SIZE = 20000
MAX_SEQUENCE_LENGTH = 50
EMBEDDING_DIM = 128
LSTM_UNITS = 64
LSTM_EPOCHS = 10
LSTM_BATCH_SIZE = 128
EARLY_STOPPING_PATIENCE = 2

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
SENTIMENT_COLORS = {"positive": "#16a34a", "neutral": "#64748b", "negative": "#dc2626"}
LOW_CONFIDENCE_THRESHOLD = 0.55
