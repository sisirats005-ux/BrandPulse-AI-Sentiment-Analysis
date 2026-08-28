"""
Re-evaluates both trained models on the same held-out test split used
during training (random_state=42, 80/20 stratified split) and writes:

  - models/model_metadata.json   summary stats used by the dashboard
  - models/error_analysis.json   misclassified examples + confusion data
                                  used by the "Error Analysis" tab

Run this once after (re)training the models:
    python scripts/generate_model_artifacts.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from sklearn.model_selection import train_test_split

from src import config
from src.model_utils import load_logreg, load_lstm, predict_logreg, predict_lstm

CLASSES = config.SENTIMENT_CLASSES


def load_dataset():
    df = pd.read_csv(config.DATA_PATH)
    text_col = next((c for c in ["text", "tweet", "content"] if c in df.columns), None)
    label_col = next(
        (c for c in ["sentiment", "label", "target", "airline_sentiment"] if c in df.columns), None
    )
    df = df[[text_col, label_col]].dropna().reset_index(drop=True)
    df.columns = ["text", "sentiment"]
    df["sentiment"] = df["sentiment"].astype(str).str.lower()
    df = df[df["sentiment"].isin(CLASSES)]

    # Match notebook 01's preprocessing exactly: drop rows that clean to
    # empty text, so the train_test_split(random_state=42) below lands on
    # the identical rows the models were actually trained/tested on.
    from src.text_utils import clean_tweet
    df["clean"] = df["text"].astype(str).map(clean_tweet)
    df = df[df["clean"].str.strip().astype(bool)].reset_index(drop=True)
    return df


def main():
    df = load_dataset()
    _, test_df = train_test_split(
        df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=df["sentiment"]
    )
    texts = test_df["text"].tolist()
    y_true = test_df["sentiment"].tolist()
    print(f"Test set: {len(texts)} tweets")

    metadata = {
        "project": "BrandPulse AI",
        "dataset": "Twitter US Airline Sentiment",
        "dataset_size": int(len(df)),
        "train_test_split": "80/20 (stratified)",
        "random_state": config.RANDOM_STATE,
        "class_distribution": df["sentiment"].value_counts(normalize=True).round(4).to_dict(),
        "models": {},
    }
    error_analysis = {"classes": CLASSES, "models": {}}

    # -- Logistic Regression -------------------------------------------------
    logreg, vectorizer = load_logreg()
    if logreg is not None:
        t0 = time.time()
        preds, conf, proba, classes = predict_logreg(texts, logreg, vectorizer)
        infer_time = time.time() - t0

        acc = accuracy_score(y_true, preds)
        macro_f1 = f1_score(y_true, preds, average="macro")
        weighted_f1 = f1_score(y_true, preds, average="weighted")
        per_class_f1 = dict(zip(CLASSES, f1_score(y_true, preds, labels=CLASSES, average=None)))
        cm = confusion_matrix(y_true, preds, labels=CLASSES).tolist()

        metadata["models"]["logistic_regression"] = {
            "display_name": "TF-IDF + Logistic Regression",
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class_f1": {k: round(float(v), 4) for k, v in per_class_f1.items()},
            "approx_training_time_seconds": 2,
            "approx_model_size_mb": round(
                (os.path.getsize(config.LOGREG_MODEL_PATH) + os.path.getsize(config.TFIDF_VECTORIZER_PATH))
                / 1e6, 2,
            ),
            "inference_time_seconds_for_test_set": round(infer_time, 3),
            "test_set_size": len(texts),
        }

        mis_idx = [i for i, (p, t) in enumerate(zip(preds, y_true)) if p != t]
        examples = [
            {"tweet": texts[i], "actual": y_true[i], "predicted": str(preds[i]), "confidence": round(float(conf[i]), 3)}
            for i in mis_idx
        ]
        confusion_pairs = pd.Series(
            [f"{t} -> {p}" for t, p in zip(y_true, preds) if t != p]
        ).value_counts().head(6).to_dict()

        error_analysis["models"]["logistic_regression"] = {
            "confusion_matrix": cm,
            "top_confusion_pairs": confusion_pairs,
            "misclassified_examples": examples[:60],
            "n_misclassified": len(mis_idx),
        }
        print("Logistic Regression: acc=%.4f macro_f1=%.4f" % (acc, macro_f1))
    else:
        print("Logistic Regression model not found, skipping.")

    # -- LSTM ------------------------------------------------------------
    lstm_model, tokenizer, lstm_config = load_lstm()
    if lstm_model is not None:
        t0 = time.time()
        preds, conf, proba, classes = predict_lstm(texts, lstm_model, tokenizer, lstm_config)
        infer_time = time.time() - t0

        acc = accuracy_score(y_true, preds)
        macro_f1 = f1_score(y_true, preds, average="macro")
        weighted_f1 = f1_score(y_true, preds, average="weighted")
        per_class_f1 = dict(zip(CLASSES, f1_score(y_true, preds, labels=CLASSES, average=None)))
        cm = confusion_matrix(y_true, preds, labels=CLASSES).tolist()

        metadata["models"]["bilstm"] = {
            "display_name": "Bidirectional LSTM",
            "accuracy": round(float(acc), 4),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "per_class_f1": {k: round(float(v), 4) for k, v in per_class_f1.items()},
            "approx_training_time_seconds": 100,
            "approx_model_size_mb": round(
                (os.path.getsize(config.LSTM_MODEL_PATH) + os.path.getsize(config.TOKENIZER_PATH)) / 1e6, 2
            ),
            "inference_time_seconds_for_test_set": round(infer_time, 3),
            "test_set_size": len(texts),
        }

        mis_idx = [i for i, (p, t) in enumerate(zip(preds, y_true)) if p != t]
        examples = [
            {"tweet": texts[i], "actual": y_true[i], "predicted": str(preds[i]), "confidence": round(float(conf[i]), 3)}
            for i in mis_idx
        ]
        confusion_pairs = pd.Series(
            [f"{t} -> {p}" for t, p in zip(y_true, preds) if t != p]
        ).value_counts().head(6).to_dict()

        error_analysis["models"]["bilstm"] = {
            "confusion_matrix": cm,
            "top_confusion_pairs": confusion_pairs,
            "misclassified_examples": examples[:60],
            "n_misclassified": len(mis_idx),
        }
        print("BiLSTM: acc=%.4f macro_f1=%.4f" % (acc, macro_f1))
    else:
        print("LSTM model not found, skipping.")

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    with open(config.MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(config.ERROR_ANALYSIS_PATH, "w") as f:
        json.dump(error_analysis, f, indent=2)

    print(f"\nWrote {config.MODEL_METADATA_PATH}")
    print(f"Wrote {config.ERROR_ANALYSIS_PATH}")


if __name__ == "__main__":
    main()
