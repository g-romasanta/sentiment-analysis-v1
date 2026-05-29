"""
Teaching Effectiveness Assessment Tool
Phase 1 — Training Pipeline

Modernized from BSU ECE Capstone 2023 (Romasanta et al.)
Run: python train.py

Steps:
  1. Load MUD Card TSV dataset
  2. Store raw data in SQLite
  3. Translate Tagalog → English
  4. Label comments (positive / negative)
  5. Clean text
  6. Split and vectorize
  7. Train Multinomial Naive Bayes
  8. Evaluate (accuracy + confusion matrix)
  9. Save model and vectorizer
"""

import os
import sys
import re
import string
import sqlite3
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import nltk
import matplotlib.pyplot as plt

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE_DIR, "data", "MUD_Card_Data.tsv")
DB_PATH     = os.path.join(BASE_DIR, "data", "training.db")
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "sentiment_model.sav")
VECTOR_PATH = os.path.join(MODELS_DIR, "tfidfvectorizer.sav")

os.makedirs(MODELS_DIR, exist_ok=True)

# ── NLTK downloads ────────────────────────────────────────────────────────────
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
STOP_WORDS = set(stopwords.words("english"))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load raw MUD Card data
# ─────────────────────────────────────────────────────────────────────────────
def step1_load_raw():
    print("\n[STEP 1] Loading MUD Card dataset...")
    df = pd.read_csv(DATA_PATH, delimiter="\t")
    print(f"         Loaded {len(df)} rows — columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Store raw data in SQLite
# ─────────────────────────────────────────────────────────────────────────────
def step2_store_raw(df: pd.DataFrame):
    print("\n[STEP 2] Storing raw data in SQLite...")
    conn = sqlite3.connect(DB_PATH)
    df.to_sql("raw_mudcard", conn, if_exists="replace", index=False)
    conn.close()
    print(f"         Saved to {DB_PATH} → table: raw_mudcard")
    return DB_PATH


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Translate Tagalog → English
# ─────────────────────────────────────────────────────────────────────────────
def _translate(text: str) -> str:
    try:
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return ""
        if detect(text) == "tl":
            return GoogleTranslator(source="tl", target="en").translate(text)
        return text
    except (LangDetectException, Exception):
        return str(text) if text else ""


def step3_translate(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 3] Translating Tagalog comments to English...")
    total = len(df)
    df = df.copy()
    for col in ["Positive Comments", "Negative Comments"]:
        translated = []
        for i, text in enumerate(df[col]):
            translated.append(_translate(text))
            if (i + 1) % 500 == 0:
                print(f"         {col}: {i+1}/{total} translated...")
        df[col] = translated
    print(f"         Translation complete.")

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("translated_mudcard", conn, if_exists="replace", index=False)
    conn.close()
    print(f"         Saved → table: translated_mudcard")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Label comments
# ─────────────────────────────────────────────────────────────────────────────
def step4_label(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 4] Labeling comments (positive / negative)...")
    features, sentiments = [], []

    for text in df["Negative Comments"]:
        t = str(text).strip().lower()
        if t and t not in ("none", "n/a", "nan"):
            features.append(str(text))
            sentiments.append("negative")

    for text in df["Positive Comments"]:
        t = str(text).strip().lower()
        if t and t not in ("none", "n/a", "nan"):
            features.append(str(text))
            sentiments.append("positive")

    labeled = pd.DataFrame({"Features": features, "Sentiments": sentiments})
    neg = (labeled["Sentiments"] == "negative").sum()
    pos = (labeled["Sentiments"] == "positive").sum()
    print(f"         Total: {len(labeled)} comments — positive: {pos}, negative: {neg}")

    conn = sqlite3.connect(DB_PATH)
    labeled.to_sql("labeled_comments", conn, if_exists="replace", index=False)
    conn.close()
    print(f"         Saved → table: labeled_comments")
    return labeled


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Clean text
# ─────────────────────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation + "*_"), "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"[" "\u2018\u2019\u201c\u201d\u2026]", "", text)
    tokens = word_tokenize(text)
    tokens = [
        t.translate(str.maketrans("", "", string.punctuation))
        for t in tokens
        if t not in STOP_WORDS
    ]
    return " ".join(tokens)


def step5_clean(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[STEP 5] Cleaning text...")
    df = df.copy()
    df["Features"] = df["Features"].apply(_clean)
    df = df[df["Features"].str.strip() != ""].reset_index(drop=True)
    print(f"         Cleaned — {len(df)} rows remaining after empty removal")

    conn = sqlite3.connect(DB_PATH)
    df.to_sql("cleaned_comments", conn, if_exists="replace", index=False)
    conn.close()
    print(f"         Saved → table: cleaned_comments")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Split and vectorize
# ─────────────────────────────────────────────────────────────────────────────
def step6_split_vectorize(df: pd.DataFrame):
    print("\n[STEP 6] Splitting data and vectorizing with TF-IDF...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["Features"], df["Sentiments"], test_size=0.2, random_state=143
    )
    print(f"         Train: {len(X_train)} | Test: {len(X_test)}")

    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words=list(STOP_WORDS)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)
    print(f"         Vocabulary size: {len(vectorizer.vocabulary_)}")
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 & 8 — Train and evaluate
# ─────────────────────────────────────────────────────────────────────────────
def step7_train_evaluate(X_train_vec, X_test_vec, y_train, y_test):
    print("\n[STEP 7] Training Multinomial Naive Bayes...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    print("\n[STEP 8] Evaluating model...")

    # Training set
    train_pred = model.predict(X_train_vec)
    train_acc  = metrics.accuracy_score(y_train, train_pred)
    print(f"\n         TRAINING SET")
    print(f"         Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    print(classification_report(y_train, train_pred, digits=3))

    # Test set
    test_pred = model.predict(X_test_vec)
    test_acc  = metrics.accuracy_score(y_test, test_pred)
    print(f"         TEST SET")
    print(f"         Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(classification_report(y_test, test_pred, digits=3))

    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (preds, labels, title) in zip(
        axes,
        [
            (train_pred, y_train, "Training Confusion Matrix"),
            (test_pred,  y_test,  "Test Confusion Matrix"),
        ],
    ):
        cm = confusion_matrix(labels, preds, labels=model.classes_)
        ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot(ax=ax)
        ax.set_title(title)

    plt.tight_layout()
    plot_path = os.path.join(BASE_DIR, "data", "confusion_matrix.png")
    plt.savefig(plot_path, dpi=150)
    print(f"\n         Confusion matrix saved → {plot_path}")
    plt.show()

    return model, test_acc


# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Save model
# ─────────────────────────────────────────────────────────────────────────────
def step9_save(model, vectorizer):
    print("\n[STEP 9] Saving model and vectorizer...")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(VECTOR_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"         Model saved     → {MODEL_PATH}")
    print(f"         Vectorizer saved → {VECTOR_PATH}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Teaching Effectiveness Assessment Tool — Phase 1 Training")
    print("=" * 60)

    df_raw        = step1_load_raw()
    step2_store_raw(df_raw)
    df_translated = step3_translate(df_raw)
    df_labeled    = step4_label(df_translated)
    df_cleaned    = step5_clean(df_labeled)
    X_train, X_test, y_train, y_test, vectorizer = step6_split_vectorize(df_cleaned)
    model, acc    = step7_train_evaluate(X_train, X_test, y_train, y_test)
    step9_save(model, vectorizer)

    print("\n" + "=" * 60)
    print(f"  Training complete! Test Accuracy: {acc*100:.2f}%")
    print(f"  Models saved to: {MODELS_DIR}")
    print("=" * 60)
