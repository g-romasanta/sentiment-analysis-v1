"""
Teaching Effectiveness Assessment Tool
Phase 2 — User-Facing GUI

Run: streamlit run app.py
"""

import os
import io
import re
import string
import pickle
import sqlite3
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import nltk

from langdetect import detect, LangDetectException
from deep_translator import GoogleTranslator
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "..", "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "sentiment_model.sav")
VECTOR_PATH = os.path.join(MODELS_DIR, "tfidfvectorizer.sav")
DB_PATH     = os.path.join(BASE_DIR, "assessment.db")
SAMPLE_FEEDBACK_PATH = os.path.join(BASE_DIR, "sample_data", "Sample_Feedback.xlsx")
SAMPLE_PERF_PATH     = os.path.join(BASE_DIR, "sample_data", "Sample_Performance.xlsx")

STOP_WORDS = set(stopwords.words("english"))

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Teaching Effectiveness Assessment Tool",
    page_icon="🎓",
    layout="wide",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2rem; font-weight: 700; color: #1a1a2e; }
    .sub-title  { font-size: 0.95rem; color: #666; margin-bottom: 1.5rem; }
    .section-label {
        font-size: 0.78rem; font-weight: 700; color: #4361ee;
        text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 0.2rem;
    }
    .optional-tag {
        display: inline-block; background: #e8f4fd; color: #1a6fa8;
        border-radius: 12px; padding: 2px 10px; font-size: 0.75rem;
        font-weight: 600; margin-left: 8px; vertical-align: middle;
    }
    .required-tag {
        display: inline-block; background: #fde8e8; color: #a81a1a;
        border-radius: 12px; padding: 2px 10px; font-size: 0.75rem;
        font-weight: 600; margin-left: 8px; vertical-align: middle;
    }
    .result-card {
        background: #f8f9fa; border-radius: 10px;
        padding: 1.2rem; margin-bottom: 0.5rem;
        border-left: 5px solid #4361ee;
    }
    .badge { display:inline-block; padding:4px 14px; border-radius:20px;
             font-weight:700; font-size:0.9rem; }
    .b5 { background:#d4edda; color:#155724; }
    .b4 { background:#cce5ff; color:#004085; }
    .b3 { background:#fff3cd; color:#856404; }
    .b2 { background:#fde8d8; color:#7d3c07; }
    .b1 { background:#f8d7da; color:#721c24; }
    .note-box {
        background: #fffbea; border: 1px solid #f0d060;
        border-radius: 8px; padding: 0.8rem 1rem;
        font-size: 0.85rem; color: #6b5900;
    }
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTOR_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

# ── Helpers ───────────────────────────────────────────────────────────────────
def translate_text(text: str) -> str:
    try:
        if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return ""
        if detect(text) == "tl":
            return GoogleTranslator(source="tl", target="en").translate(text)
        return text
    except (LangDetectException, Exception):
        return str(text) if text else ""


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation + "*_"), "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    text = re.sub(r"[\u2018\u2019\u201c\u201d\u2026]", "", text)
    tokens = word_tokenize(text)
    tokens = [t.translate(str.maketrans("", "", string.punctuation))
              for t in tokens if t not in STOP_WORDS]
    return " ".join(tokens)


def get_rating(score: float) -> int:
    if score <= 1.8:   return 1
    elif score <= 2.6: return 2
    elif score <= 3.4: return 3
    elif score <= 4.2: return 4
    return 5


def get_descriptive(rating: int) -> str:
    return {1:"Poor", 2:"Fair", 3:"Satisfactory",
            4:"Very Satisfactory", 5:"Outstanding"}.get(rating, "N/A")


def badge_class(rating: int) -> str:
    return {1:"b1", 2:"b2", 3:"b3", 4:"b4", 5:"b5"}.get(rating, "b3")


def read_file(uploaded) -> pd.DataFrame:
    if uploaded.name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def save_to_db(df: pd.DataFrame, table: str):
    conn = sqlite3.connect(DB_PATH)
    df.to_sql(table, conn, if_exists="replace", index=False)
    conn.close()


def create_sample_files():
    """Create sample files if they don't exist yet."""
    os.makedirs(os.path.join(BASE_DIR, "sample_data"), exist_ok=True)

    if not os.path.exists(SAMPLE_FEEDBACK_PATH):
        pd.DataFrame({
            "Teacher No.": [1,1,1,2,2,2,3,3,3],
            "Feedback": [
                "mabait po siya at laging handang tumulong",
                "she explains topics clearly",
                "very approachable and kind",
                "magaling magturo",
                "she is always prepared",
                "she gives clear instructions",
                "sometimes hard to follow",
                "she is often absent",
                "needs to slow down during discussions",
            ]
        }).to_excel(SAMPLE_FEEDBACK_PATH, index=False)

    if not os.path.exists(SAMPLE_PERF_PATH):
        pd.DataFrame({
            "Teacher No.": [1, 2, 3],
            "Performance Evaluation Rating": [4.05, 4.80, 3.38]
        }).to_excel(SAMPLE_PERF_PATH, index=False)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🎓 Teaching Effectiveness Assessment Tool</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload student feedback to generate teacher effectiveness ratings automatically.</p>', unsafe_allow_html=True)
st.divider()

# ── Model check ───────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTOR_PATH):
    st.error("⚠️ Model files not found. Please run Phase 1 training first: `python phase1/train.py`")
    st.stop()

model, vectorizer = load_model()
create_sample_files()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    enable_translation = st.toggle(
        "Translate Tagalog → English", value=True,
        help="Turn off if your feedback data is already in English"
    )
    st.divider()

    st.markdown("**Sample files**")
    with open(SAMPLE_FEEDBACK_PATH, "rb") as f:
        st.download_button("📥 Sample Feedback File", data=f.read(),
            file_name="Sample_Feedback.xlsx", use_container_width=True,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with open(SAMPLE_PERF_PATH, "rb") as f:
        st.download_button("📥 Sample Performance File", data=f.read(),
            file_name="Sample_Performance.xlsx", use_container_width=True,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    st.divider()
    st.markdown("**How scoring works**")
    st.caption(
        "If you provide a performance evaluation file, "
        "the final rating combines your sentiment score and "
        "performance rating equally (50/50).\n\n"
        "Without it, the sentiment score alone determines the rating."
    )

# ── Step 1 — Feedback upload (required) ──────────────────────────────────────
st.markdown('<p class="section-label">Step 1 — Student Feedback <span class="required-tag">Required</span></p>', unsafe_allow_html=True)
st.caption("Upload a CSV or Excel file with columns: `Teacher No.` and `Feedback`")

feedback_file = st.file_uploader(
    "Feedback file", type=["xlsx", "xls", "csv"],
    key="feedback", label_visibility="collapsed"
)

df_feedback = None
if feedback_file:
    try:
        df_feedback = read_file(feedback_file)
        missing = {"Teacher No.", "Feedback"} - set(df_feedback.columns)
        if missing:
            st.error(f"Missing column(s): {', '.join(missing)}")
            df_feedback = None
        else:
            st.success(f"✅ {len(df_feedback)} rows · {df_feedback['Teacher No.'].nunique()} teachers")
            with st.expander("Preview", expanded=False):
                st.dataframe(df_feedback.head(8), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read file: {e}")

st.divider()

# ── Step 2 — Performance evaluation upload (optional) ────────────────────────
st.markdown('<p class="section-label">Step 2 — Performance Evaluation Ratings <span class="optional-tag">Optional</span></p>', unsafe_allow_html=True)
st.caption("Upload a CSV or Excel file with columns: `Teacher No.` and `Performance Evaluation Rating`")

perf_file = st.file_uploader(
    "Performance file", type=["xlsx", "xls", "csv"],
    key="performance", label_visibility="collapsed"
)

df_perf = None
if perf_file:
    try:
        df_perf = read_file(perf_file)
        missing = {"Teacher No.", "Performance Evaluation Rating"} - set(df_perf.columns)
        if missing:
            st.error(f"Missing column(s): {', '.join(missing)}")
            df_perf = None
        else:
            st.success(f"✅ {len(df_perf)} teachers with performance ratings")
            with st.expander("Preview", expanded=False):
                st.dataframe(df_perf.head(8), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read file: {e}")

if not perf_file:
    st.markdown(
        '<div class="note-box">ℹ️ No performance evaluation file uploaded — '
        'ratings will be based on sentiment scores only.</div>',
        unsafe_allow_html=True
    )

st.divider()

# ── Run ───────────────────────────────────────────────────────────────────────
run_disabled = df_feedback is None
if st.button("▶️ Run Assessment", type="primary",
             use_container_width=True, disabled=run_disabled):

    df = df_feedback.copy()
    df["Feedback"] = df["Feedback"].fillna("").astype(str)
    progress = st.progress(0, text="Starting...")

    # Save raw input
    save_to_db(df, "raw_input")
    progress.progress(10, text="Raw data saved...")

    # Translation
    if enable_translation:
        progress.progress(20, text="Translating Tagalog feedback...")
        df["Feedback"] = df["Feedback"].apply(translate_text)
        save_to_db(df, "translated_input")

    # Cleaning
    progress.progress(40, text="Cleaning text...")
    df["_cleaned"] = df["Feedback"].apply(clean_text)
    save_to_db(df[["Teacher No.", "_cleaned"]], "cleaned_input")

    # MNB inference
    progress.progress(60, text="Running sentiment analysis...")
    vectors = vectorizer.transform(df["_cleaned"].tolist())
    probs   = model.predict_proba(vectors)
    df["_sentiment_score"]   = [float(p[1]) for p in probs]
    df["_sentiment_score_5"] = df["_sentiment_score"] * 5

    # Aggregate per teacher
    progress.progress(80, text="Computing ratings...")
    sentiment_agg = df.groupby("Teacher No.").agg(
        sentiment_mean   =("_sentiment_score",   "mean"),
        sentiment_mean_5 =("_sentiment_score_5", "mean"),
    ).reset_index()

    # Merge performance ratings if provided
    if df_perf is not None:
        df_perf_clean = df_perf.copy()
        df_perf_clean["Performance Evaluation Rating"] = pd.to_numeric(
            df_perf_clean["Performance Evaluation Rating"], errors="coerce"
        )
        merged = sentiment_agg.merge(df_perf_clean, on="Teacher No.", how="left")
        has_perf = True
    else:
        merged = sentiment_agg.copy()
        merged["Performance Evaluation Rating"] = None
        has_perf = False

    # Compute final score
    results = []
    for _, row in merged.iterrows():
        sent_5 = row["sentiment_mean_5"]
        perf   = row["Performance Evaluation Rating"]

        if has_perf and pd.notna(perf):
            final = (sent_5 + float(perf)) / 2
            method = "Sentiment + Performance (50/50)"
        else:
            final  = sent_5
            method = "Sentiment only"

        rating = get_rating(final)
        results.append({
            "Teacher No."       : row["Teacher No."],
            "Numerical Rating"  : rating,
            "Descriptive Rating": get_descriptive(rating),
            "_method"           : method,
            "_final_score"      : round(final, 4),
        })

    results_df = pd.DataFrame(results)
    save_to_db(results_df.drop(columns=["_method","_final_score"]), "results")
    progress.progress(100, text="Done!")
    progress.empty()

    # ── Display ───────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Results")

    # Summary metrics
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Teachers",    len(results_df))
    c2.metric("Outstanding",       (results_df["Numerical Rating"] == 5).sum())
    c3.metric("Very Satisfactory", (results_df["Numerical Rating"] == 4).sum())
    c4.metric("Satisfactory",      (results_df["Numerical Rating"] == 3).sum())
    c5.metric("Fair",              (results_df["Numerical Rating"] == 2).sum())
    c6.metric("Poor",              (results_df["Numerical Rating"] == 1).sum())

    st.divider()

    # Clean results table
    display_df = results_df[["Teacher No.", "Numerical Rating", "Descriptive Rating"]].copy()
    display_df["Rating"] = display_df.apply(
        lambda r: f"{r['Numerical Rating']} – {r['Descriptive Rating']}", axis=1
    )
    st.dataframe(
        display_df[["Teacher No.", "Rating"]],
        use_container_width=True,
        hide_index=True,
    )

    # Rating cards
    st.divider()
    cols = st.columns(4)
    for i, row in results_df.iterrows():
        cls = badge_class(row["Numerical Rating"])
        with cols[i % 4]:
            st.markdown(
                f"**Teacher {row['Teacher No.']}**<br>"
                f'<span class="badge {cls}">{row["Numerical Rating"]} – {row["Descriptive Rating"]}</span><br>'
                f'<small style="color:#888">{row["_method"]}</small>',
                unsafe_allow_html=True,
            )
            st.write("")

    # ── Export ────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📥 Download Results")

    export_df = results_df[["Teacher No.", "Numerical Rating", "Descriptive Rating"]]

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download as CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="assessment_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        buf = io.BytesIO()
        export_df.to_excel(buf, index=False, sheet_name="Results")
        buf.seek(0)
        st.download_button(
            "⬇️ Download as Excel",
            data=buf.read(),
            file_name="assessment_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
