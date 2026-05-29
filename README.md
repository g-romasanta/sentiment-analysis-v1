# Teaching Effectiveness Assessment Tool v1

Modernized rebuild of the BSU ECE Capstone 2023 project.
*Design and Development of a Teaching Effectiveness Assessment Tool with Sentiment Analysis*
— Romasanta et al.

## Project Structure

```
sentiment-v1/
├── phase1/
│   ├── train.py              # Training pipeline — run once as a script
│   ├── train.ipynb           # Same pipeline — step-by-step notebook
│   └── data/
│       └── MUD_Card_Data.tsv # Training dataset (MUD Card feedback)
├── phase2/
│   ├── app.py                # Streamlit GUI — user-facing assessment tool
│   └── sample_data/
│       └── Input.xlsx        # Sample input file
├── models/                   # Shared — Phase 1 writes, Phase 2 reads
│   ├── sentiment_model.sav   # Trained MNB classifier
│   └── tfidfvectorizer.sav   # Fitted TF-IDF vectorizer
├── requirements.txt
└── README.md
```

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the GUI (pre-trained model included)
```bash
streamlit run phase2/app.py
```
The model is already trained and included — the GUI works immediately.

### Retrain the model (optional)
```bash
# As a script
python phase1/train.py

# Or open the notebook
jupyter notebook phase1/train.ipynb
```

## Phase 1 — Training Pipeline

Runs on the MUD Card TSV dataset (student feedback collected via Most Useful / Difficult Cards).

| Step | What happens | Stored in SQLite |
|------|-------------|-----------------|
| 1 | Load MUD Card TSV | `raw_mudcard` |
| 2 | Store raw data | `raw_mudcard` |
| 3 | Translate Tagalog → English | `translated_mudcard` |
| 4 | Label comments (positive/negative) | `labeled_comments` |
| 5 | Clean text | `cleaned_comments` |
| 6 | TF-IDF vectorization + train/test split | — |
| 7 | Train Multinomial Naive Bayes | — |
| 8 | Evaluate (accuracy + confusion matrix) | — |
| 9 | Save model + vectorizer as `.sav` files | — |

## Phase 2 — GUI

Clean Streamlit web app. Upload any CSV or Excel with these columns:

| Column | Description |
|--------|-------------|
| `Teacher No.` | Teacher identifier |
| `Feedback` | Student comment (Tagalog or English) |
| `Performance Evaluation Rating` | Existing numerical rating (1–5) |

Each uploaded file is stored in SQLite at each pipeline stage. Final results are downloadable as CSV or Excel.

## What Changed from the Original (2023)

| Issue | Original | Modernized |
|-------|----------|------------|
| Translation | `googletrans==4.0.0-rc1` (broken) | `deep-translator` (stable) |
| Environment | Google Colab + Drive paths | Fully local, any machine |
| UI | None | Streamlit GUI |
| Inference bug | `vectorizer.fit_transform()` on new data | `vectorizer.transform()` (correct) |
| Module structure | Code runs on import | Clean modular functions |
| `requirements.txt` | 3 packages (incomplete) | All dependencies pinned |
| File paths | Hardcoded Drive paths | Relative paths |
| Storage | No persistence | SQLite at every stage |

## Original Capstone

Alcantara, D.G.P., Diezmos, K.D.A., Romasanta, G.A.I., Verano, N.B. (2023).
*Design and Development of a Teaching Effectiveness Assessment Tool with Sentiment Analysis.*
BS Electronics Engineering Capstone Project, Batangas State University – Alangilan.
