#  BloomLens

**BloomLens** is a web app that reads exam questions and automatically figures out which level of *Bloom's Taxonomy* each question belongs to — Remember, Understand, Apply, Analyze, Evaluate, or Create. It also runs sentiment analysis, discovers topic clusters, translates questions to Hindi, and can generate or rewrite questions using an AI language model.

Everything runs in a single Streamlit interface with four tabs, one for each course outcome (CO1–CO4).

---

## What is Bloom's Taxonomy?

Bloom's Taxonomy is a framework that sorts exam questions by the thinking skill they require, from basic recall at the bottom to creative design at the top.

| Level | What the student does | Example action verbs |
|---|---|---|
| **Remember** | Recalls facts | define, list, identify, name |
| **Understand** | Explains in own words | explain, describe, summarize, compare |
| **Apply** | Uses knowledge to solve | apply, solve, implement, calculate |
| **Analyze** | Breaks down and examines | analyze, differentiate, examine |
| **Evaluate** | Makes judgments | evaluate, justify, critique, assess |
| **Create** | Builds something new | design, formulate, construct, develop |

---

## What the App Does

1. **Paste text or upload a PDF** — the app extracts individual questions automatically.
2. **Classifies each question** into one of the six Bloom levels using verb-rule matching, a TF-IDF + SVM model, or an LLM.
3. **Runs sentiment analysis** (VADER) to label each question as Positive, Negative, or Neutral and shows the polarity score.
4. **Extracts action verbs** that map to Bloom levels.
5. **Finds topic clusters** using LDA (Latent Dirichlet Allocation) and shows a topic–Bloom heatmap.
6. **Translates questions to Hindi** using a local Helsinki-NLP model or the Groq Cloud API.
7. **Generates new exam questions** at any Bloom level for any topic.
8. **Rewrites ("levels up") existing questions** to a higher Bloom level.
9. **Compares prompt strategies** (Zero-Shot vs Few-Shot vs Chain-of-Thought) on LLM classification.

---

## Model Accuracy

| Model | Macro-F1 | Speed |
|---|---|---|
| **Verb Lexicon Rules** | **0.9687** | Instant |
| TF-IDF + Linear SVM | 0.9056 | Fast |
| SBERT + Logistic Regression | 0.8498 | Medium |
| DistilBERT (fine-tuned) | 0.6567 | Slow (GPU recommended) |
| VADER Sentiment | 0.7435 | Instant |
| XLM-RoBERTa (Hindi) | ~0.73 | Medium |

The simple verb-rule classifier achieves the highest accuracy (0.97) because exam questions almost always start with a Bloom action verb.

---

## Project Structure

```
BloomLens/
│
├── app.py                        # Main Streamlit app — 4 tabs (CO1–CO4)
├── requirements.txt              # Python dependencies
├── .env                          # Your API keys go here
│
├── src/
│   ├── preprocess.py             # Extracts questions from text/PDF, cleans, TF-IDF
│   ├── classify.py               # Verb-rule classifier + TF-IDF SVM classifier
│   ├── sentiment_tools.py        # VADER sentiment analysis + verb extraction
│   ├── topics_multilingual.py    # LDA topic modeling + Hindi translation
│   └── llm_tools.py              # Groq API: generate, classify, level-up, explain
│
├── data/
│   └── annotations/
│       └── gold_extraction_100.csv   # Hand-labelled evaluation set (100 questions)
│
├── models/
│   └── checkpoints/bloom/        # Saved fine-tuned model weights (auto-generated)
│
├── reports/
│   ├── figures/                  # Charts: confusion matrices, word clouds, heatmaps
│   └── tables/                   # CSV results: model comparison, translations, LLM experiments
│
└── .cache/
    ├── llm/                      # Cached LLM API responses (saves API calls)
    └── translations/             # Cached translation results
```

---

## App Tabs

### Tab 1 — Text Extraction & Processing 

Paste an exam paper or upload a PDF. The app:
- Splits the text into individual questions using regex rules (handles numbered lists, sub-parts like `(a)`, `(b)`, mark hints like `[5 marks]`)
- Cleans each question (lowercase, removes punctuation, keeps Bloom verbs)
- Shows the top TF-IDF terms per question
- Shows a bar chart of Bloom level distribution based on verb rules

### Tab 2 — Sentiment Analysis & Information Extraction 

Enter questions (or they auto-populate from Tab 1). The app:
- Classifies each question into a Bloom level
- Runs VADER sentiment analysis (Positive / Negative / Neutral + polarity score)
- Extracts Bloom action verbs from each question
- Shows a side-by-side model accuracy comparison table

### Tab 3 — Topic Modeling & Multilingual NLP 

**LDA Topic Modeling sub-tab:**
- Train a Gensim LDA model on your questions and explore the discovered topic clusters
- Each topic is shown as a set of keyword chips
- Optionally view a saved topic–Bloom heatmap and coverage gap report

**Multilingual (Hindi) sub-tab:**
- Translate English questions to Hindi
- Choose between Groq Cloud API (fast, requires key) or local Helsinki-NLP `opus-mt-en-hi` model
- Run a live side-by-side cloud vs local latency benchmark

### Tab 4 — Generative AI & Prompt Engineering 

Four sub-tabs:

| Sub-tab | What it does |
|---|---|
| **Generate Questions** | Give a topic + Bloom level → LLM writes exam questions; verb classifier cross-checks them |
| **Level-Up** | Paste any question → LLM rewrites it at a higher Bloom level |
| **LLM Classify** | Classify a question using Zero-Shot, Few-Shot (3 examples), or Chain-of-Thought prompting, with explanation |
| **Prompt Comparison** | Run all three prompt strategies on a question sample and compare Macro-F1 scores |

---

## Setup

### Prerequisites

- Python 3.10 or later
- pip
- A free Groq API key for the LLM and translation features (Tabs 3 and 4)

### Step 1 — Clone or download the project

```bash
git clone <your-repo-url>
cd BloomLens
```

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

Then download the spaCy English model and NLTK data:

```bash
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('punkt')"
```

### Step 3 — Set your API key

Copy the example env file and fill in your Groq API key:

```bash
# Linux / macOS
cp .env.example .env

# Windows
copy .env.example .env
```

Then open `.env` and replace the placeholder:

```
GROQ_API_KEY=gsk_your_key_here
LLM_PROVIDER=groq
DEMO_MODE=false
```

Get a free key at [console.groq.com](https://console.groq.com). Tabs 1 and 2 work without it; Tabs 3 (cloud translation) and 4 (generative AI) need it.

> Set `DEMO_MODE=true` to run the app fully offline using cached responses — no API key required.

### Step 4 (optional) — Tesseract OCR for scanned PDFs

Only needed if you want to extract text from scanned (image-based) PDFs.

```bash
# Windows: download installer from https://github.com/UB-Mannheim/tesseract/wiki
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr tesseract-ocr-hin
```

### Step 5 — Run the app

```bash
streamlit run app.py
```

The app opens at [http://localhost:8501](http://localhost:8501).

---

## How the Classifier Works

The default classifier (Verb Lexicon Rules) is very simple and very accurate:

1. Take the first word of the question (e.g. `"Define the term recursion"` → first word is `define`)
2. Look it up in a hand-curated dictionary of ~150 Bloom action verbs
3. Return the mapped level (`define` → `Remember`)
4. If the first word is not a verb, scan the full question for any Bloom verb and return the highest-priority level found

This achieves 0.9687 Macro-F1 because exam questions nearly always start with an action verb.

The TF-IDF SVM classifier trains a Linear SVM on bigram TF-IDF features from `reports/tables/processed_questions.csv` when that file is present, achieving 0.9056 Macro-F1.

---

## LLM Integration

All LLM calls go through `src/llm_tools.py` and use the Groq API with:
- **Model cascade**: tries `qwen/qwen3.8-27b` first, falls back to `openai/gpt-oss-120b` and `openai/gpt-oss-20b`
- **Local cache**: responses are saved in `.cache/llm/` keyed by MD5 of the prompt, so repeated identical calls are free
- **Rate limit handling**: automatically retries with a short wait on 429 errors

Three prompt strategies are implemented:

| Strategy | Description |
|---|---|
| Zero-Shot | "Classify this question. Reply with ONLY the level name." |
| Few-Shot (3) | Provides 3 labelled examples before asking for classification |
| Chain-of-Thought | Asks the model to reason step-by-step (verb → skill → level) |

---

## Reports and Output Files

After running the app or any scripts, results are saved here:

| File | What it contains |
|---|---|
| `reports/tables/processed_questions.csv` | Full dataset with questions, labels, source |
| `reports/tables/model_comparison.csv` | Macro-F1, QWK, and speed for all models |
| `reports/tables/multilingual_results.csv` | XLM-RoBERTa Hindi classification results |
| `reports/tables/cloud_vs_local.csv` | Cloud vs local translation latency benchmark |
| `reports/tables/llm_experiments.csv` | Prompt strategy comparison (Zero-Shot / Few-Shot / CoT) |
| `reports/tables/generated_questions.csv` | LLM-generated exam questions |
| `reports/tables/gap_report.md` | Bloom level gap analysis per topic |
| `reports/figures/finetuned_confusion.png` | DistilBERT confusion matrix |
| `reports/figures/topic_bloom_heatmap.png` | Topic–Bloom level heatmap |
| `reports/figures/wordclouds.png` | Word clouds per Bloom level |
| `reports/figures/top_verbs_per_level.png` | Most common action verbs per level |
| `reports/figures/sentiment_confusion_matrix.png` | Sentiment model confusion matrix |
| `reports/figures/vectorization_comparison.png` | TF-IDF vs SBERT vectorization comparison |

---

## Limitations

- The verb-rule classifier is accurate but only works when questions start with an action verb. Topic headings or vague phrasing can cause mis-classification.
- Hindi and multilingual translations are machine-generated and may contain errors, especially for technical terms.
- LLM features (Tabs 3 and 4) require a valid Groq API key; without one these tabs show an error.
- Fine-tuning DistilBERT on the full dataset requires a GPU for practical training time.
- Accuracy figures are based on synthetic and auto-generated data; performance on real exam papers may differ.

## Ethical Notes

- No student data is collected or stored anywhere.
- All datasets are either synthetic or public domain.
- Pre-trained model biases may affect classification accuracy across different subjects and languages.
