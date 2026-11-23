"""
src/topics_multilingual.py — LDA topic modeling + Hindi translation (CO3)
Translation: Groq Cloud API (primary) with Helsinki-NLP local model as fallback.
"""
from __future__ import annotations
import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(".cache/translations")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── LDA Topic Modeling ─────────────────────────────────────────────────────

def _preprocess_lda(texts: list[str]) -> list[list[str]]:
    import nltk
    for pkg in ["stopwords", "wordnet", "punkt"]:
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    stop = set(stopwords.words("english"))
    lem  = WordNetLemmatizer()
    result = []
    for text in texts:
        text = re.sub(r"[^a-z\s]", " ", text.lower())
        tokens = [lem.lemmatize(t) for t in text.split()
                  if len(t) >= 3 and t.isalpha() and t not in stop]
        result.append(tokens or ["question"])
    return result


def run_lda_model(texts: list[str], n_topics: int = 5) -> dict:
    """Train LDA on texts. Returns {n_topics, topics: [(id, words_str), ...]}"""
    try:
        import gensim
        from gensim import corpora
        from gensim.models import LdaModel
    except ImportError:
        return {"n_topics": 0, "topics": [], "error": "gensim not installed"}

    tokenized  = _preprocess_lda(texts)
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=1, no_above=0.9)
    corpus = [dictionary.doc2bow(doc) for doc in tokenized]

    lda = LdaModel(
        corpus=corpus, id2word=dictionary,
        num_topics=n_topics, random_state=42,
        passes=15, alpha="auto",
    )

    topics = []
    for i in range(n_topics):
        words = lda.print_topic(i, topn=8)
        topics.append((i, words))

    return {"n_topics": n_topics, "topics": topics, "model": lda, "corpus": corpus, "dictionary": dictionary}


# ── Cloud Translation via Groq API ─────────────────────────────────────────

def translate_via_groq(text: str) -> tuple[str, float]:
    """
    Translate English → Hindi using Groq Cloud API (LLM-based).
    Returns (translated_text, latency_seconds).
    """
    import os, hashlib, json
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")

    # Cache cloud translations too
    key = hashlib.md5(f"groq-en-hi:{text}".encode()).hexdigest()
    cf  = CACHE_DIR / f"{key}.json"
    if cf.exists():
        cached = json.loads(cf.read_text())
        return cached["translation"], cached["latency"]

    prompt = (
        "Translate the following English sentence into Hindi. "
        "Reply with ONLY the Hindi translation, no explanation, no romanization:\n\n"
        f"{text}"
    )

    client = Groq(api_key=api_key)
    t0 = time.time()
    # Use only models confirmed available on this Groq account
    for model in ["qwen/qwen3.8-27b", "openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )
            latency = round(time.time() - t0, 3)
            result  = resp.choices[0].message.content.strip()
            cf.write_text(json.dumps({"translation": result, "latency": latency}))
            logger.info("Groq translation OK with model %s (%.2fs)", model, latency)
            return result, latency
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                time.sleep(1)
                continue
            if "404" in err or "model" in err.lower():
                continue
            raise RuntimeError(f"Groq translation error: {err}")

    raise RuntimeError("All Groq models failed for translation.")


# ── Local Translation via Helsinki-NLP ─────────────────────────────────────

_trans_model = None


def _load_local_translation_model():
    global _trans_model
    if _trans_model is None:
        try:
            from transformers import MarianMTModel, MarianTokenizer
            name  = "Helsinki-NLP/opus-mt-en-hi"
            tok   = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name)
            _trans_model = (tok, model)
            logger.info("Helsinki-NLP translation model loaded.")
        except Exception as e:
            logger.warning("Local translation model load failed: %s", e)
            _trans_model = None
    return _trans_model


def translate_via_local(text: str) -> tuple[str, float]:
    """
    Translate English → Hindi using local Helsinki-NLP MarianMT model.
    Returns (translated_text, latency_seconds).
    """
    import hashlib, json, torch
    key = hashlib.md5(f"en-hi:{text}".encode()).hexdigest()
    cf  = CACHE_DIR / f"{key}.json"
    if cf.exists():
        cached = json.loads(cf.read_text())
        return cached["translation"], cached.get("latency", 0.0)

    t0     = time.time()
    loaded = _load_local_translation_model()
    if loaded is None:
        raise RuntimeError("Local translation model not available.")

    tok, model = loaded
    inputs = tok([text], return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        translated = model.generate(**inputs)
    result  = tok.decode(translated[0], skip_special_tokens=True)
    latency = round(time.time() - t0, 3)
    cf.write_text(json.dumps({"translation": result, "latency": latency}))
    return result, latency


# ── Primary API: translate_to_hindi (Cloud-first, local fallback) ───────────

def translate_to_hindi(texts: list[str], provider: str = "groq") -> list[str]:
    """
    Translate a list of English texts to Hindi.
    provider = 'groq'  → Groq Cloud API (primary, satisfies CO3 cloud requirement)
    provider = 'local' → Helsinki-NLP local model
    Falls back to local if Groq fails.
    """
    results = []
    for text in texts:
        if provider == "groq":
            try:
                translation, _ = translate_via_groq(text)
                results.append(translation)
                continue
            except Exception as e:
                logger.warning("Groq translation failed, falling back to local: %s", e)
        try:
            translation, _ = translate_via_local(text)
            results.append(translation)
        except Exception as e:
            logger.warning("Local translation also failed: %s", e)
            results.append(f"[Translation unavailable] {text}")
    return results


# ── Cloud vs Local Benchmark (live) ────────────────────────────────────────

def benchmark_cloud_vs_local(texts: list[str]) -> list[dict]:
    """
    Run both Groq cloud and local Helsinki-NLP translation on each text.
    Returns list of dicts with cloud/local translation and latencies.
    """
    rows = []
    for text in texts:
        row = {"english": text}

        # Cloud (Groq)
        try:
            cloud_trans, cloud_lat = translate_via_groq(text)
            row["cloud_translation"] = cloud_trans
            row["cloud_latency_ms"]  = round(cloud_lat * 1000)
            row["cloud_provider"]    = "Groq (qwen3.8-27b)"
        except Exception as e:
            row["cloud_translation"] = f"[Error] {e}"
            row["cloud_latency_ms"]  = -1
            row["cloud_provider"]    = "Groq (failed)"

        # Local (Helsinki-NLP)
        try:
            local_trans, local_lat = translate_via_local(text)
            row["local_translation"] = local_trans
            row["local_latency_ms"]  = round(local_lat * 1000)
            row["local_provider"]    = "Helsinki-NLP (opus-mt-en-hi)"
        except Exception as e:
            row["local_translation"] = f"[Error] {e}"
            row["local_latency_ms"]  = -1
            row["local_provider"]    = "Local (failed)"

        rows.append(row)
    return rows
