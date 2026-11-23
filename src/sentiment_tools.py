"""
src/sentiment_tools.py — Sentiment analysis + information extraction (CO2)
VADER for fast sentiment; extracts Bloom verbs and named concepts.
"""
from __future__ import annotations
import re

_vader = None


def _get_vader():
    global _vader
    if _vader is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            _vader = SentimentIntensityAnalyzer()
        except ImportError:
            _vader = None
    return _vader


def analyse_sentiment(text: str) -> dict:
    """
    Returns {label: POSITIVE/NEGATIVE/NEUTRAL, score: compound, scores: {...}}
    Falls back to simple lexicon if VADER unavailable.
    """
    vader = _get_vader()
    if vader:
        scores = vader.polarity_scores(text)
        compound = scores["compound"]
        label = "POSITIVE" if compound >= 0.05 else ("NEGATIVE" if compound <= -0.05 else "NEUTRAL")
        return {"label": label, "score": compound, "scores": scores}
    # Simple fallback
    pos_words = {"good","great","excellent","positive","correct","understand","explain","create"}
    neg_words = {"hard","difficult","complex","fail","wrong","mistake","error"}
    words = set(re.findall(r"\b\w+\b", text.lower()))
    pos = len(words & pos_words)
    neg = len(words & neg_words)
    score = (pos - neg) / max(pos + neg, 1)
    label = "POSITIVE" if score > 0 else ("NEGATIVE" if score < 0 else "NEUTRAL")
    return {"label": label, "score": round(score, 3), "scores": {}}


def extract_verbs(text: str) -> list[str]:
    """Extract Bloom action verbs from text."""
    from src.classify import VERB_TO_LEVEL
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in words if w in VERB_TO_LEVEL]


def extract_concepts(text: str) -> list[str]:
    """Extract key noun phrases (simple heuristic: 2–3 word sequences after action verbs)."""
    text_clean = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text_clean.split()
    concepts = []
    skip_words = {"the", "a", "an", "in", "of", "to", "for", "and", "or", "how", "why", "what", "which"}
    noun_phrase = []
    for w in words:
        wl = w.lower()
        if wl not in skip_words and len(wl) > 2:
            noun_phrase.append(w)
            if len(noun_phrase) == 2:
                concepts.append(" ".join(noun_phrase))
                noun_phrase = []
        else:
            noun_phrase = []
    return list(dict.fromkeys(concepts))[:5]  # deduplicated top 5
