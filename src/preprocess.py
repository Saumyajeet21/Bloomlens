"""
src/preprocess.py — Text extraction, cleaning, TF-IDF vectorization (CO1)
"""
from __future__ import annotations
import re
import pandas as pd

BLOOM_VERBS = {
    "Remember":   ["define","list","recall","recite","recognize","identify","name","state","label","match","memorize","repeat","reproduce","select","locate","quote","outline","enumerate"],
    "Understand": ["explain","describe","summarize","paraphrase","classify","compare","contrast","interpret","discuss","illustrate","express","indicate","report","restate","review","translate","infer","convert","extend","generalize","predict","relate","rephrase","rewrite","show","sort"],
    "Apply":      ["apply","demonstrate","use","implement","solve","calculate","operate","execute","carry","choose","compute","construct","employ","examine","modify","practice","produce","manipulate","prepare","schedule","experiment","draw"],
    "Analyze":    ["analyze","analyse","differentiate","examine","distinguish","break","diagram","deconstruct","investigate","question","test","relate","separate","categorize","deduce","detect","inspect","subdivide","survey","appraise","criticize","debate","focus","research"],
    "Evaluate":   ["evaluate","assess","judge","justify","argue","defend","critique","appraise","conclude","decide","determine","discriminate","measure","prioritize","prove","rank","rate","recommend","support","value","check","monitor","validate","verify"],
    "Create":     ["create","design","develop","formulate","build","construct","plan","produce","invent","compose","devise","propose","arrange","assemble","collect","combine","compile","generate","hypothesize","integrate","organize","perform","revise","set","synthesize","draft","originate","initiate","make","model"],
}

# ── Regex helpers ────────────────────────────────────────────────────────────

# MAIN question label: Q1. / Q1) / 1. / 1) / (1)
_MAIN_LABEL_RE = re.compile(
    r"^(?:"
    r"Q\.?\s*\d+[\.\):\s]*|"   # Q1, Q.1, Q1.
    r"\d+[\.\)]\s+|"            # 1.  or  1)
    r"\(\s*\d+\s*\)"            # (1)
    r")",
    re.IGNORECASE,
)

# SUBPART label: (a) / (b) / (i) / (ii) / a. / a)
_SUB_LABEL_RE = re.compile(
    r"^(?:"
    r"\(\s*[a-z]\s*\)|"           # (a), (b)
    r"\(\s*[ivxlIVXL]+\s*\)|"     # (i), (ii)
    r"[a-z][\.\)]\s+"             # a.  or  a)
    r")",
    re.IGNORECASE,
)

# Lines that are definitely NOT questions (headers, instructions, metadata …)
_SKIP_RE = re.compile(
    r"(?i)^("
    r"department|university|college|institute|exam(ination)?|course|"
    r"subject|semester|date\b|time\b|duration|max\.?\s*marks?|total\s*marks?|"
    r"note:|instructions?|answer\s+(all|any)|section|part\s+[a-z]:|"
    r"reg\.?\s*no|roll\s*no|name\s*:|page\s*\d|www\.|https?:|"
    r"marks?\s*:\s*\d|co\s*\d+|unit\s+\d|module\s+\d"
    r")"
)


def _is_topic_heading(text: str) -> bool:
    """
    Return True if a line looks like a section topic heading rather than
    a real question — e.g. 'Finite Automata', 'Regular Languages 10 Marks'.
    Heuristic: short, no Bloom action verb, no '?'.
    """
    if text.endswith("?"):
        return False
    words = text.lower().split()
    if len(words) > 10:
        return False
    all_bloom = {v for verbs in BLOOM_VERBS.values() for v in verbs}
    if any(w.rstrip(".,;:") in all_bloom for w in words):
        return False
    return True


def extract_questions(text: str) -> list[str]:
    """
    Extract only question sentences from raw PDF / pasted text.

    Handles two common exam-paper layouts:

    Layout A — Two-level hierarchy (main Q + subparts):
        Q1. Finite Automata          ← topic heading, NOT a real question
            (a) Construct a DFA …    ← real question
            (b) Trace the string …   ← real question

    Layout B — Flat numbered questions (no subparts):
        Q1. Define recursion.
        Q2. Explain stack overflow.

    Strategy:
      1. Parse lines into a hierarchical structure tracking main Qs and subparts.
      2. If subparts exist → return subparts only (prefixed with parent topic).
      3. If no subparts → return main question bodies.
      4. Heuristic fallback for unlabelled papers (Bloom verbs / '?').
    """
    # Strip mark hints like [5 marks] [6] [4]
    text = re.sub(r"\[\s*\d+\s*(?:marks?)?\s*\]", "", text)
    # Also strip trailing standalone marks like "   [6]" at end of line
    text = re.sub(r"\s+\[\d+\]", "", text)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # ── Pass 1: hierarchical parse ───────────────────────────────────────────
    # Each entry: {"topic": str, "body": str|None, "subs": [str]}
    sections: list[dict] = []
    current_section: dict | None = None
    pending_sub: str | None = None      # accumulate multi-line subparts

    for line in lines:
        if _SKIP_RE.match(line):
            continue
        is_section_heading = line.isupper() and len(line.split()) <= 6
        if is_section_heading:
            continue

        if _MAIN_LABEL_RE.match(line):
            # Flush any pending subpart
            if pending_sub and current_section is not None:
                current_section["subs"].append(pending_sub)
                pending_sub = None

            body = _MAIN_LABEL_RE.sub("", line, count=1).strip()
            current_section = {"topic": body, "body": None, "subs": []}
            sections.append(current_section)

            # If the body itself is a real question (not a topic heading),
            # treat it as the question body too
            if body and not _is_topic_heading(body):
                current_section["body"] = body

        elif _SUB_LABEL_RE.match(line) and current_section is not None:
            # Flush previous subpart
            if pending_sub:
                current_section["subs"].append(pending_sub)
            pending_sub = _SUB_LABEL_RE.sub("", line, count=1).strip()

        elif pending_sub is not None:
            # Continuation of a subpart (line-wrapped in PDF)
            if len(line) < 200:
                pending_sub += " " + line

        elif current_section is not None and current_section["body"] is None:
            # Continuation of a main question body (no subparts yet)
            if len(line) < 200:
                current_section["topic"] += " " + line
                if not _is_topic_heading(current_section["topic"]):
                    current_section["body"] = current_section["topic"]

    # Flush last pending subpart
    if pending_sub and current_section is not None:
        current_section["subs"].append(pending_sub)

    # ── Decide what to return ────────────────────────────────────────────────
    all_subs_exist = any(s["subs"] for s in sections)

    results: list[str] = []

    if all_subs_exist:
        # Layout A: return subparts, prefixed with parent topic for context
        for sec in sections:
            topic = sec["topic"].split("10 Marks")[0].split("  ")[0].strip()
            for sub in sec["subs"]:
                if len(sub) > 8:
                    # Prefix with parent topic so classifier has context
                    results.append(f"[{topic}] {sub}" if topic else sub)
    else:
        # Layout B: return main question bodies
        for sec in sections:
            body = sec["body"] or sec["topic"]
            if body and len(body) > 8:
                results.append(body)

    if results:
        return results

    # ── Pass 2: heuristic fallback (no labels found) ─────────────────────────
    all_bloom_verbs = {v for verbs in BLOOM_VERBS.values() for v in verbs}
    heuristic: list[str] = []

    for line in lines:
        if len(line) < 10 or _SKIP_RE.match(line):
            continue
        first_word = line.split()[0].lower().rstrip(".,;:")
        if line.rstrip().endswith("?") or first_word in all_bloom_verbs:
            heuristic.append(line)

    return heuristic if heuristic else [text.strip()]


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation, preserve Bloom verbs."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    all_bloom = {v for verbs in BLOOM_VERBS.values() for v in verbs}
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("english")) - all_bloom
    except Exception:
        import nltk; nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        sw = set(stopwords.words("english")) - all_bloom
    tokens = [t for t in tokens if t not in sw and len(t) >= 2]
    return " ".join(tokens)


def vectorize_tfidf(texts: list[str], top_n: int = 10) -> pd.DataFrame:
    """Return top TF-IDF terms per document as a tidy DataFrame."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    cleaned = [clean_text(t) for t in texts]
    vec = TfidfVectorizer(max_features=100, ngram_range=(1, 2))
    X   = vec.fit_transform(cleaned)
    terms = vec.get_feature_names_out()
    rows  = []
    for i, text in enumerate(texts):
        scores = X[i].toarray()[0]
        top_idx = scores.argsort()[-top_n:][::-1]
        top_terms = [(terms[j], round(float(scores[j]), 4)) for j in top_idx if scores[j] > 0]
        rows.append({
            "Question": text[:50] + "…",
            "Top Terms": ", ".join(f"{t}({s})" for t, s in top_terms[:5]),
        })
    return pd.DataFrame(rows)
