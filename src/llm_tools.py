"""
src/llm_tools.py — Generative AI: classify, generate, level-up, explain (CO4)
Uses Groq API (qwen/qwen3-32b) with local caching and fallback.
"""
from __future__ import annotations
import os, re, time, json, hashlib, logging
from pathlib import Path

logger   = logging.getLogger(__name__)
BLOOM    = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
CACHE    = Path(".cache/llm")
CACHE.mkdir(parents=True, exist_ok=True)

PRIORITY = ["Create", "Evaluate", "Analyze", "Apply", "Understand", "Remember"]

# ── LLM call ───────────────────────────────────────────────────────────────

def _cache_get(key: str):
    f = CACHE / f"{key}.json"
    return json.loads(f.read_text()) if f.exists() else None


def _cache_set(key: str, data: dict):
    (CACHE / f"{key}.json").write_text(json.dumps(data))


def call_llm(prompt: str, max_tokens: int = 300) -> tuple[str, float]:
    """Call Groq API with caching. Returns (text, latency_s). Raises on failure so caller can show error."""
    key    = hashlib.md5(prompt.encode()).hexdigest()
    cached = _cache_get(key)
    if cached:
        return cached["text"], cached["latency"]

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set in .env")

    from groq import Groq
    client = Groq(api_key=api_key)
    t0 = time.time()
    for model in ["qwen/qwen3.8-27b", "openai/gpt-oss-120b", "openai/gpt-oss-20b"]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            latency = round(time.time() - t0, 3)
            text    = resp.choices[0].message.content.strip()
            _cache_set(key, {"text": text, "latency": latency})
            logger.info("LLM success with model %s", model)
            return text, latency
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                time.sleep(2)
                continue
            if "404" in err or "model" in err.lower():
                continue
            raise RuntimeError(f"Groq error ({model}): {err}")
    raise RuntimeError("All Groq models failed — check API key at console.groq.com")


def _parse_level(text: str) -> str:
    for level in BLOOM:
        if level.lower() in text.lower():
            return level
    return "Understand"


# ── Classify ───────────────────────────────────────────────────────────────

PROMPTS = {
    "zero_shot_v1": (
        "Classify the following exam question into exactly one of these Bloom's Taxonomy levels: "
        "Remember, Understand, Apply, Analyze, Evaluate, Create.\n"
        "Reply with ONLY the level name.\n\nQuestion: {question}"
    ),
    "few_shot_k3_v1": (
        "Classify exam questions into Bloom's Taxonomy levels.\n\n"
        "Examples:\n"
        "Q: Define recursion. → Remember\n"
        "Q: Explain how TCP/IP works. → Understand\n"
        "Q: Analyze the time complexity of merge sort. → Analyze\n\n"
        "Now classify (reply with ONLY the level):\n"
        "Q: {question} →"
    ),
    "cot_v1": (
        "Classify this exam question into Bloom's Taxonomy (Remember/Understand/Apply/Analyze/Evaluate/Create).\n"
        "Think step by step:\n"
        "1. What is the key action verb?\n"
        "2. What cognitive skill does it require?\n"
        "3. Which Bloom level fits best?\n\n"
        "Question: {question}\n\n"
        "Reply in format: Level: <level name>"
    ),
}


def classify_with_llm(question: str, prompt_name: str = "few_shot_k3_v1") -> dict:
    prompt = PROMPTS.get(prompt_name, PROMPTS["few_shot_k3_v1"]).format(question=question)
    text, latency = call_llm(prompt, max_tokens=50)
    level = _parse_level(text)
    return {"level": level, "raw": text, "latency": latency, "prompt": prompt_name}


def explain_level(question: str, level: str) -> str:
    prompt = (
        f"In 2–3 sentences, explain why this exam question belongs to the '{level}' level "
        f"of Bloom's Taxonomy:\n\nQuestion: {question}"
    )
    text, _ = call_llm(prompt, max_tokens=150)
    return text


# ── Generate ───────────────────────────────────────────────────────────────

def generate_questions(topic: str, level: str, subject: str = "Computer Science", n: int = 3) -> list[str]:
    """Generate n exam questions at the given Bloom level."""
    prompt = (
        f"Generate exactly {n} exam questions about '{topic}' in {subject} "
        f"at the '{level}' level of Bloom's Taxonomy.\n"
        f"Use strong action verbs for {level} (e.g., {_level_verbs(level)}).\n"
        f"Format: numbered list only, one question per line. No explanations."
    )
    text, _ = call_llm(prompt, max_tokens=400)
    questions = []
    for line in text.split("\n"):
        line = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
        if len(line) > 15:
            questions.append(line)
    return questions[:n]


def _level_verbs(level: str) -> str:
    verbs = {
        "Remember": "define, list, recall",
        "Understand": "explain, describe, summarize",
        "Apply": "apply, solve, implement",
        "Analyze": "analyze, differentiate, examine",
        "Evaluate": "evaluate, justify, critique",
        "Create": "design, construct, formulate",
    }
    return verbs.get(level, "describe")


# ── Level-Up ───────────────────────────────────────────────────────────────

def levelup_question(question: str, from_level: str, to_level: str = None) -> dict:
    """Rewrite a question to a higher Bloom level."""
    if to_level is None:
        idx      = BLOOM.index(from_level) if from_level in BLOOM else 0
        to_level = BLOOM[min(idx + 1, len(BLOOM) - 1)]

    prompt = (
        f"Rewrite this question from Bloom's '{from_level}' level to '{to_level}' level.\n"
        f"Use action verbs appropriate for {to_level}: {_level_verbs(to_level)}.\n"
        f"Return ONLY the rewritten question.\n\n"
        f"Original: {question}"
    )
    rewritten, latency = call_llm(prompt, max_tokens=200)
    return {
        "original":  question,
        "rewritten": rewritten.strip(),
        "from_level": from_level,
        "to_level":   to_level,
        "latency_s":  latency,
    }
