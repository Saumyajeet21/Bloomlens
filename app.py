"""
BloomLens — Streamlit App
Bloom's Taxonomy Question Intelligence Platform
4 Tabs covering all 4 Course Outcomes (CO1–CO4)
"""
import streamlit as st
import pandas as pd
import numpy as np
import os, re, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BloomLens",
    layout="wide",
    initial_sidebar_state="expanded",
)

BLOOM  = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
COLORS = {
    "Remember":   "#6366f1",
    "Understand": "#3b82f6",
    "Apply":      "#10b981",
    "Analyze":    "#f59e0b",
    "Evaluate":   "#ef4444",
    "Create":     "#8b5cf6",
}

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0f172a; }
  [data-testid="stSidebar"]          { background: #1e293b; }
  h1,h2,h3,h4                        { color: #a5b4fc; }
  .stTabs [data-baseweb="tab"]       { color: #94a3b8; font-weight:600; }
  .stTabs [aria-selected="true"]     { color: #a5b4fc !important; border-bottom: 2px solid #6366f1; }
  .metric-card { background:#1e293b; border:1px solid #334155; border-radius:12px;
                 padding:16px; text-align:center; margin:4px; }
  .bloom-badge { border-radius:20px; padding:4px 14px; font-weight:700;
                 font-size:13px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  BloomLens")
    st.caption("Bloom's Taxonomy Question Intelligence")
    st.divider()
    st.markdown("### 📊 Project Accuracy")
    acc_data = {
        "Model":      ["Verb Rules", "TF-IDF SVM", "SBERT+LR", "DistilBERT"],
        "Macro-F1":   [0.9687, 0.9056, 0.8498, 0.6567],
    }
    acc_df = pd.DataFrame(acc_data)
    st.dataframe(acc_df.style.highlight_max("Macro-F1", color="#1a3a2a"), use_container_width=True)
    st.divider()
    st.markdown("### ⚙️ Settings")
    classifier_choice = st.selectbox("Classifier", ["Verb Rules (Fast · 0.97)", "TF-IDF SVM (0.91)", "DistilBERT (Transformer)"])
    llm_provider = st.selectbox("LLM Provider", ["groq", "openai", "mock"])
    st.caption(f"LLM: `{llm_provider}` | API key: {'✅' if os.getenv('GROQ_API_KEY') else '❌'}")

# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📄 Text Processing",
    "🔍 Sentiment & Extraction",
    "🌐 Topics & Multilingual",
    "✨ Generative AI",
])

# ═══════════════════════════════════════════════════════════════════════════
# CO1 — Text Extraction, Preprocessing, Cleaning, Vectorization
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📄 Text Extraction & Processing")
    st.caption("Extract questions from raw text or PDF, clean, and vectorize.")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        input_mode = st.radio("Input mode", ["Paste Text", "Upload PDF"], horizontal=True)
        raw_text = ""

        if input_mode == "Paste Text":
            raw_text = st.text_area(
                "Paste exam paper / questions",
                height=220,
                value="""Q1. Define the term recursion in programming. [5 marks]
Q2. Explain how a stack data structure works. [5 marks]
Q3. Apply dynamic programming to solve the knapsack problem. [10 marks]
Q4. Analyze the time complexity of merge sort vs quick sort. [10 marks]
Q5. Evaluate the trade-offs between SQL and NoSQL databases. [10 marks]
Q6. Design a fault-tolerant distributed caching system. [15 marks]""",
            )
        else:
            uploaded = st.file_uploader("Upload PDF", type=["pdf"])
            if uploaded:
                try:
                    import pdfplumber
                    with pdfplumber.open(uploaded) as pdf:
                        raw_text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                    st.success(f"✅ Extracted {len(raw_text)} characters from PDF")
                    with st.expander("🔎 Raw PDF text (debug)", expanded=False):
                        st.text(raw_text[:3000] + ("…" if len(raw_text) > 3000 else ""))
                except Exception as e:
                    st.error(f"PDF error: {e}")

        extract_btn = st.button("🔍 Extract & Process", type="primary", use_container_width=True)

    if extract_btn and raw_text:
        from src.preprocess import extract_questions, clean_text, vectorize_tfidf

        with st.spinner("Processing..."):
            questions = extract_questions(raw_text)
            cleaned   = [clean_text(q) for q in questions]

        with col_out:
            st.markdown(f"### ✅ Extracted {len(questions)} Questions")
            for i, (q, c) in enumerate(zip(questions, cleaned), 1):
                with st.expander(f"Q{i}. {q[:60]}..."):
                    st.markdown(f"**Original:** {q}")
                    st.markdown(f"**Cleaned:** `{c}`")
            st.session_state["questions"]      = questions
            st.session_state["cleaned"]        = cleaned
            st.session_state["questions_text"] = "\n".join(questions)
            st.success("💡 Questions auto-populated in other tabs!", icon="🔗")

        # Vectorization display
        st.divider()
        st.markdown("### 📐 TF-IDF Vectorization (Top Terms)")
        if len(questions) >= 2:
            tfidf_df = vectorize_tfidf(questions)
            st.dataframe(tfidf_df, use_container_width=True)
        else:
            st.info("Need ≥ 2 questions for TF-IDF.")

        # Class distribution
        st.divider()
        st.markdown("### 🏷️ Bloom Level Distribution (Verb-Rule Preview)")
        from src.classify import classify_verb
        preds = [classify_verb(q) or "Unknown" for q in questions]
        dist  = pd.Series(preds).value_counts().reset_index()
        dist.columns = ["Level", "Count"]
        dist["Color"] = dist["Level"].map(lambda l: COLORS.get(l, "#64748b"))
        st.bar_chart(dist.set_index("Level")["Count"])

    elif extract_btn:
        st.warning("Please provide some text first.")

    # Show loaded dataset stats
    st.divider()
    with st.expander("📊 View Full Dataset (processed_questions.csv)", expanded=False):
        data_path = Path("reports/tables/processed_questions.csv")
        if data_path.exists():
            df_all = pd.read_csv(data_path)
            st.metric("Total Questions", len(df_all))
            dist = df_all["label"].value_counts()
            st.bar_chart(dist)
            st.dataframe(df_all[["text","label","source"]].head(20), use_container_width=True)
        else:
            st.info("Run Phase 1 to generate processed_questions.csv")


# ═══════════════════════════════════════════════════════════════════════════
# CO2 — Sentiment Analysis + Information Extraction
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Sentiment Analysis & Information Extraction")

    # Auto-populate from CO1 if questions were extracted
    _co2_default = st.session_state.get("questions_text", "") or \
        "Define the term recursion in programming.\nExplain how a stack data structure works.\nAnalyze the time complexity of merge sort vs quick sort.\nDesign a fault-tolerant distributed caching system.\nEvaluate the trade-offs between SQL and NoSQL databases."

    if st.session_state.get("questions_text"):
        st.info("✅ Questions auto-populated from extracted text", icon="🔗")

    q_input = st.text_area(
        "Enter questions (one per line)",
        height=160,
        value=_co2_default,
    )
    run_analysis = st.button("🧠 Analyse", type="primary")

    if run_analysis and q_input.strip():
        questions_list = [q.strip() for q in q_input.strip().split("\n") if q.strip()]

        from src.classify import classify_verb, classify_svm
        from src.sentiment_tools import analyse_sentiment, extract_verbs

        rows = []
        for q in questions_list:
            level    = classify_verb(q) or "Understand"
            sent     = analyse_sentiment(q)
            verbs    = extract_verbs(q)
            rows.append({
                "Question":    q[:70],
                "Bloom Level": level,
                "Sentiment":   sent["label"],
                "Polarity":    round(sent["score"], 3),
                "Action Verb": ", ".join(verbs[:3]) if verbs else "—",
            })

        df_res = pd.DataFrame(rows)

        # Colour-coded table
        st.markdown("### 📋 Classification + Sentiment Results")

        def color_level(val):
            c = COLORS.get(val, "#64748b")
            return f"background-color:{c}22; color:{c}; font-weight:700"

        def color_sent(val):
            if val == "POSITIVE": return "color:#10b981; font-weight:700"
            if val == "NEGATIVE": return "color:#ef4444; font-weight:700"
            return "color:#94a3b8"

        styled = (df_res.style
                  .map(color_level, subset=["Bloom Level"])
                  .map(color_sent,  subset=["Sentiment"]))
        st.dataframe(styled, use_container_width=True)

        # Sentiment distribution
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 😊 Sentiment Distribution")
            sent_dist = df_res["Sentiment"].value_counts()
            st.bar_chart(sent_dist)
        with c2:
            st.markdown("#### 🏷️ Bloom Level Distribution")
            level_dist = df_res["Bloom Level"].value_counts()
            st.bar_chart(level_dist)

        # Model comparison
        st.divider()
        st.markdown("### 🏆 Model Accuracy Comparison")
        comp_df = pd.DataFrame([
            {"Model": "Verb Lexicon Rules", "Macro-F1": 0.9687, "Adj-Error": "0.0",  "QWK": "0.9791", "Speed": "Fast"},
            {"Model": "TF-IDF + SVM",       "Macro-F1": 0.9056, "Adj-Error": "0.0",  "QWK": "0.8780", "Speed": "Fast"},
            {"Model": "SBERT + LR",         "Macro-F1": 0.8498, "Adj-Error": "0.8",  "QWK": "0.9327", "Speed": "Medium"},
            {"Model": "DistilBERT",         "Macro-F1": 0.6567, "Adj-Error": "0.5",  "QWK": "0.8458", "Speed": "Slow"},
            {"Model": "VADER (Sentiment)",  "Macro-F1": 0.7435, "Adj-Error": "—",    "QWK": "—",      "Speed": "Fast"},
        ])
        st.dataframe(comp_df.style.highlight_max(subset=["Macro-F1"], color="#1a3a2a"), use_container_width=True)

        # Extracted verbs
        st.divider()
        st.markdown("### 🔑 Extracted Bloom Action Verbs")
        all_verbs = []
        for q in questions_list:
            vbs = extract_verbs(q)
            for v in vbs:
                from src.classify import VERB_TO_LEVEL
                all_verbs.append({"Verb": v, "Bloom Level": VERB_TO_LEVEL.get(v.lower(), "Unknown"), "Question": q[:50]})
        if all_verbs:
            st.dataframe(pd.DataFrame(all_verbs), use_container_width=True)
        else:
            st.info("No Bloom action verbs detected.")


# ═══════════════════════════════════════════════════════════════════════════
# CO3 — Topic Modeling + Multilingual NLP + Cloud AI
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🌐 Topic Modeling & Multilingual NLP")

    sub1, sub2 = st.tabs(["📊 LDA Topic Modeling", "🌍 Multilingual (Hindi)"])

    with sub1:
        st.markdown("### LDA Topic Modeling")

        _lda_default_list = [
            "Define recursion in programming","Explain binary search trees",
            "Apply dynamic programming to knapsack","Analyze merge sort complexity",
            "Evaluate SQL vs NoSQL trade-offs","Design a caching system",
            "Explain neural network learning","Analyze graph traversal algorithms",
            "Design a distributed database","Evaluate encryption algorithms",
            "Apply sorting algorithms to datasets","Create a machine learning pipeline",
        ]
        _lda_default = st.session_state.get("questions_text") or "\n".join(_lda_default_list)

        if st.session_state.get("questions_text"):
            st.info("✅ Questions auto-populated from extracted text", icon="🔗")

        lda_text = st.text_area(
            "Questions for topic modeling (one per line, need ≥ 10)",
            height=180,
            value=_lda_default,
        )
        n_topics_input = st.slider("Number of topics", 2, 10, 5)
        run_lda = st.button("🔬 Run LDA", type="primary")

        if run_lda:
            texts = [t.strip() for t in lda_text.strip().split("\n") if t.strip()]
            if len(texts) < 5:
                st.warning("Need at least 5 questions.")
            else:
                with st.spinner("Training LDA model..."):
                    from src.topics_multilingual import run_lda_model
                    result = run_lda_model(texts, n_topics=n_topics_input)

                st.success(f"✅ Discovered {result['n_topics']} topics")
                for tid, words_str in result["topics"]:
                    # Parse "0.12*\"word\" + 0.08*\"word2\" ..." → list of words
                    import re as _re
                    word_list = _re.findall(r'"([^"]+)"', words_str)
                    with st.expander(f"📌 Topic {tid+1}  —  {' · '.join(word_list[:4])}", expanded=tid<2):
                        chips = "  ".join(
                            f'<span style="background:#1e3a5f;color:#93c5fd;border-radius:20px;padding:4px 12px;font-size:13px;margin:3px;display:inline-block">{w}</span>'
                            for w in word_list
                        )
                        st.markdown(chips, unsafe_allow_html=True)

                # Heatmap from saved file
                heatmap_path = Path("reports/figures/topic_bloom_heatmap.png")
                if heatmap_path.exists():
                    st.image(str(heatmap_path), caption="Topic–Bloom Heatmap", use_container_width=True)

                # Gap report
                gap_path = Path("reports/tables/gap_report.md")
                if gap_path.exists():
                    st.markdown("### ⚠️ Bloom Gap Report")
                    st.markdown(gap_path.read_text(encoding="utf-8"))

    with sub2:
        st.markdown("### 🌍 Multilingual — English → Hindi Translation")
        st.caption("☁️ **Primary:** Groq Cloud API (qwen3.8-27b)  |  🖥️ **Fallback:** Helsinki-NLP opus-mt-en-hi (local)")

        _hi_default = st.session_state.get("questions_text") or \
            "Define the term recursion in programming.\nExplain how a neural network learns.\nAnalyze the time complexity of merge sort."

        if st.session_state.get("questions_text"):
            st.info("✅ Questions auto-populated from extracted text", icon="🔗")

        hi_questions = st.text_area(
            "English questions to translate",
            height=140,
            value=_hi_default,
        )

        # Provider selector
        provider_choice = st.radio(
            "Translation provider",
            ["☁️ Groq Cloud API (LLM)", "🖥️ Local Model (Helsinki-NLP)"],
            horizontal=True,
        )
        provider = "groq" if "Groq" in provider_choice else "local"

        col_a, col_b = st.columns(2)
        with col_a: run_translate = st.button("🌐 Translate to Hindi", type="primary")
        with col_b: run_cloud_compare = st.button("☁️ Live Cloud vs Local Benchmark")

        if run_translate:
            q_list = [q.strip() for q in hi_questions.strip().split("\n") if q.strip()]
            provider_label = "Groq Cloud API" if provider == "groq" else "Helsinki-NLP (local)"
            with st.spinner(f"Translating via {provider_label}..."):
                from src.topics_multilingual import translate_to_hindi
                translated = translate_to_hindi(q_list, provider=provider)

            st.success(f"✅ Translated {len(q_list)} questions using **{provider_label}**")
            df_trans = pd.DataFrame({"English": q_list, "Hindi (Translation)": translated})
            st.dataframe(df_trans, use_container_width=True)

            # XLM-R results
            st.divider()
            st.markdown("### 🤖 Multilingual Classification Results")
            ml_path = Path("reports/tables/multilingual_results.csv")
            if ml_path.exists():
                ml_df = pd.read_csv(ml_path)
                st.dataframe(ml_df, use_container_width=True)
            else:
                st.info("Run `python run_phase4.py` to generate XLM-R results.")

        if run_cloud_compare:
            q_list = [q.strip() for q in hi_questions.strip().split("\n") if q.strip()]
            if not q_list:
                st.warning("Enter at least one question first.")
            else:
                st.markdown("### ☁️ Live Cloud vs Local Translation Benchmark")
                st.caption("Running both Groq Cloud API and Helsinki-NLP on the same questions in real time.")

                from src.topics_multilingual import benchmark_cloud_vs_local
                with st.spinner("Running live cloud & local translation benchmark..."):
                    rows = benchmark_cloud_vs_local(q_list)

                import pandas as _pd
                bdf = _pd.DataFrame(rows)

                # Display side-by-side
                for row in rows:
                    with st.expander(f"📝 {row['english'][:70]}"):
                        bc1, bc2 = st.columns(2)
                        with bc1:
                            st.markdown("**☁️ Groq Cloud API**")
                            st.markdown(f"*{row.get('cloud_provider', 'Groq')}*")
                            st.info(row.get("cloud_translation", "—"))
                            st.caption(f"⏱ {row.get('cloud_latency_ms', '—')} ms")
                        with bc2:
                            st.markdown("**🖥️ Local Model**")
                            st.markdown(f"*{row.get('local_provider', 'Helsinki-NLP')}*")
                            st.info(row.get("local_translation", "—"))
                            st.caption(f"⏱ {row.get('local_latency_ms', '—')} ms")

                # Latency comparison metrics
                st.divider()
                valid_cloud = [r["cloud_latency_ms"] for r in rows if r.get("cloud_latency_ms", -1) >= 0]
                valid_local = [r["local_latency_ms"] for r in rows if r.get("local_latency_ms", -1) >= 0]
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("☁️ Avg Cloud Latency", f"{sum(valid_cloud)//max(len(valid_cloud),1)} ms" if valid_cloud else "—")
                mc2.metric("🖥️ Avg Local Latency", f"{sum(valid_local)//max(len(valid_local),1)} ms" if valid_local else "—")
                mc3.metric("Questions Translated", len(rows))

                # Save to CSV for reference
                save_path = Path("reports/tables/cloud_vs_local.csv")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                bdf.to_csv(save_path, index=False)
                st.caption(f"✅ Results saved to `{save_path}`")


# ═══════════════════════════════════════════════════════════════════════════
# CO4 — Generative AI / LLM / Prompt Engineering
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## ✨ Generative AI & Prompt Engineering")

    gen_tab, levelup_tab, classify_tab, compare_tab = st.tabs([
        "🚀 Generate Questions",
        "⬆️ Level-Up",
        "🏷️ LLM Classify",
        "📊 Prompt Comparison",
    ])

    # ── Generate ───────────────────────────────────────────────────────────
    with gen_tab:
        st.markdown("### Generate Exam Questions at Any Bloom Level")
        g1, g2, g3 = st.columns(3)
        topic    = g1.text_input("Topic", value="sorting algorithms")
        level    = g2.selectbox("Bloom Level", BLOOM, index=2)
        n_gen    = g3.number_input("Count", 1, 8, 3)
        subject  = st.text_input("Subject", value="Computer Science")

        if st.button("🚀 Generate", type="primary"):
            if not topic.strip():
                st.warning("Enter a topic.")
            else:
                with st.spinner("Generating with Groq LLM..."):
                    try:
                        from src.llm_tools import generate_questions
                        qs = generate_questions(topic, level, subject, int(n_gen))
                        if not qs:
                            st.error("LLM returned empty response. Try again.")
                        else:
                            color = COLORS.get(level, "#6366f1")
                            st.markdown(f'<span class="bloom-badge" style="background:{color}22;color:{color}">{level}</span>', unsafe_allow_html=True)
                            for i, q in enumerate(qs, 1):
                                st.markdown(f"**{i}.** {q}")
                            # Verify
                            st.divider()
                            st.markdown("#### ✅ Verification (rule-based classifier agreement)")
                            from src.classify import classify_verb
                            match = sum(1 for q in qs if (classify_verb(q) or "Understand") == level)
                            st.progress(match / max(len(qs), 1))
                            st.caption(f"{match}/{len(qs)} questions verified as **{level}** by verb rules")
                    except Exception as e:
                        st.error(f"❌ LLM Error: {e}")
                        st.info("💡 Get a free API key at **console.groq.com** and update GROQ_API_KEY in `.env`")

    # ── Level-Up ──────────────────────────────────────────────────────────
    with levelup_tab:
        st.markdown("### Rewrite a Question to a Higher Bloom Level")
        orig_q = st.text_area("Original question", value="Define the term recursion.")
        lu1, lu2 = st.columns(2)
        from_lvl = lu1.selectbox("Current Level", BLOOM, index=0)
        to_lvl   = lu2.selectbox("Target Level",  BLOOM, index=3)

        if st.button("⬆️ Level Up", type="primary"):
            if from_lvl == to_lvl:
                st.warning("Choose a different target level.")
            else:
                with st.spinner("Rewriting..."):
                    try:
                        from src.llm_tools import levelup_question
                        result = levelup_question(orig_q, from_lvl, to_lvl)
                        c_from = COLORS.get(from_lvl, "#64748b")
                        c_to   = COLORS.get(to_lvl,   "#64748b")
                        st.markdown(
                            f'<span class="bloom-badge" style="background:{c_from}22;color:{c_from}">{from_lvl}</span>'
                            f' → '
                            f'<span class="bloom-badge" style="background:{c_to}22;color:{c_to}">{to_lvl}</span>',
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Original:**"); st.info(result["original"])
                        st.markdown("**Rewritten:**"); st.success(result["rewritten"])
                        st.caption(f"Latency: {result.get('latency_s','—')}s")
                    except Exception as e:
                        st.error(f"❌ LLM Error: {e}")
                        st.info("💡 Update GROQ_API_KEY in `.env` with a fresh key from **console.groq.com**")

    # ── LLM Classify ──────────────────────────────────────────────────────
    with classify_tab:
        st.markdown("### LLM-Based Classification with Explanation")
        q_classify = st.text_area(
            "Enter a question",
            value="Design a fault-tolerant distributed caching system."
        )
        prompt_style = st.radio(
            "Prompt strategy",
            ["Zero-Shot", "Few-Shot (3)", "Chain-of-Thought"],
            horizontal=True,
        )

        if st.button("🏷️ Classify with LLM", type="primary"):
            with st.spinner("Calling LLM..."):
                from src.llm_tools import classify_with_llm, explain_level
                prompt_map = {
                    "Zero-Shot": "zero_shot_v1",
                    "Few-Shot (3)": "few_shot_k3_v1",
                    "Chain-of-Thought": "cot_v1",
                }
                result = classify_with_llm(q_classify, prompt_map[prompt_style])
                explanation = explain_level(q_classify, result["level"])

            color = COLORS.get(result["level"], "#6366f1")
            st.markdown(
                f'### Classified as: <span class="bloom-badge" style="background:{color}22;color:{color};font-size:20px">'
                f'{result["level"]}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Explanation:** {explanation}")
            st.caption(f"Latency: {result.get('latency', '—')}s | Prompt: {prompt_style}")

    # ── Prompt Comparison ─────────────────────────────────────────────────
    with compare_tab:
        st.markdown("### Prompt Engineering Comparison")
        st.caption("Compare Zero-Shot vs Few-Shot vs Chain-of-Thought on a sample of questions")

        # Show cached results if available
        exp_path = Path("reports/tables/llm_experiments.csv")
        if exp_path.exists():
            st.markdown("#### 📊 Saved Experiment Results")
            exp_df = pd.read_csv(exp_path)
            st.dataframe(exp_df, use_container_width=True)
            if "macro_f1" in exp_df.columns:
                st.bar_chart(exp_df.set_index("prompt_name")["macro_f1"])
        else:
            st.info("No experiment results yet.")

        if st.button("🔬 Run Prompt Comparison (uses API calls)", type="secondary"):
            data_path = Path("reports/tables/processed_questions.csv")
            if data_path.exists():
                df_data = pd.read_csv(data_path)
                df_data = df_data[df_data["label"].isin(BLOOM)].dropna(subset=["text","label"])
                sample_q = df_data["text"].tolist()[:10]
                sample_l = df_data["label"].tolist()[:10]

                results = []
                for strategy, pname in [("Zero-Shot","zero_shot_v1"),("Few-Shot","few_shot_k3_v1"),("CoT","cot_v1")]:
                    with st.spinner(f"Running {strategy}..."):
                        from src.llm_tools import classify_with_llm
                        from sklearn.metrics import f1_score
                        preds = []
                        for q in sample_q:
                            r = classify_with_llm(q, pname)
                            preds.append(r["level"])
                            time.sleep(0.5)
                        f1 = f1_score(sample_l, preds, average="macro",
                                      labels=BLOOM, zero_division=0)
                        results.append({"Strategy": strategy, "Macro-F1": round(f1, 4), "Prompt": pname})

                res_df = pd.DataFrame(results)
                st.dataframe(res_df, use_container_width=True)
                st.bar_chart(res_df.set_index("Strategy")["Macro-F1"])
            else:
                st.error("No question data found.")
