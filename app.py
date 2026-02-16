import os
import json
import pandas as pd
import requests
import streamlit as st

from scoring import (
    GOEMOTIONS_TAXONOMY,
    escalation_score,
    misunderstanding_score,
    empathy_score,
)

st.set_page_config(page_title="Diffuser", layout="centered")
st.title("Diffuser")
st.caption("Chat analyzer + local LLM rewrite (Ollama). Alternates Me ↔ Them.")

# Streamlit Cloud cannot reach your laptop's Ollama (localhost there != your localhost)
IS_CLOUD = bool(os.getenv("STREAMLIT_CLOUD") or os.getenv("STREAMLIT_SERVER_RUNNING"))

# session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "next_speaker" not in st.session_state:
    st.session_state.next_speaker = "Me"

if "analysis_df" not in st.session_state:
    st.session_state.analysis_df = None
if "show_analysis" not in st.session_state:
    st.session_state.show_analysis = False
if "trend_metric" not in st.session_state:
    st.session_state.trend_metric = "escalation_risk"

if "overall_esc" not in st.session_state:
    st.session_state.overall_esc = 0
if "overall_mis" not in st.session_state:
    st.session_state.overall_mis = 0
if "overall_emp" not in st.session_state:
    st.session_state.overall_emp = 0

# controls
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Reset conversation"):
        st.session_state.messages = []
        st.session_state.next_speaker = "Me"
        st.session_state.analysis_df = None
        st.session_state.show_analysis = False
        st.session_state.trend_metric = "escalation_risk"
        st.session_state.overall_esc = 0
        st.session_state.overall_mis = 0
        st.session_state.overall_emp = 0
        st.rerun()

with col2:
    if st.button("Undo last message") and st.session_state.messages:
        st.session_state.messages.pop()
        if not st.session_state.messages:
            st.session_state.next_speaker = "Me"
        else:
            st.session_state.next_speaker = "Them" if st.session_state.messages[-1]["speaker"] == "Me" else "Me"
        st.session_state.analysis_df = None
        st.session_state.show_analysis = False
        st.rerun()

st.divider()

# chat styling
st.markdown(
    """
    <style>
    .chat-row { display:flex; width:100%; margin:0.25rem 0; }
    .chat-row.me { justify-content:flex-end; }
    .chat-row.them { justify-content:flex-start; }

    .bubble {
        width: fit-content;
        min-width: 260px;
        max-width: 85%;
        padding: 0.8rem 1.0rem;
        border-radius: 18px;
        line-height: 1.25rem;
        font-size: 1rem;
        word-wrap: break-word;
        white-space: pre-wrap;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .bubble.me {
        border-bottom-right-radius: 6px;
        background: rgba(255,255,255,0.10);
    }
    .bubble.them {
        border-bottom-left-radius: 6px;
        background: rgba(255,255,255,0.06);
    }
    .small-label {
        font-size: 0.75rem;
        opacity: 0.7;
        margin-bottom: 0.15rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# make chat
for m in st.session_state.messages:
    cls = "me" if m["speaker"] == "Me" else "them"
    speaker_label = m["speaker"]
    safe_text = m["text"].replace("<", "&lt;").replace(">", "&gt;")
    st.markdown(
        f"""
        <div class="chat-row {cls}">
          <div>
            <div class="small-label">{speaker_label}</div>
            <div class="bubble {cls}">{safe_text}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# input
placeholder = f"Type {st.session_state.next_speaker}'s message and press Enter…"
new_text = st.chat_input(placeholder)

if new_text:
    st.session_state.messages.append({"speaker": st.session_state.next_speaker, "text": new_text.strip()})
    st.session_state.next_speaker = "Them" if st.session_state.next_speaker == "Me" else "Me"
    st.session_state.analysis_df = None
    st.session_state.show_analysis = False
    st.rerun()

# convo display
st.divider()
st.subheader("Conversation (for analysis)")
if st.session_state.messages:
    formatted = "\n".join([f'{m["speaker"]}: {m["text"]}' for m in st.session_state.messages])
    st.code(formatted)
else:
    st.info("No messages yet. Start typing above.")

# goEmotions taxonomy context
with st.expander("Emotion context (GoEmotions taxonomy)"):
    st.caption("Local LLM picks 1–3 likely emotions from this list (soft/uncertain).")
    st.write("**Taxonomy:** " + ", ".join(GOEMOTIONS_TAXONOMY))

# ollama
def ollama_generate_json(model: str, prompt: str, timeout_s: int = 120) -> dict:
    """
    Calls Ollama via OpenAI-compatible endpoint:
    http://localhost:11434/v1/chat/completions

    This avoids /api/generate vs /api/chat mismatches and is the most stable option.
    """
    r = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": "You output ONLY valid JSON. No markdown. No extra text."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        },
        timeout=timeout_s,
    )

    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:800]}")

    data = r.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except Exception:
        raise RuntimeError(f"Unexpected response shape: {str(data)[:800]}")

    try:
        return json.loads(text)
    except Exception as e:
        raise RuntimeError(f"Model did not return valid JSON: {e}\nRaw:\n{text[:800]}")

def llm_rewrite_local(conversation_text: str, overall_esc: int, overall_mis: int, overall_emp: int, model: str) -> dict:
    taxonomy = ", ".join(GOEMOTIONS_TAXONOMY)

    prompt = f"""
You are an assistant that helps write calm replies in friend conflicts.

Rules:
- Assume good intent. Do NOT accuse or mind-read.
- If misunderstanding risk is high, ask a clarifying question.
- If empathy is low, include a short validating line.
- Keep the reply short like real texting (1–3 short lines).
- Handle slang/typos naturally.
- Choose likely emotions ONLY from this GoEmotions taxonomy (do not invent labels):
[{taxonomy}]
- Do NOT claim certainty. Use soft language like "it seems like" or "maybe".

Return ONLY valid JSON with EXACT keys:
- likely_emotions: array of 1–3 strings from the taxonomy
- empathy_line: short validating sentence
- clarifying_question: one clear question
- next_message: 1–3 short lines total
- why_this_works: 1–2 sentences

Conversation:
{conversation_text}

Recent scores:
Escalation Risk: {overall_esc}/100
Misunderstanding Risk: {overall_mis}/100
Empathy Score: {overall_emp}/100
""".strip()

    out = ollama_generate_json(model=model, prompt=prompt)

    # Defensive defaults
    out.setdefault("likely_emotions", [])
    out.setdefault("empathy_line", "")
    out.setdefault("clarifying_question", "")
    out.setdefault("next_message", "")
    out.setdefault("why_this_works", "")

    # Clamp likely_emotions to taxonomy and max 3
    cleaned = []
    for e in out.get("likely_emotions", []):
        if isinstance(e, str) and e in GOEMOTIONS_TAXONOMY and e not in cleaned:
            cleaned.append(e)
        if len(cleaned) == 3:
            break
    out["likely_emotions"] = cleaned

    return out

# Offline fallback (so cloud demo still "works")
def fallback_suggestion(overall_esc: int, overall_mis: int, overall_emp: int) -> dict:
    need_empathy = overall_emp <= 18
    need_clarity = overall_mis >= 45
    need_deescalate = overall_esc >= 35

    empathy_line = "I hear you — that makes sense." if need_empathy else "Got you."
    clarifying_question = "Can you help me understand what you meant by that?" if need_clarity else "What did you mean by that?"
    deescalate = "I’m not trying to argue — I just want to understand." if need_deescalate else ""

    next_message = "\n".join([p for p in [empathy_line, clarifying_question, deescalate] if p]).strip()

    return {
        "likely_emotions": ["confusion"] if need_clarity else [],
        "empathy_line": empathy_line,
        "clarifying_question": clarifying_question,
        "next_message": next_message,
        "why_this_works": "Uses validation + a clarifying question to reduce assumptions and keep the tone calm.",
    }

# analysis UI
st.divider()
st.subheader("Analysis")

if st.button("Analyze conversation"):
    if not st.session_state.messages:
        st.warning("Add some messages first.")
    else:
        rows = []
        for idx, m in enumerate(st.session_state.messages, start=1):
            rows.append({
                "turn": idx,
                "speaker": m["speaker"],
                "text": m["text"],
                "escalation_risk": escalation_score(m["text"]),
                "misunderstanding_risk": misunderstanding_score(m["text"]),
                "empathy_level": empathy_score(m["text"]),
            })

        st.session_state.analysis_df = pd.DataFrame(rows)
        st.session_state.show_analysis = True

if st.session_state.show_analysis and st.session_state.analysis_df is not None:
    df = st.session_state.analysis_df

    tail = df.tail(3)
    overall_esc = int(round(tail["escalation_risk"].mean()))
    overall_mis = int(round(tail["misunderstanding_risk"].mean()))
    overall_emp = int(round(tail["empathy_level"].mean()))

    st.session_state.overall_esc = overall_esc
    st.session_state.overall_mis = overall_mis
    st.session_state.overall_emp = overall_emp

    overall_conflict = int(round(
        0.50 * overall_esc +
        0.35 * overall_mis +
        0.15 * (100 - overall_emp)
    ))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Escalation", overall_esc)
    c2.metric("Misunderstanding", overall_mis)
    c3.metric("Empathy", overall_emp)
    c4.metric("Overall Risk", overall_conflict)

    metric_choice = st.selectbox(
        "Trend chart",
        ["escalation_risk", "misunderstanding_risk", "empathy_level"],
        index=["escalation_risk", "misunderstanding_risk", "empathy_level"].index(st.session_state.trend_metric),
        key="trend_metric"
    )
    st.line_chart(df.set_index("turn")[metric_choice])

    st.write("Per-message breakdown:")
    st.dataframe(
        df[["turn", "speaker", "escalation_risk", "misunderstanding_risk", "empathy_level", "text"]],
        use_container_width=True
    )

# suggested next message
st.divider()
st.subheader("Suggested next message")

mode = st.radio(
    "Suggestion mode",
    ["Local LLM (Ollama)", "Offline fallback (no LLM)"],
    index=0 if not IS_CLOUD else 1,
    horizontal=True
)

model = st.selectbox(
    "Ollama model",
    ["llama3.1:8b", "qwen2.5:7b", "qwen2.5:3b"],
    index=0,
    disabled=(mode != "Local LLM (Ollama)"),
)

if st.button("Generate suggestion"):
    if not st.session_state.messages:
        st.info("Add some messages first.")
    elif st.session_state.analysis_df is None:
        st.info("Click Analyze conversation first so we have scores.")
    else:
        conversation_text = "\n".join([f'{m["speaker"]}: {m["text"]}' for m in st.session_state.messages])

        if mode == "Offline fallback (no LLM)":
            out = fallback_suggestion(
                st.session_state.overall_esc,
                st.session_state.overall_mis,
                st.session_state.overall_emp
            )
        else:
            if IS_CLOUD:
                st.warning("Streamlit Cloud can’t access your laptop’s Ollama. Switch to Offline fallback or run locally.")
                st.stop()

            with st.spinner("Generating (local)…"):
                try:
                    out = llm_rewrite_local(
                        conversation_text,
                        st.session_state.overall_esc,
                        st.session_state.overall_mis,
                        st.session_state.overall_emp,
                        model=model,
                    )
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach Ollama at http://localhost:11434 from THIS machine.")
                    st.info("Open the Ollama app and ensure it’s running locally, then retry.")
                    st.stop()
                except requests.exceptions.Timeout:
                    st.error("Ollama timed out. Try a smaller model (qwen2.5:3b).")
                    st.stop()
                except Exception as e:
                    st.error(f"Local LLM error: {e}")
                    st.stop()

        if out.get("likely_emotions"):
            st.write("**Likely emotions:** " + ", ".join(out["likely_emotions"]))

        st.text_area("Empathy line", out.get("empathy_line", ""), height=60)
        st.text_area("Clarifying question", out.get("clarifying_question", ""), height=80)
        st.text_area("Next message", out.get("next_message", ""), height=140)
        st.caption(out.get("why_this_works", ""))

with st.expander("Troubleshooting (Ollama)"):
    st.markdown(
        """
**Local checks (run in Terminal):**
- `curl http://localhost:11434/api/tags`
- Test OpenAI-compatible endpoint:
  - `curl http://localhost:11434/v1/models`

If you're on Streamlit Cloud:
- Local Ollama will NOT be reachable there.
- Use Offline fallback in the deployed demo.
"""
    )