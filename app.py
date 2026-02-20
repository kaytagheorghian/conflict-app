import os
import re
import json
import html
import pandas as pd
import requests
import streamlit as st
import altair as alt
from transformers import pipeline


from goemotions_scoring import (
   GOEMOTIONS_TAXONOMY,
   scores_from_emotion_probs,
   top_emotions,
)


from llm_ontology import analyze_conversation_llm




# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Diffuser", layout="centered")
IS_CLOUD = bool(os.getenv("STREAMLIT_CLOUD") or os.getenv("STREAMLIT_SERVER_RUNNING"))


st.title("Diffuser")
st.caption("Type messages and press Enter. The app alternates Me ↔ Them.")




# ---------------------------
# Session state
# ---------------------------
if "messages" not in st.session_state:
   st.session_state.messages = []
if "next_speaker" not in st.session_state:
   st.session_state.next_speaker = "Me"


# NEW: customizable "Them" name
if "them_name" not in st.session_state:
   st.session_state.them_name = "Them"


# analysis persistence
if "analysis_df" not in st.session_state:
   st.session_state.analysis_df = None
if "show_analysis" not in st.session_state:
   st.session_state.show_analysis = False
if "trend_metric" not in st.session_state:
   st.session_state.trend_metric = "escalation_risk"


# suggestion persistence
if "suggestion" not in st.session_state:
   st.session_state.suggestion = None
if "suggestion_error" not in st.session_state:
   st.session_state.suggestion_error = None

# If the last message is from Me, we should NOT suggest a next message
last_from_me = bool(st.session_state.messages) and st.session_state.messages[-1]["speaker"] == "Me"




# ---------------------------
# Sidebar: name input
# ---------------------------
with st.sidebar:
   st.subheader("Settings")
   them = st.text_input("Their name (Them)", value=st.session_state.them_name).strip()
   st.session_state.them_name = them if them else "Them"
   st.caption("This only changes labels + prompts. It doesn’t affect scoring models.")




# ---------------------------
# Controls
# ---------------------------
colA, colB = st.columns([1, 1])


with colA:
   if st.button("Reset conversation"):
       st.session_state.messages = []
       st.session_state.next_speaker = "Me"
       st.session_state.analysis_df = None
       st.session_state.show_analysis = False
       st.session_state.suggestion = None
       st.session_state.suggestion_error = None
       st.rerun()


with colB:
   if st.button("Undo last message") and st.session_state.messages:
       st.session_state.messages.pop()
       if not st.session_state.messages:
           st.session_state.next_speaker = "Me"
       else:
           st.session_state.next_speaker = (
               "Them" if st.session_state.messages[-1]["speaker"] == "Me" else "Me"
           )
       st.session_state.analysis_df = None
       st.session_state.show_analysis = False
       st.session_state.suggestion = None
       st.session_state.suggestion_error = None
       st.rerun()


st.divider()




# ---------------------------
# Models (cached)
# ---------------------------
@st.cache_resource
def load_goemotions_pipeline():
   return pipeline(
       "text-classification",
       model="SamLowe/roberta-base-go_emotions",
       top_k=None,
       truncation=True,
       device=-1,  # CPU
   )


@st.cache_resource
def load_toxicity_pipeline():
   return pipeline(
       "text-classification",
       model="unitary/toxic-bert",
       truncation=True,
       device=-1,
   )


@st.cache_data(show_spinner=False)
def goemotions_probs(text: str) -> dict:
   clf = load_goemotions_pipeline()
   out = clf(text)
   items = out[0] if isinstance(out, list) else out
   probs = {}
   for d in items:
       label = d.get("label")
       score = d.get("score", 0.0)
       if isinstance(label, str):
           probs[label] = float(score)
   return probs


@st.cache_data(show_spinner=False)
def toxicity_score(text: str) -> int:
   tox = load_toxicity_pipeline()
   out = tox(text)
   if isinstance(out, list) and out and isinstance(out[0], dict):
       score = float(out[0].get("score", 0.0))
       label = str(out[0].get("label", "")).lower()
       # toxic-bert is binary; normalize so higher always means "more toxic"
       if "non" in label and "toxic" in label:
           score = 1.0 - score
       return int(round(100 * max(0.0, min(1.0, score))))
   return 0




# ---------------------------
# Misunderstanding (A) + Clarification attempt (still useful)
# ---------------------------
MIND_READING = [
   "you meant", "you were trying", "you were tryna", "you did that because",
   "you just wanted", "you only", "you think", "you don't even", "you dont even"
]
ASSUMPTION_STARTERS = ["so you're", "so youre", "so you are", "clearly", "obviously", "i guess"]
OVERGENERAL = ["always", "never", "every time", "as usual"]
CONTEXT_REFERENCES = ["earlier", "before", "last time", "again", "like always", "as usual", "the other day"]
VAGUE_WORDS = ["that", "this", "it", "stuff", "things", "whatever"]


CLARIFYING_PHRASES = [
   "help me understand", "can you help me understand", "what do you mean",
   "can we talk", "can we talk about it", "can we talk about this",
   "i'm trying to understand", "im trying to understand",
   "i might be misunderstanding", "maybe i'm misunderstanding",
   "to be clear", "can you clarify", "clarify", "what happened"
]


INSULT_WORDS = ["crazy", "psycho", "insane", "delusional", "pathetic", "stupid", "dumb"]
DISMISSIVE = ["whatever", "k", "fine", "stop", "idc", "i don't care", "i dont care"]




def misunderstanding_risk_A(text: str) -> int:
   t = text.lower().strip()
   score = 0
   score += 18 * sum(1 for p in MIND_READING if p in t)
   score += 12 * sum(1 for p in ASSUMPTION_STARTERS if p in t)
   score += 10 * sum(1 for p in OVERGENERAL if p in t)
   score += 8 * sum(1 for p in CONTEXT_REFERENCES if p in t)


   tokens = re.findall(r"[a-zA-Z']+", t)
   vague_hits = sum(1 for w in VAGUE_WORDS if re.search(rf"\b{re.escape(w)}\b", t))
   if len(tokens) <= 10 and vague_hits >= 2:
       score += 18
   elif vague_hits >= 5:
       score += 12


   you_count = len(re.findall(r"\byou\b", t))
   if you_count >= 3 and len(tokens) <= 14:
       score += 8


   return max(0, min(100, score))




def clarification_attempt(text: str) -> int:
   t = text.lower().strip()
   score = 0
   score += 25 * sum(1 for p in CLARIFYING_PHRASES if p in t)
   score += min(10, 3 * t.count("?"))
   return max(0, min(100, score))




def escalation_override(text: str) -> int:
   """
   Small deterministic bump for obvious escalation words (helps correct cases like 'you're being crazy').
   """
   t = text.lower().strip()
   bump = 0
   for w in INSULT_WORDS:
       if re.search(rf"\b{re.escape(w)}\b", t):
           bump = max(bump, 50)
   for p in DISMISSIVE:
       if re.search(rf"\b{re.escape(p)}\b", t):
           bump = max(bump, 25)
   return bump




# ---------------------------
# Score + “why” explanation (cached per text)
# ---------------------------
@st.cache_data(show_spinner=False)
def score_and_explain(text: str) -> dict:
   probs = goemotions_probs(text)
   base = scores_from_emotion_probs(probs)  # escalation_risk + empathy_level
   tops = top_emotions(probs, k=3)


   tox = toxicity_score(text)
   mis = misunderstanding_risk_A(text)
   clar = clarification_attempt(text)


   esc = max(int(base.get("escalation_risk", 0)), tox)
   esc = min(100, esc + escalation_override(text))


   emp = int(base.get("empathy_level", 0))


   t = text.lower()
   reasons = []
   if any(w in t for w in OVERGENERAL):
       reasons.append("Overgeneralizing (always/never/as usual)")
   if any(p in t for p in MIND_READING):
       reasons.append("Mind-reading / assuming intent")
   if any(p in t for p in ASSUMPTION_STARTERS):
       reasons.append("Assumption starter (clearly/obviously/so you’re...)")
   if any(re.search(rf"\b{re.escape(w)}\b", t) for w in INSULT_WORDS):
       reasons.append("Name-calling / labeling")
   if any(re.search(rf"\b{re.escape(p)}\b", t) for p in DISMISSIVE):
       reasons.append("Dismissive / shutdown phrase")
   if "!" in text:
       reasons.append("Exclamation intensity")
   if text.count("?") >= 2:
       reasons.append("Multiple question marks")
   if clar >= 40:
       reasons.append("Repair language present (clarifying / de-escalating)")
   if not reasons:
       reasons.append("No strong red flags detected (mostly neutral wording)")


   return {
       "escalation_risk": int(esc),
       "toxicity": int(tox),
       "misunderstanding_risk": int(mis),
       "clarification_attempt": int(clar),
       "empathy_level": int(emp),
       "top_emotions": tops,     # list[(label, prob)]
       "reasons": reasons[:4],   # top reasons only
   }




def tooltip_text_for_message(s: dict) -> str:
   emo_str = ", ".join([f"{e} ({p:.2f})" for e, p in s["top_emotions"]]) if s["top_emotions"] else "—"
   why_str = " • ".join(s["reasons"]) if s["reasons"] else "—"
   return (
       f"Escalation: {s['escalation_risk']}/100\n"
       f"Toxicity: {s['toxicity']}/100\n"
       f"Misunderstanding (A): {s['misunderstanding_risk']}/100\n"
       f"Clarification: {s['clarification_attempt']}/100\n"
       f"Empathy: {s['empathy_level']}/100\n\n"
       f"Top emotions: {emo_str}\n"
       f"Why: {why_str}"
   )




def heat_style(score_0_100: int) -> str:
   v = max(0, min(100, int(score_0_100)))
   a = 0.04 + 0.28 * (v / 100.0)     # alpha
   blur = 2 + int(10 * (v / 100.0))  # blur radius
   border = 0.10 + 0.35 * (v / 100.0)
   return f"box-shadow: 0 0 {blur}px rgba(255, 120, 0, {a}); border-color: rgba(255, 160, 0, {border});"




# ---------------------------
# Heatmap overlay ALWAYS ON (no toggle)
# ---------------------------
st.session_state.setdefault("heatmap_metric", "escalation_risk")
st.session_state.heatmap_metric = st.selectbox(
   "Heatmap metric",
   ["escalation_risk", "toxicity", "misunderstanding_risk", "clarification_attempt", "empathy_level"],
   index=["escalation_risk", "toxicity", "misunderstanding_risk", "clarification_attempt", "empathy_level"].index(
       st.session_state.heatmap_metric
   ),
)


st.divider()




# ---------------------------
# Chat bubble styles
# ---------------------------
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
       transition: box-shadow 150ms ease, border-color 150ms ease, transform 150ms ease;
   }
   .bubble:hover { transform: translateY(-1px); }


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




# ---------------------------
# Render chat messages
# ---------------------------
for m in st.session_state.messages:
   cls = "me" if m["speaker"] == "Me" else "them"
   speaker_label = "Me" if cls == "me" else st.session_state.them_name


   raw_text = m["text"]
   safe_text = html.escape(raw_text)


   s = score_and_explain(raw_text)
   tip_raw = tooltip_text_for_message(s)
   tip_attr = html.escape(tip_raw, quote=True).replace("\n", "&#10;")


   metric = st.session_state.heatmap_metric
   extra_style = heat_style(int(s.get(metric, 0)))
   style_attr = html.escape(extra_style, quote=True)


   st.markdown(
       f"""
       <div class="chat-row {cls}">
         <div>
           <div class="small-label">{speaker_label}</div>
           <div class="bubble {cls}" title="{tip_attr}" style="{style_attr}">{safe_text}</div>
         </div>
       </div>
       """,
       unsafe_allow_html=True,
   )




# ---------------------------
# Input
# ---------------------------
placeholder_name = "Me" if st.session_state.next_speaker == "Me" else st.session_state.them_name
placeholder = f"Type {placeholder_name}'s message and press Enter…"
new_text = st.chat_input(placeholder)


if new_text:
   st.session_state.messages.append({"speaker": st.session_state.next_speaker, "text": new_text.strip()})
   st.session_state.next_speaker = "Them" if st.session_state.next_speaker == "Me" else "Me"
   st.session_state.analysis_df = None
   st.session_state.show_analysis = False
   st.session_state.suggestion = None
   st.session_state.suggestion_error = None
   st.rerun()




# ---------------------------
# Conversation export
# ---------------------------
st.divider()
st.subheader("Conversation (for analysis)")


if st.session_state.messages:
   lines = []
   for m in st.session_state.messages:
       name = "Me" if m["speaker"] == "Me" else st.session_state.them_name
       lines.append(f"{name}: {m['text']}")
   st.code("\n".join(lines))
else:
   st.info("No messages yet. Start typing above.")




# ---------------------------
# Analysis + 3-line trend chart
# ---------------------------
st.divider()
st.subheader("Analysis")


N = st.slider("Analyze last N messages", min_value=5, max_value=60, value=20, step=5)


if st.button("Analyze conversation"):
   if not st.session_state.messages:
       st.warning("Add some messages first.")
   else:
       msgs = st.session_state.messages[-N:]
       rows = []
       with st.spinner("Scoring…"):
           for idx, m in enumerate(msgs, start=1):
               s = score_and_explain(m["text"])
               tops = s["top_emotions"]
               rows.append({
                   "turn": idx,
                   "speaker": ("Me" if m["speaker"] == "Me" else st.session_state.them_name),
                   "text": m["text"],
                   "escalation_risk": s["escalation_risk"],
                   "toxicity": s["toxicity"],
                   "misunderstanding_risk": s["misunderstanding_risk"],
                   "clarification_attempt": s["clarification_attempt"],
                   "empathy_level": s["empathy_level"],
                   "top_emotions": ", ".join([f"{e}({p:.2f})" for e, p in tops]),
               })


       st.session_state.analysis_df = pd.DataFrame(rows)
       st.session_state.show_analysis = True

       # Clear old suggestion if we are waiting on Them
       if last_from_me:
           st.session_state.suggestion = None
           st.session_state.suggestion_error = None

       # ALSO generate suggestion automatically right after analysis (only if Them spoke last)
       if (not IS_CLOUD) and (not last_from_me):
           convo = "\n".join(
               [f'{"Me" if m["speaker"]=="Me" else st.session_state.them_name}: {m["text"]}' for m in st.session_state.messages]
           )
           try:
               st.session_state.suggestion = analyze_conversation_llm(convo, model="llama3.1:8b")
               st.session_state.suggestion_error = None
           except Exception as e:
               st.session_state.suggestion = None
               st.session_state.suggestion_error = str(e)




if st.session_state.show_analysis and st.session_state.analysis_df is not None:
   df = st.session_state.analysis_df


   tail = df.tail(3)
   overall_esc = int(round(tail["escalation_risk"].mean()))
   overall_tox = int(round(tail["toxicity"].mean()))
   overall_mis = int(round(tail["misunderstanding_risk"].mean()))
   overall_clar = int(round(tail["clarification_attempt"].mean()))
   overall_emp = int(round(tail["empathy_level"].mean()))
   overall_risk = int(round(0.45 * overall_esc + 0.30 * overall_mis + 0.25 * overall_tox))


   c1, c2, c3, c4, c5, c6 = st.columns(6)
   c1.metric("Escalation", overall_esc)
   c2.metric("Toxicity", overall_tox)
   c3.metric("Misunderstanding (A)", overall_mis)
   c4.metric("Clarification", overall_clar)
   c5.metric("Empathy", overall_emp)
   c6.metric("Overall Risk", overall_risk)


   metric_choice = st.selectbox(
       "Trend metric",
       ["escalation_risk", "toxicity", "misunderstanding_risk", "clarification_attempt", "empathy_level"],
       index=["escalation_risk", "toxicity", "misunderstanding_risk", "clarification_attempt", "empathy_level"].index(
           st.session_state.trend_metric
       ),
       key="trend_metric",
   )


   chart_df = df[["turn", "speaker", metric_choice]].copy()
   chart_df["Overall"] = chart_df[metric_choice]
   chart_df["Me"] = chart_df[metric_choice].where(chart_df["speaker"] == "Me")
   chart_df[st.session_state.them_name] = chart_df[metric_choice].where(chart_df["speaker"] == st.session_state.them_name)


   chart_df["Me"] = chart_df["Me"].ffill()
   chart_df[st.session_state.them_name] = chart_df[st.session_state.them_name].ffill()


   melted = chart_df.melt(
       id_vars=["turn"],
       value_vars=["Overall", "Me", st.session_state.them_name],
       var_name="series",
       value_name="value",
   ).dropna()


   line = (
       alt.Chart(melted)
       .mark_line()
       .encode(
           x=alt.X("turn:Q", title="Turn"),
           y=alt.Y("value:Q", title=metric_choice, scale=alt.Scale(domain=[0, 100])),
           color=alt.Color("series:N", title=""),
           tooltip=["turn:Q", "series:N", "value:Q"],
       )
   )
   st.altair_chart(line, use_container_width=True)


   st.write("Per-message breakdown:")
   st.dataframe(df, use_container_width=True)




# ---------------------------
# Suggested next message (Me-only) — ALWAYS ON, no button
# ---------------------------
st.divider()
st.subheader("Suggested next message (Me-only)")


if last_from_me:
   st.info("Waiting on a response from " + st.session_state.them_name + " — no suggestion yet.")
elif IS_CLOUD:
   st.info("Local LLM suggestions require running locally (Streamlit Cloud can’t reach your Ollama).")
else:
   if st.session_state.suggestion_error:
       st.error(f"LLM ontology unavailable: {st.session_state.suggestion_error}")
       st.info("No suggestion yet — click Analyze again (or add another message and analyze).")
   elif not st.session_state.suggestion:
       st.info("No suggestion yet — click Analyze conversation to generate an LLM-based suggestion.")
   else:
       out = st.session_state.suggestion
       if out.get("likely_emotions_them"):
           st.write("**Likely emotions (" + st.session_state.them_name + "):** " + ", ".join(out["likely_emotions_them"]))
       st.text_area("Validate MY feeling (Me)", out.get("self_validation_line", ""), height=60)
       if out.get("clarifying_question", ""):
           st.text_area("Clarifying question (Me)", out.get("clarifying_question", ""), height=80)
       st.text_area("Next message to send (Me)", out.get("next_message", ""), height=140)
       st.caption(out.get("why_this_works", ""))