# Diffuser (Local LLM)

A Streamlit app that:
- lets a user enter a conversation (Me ↔ Them),
- computes interpretable scores:
  - Escalation Risk
  - Misunderstanding Risk
  - Empathy Score
- generates a de-escalating next message using a local LLM via Ollama.

## Setup

```bash
pip install -r requirements.txt