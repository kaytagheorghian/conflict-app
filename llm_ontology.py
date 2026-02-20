# llm_ontology.py
import json
import requests
from typing import Any, Dict


def safe_json_from_text(text: str) -> Dict[str, Any]:
   """
   Parse JSON even if the model adds extra text.
   Strategy: find first '{' and last '}'.
   """
   text = text.strip()
   try:
       return json.loads(text)
   except Exception:
       pass


   l = text.find("{")
   r = text.rfind("}")
   if l != -1 and r != -1 and r > l:
       return json.loads(text[l:r+1])
   raise ValueError("Model did not return valid JSON.")


def ollama_chat_json(model: str, system: str, user: str, timeout_s: int = 120) -> Dict[str, Any]:
   """
   Uses Ollama OpenAI-compatible endpoint and forces JSON output using response_format.
   """
   r = requests.post(
       "http://localhost:11434/v1/chat/completions",
       json={
           "model": model,
           "messages": [
               {"role": "system", "content": system},
               {"role": "user", "content": user},
           ],
           "temperature": 0.2,
           # IMPORTANT: force JSON
           "response_format": {"type": "json_object"},
       },
       timeout=timeout_s,
   )
   if r.status_code != 200:
       raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:800]}")


   data = r.json()
   content = data["choices"][0]["message"]["content"]
   return safe_json_from_text(content)


def analyze_conversation_llm(
   conversation_text: str,
   model: str = "llama3.1:8b",
   timeout_s: int = 120
) -> Dict[str, Any]:
   """
   Returns a Me-only suggestion JSON.
   """


   system = (
       "You generate the NEXT message for the speaker named 'Me'. "
       "Return ONLY a JSON object. Do not include markdown, backticks, or extra text.\n\n"
       "Hard rules:\n"
       "- Write as ME (first-person).\n"
       "- Do NOT write as Them.\n"
       "- Validate both people's feelings and perspectives.\n"
       "- Keep next_message short like real texting (1–3 short lines).\n"
       "- Do NOT insult, threaten, or escalate.\n"
       "- If an insult was already said by ME, ALWAYS apologize and de-escalate in the next message.\n"
       "- If they told Me to stop texting / leave them alone, the best response is to respect the boundary.\n"
       "- Replace always and never with specific instances and prompt ME to fill in those instances.\n"
   )


   user = f"""
Return a JSON object with EXACT keys:
- likely_emotions_them (array of 1–3 strings)
- self_validation_line (string)
- clarifying_question (string)
- next_message (string)
- why_this_works (string)


Conversation:
{conversation_text}
""".strip()


   out = ollama_chat_json(model=model, system=system, user=user, timeout_s=timeout_s)


   # Ensure keys exist
   out.setdefault("likely_emotions_them", [])
   out.setdefault("self_validation_line", "")
   out.setdefault("clarifying_question", "")
   out.setdefault("next_message", "")
   out.setdefault("why_this_works", "")
   return out

