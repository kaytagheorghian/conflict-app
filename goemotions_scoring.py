# goemotions_scoring.py
from typing import Dict, List, Tuple


# Reference taxonomy (GoEmotions has 27 + neutral)
GOEMOTIONS_TAXONOMY = [
   "admiration", "amusement", "anger", "annoyance", "approval", "caring",
   "confusion", "curiosity", "desire", "disappointment", "disapproval",
   "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
   "joy", "love", "nervousness", "optimism", "pride", "realization",
   "relief", "remorse", "sadness", "surprise", "neutral"
]


def top_emotions(probs: Dict[str, float], k: int = 3) -> List[Tuple[str, float]]:
   items = [(e, float(probs.get(e, 0.0))) for e in GOEMOTIONS_TAXONOMY]
   items.sort(key=lambda x: x[1], reverse=True)
   return items[:k]


def emotion_vector_from_probs(probs: Dict[str, float]) -> Dict[str, float]:
   def g(name: str) -> float:
       return float(probs.get(name, 0.0))


   vec = {
       "anger": g("anger"),
       "annoyance": g("annoyance"),
       "sadness": g("sadness") + 0.5 * g("disappointment"),
       "anxiety": g("nervousness") + 0.5 * g("fear"),
       "defensiveness": g("disapproval") + 0.5 * g("annoyance"),
       "hurt": g("sadness") + 0.5 * g("remorse") + 0.25 * g("disappointment"),
       "calm_positive": g("approval") + g("gratitude") + 0.5 * g("optimism"),
       "confusion": g("confusion"),
   }


   for k2 in list(vec.keys()):
       vec[k2] = max(0.0, min(1.0, float(vec[k2])))
   return vec


def scores_from_emotion_probs(probs: Dict[str, float]) -> Dict[str, int]:
   """
   Convert GoEmotions probabilities into 0–100 scores:
   - escalation_risk: anger/annoyance/disapproval/disgust/contempt-like mix
   - empathy_level: caring/gratitude/approval/love + remorse
   """
   def g(name: str) -> float:
       return float(probs.get(name, 0.0))


   # Escalation: weight conflict-heavy emotions
   esc = (
       1.00 * g("anger") +
       0.90 * g("annoyance") +
       0.85 * g("disapproval") +
       0.65 * g("disgust") +
       0.50 * g("sadness") +
       0.35 * g("fear")
   )
   esc = max(0.0, min(1.0, esc))


   # Empathy: weight prosocial/repair emotions
   emp = (
       1.00 * g("caring") +
       0.80 * g("gratitude") +
       0.70 * g("approval") +
       0.60 * g("love") +
       0.45 * g("remorse") +
       0.35 * g("optimism")
   )
   emp = max(0.0, min(1.0, emp))


   return {
       "escalation_risk": int(round(100 * esc)),
       "empathy_level": int(round(100 * emp)),
   }

