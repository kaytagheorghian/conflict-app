import re

# GoEmotions taxonomy (27 emotions + neutral)
GOEMOTIONS_TAXONOMY = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Signals for scoring
MIND_READING = [
    "you think", "you just", "you only", "you wanted", "you meant to",
    "you were trying to", "you did that because", "you don't even", "you dont even"
]
VAGUE = ["whatever", "stuff", "things", "that", "this", "it"]
ASSUMPTION_STARTERS = ["so you're", "so youre", "so you are", "i guess", "clearly", "obviously"]
NEEDS_CONTEXT = ["earlier", "before", "last time", "again", "as usual"]

EMPATHY = [
    "i get why", "i get it", "that makes sense", "i understand",
    "i hear you", "i'm sorry", "im sorry", "my bad", "thanks for telling me",
    "i appreciate", "can you help me understand", "help me understand", "what do you mean",
    "thank you", "thanks", "i see", "fair", "you're right", "youre right"
]
I_STATEMENTS = ["i feel", "i felt", "i was worried", "i'm confused", "im confused", "i might be wrong", "maybe i'm"]

ABSOLUTES = ["always", "never", "everyone", "no one", "nobody", "nothing"]
ACCUSATIONS = [
    "you lied", "youre lying", "you are lying", "you don't care", "you dont care",
    "you hate", "you hate me", "you did this", "you did that", "you always", "you never",
]
DISMISSIVE = ["k", "ok.", "whatever", "fine", "sure", "lol", "lmao", "bruh"]
INSULTS = ["crazy", "stupid", "pathetic", "weird", "annoying", "toxic", "idiot", "dumb"]

def normalize(text: str) -> str:
    """
    Light normalization to help texting shorthand.
    This is NOT spellcheck; it's just robustifying common chat forms.
    """
    t = text.lower()

    # common shortcuts
    t = re.sub(r"\bu\b", "you", t)
    t = re.sub(r"\bur\b", "your", t)
    t = re.sub(r"\br\b", "are", t)
    t = re.sub(r"\bim\b", "i'm", t)
    t = re.sub(r"\bidk\b", "i don't know", t)
    t = re.sub(r"\bomw\b", "on my way", t)
    t = re.sub(r"\bwyd\b", "what are you doing", t)
    t = re.sub(r"\bwtf\b", "what the", t)

    # reduce stretched letters: soooo -> soo
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)

    return t

def escalation_score(text: str) -> int:
    raw = text.strip()
    t = normalize(raw).strip()

    exclam = t.count("!")
    qmarks = t.count("?")
    allcaps = sum(1 for w in re.findall(r"[A-Za-z]+", raw) if len(w) >= 4 and w.isupper())

    abs_hits = sum(1 for p in ABSOLUTES if p in t)
    acc_hits = sum(1 for p in ACCUSATIONS if p in t)
    dis_hits = sum(1 for p in DISMISSIVE if re.search(rf"\b{re.escape(p)}\b", t))
    ins_hits = sum(1 for p in INSULTS if re.search(rf"\b{re.escape(p)}\b", t))

    you_count = len(re.findall(r"\byou\b", t))
    i_count = len(re.findall(r"\bi\b", t))

    score = 0
    score += 6 * acc_hits
    score += 4 * ins_hits
    score += 3 * abs_hits
    score += 2 * dis_hits
    score += 1 * min(exclam, 3)
    score += 1 * min(qmarks, 4)
    score += 2 * min(allcaps, 3)
    score += max(0, you_count - i_count)

    return max(0, min(100, score * 5))

def misunderstanding_score(text: str) -> int:
    t = normalize(text).strip()

    score = 0
    score += 6 * sum(1 for p in MIND_READING if p in t)
    score += 4 * sum(1 for p in ASSUMPTION_STARTERS if p in t)
    score += 2 * sum(1 for p in NEEDS_CONTEXT if re.search(rf"\b{re.escape(p)}\b", t))
    score += min(t.count("?"), 4)

    vague_hits = sum(1 for p in VAGUE if re.search(rf"\b{re.escape(p)}\b", t))
    score += 1 if vague_hits >= 6 else 0

    return max(0, min(100, score * 6))

def empathy_score(text: str) -> int:
    """
    Baseline + boosts for empathy cues and "I" framing.
    Penalize heavy blame language.
    """
    t = normalize(text).strip()

    score = 20  # baseline so it's not always 0

    score += 12 * sum(1 for p in EMPATHY if p in t)
    score += 8 * sum(1 for p in I_STATEMENTS if p in t)

    you_count = len(re.findall(r"\byou\b", t))
    i_count = len(re.findall(r"\bi\b", t))
    if you_count >= 3 and i_count == 0:
        score -= 15

    # curiosity bonus (small)
    score += min(t.count("?"), 3) * 2

    return max(0, min(100, score))