from typing import Dict

POSITIVE = {"great", "strong", "bull", "beat", "up", "growth", "win"}
NEGATIVE = {"weak", "down", "loss", "miss", "bear", "risk", "fail"}


def analyze_text(text: str) -> Dict[str, float]:
    tokens = text.lower().split()
    pos = sum(token in POSITIVE for token in tokens)
    neg = sum(token in NEGATIVE for token in tokens)
    total = max(len(tokens), 1)
    score = (pos - neg) / total
    return {"positive": pos / total, "negative": neg / total, "score": score}


