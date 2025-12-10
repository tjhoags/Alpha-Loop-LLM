from typing import List

import numpy as np
from loguru import logger
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

_pipeline_cache = None


def _get_pipeline():
    global _pipeline_cache
    if _pipeline_cache is None:
        model_name = "ProsusAI/finbert"
        logger.info(f"Loading sentiment model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        _pipeline_cache = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    return _pipeline_cache


def score_texts(texts: List[str]) -> List[float]:
    """Returns sentiment scores in [-1,1], where positive=1, negative=-1.
    """
    if not texts:
        return []
    clf = _get_pipeline()
    outputs = clf(texts)
    scores = []
    for out in outputs:
        label = out["label"].upper()
        score = float(out["score"])
        if label == "POSITIVE":
            scores.append(score)
        elif label == "NEGATIVE":
            scores.append(-score)
        else:
            scores.append(0.0)
    return scores


def aggregate_sentiment(texts: List[str]) -> float:
    scores = score_texts(texts)
    if not scores:
        return 0.0
    return float(np.mean(scores))

