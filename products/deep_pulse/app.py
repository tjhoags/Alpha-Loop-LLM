from fastapi import FastAPI, Query
from typing import Dict

from .sentiment import analyze_text

app = FastAPI(title="DeepPulse", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok", "service": "deep_pulse"}


@app.get("/sentiment")
def sentiment(topic: str = Query(..., description="Topic, ticker, or phrase")) -> Dict:
    scores = analyze_text(topic)
    return {"topic": topic, "scores": scores}


