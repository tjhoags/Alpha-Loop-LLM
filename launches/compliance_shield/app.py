from fastapi import FastAPI

app = FastAPI(title="Creator Compliance Shield")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/scan")
def scan(text: str):
    lower = text.lower()
    keywords = ["sponsor", "affiliate", "ai-generated", "promo code"]
    needs_disclosure = any(k in lower for k in keywords)
    banner = "Ad/AI Disclosure Required" if needs_disclosure else None
    return {"needs_disclosure": needs_disclosure, "banner": banner}

