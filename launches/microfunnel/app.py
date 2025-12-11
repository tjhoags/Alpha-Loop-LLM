from fastapi import FastAPI

app = FastAPI(title="MicroFunnel Autopilot")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/funnel")
def funnel(doc_url: str, price_a: float = 9.0, price_b: float = 19.0):
    return {
        "landing_url": "https://example.com/funnel/demo",
        "price_test": [price_a, price_b],
    }

