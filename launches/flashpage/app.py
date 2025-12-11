from fastapi import FastAPI

app = FastAPI(title="FlashPage Waitlist")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/signup")
def signup(email: str):
    # In production this would store the lead and trigger referral logic.
    return {"status": "captured", "email": email}


@app.get("/referral/{code}")
def referral(code: str):
    # In production this would validate and credit referral rewards.
    return {"status": "ok", "code": code}


