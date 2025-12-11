# InstantMeet Backend (FastAPI)

Clean-room backend for scheduling + payments + team routing.

## Run locally
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

## Environment
- `DATABASE_URL` (default: `sqlite:///./instantmeet.db`)
- `JWT_SECRET` (optional; not required for stub token issuance)
- `STRIPE_SECRET_KEY` (optional; stubbed in MVP)

## Notes
- SQLite by default; swap `DATABASE_URL` to Postgres for production.
- Stripe/webhooks are stubbed; integrate real payment + signature verification later.
- DB schema uses SQLModel; tables are auto-created on startup.

