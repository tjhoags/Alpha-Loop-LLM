# InstantMeet (Clean Room)

Clean-room B2B scheduling with payments and team routing. No Alpha Loop code or dependencies.

## Quick Start

```bash
# Backend
cd backend
python -m venv .venv
source .venv/Scripts/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8001

# Frontend
cd ../frontend
npm install
npm run dev -- --port 3000
```

## Environment Variables (examples)
- `STRIPE_SECRET_KEY`: for checkout sessions (stub OK for dev).
- `JWT_SECRET`: token signing secret.
- `FRONTEND_URL`: e.g., http://localhost:3000
- `BACKEND_URL`: e.g., http://localhost:8001

## Deployment (tonight)
- Backend: Railway/Render (`backend/requirements.txt`, `uvicorn main:app`).
- Frontend: Vercel (Next.js App Router).
- Database: SQLite for MVP; swap to Postgres when ready.

## Zero Alpha Loop Code
- New codebase in `clone-project/`.
- No imports or reuse from `tjhoags/alpha-loop-llm`.

