# Launch Readiness Report (Tonight)

## Status
- Backend: FastAPI ready; run `uvicorn main:app --port 8001`.
- Frontend: Next.js ready; `npm run dev` or deploy to Vercel.
- Database: SQLite default; swap `DATABASE_URL` to Postgres for production.
- Payments: Stripe stub; integrate live Checkout + webhook next.
- Analytics: Add PostHog snippet before launch.

## Smoke Tests
- `GET /health` → 200
- `POST /meeting-types` with user_id=1 → 200
- `POST /book` with meeting_type_id → 200
- Frontend: load `/`, `/dashboard`, `/book/1`

## Go-Live Steps
1) Deploy backend to Railway/Render with `DATABASE_URL`.
2) Deploy frontend to Vercel with `NEXT_PUBLIC_BACKEND_URL`.
3) Configure Stripe test key; verify `POST /book` still succeeds.
4) Run smoke tests.
5) Publish Product Hunt + Show HN + social posts.

