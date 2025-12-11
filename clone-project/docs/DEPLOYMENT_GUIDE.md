# Deployment Guide (Tonight)

## Backend (Railway/Render)
1. Set `DATABASE_URL` (SQLite file or Postgres URL).
2. Command: `uvicorn main:app --host 0.0.0.0 --port 8000`.
3. Expose HTTP port; enable health check on `/health`.

## Frontend (Vercel)
1. Set env: `NEXT_PUBLIC_BACKEND_URL=https://your-backend.example.com`.
2. `npm install && npm run build`.
3. Deploy project; ensure `/book/[id]` and `/dashboard` are enabled.

## Domains
- Temporary: use Vercel default domain for tonight.
- Later: map custom domain via Cloudflare/Namecheap.

## Stripe (stub tonight)
- Create test key; set `STRIPE_SECRET_KEY`.
- Replace stub in backend `/book` with Checkout session creation; add webhook.

## Email (stub tonight)
- Use console logs; later integrate Resend/SendGrid.

