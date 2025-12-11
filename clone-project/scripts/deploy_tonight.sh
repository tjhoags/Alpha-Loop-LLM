#!/usr/bin/env bash
set -e

echo "Starting tonight deploy (placeholder commands)."

echo "1) Backend → Railway/Render"
echo "   - Set DATABASE_URL"
echo "   - Command: uvicorn main:app --host 0.0.0.0 --port 8000"

echo "2) Frontend → Vercel"
echo "   - NEXT_PUBLIC_BACKEND_URL=https://your-backend.example.com"
echo "   - npm install && npm run build"

echo "3) Stripe (stub tonight)"
echo "   - Set STRIPE_SECRET_KEY (test)"

echo "4) Smoke tests"
echo "   - GET /health"
echo "   - POST /meeting-types"
echo "   - POST /book"

echo "Deploy script is a checklist; hook into CI/CD as needed."

