# Architecture - InstantMeet (Clean Room)

## Goals
- Launch tonight with a functional MVP.
- Clean-room build: zero reuse of Alpha Loop code.
- B2B scheduling with payments and team routing.

## Tech Stack
- **Backend:** FastAPI (Python), SQLModel + SQLite (swap to Postgres later), Redis optional for rate limiting/locks.
- **Auth:** JSON Web Tokens (simple) with optional Clerk/Auth0 swap.
- **Payments:** Stripe Checkout (per-booking payment).
- **Email:** Placeholder webhook; wire to Resend/SendGrid.
- **Frontend:** Next.js 14 (App Router), TypeScript, Tailwind + shadcn/ui.
- **Analytics:** PostHog snippet on frontend; API logs via FastAPI logging.
- **Hosting:** Vercel (frontend), Railway/Render (backend + SQLite/Postgres).

## Data Model (MVP)
- **User**: id, email, name, role (organizer/admin).
- **AvailabilitySlot**: id, user_id, day_of_week, start_time, end_time, timezone.
- **MeetingType**: id, user_id, name, price_cents, duration_minutes, description, is_public, team_routing (round_robin/priority).
- **Booking**: id, meeting_type_id, attendee_name, attendee_email, start_time, end_time, status (pending/confirmed/cancelled), payment_status, stripe_session_id.

## API Surface (Backend)
- `GET /health`: health check.
- `POST /auth/signup` / `POST /auth/login`: basic JWT issuance.
- `GET /availability`: list slots for a user.
- `POST /availability`: create/update slots.
- `GET /meeting-types`: list meeting types.
- `POST /meeting-types`: create meeting type.
- `POST /book`: create booking, create Stripe Checkout session (stub), hold slot.
- `POST /webhooks/stripe`: handle payment confirmation (stubbed).
- `GET /bookings`: list bookings for organizer/admin.

## Frontend Pages
- `/` Landing: hero, pricing, CTA to start.
- `/dashboard` (organizer): create meeting type, view bookings, copy booking link.
- `/book/[meetingTypeId]`: attendee flow to pick time, pay, and confirm.

## Directory Layout
```
clone-project/
├── backend/
│   ├── main.py            # FastAPI app entrypoint
│   ├── models.py          # SQLModel models
│   ├── api/
│   │   ├── auth.py        # auth endpoints
+│   │   └── core.py        # booking/availability endpoints
│   ├── requirements.txt   # backend deps
│   └── README.md          # backend setup
├── frontend/
│   ├── package.json
│   ├── next.config.js
│   ├── app/
│   │   ├── page.tsx           # landing
│   │   ├── dashboard/page.tsx # organizer dashboard
│   │   └── book/[id]/page.tsx # booking page
│   ├── app/globals.css
│   └── tsconfig.json
├── docs/
│   ├── PRD.md
│   ├── USER_FLOWS.md
│   ├── ROADMAP.md
│   ├── MARKETING_STRATEGY.md
│   ├── LAUNCH_PLAN.md
│   └── DESIGN_SYSTEM.md
├── scripts/
│   └── deploy_tonight.sh
└── README.md
```

## Security & Isolation
- No Alpha Loop imports, code, or assets.
- New git history; separate directory.
- Environment variables isolated in `.env` under `clone-project/`.

## Tonight Scope vs Later
- **Tonight:** Single-user + simple team routing (round-robin), Stripe checkout stub, SQLite, basic email stub, public booking pages, dashboard list of bookings.
- **Later:** Multi-tenant with Postgres, real email, full Stripe integration, webhooks with signature verification, full team roles, calendar sync (Google/Microsoft), lead enrichment.

