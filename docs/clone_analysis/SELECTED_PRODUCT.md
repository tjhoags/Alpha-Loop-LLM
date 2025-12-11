# Selected Product to Clone: InstantMeet

## Summary
- **Product:** InstantMeet — B2B scheduling with payments and team routing
- **Why:** Fastest to build tonight, immediate monetization via Stripe, strong product-led growth via shareable links and embeddable widgets.
- **Target Users:** Agencies, consultants, sales teams, customer success teams.
- **Revenue Model:** SaaS subscription (Starter/Pro/Team) + per-transaction Stripe fees passthrough; optional usage-based overages on booked meetings.

## Scoring (1-10, weighted)
- Build Speed (30%): 9
- Revenue Potential (25%): 8
- Marketing Potential (20%): 8
- Market Size (15%): 7
- Defensibility (10%): 6
- **Composite Score:** 7.85

## Differentiation Angle
- Payments-first scheduling (collect payment before confirming).
- Team routing (round-robin / priority routing) baked in from day one.
- Lead capture + enrichment on every booking.
- Embeddable widget for websites and proposals.

## Monetization
- **Starter ($19/mo):** 1 calendar, 1 payment flow, basic branding.
- **Pro ($49/mo):** Team routing, advanced branding, webhooks, CRM export.
- **Team ($99/mo):** Multiple workspaces, audit logs, priority support.

## Launch Tonight Scope (MVP)
- Host public booking pages with slots pulled from static availability (config file/DB).
- Accept Stripe checkout for paid meetings (single price per link).
- Create bookings, send confirmation email (stub), and store in SQLite.
- Admin dashboard (lightweight) to view bookings and toggle availability.

## Go-to-Market (World-Class Marketing)
- Product Hunt launch with “payments-first scheduling” angle.
- Templates for agencies/consultants to share instantly.
- SEO landing focused on “paid discovery calls” and “team round robin scheduler”.
- Viral loop: booking confirmation includes referral link for the organizer to invite teammates.

## Zero Alpha Loop Code
- Clean-room implementation in `clone-project/` with new git history.
- No imports or reuse from `tjhoags/alpha-loop-llm`.

## Next Steps
1. Finalize architecture (backend FastAPI + frontend Next.js).
2. Implement core features and deploy.
3. Execute launch plan (PH, HN, social, email).

