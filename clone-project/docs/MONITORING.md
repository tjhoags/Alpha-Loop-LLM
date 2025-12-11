# Monitoring & Metrics

## Frontend
- Add PostHog snippet with project key.
- Track events:
  - `landing_view`
  - `cta_start_free`
  - `booking_started`
  - `booking_completed`

## Backend
- Enable access logs in uvicorn.
- Log booking creation and meeting type creation.

## KPIs Tonight
- Visitors → booking page views
- Booking attempts (`POST /book`)
- Meeting types created
- Pricing page clicks

## Alerts (later)
- Error rate on backend endpoints
- Stripe webhook failures
- Booking drop-off >30%

