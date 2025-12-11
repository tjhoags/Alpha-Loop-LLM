# API Reference (MVP)

Base URL: `http://localhost:8001`

## Auth
- `POST /auth/signup` — form data: `email`, `name`, `password`
- `POST /auth/login` — form data: `email`, `password`

## Availability
- `GET /availability?user_id={id}`
- `POST /availability` — form data: `user_id`, `day_of_week`, `start_time` (HH:MM), `end_time` (HH:MM), `timezone`

## Meeting Types
- `GET /meeting-types?user_id={id}`
- `POST /meeting-types` — form data: `user_id`, `name`, `description?`, `price_cents`, `duration_minutes`, `team_routing`

## Booking
- `POST /book` — form data: `meeting_type_id`, `attendee_name`, `attendee_email`, `start_time` (ISO)
- `GET /bookings?user_id={id}`

## Health
- `GET /health`

