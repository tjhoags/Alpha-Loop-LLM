from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel import Session, select

from ..models import AvailabilitySlot, Booking, MeetingType

router = APIRouter(prefix="", tags=["core"])


def get_db(session: Session = Depends()):
    # Placeholder dependency; will be injected from main with partial
    return session


@router.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# Availability
@router.get("/availability", response_model=List[AvailabilitySlot])
def list_availability(user_id: int, db: Session = Depends(get_db)):
    return db.exec(select(AvailabilitySlot).where(AvailabilitySlot.user_id == user_id)).all()


@router.post("/availability", response_model=AvailabilitySlot)
def upsert_availability(
    user_id: int,
    day_of_week: int,
    start_time: str,
    end_time: str,
    timezone: str = "UTC",
    db: Session = Depends(get_db),
):
    slot = AvailabilitySlot(
        user_id=user_id,
        day_of_week=day_of_week,
        start_time=datetime.strptime(start_time, "%H:%M").time(),
        end_time=datetime.strptime(end_time, "%H:%M").time(),
        timezone=timezone,
    )
    db.add(slot)
    db.commit()
    db.refresh(slot)
    return slot


# Meeting Types
@router.get("/meeting-types", response_model=List[MeetingType])
def list_meeting_types(user_id: int, db: Session = Depends(get_db)):
    return db.exec(select(MeetingType).where(MeetingType.user_id == user_id)).all()


@router.post("/meeting-types", response_model=MeetingType)
def create_meeting_type(
    user_id: int,
    name: str,
    description: Optional[str] = "",
    price_cents: int = 0,
    duration_minutes: int = 30,
    team_routing: str = "solo",
    db: Session = Depends(get_db),
):
    mt = MeetingType(
        user_id=user_id,
        name=name,
        description=description,
        price_cents=price_cents,
        duration_minutes=duration_minutes,
        team_routing=team_routing,
    )
    db.add(mt)
    db.commit()
    db.refresh(mt)
    return mt


# Booking
@router.post("/book", response_model=Booking)
def book(
    meeting_type_id: int,
    attendee_name: str,
    attendee_email: str,
    start_time: str,
    db: Session = Depends(get_db),
):
    mt = db.get(MeetingType, meeting_type_id)
    if not mt:
        raise HTTPException(status_code=404, detail="Meeting type not found")

    start_dt = datetime.fromisoformat(start_time)
    end_dt = start_dt + timedelta(minutes=mt.duration_minutes)

    booking = Booking(
        meeting_type_id=meeting_type_id,
        attendee_name=attendee_name,
        attendee_email=attendee_email,
        start_time=start_dt,
        end_time=end_dt,
        status="pending" if mt.price_cents > 0 else "confirmed",
        payment_status="unpaid" if mt.price_cents > 0 else "free",
    )
    db.add(booking)
    db.commit()
    db.refresh(booking)

    # Placeholder: integrate Stripe Checkout here; return session_id if needed
    return booking


@router.get("/bookings", response_model=List[Booking])
def list_bookings(user_id: int, db: Session = Depends(get_db)):
    # list bookings for meeting types owned by user
    m_ids = db.exec(select(MeetingType.id).where(MeetingType.user_id == user_id)).all()
    return db.exec(select(Booking).where(Booking.meeting_type_id.in_(m_ids))).all()

