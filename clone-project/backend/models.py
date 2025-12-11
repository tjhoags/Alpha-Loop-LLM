from datetime import datetime, time
from typing import Optional

from sqlmodel import Field, SQLModel, create_engine, Session, select


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    name: str
    hashed_password: str
    role: str = Field(default="organizer")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AvailabilitySlot(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    day_of_week: int  # 0=Mon
    start_time: time
    end_time: time
    timezone: str = "UTC"
    created_at: datetime = Field(default_factory=datetime.utcnow)


class MeetingType(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id", index=True)
    name: str
    description: Optional[str] = ""
    price_cents: int = 0
    duration_minutes: int = 30
    is_public: bool = True
    team_routing: str = "solo"  # solo | round_robin | priority
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Booking(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    meeting_type_id: int = Field(foreign_key="meetingtype.id", index=True)
    attendee_name: str
    attendee_email: str
    start_time: datetime
    end_time: datetime
    status: str = "pending"  # pending|confirmed|cancelled
    payment_status: str = "unpaid"  # unpaid|paid
    stripe_session_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


def get_session(db_url: str):
    engine = create_engine(db_url, echo=False)
    SQLModel.metadata.create_all(engine)
    return Session(engine)

