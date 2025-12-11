import os
from functools import partial

from fastapi import FastAPI, Depends
from sqlmodel import Session, create_engine, SQLModel

from .models import User, AvailabilitySlot, MeetingType, Booking
from .api import auth, core

DB_URL = os.getenv("DATABASE_URL", "sqlite:///./instantmeet.db")

engine = create_engine(DB_URL, echo=False, connect_args={"check_same_thread": False} if DB_URL.startswith("sqlite") else {})


def get_session():
    with Session(engine) as session:
        yield session


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


app = FastAPI(title="InstantMeet API", version="0.1.0")

# Inject DB dependency into routers
auth.router.dependencies = [Depends(get_session)]
core.router.dependencies = [Depends(get_session)]

app.include_router(auth.router)
app.include_router(core.router)


@app.on_event("startup")
def on_startup():
    create_db_and_tables()


@app.get("/")
def root():
    return {"service": "instantmeet-api", "status": "ok"}

