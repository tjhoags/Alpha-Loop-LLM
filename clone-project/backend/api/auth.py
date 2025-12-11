from datetime import datetime, timedelta
import secrets
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from passlib.context import CryptContext
from sqlmodel import Session, select

from ..models import User

router = APIRouter(prefix="/auth", tags=["auth"])

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
TOKEN_TTL_MINUTES = 60 * 24


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)


def issue_token() -> str:
    return secrets.token_hex(32)


def get_db(session: Session = Depends()):
    # Placeholder dependency; will be injected from main with partial
    return session


@router.post("/signup")
def signup(email: str, name: str, password: str, db: Session = Depends(get_db)):
    existing = db.exec(select(User).where(User.email == email)).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(email=email, name=name, hashed_password=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return {"user_id": user.id, "token": issue_token(), "expires_in_minutes": TOKEN_TTL_MINUTES}


@router.post("/login")
def login(email: str, password: str, db: Session = Depends(get_db)):
    user = db.exec(select(User).where(User.email == email)).first()
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"user_id": user.id, "token": issue_token(), "expires_in_minutes": TOKEN_TTL_MINUTES}

