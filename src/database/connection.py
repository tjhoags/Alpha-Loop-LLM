from functools import lru_cache

from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config.settings import get_settings


@lru_cache
def get_engine() -> Engine:
    settings = get_settings()
    url = settings.sqlalchemy_url
    logger.info(f"Creating SQLAlchemy engine for {url}")
    engine = create_engine(url, pool_pre_ping=True, pool_recycle=3600)
    return engine


def healthcheck() -> bool:
    """Simple connectivity check."""
    engine = get_engine()
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connectivity check passed.")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Database connectivity check failed: {exc}")
        return False


