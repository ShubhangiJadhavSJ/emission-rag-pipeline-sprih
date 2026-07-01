"""Automatic schema creation + ground-truth seeding.

Called from the FastAPI lifespan handler so `docker compose up` is the only
step needed — no manual migrations. Safe to run repeatedly (idempotent).
"""
import json
import logging
import os
import time

from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from app.config import settings
from app.db.database import Base, engine

logger = logging.getLogger(__name__)


def _wait_for_db(max_tries: int = 30, delay: float = 2.0) -> None:
    """Block until MariaDB accepts connections (compose healthcheck already
    gates this, but we retry defensively for local/non-compose runs)."""
    for attempt in range(1, max_tries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return
        except OperationalError:
            logger.info("Waiting for database... (%d/%d)", attempt, max_tries)
            time.sleep(delay)
    raise RuntimeError("Database did not become available in time")


def init_db() -> None:
    _wait_for_db()
    # Import models so they register on the metadata before create_all.
    from app.db import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    logger.info("Database tables ensured.")


def ground_truth_count() -> int:
    path = settings.ground_truth_path
    if not os.path.exists(path):
        logger.warning("Ground-truth file not found at %s", path)
        return 0
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    return len(data)
