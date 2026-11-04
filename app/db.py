"""Task 5 — the database trail. One row per request: what was decided, when, by which model,
how confidently. Rejected uploads are recorded too (status != 'ok') so the trail is complete."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from sqlalchemy import (
    DateTime, Float, Integer, String, create_engine, func, select, text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

log = logging.getLogger("visionguard.db")


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    status: Mapped[str] = mapped_column(String(20), default="ok", index=True)  # ok | rejected | error
    filename: Mapped[str] = mapped_column(String(128))
    image_sha256: Mapped[str | None] = mapped_column(String(64), index=True)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    size_kb: Mapped[float | None] = mapped_column(Float)
    label: Mapped[str | None] = mapped_column(String(20), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    prob_ai: Mapped[float | None] = mapped_column(Float)
    threshold: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(32))
    inference_mode: Mapped[str] = mapped_column(String(10))
    latency_ms: Mapped[float | None] = mapped_column(Float)
    error: Mapped[str | None] = mapped_column(String(255))


class Database:
    def __init__(self, url: str, retries: int = 10):
        kw = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kw["connect_args"] = {"check_same_thread": False}
        self.engine = create_engine(url, **kw)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)
        for attempt in range(1, retries + 1):  # Compose waits for health, this is belt+braces
            try:
                Base.metadata.create_all(self.engine)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == retries:
                    raise
                log.warning("database not ready (%s), retry %d/%d", exc.__class__.__name__,
                            attempt, retries)
                time.sleep(2)
        log.info("database ready: %s", self.engine.url.render_as_string(hide_password=True))

    def add(self, **fields) -> PredictionRecord:
        with self.Session.begin() as s:
            rec = PredictionRecord(**fields)
            s.add(rec)
        return rec

    def get(self, rec_id: int) -> PredictionRecord | None:
        with self.Session() as s:
            return s.get(PredictionRecord, rec_id)

    def page(self, page: int, size: int, label: str | None = None, include_rejected=False):
        with self.Session() as s:
            q = select(PredictionRecord)
            if not include_rejected:
                q = q.where(PredictionRecord.status == "ok")
            if label:
                q = q.where(PredictionRecord.label == label)
            total = s.scalar(select(func.count()).select_from(q.subquery()))
            rows = s.scalars(q.order_by(PredictionRecord.created_at.desc(),
                                        PredictionRecord.id.desc())
                             .offset((page - 1) * size).limit(size)).all()
            return rows, total

    def stats(self) -> dict:
        with self.Session() as s:
            rows = s.execute(select(PredictionRecord.label, func.count(),
                                    func.avg(PredictionRecord.confidence))
                             .where(PredictionRecord.status == "ok")
                             .group_by(PredictionRecord.label)).all()
            rejected = s.scalar(select(func.count()).where(PredictionRecord.status != "ok"))
        return {"by_label": {r[0]: {"count": r[1], "avg_conf": r[2]} for r in rows},
                "rejected": rejected or 0}

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001
            return False
