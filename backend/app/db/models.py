"""ORM models.

Four tables map to the four concerns the assignment asks us to persist:

  documents     — one row per uploaded PDF (this is the frontend table).
  traces        — one row per extraction run (observability).
  evaluations   — per-PDF prediction vs ground truth, per experiment.
  experiments   — aggregate metrics for a tagged experiment (trend report).
"""
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pdf_name: Mapped[str] = mapped_column(String(512), index=True)
    blob_path: Mapped[str] = mapped_column(String(1024))
    # processing | done | error
    status: Mapped[str] = mapped_column(String(32), default="processing", index=True)

    # Extracted values. Stored as strings so we can keep original formatting
    # ("12,500") and the literal "NA"; the unit lives in its own column.
    scope1_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scope1_unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    scope2_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scope2_unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    scope3_value: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scope3_unit: Mapped[str | None] = mapped_column(String(32), nullable=True)
    reporting_year: Mapped[str | None] = mapped_column(String(16), nullable=True)

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    experiment_tag: Mapped[str] = mapped_column(String(128), default="live", index=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    traces: Mapped[list["Trace"]] = relationship(back_populates="document")


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # uuid hex
    document_id: Mapped[int | None] = mapped_column(
        ForeignKey("documents.id"), nullable=True, index=True
    )
    pdf_name: Mapped[str] = mapped_column(String(512))
    experiment_tag: Mapped[str] = mapped_column(String(128), default="live", index=True)

    # Pipeline configuration captured for reproducibility.
    prompt_version: Mapped[str] = mapped_column(String(16))
    model: Mapped[str] = mapped_column(String(128))
    retrieval_k: Mapped[int] = mapped_column(Integer)
    chunk_size: Mapped[int] = mapped_column(Integer)
    chunk_overlap: Mapped[int] = mapped_column(Integer)

    # What actually happened, so a failed run is debuggable end to end.
    queries: Mapped[list | None] = mapped_column(JSON, nullable=True)
    retrieved_chunks: Mapped[list | None] = mapped_column(JSON, nullable=True)
    prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_output: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_output: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)

    status: Mapped[str] = mapped_column(String(32), default="ok")  # ok | error
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    document: Mapped["Document"] = relationship(back_populates="traces")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_tag: Mapped[str] = mapped_column(String(128), index=True)
    pdf_name: Mapped[str] = mapped_column(String(512), index=True)

    predicted: Mapped[dict] = mapped_column(JSON)
    ground_truth: Mapped[dict] = mapped_column(JSON)
    field_scores: Mapped[dict] = mapped_column(JSON)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    tag: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    config: Mapped[dict] = mapped_column(JSON)
    metrics: Mapped[dict] = mapped_column(JSON)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
