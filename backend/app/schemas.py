"""API request/response schemas."""
from datetime import datetime

from pydantic import BaseModel


class DocumentOut(BaseModel):
    id: int
    pdf_name: str
    status: str
    scope1_value: str | None = None
    scope1_unit: str | None = None
    scope2_value: str | None = None
    scope2_unit: str | None = None
    scope3_value: str | None = None
    scope3_unit: str | None = None
    reporting_year: str | None = None
    error: str | None = None
    experiment_tag: str
    trace_id: str | None = None
    created_at: datetime | None = None

    class Config:
        from_attributes = True


class TraceOut(BaseModel):
    id: str
    document_id: int | None
    pdf_name: str
    experiment_tag: str
    prompt_version: str
    model: str
    retrieval_k: int
    chunk_size: int
    chunk_overlap: int
    queries: list | None
    retrieved_chunks: list | None
    prompt: str | None
    raw_output: str | None
    parsed_output: dict | None
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int
    status: str
    error: str | None
    created_at: datetime | None

    class Config:
        from_attributes = True


class ExperimentOut(BaseModel):
    tag: str
    config: dict
    metrics: dict
    notes: str | None
    created_at: datetime | None

    class Config:
        from_attributes = True


class EvaluateRequest(BaseModel):
    experiment_tag: str = "live"
