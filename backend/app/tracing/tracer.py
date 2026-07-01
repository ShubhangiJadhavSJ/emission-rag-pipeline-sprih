"""Per-upload tracing.

Every upload produces exactly one trace. The trace is ALWAYS persisted to the
`traces` table (so observability works with zero external setup and is viewable
in the UI), and ADDITIONALLY mirrored to Langfuse when keys are configured.

A trace captures everything needed to debug a failed extraction: the retrieval
queries, the retrieved chunks (with scores), the exact prompt, model, token
counts, cost, latency, the raw model output, and the final parsed result.
"""
import logging
import time
import uuid

from app.config import settings
from app.db.models import Trace

logger = logging.getLogger(__name__)


class UploadTrace:
    def __init__(self, pdf_name: str, experiment_tag: str, config: dict):
        self.id = uuid.uuid4().hex
        self.pdf_name = pdf_name
        self.experiment_tag = experiment_tag
        self.config = config
        self._start = time.perf_counter()
        self.data: dict = {
            "queries": None,
            "retrieved_chunks": None,
            "prompt": None,
            "raw_output": None,
            "parsed_output": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
            "used_ocr": False,
            "status": "ok",
            "error": None,
        }

    def set(self, **kwargs) -> None:
        self.data.update(kwargs)

    def fail(self, error: str) -> None:
        self.data["status"] = "error"
        self.data["error"] = error

    def to_model(self, document_id: int | None) -> Trace:
        latency_ms = int((time.perf_counter() - self._start) * 1000)
        return Trace(
            id=self.id,
            document_id=document_id,
            pdf_name=self.pdf_name,
            experiment_tag=self.experiment_tag,
            prompt_version=self.config.get("prompt_version", ""),
            model=self.config.get("model", ""),
            retrieval_k=self.config.get("retrieval_k", 0),
            chunk_size=self.config.get("chunk_size", 0),
            chunk_overlap=self.config.get("chunk_overlap", 0),
            queries=self.data["queries"],
            retrieved_chunks=self.data["retrieved_chunks"],
            prompt=self.data["prompt"],
            raw_output=self.data["raw_output"],
            parsed_output=self.data["parsed_output"],
            input_tokens=self.data["input_tokens"],
            output_tokens=self.data["output_tokens"],
            cost_usd=self.data["cost_usd"],
            latency_ms=latency_ms,
            status=self.data["status"],
            error=self.data["error"],
        )

    def send_to_langfuse(self) -> None:
        if not settings.langfuse_enabled:
            return
        try:
            from langfuse import Langfuse

            lf = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            trace = lf.trace(
                id=self.id,
                name="emission-extraction",
                tags=[self.experiment_tag, self.config.get("prompt_version", "")],
                input={"pdf_name": self.pdf_name, "queries": self.data["queries"]},
                output=self.data["parsed_output"],
                metadata=self.config,
            )
            trace.generation(
                name="llm-extraction",
                model=self.config.get("model", ""),
                input=self.data["prompt"],
                output=self.data["raw_output"],
                usage={
                    "input": self.data["input_tokens"],
                    "output": self.data["output_tokens"],
                    "total_cost": self.data["cost_usd"],
                },
            )
            lf.flush()
        except Exception as exc:  # pragma: no cover - never break a run on tracing
            logger.warning("Langfuse export failed: %s", exc)
