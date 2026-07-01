"""End-to-end RAG extraction pipeline.

    raw text -> chunks -> embeddings -> Qdrant -> retrieval -> LLM -> parsed JSON

`run_extraction` is the single entry point used by both the live upload flow and
the offline experiment runner, so they exercise identical code with different
configs. It records one trace and returns the parsed result + trace.
"""
import logging
from dataclasses import asdict, dataclass

from app.ingestion.chunker import chunk_text
from app.ingestion.pdf_extract import extract_text
from app.rag import prompts
from app.rag.extractor import extract
from app.tracing.tracer import UploadTrace
from app.vectorstore.qdrant_store import QdrantStore

logger = logging.getLogger(__name__)

# Safety cap on chunks indexed per document (bounds embedding time/memory).
MAX_CHUNKS = 500


@dataclass
class PipelineConfig:
    prompt_version: str = "v3"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    retrieval_k: int = 6
    query_set: str = "expanded"  # "default" | "expanded"


@dataclass
class PipelineOutput:
    parsed: dict
    trace: UploadTrace


def _retrieve(store: QdrantStore, document_id: int, cfg: PipelineConfig):
    queries = prompts.RETRIEVAL_QUERIES[cfg.query_set]
    seen: dict[int, dict] = {}
    for q in queries:
        for hit in store.search(document_id, q, cfg.retrieval_k):
            idx = hit["chunk_index"]
            # Keep the highest-scoring occurrence of each chunk across queries.
            if idx not in seen or hit["score"] > seen[idx]["score"]:
                seen[idx] = hit
    chunks = sorted(seen.values(), key=lambda h: h["score"], reverse=True)
    return queries, chunks[: max(cfg.retrieval_k, len(queries))]


def run_extraction(
    *,
    document_id: int,
    pdf_name: str,
    pdf_bytes: bytes,
    cfg: PipelineConfig,
    experiment_tag: str,
    store: QdrantStore | None = None,
    model_name: str = "",
) -> PipelineOutput:
    store = store or QdrantStore()
    trace = UploadTrace(
        pdf_name=pdf_name,
        experiment_tag=experiment_tag,
        config={
            "prompt_version": cfg.prompt_version,
            "model": model_name,
            "retrieval_k": cfg.retrieval_k,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "query_set": cfg.query_set,
        },
    )

    try:
        # 1. Extract text (with OCR fallback for scanned PDFs).
        text, used_ocr = extract_text(pdf_bytes)
        trace.set(used_ocr=used_ocr)

        # 2. Chunk + 3. embed + index (re-index is idempotent per document).
        chunks = chunk_text(text, cfg.chunk_size, cfg.chunk_overlap)
        # Bound worst-case embedding time on very large reports. Emission tables
        # appear early/mid-document, so the cap rarely drops relevant content.
        if len(chunks) > MAX_CHUNKS:
            logger.warning(
                "%s produced %d chunks; capping to %d", pdf_name, len(chunks), MAX_CHUNKS
            )
            chunks = chunks[:MAX_CHUNKS]
        store.delete_document(document_id)
        store.index_chunks(document_id, chunks)

        # 4. Retrieve.
        queries, retrieved = _retrieve(store, document_id, cfg)
        trace.set(
            queries=queries,
            retrieved_chunks=[
                {"chunk_index": h["chunk_index"], "score": h["score"], "text": h["text"]}
                for h in retrieved
            ],
        )

        # 5. LLM extraction.
        result = extract(cfg.prompt_version, retrieved)
        trace.set(
            prompt=result.prompt,
            raw_output=result.raw_output,
            parsed_output=result.parsed,
            input_tokens=result.llm.input_tokens,
            output_tokens=result.llm.output_tokens,
            cost_usd=result.llm.cost_usd,
        )
        trace.config["model"] = result.llm.model
        return PipelineOutput(parsed=result.parsed, trace=trace)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed for %s", pdf_name)
        trace.fail(str(exc))
        # Degrade to all-NA rather than propagate (assignment: never crash a row).
        from app.rag.extractor import _EMPTY

        return PipelineOutput(parsed=dict(_EMPTY), trace=trace)
