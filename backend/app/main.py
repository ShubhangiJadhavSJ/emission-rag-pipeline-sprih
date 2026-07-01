"""FastAPI application: ingestion + RAG extraction + observability + evaluation.

Routes
  POST /api/documents/upload     upload a PDF -> background extraction
  GET  /api/documents            list rows (the frontend table)
  GET  /api/documents/{id}       one row
  GET  /api/documents/{id}/trace the trace for a row (observability)
  GET  /api/traces/{trace_id}    a trace by id
  POST /api/evaluate             score the "live" rows against ground truth
  GET  /api/experiments          list experiment metrics (trend report data)
  GET  /api/ground-truth         the labelled ground truth
"""
import logging
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.database import get_db, session_scope
from app.db.init_db import init_db
from app.db.models import Document, Experiment, Trace
from app.eval.evaluate import evaluate_predictions, load_ground_truth
from app.rag.pipeline import PipelineConfig, run_extraction
from app.schemas import DocumentOut, EvaluateRequest, ExperimentOut, TraceOut
from app.storage.blob import read_pdf, save_pdf
from app.vectorstore.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Auto-create tables on startup — no manual migration step.
    init_db()
    # Recover rows whose background task was interrupted (e.g. server restart /
    # OOM) so the UI never hangs on "processing" forever.
    with session_scope() as db:
        stuck = (
            db.query(Document).filter(Document.status == "processing").all()
        )
        for doc in stuck:
            doc.status = "error"
            doc.error = "Extraction was interrupted (server restarted). Re-upload to retry."
        if stuck:
            logger.warning("Reset %d stuck 'processing' rows to 'error'.", len(stuck))
    logger.info("Startup complete.")
    yield


app = FastAPI(title="Emission Data Extraction RAG", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Background extraction worker
# -----------------------------------------------------------------------------
def _process_document(document_id: int, pdf_name: str, blob_path: str) -> None:
    cfg = PipelineConfig(
        prompt_version=settings.default_prompt_version,
        chunk_size=settings.default_chunk_size,
        chunk_overlap=settings.default_chunk_overlap,
        retrieval_k=settings.default_retrieval_k,
        query_set="expanded" if settings.default_prompt_version == "v3" else "default",
    )
    try:
        pdf_bytes = read_pdf(blob_path)
        output = run_extraction(
            document_id=document_id,
            pdf_name=pdf_name,
            pdf_bytes=pdf_bytes,
            cfg=cfg,
            experiment_tag="live",
            store=QdrantStore(),
        )
        parsed = output.parsed
        output.trace.send_to_langfuse()

        with session_scope() as db:
            db.add(output.trace.to_model(document_id))
            doc = db.get(Document, document_id)
            doc.scope1_value = parsed["scope1"]["value"]
            doc.scope1_unit = parsed["scope1"]["unit"]
            doc.scope2_value = parsed["scope2"]["value"]
            doc.scope2_unit = parsed["scope2"]["unit"]
            doc.scope3_value = parsed["scope3"]["value"]
            doc.scope3_unit = parsed["scope3"]["unit"]
            doc.reporting_year = parsed.get("reporting_year")
            doc.trace_id = output.trace.id
            if output.trace.data["status"] == "error":
                doc.status = "error"
                doc.error = output.trace.data["error"]
            else:
                doc.status = "done"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed processing document %s", document_id)
        with session_scope() as db:
            doc = db.get(Document, document_id)
            if doc:
                doc.status = "error"
                doc.error = str(exc)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.post("/api/documents/upload", response_model=DocumentOut)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    experiment_tag: str = Form("live"),
    db: Session = Depends(get_db),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    content = await file.read()
    blob_path = save_pdf(content, file.filename)

    doc = Document(
        pdf_name=file.filename,
        blob_path=blob_path,
        status="processing",
        experiment_tag=experiment_tag,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(_process_document, doc.id, doc.pdf_name, blob_path)
    return doc


@app.get("/api/documents", response_model=list[DocumentOut])
def list_documents(experiment_tag: str = "live", db: Session = Depends(get_db)):
    rows = db.execute(
        select(Document)
        .where(Document.experiment_tag == experiment_tag)
        .order_by(Document.created_at.desc())
    ).scalars().all()
    return rows


@app.get("/api/documents/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc:
        raise HTTPException(404, "Document not found")
    return doc


@app.get("/api/documents/{doc_id}/trace", response_model=TraceOut)
def get_document_trace(doc_id: int, db: Session = Depends(get_db)):
    doc = db.get(Document, doc_id)
    if not doc or not doc.trace_id:
        raise HTTPException(404, "Trace not found")
    trace = db.get(Trace, doc.trace_id)
    if not trace:
        raise HTTPException(404, "Trace not found")
    return trace


@app.get("/api/traces/{trace_id}", response_model=TraceOut)
def get_trace(trace_id: str, db: Session = Depends(get_db)):
    trace = db.get(Trace, trace_id)
    if not trace:
        raise HTTPException(404, "Trace not found")
    return trace


@app.post("/api/evaluate")
def evaluate_live(req: EvaluateRequest, db: Session = Depends(get_db)):
    """Score the rows of an experiment_tag against ground truth.

    Predictions are keyed by pdf_name, so the uploaded files must match the
    ground-truth file names to be scored.
    """
    rows = db.execute(
        select(Document).where(Document.experiment_tag == req.experiment_tag)
    ).scalars().all()
    predictions = {
        d.pdf_name: {
            "reporting_year": d.reporting_year or "NA",
            "scope1": {"value": d.scope1_value or "NA", "unit": d.scope1_unit or "NA"},
            "scope2": {"value": d.scope2_value or "NA", "unit": d.scope2_unit or "NA"},
            "scope3": {"value": d.scope3_value or "NA", "unit": d.scope3_unit or "NA"},
        }
        for d in rows
    }
    metrics = evaluate_predictions(db, req.experiment_tag, predictions)
    return {"experiment_tag": req.experiment_tag, "metrics": metrics}


@app.get("/api/experiments", response_model=list[ExperimentOut])
def list_experiments(db: Session = Depends(get_db)):
    rows = db.execute(
        select(Experiment).order_by(Experiment.created_at.asc())
    ).scalars().all()
    return rows


@app.get("/api/ground-truth")
def get_ground_truth():
    try:
        return load_ground_truth()
    except FileNotFoundError:
        raise HTTPException(404, "Ground truth file not found")
