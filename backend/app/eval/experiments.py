"""Experiment runner + trend report generator.

Runs a sequence of >=3 experiment configurations over the corpus of
ground-truth PDFs, evaluates each against the labels, stores per-experiment
metrics in the `experiments` table, and writes a trend report
(Markdown + PNG chart) to /data/reports.

Run inside the backend container:

    docker compose exec backend python -m app.eval.experiments

Each experiment is a single source of truth for prompt/chunking/retrieval
changes, tagged so its trace and metrics line up in the trend report.
"""
import json
import logging
import os

from app.config import settings
from app.db.database import session_scope
from app.db.init_db import init_db
from app.db.models import Experiment
from app.eval.evaluate import evaluate_predictions, load_ground_truth
from app.rag.pipeline import PipelineConfig, run_extraction
from app.storage.blob import read_pdf
from app.vectorstore.qdrant_store import QdrantStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PDF_DIR = "/data/pdfs"
REPORT_DIR = "/data/reports"

# -----------------------------------------------------------------------------
# The >=3 experiments. Each row changes one lever and documents WHY.
# -----------------------------------------------------------------------------
EXPERIMENTS = [
    {
        "tag": "exp1_baseline",
        "notes": "Naive prompt v1, fixed 800-char chunks, single generic query, k=5. Baseline floor.",
        "config": PipelineConfig(
            prompt_version="v1",
            chunk_size=800,
            chunk_overlap=100,
            retrieval_k=5,
            query_set="default",
        ),
    },
    {
        "tag": "exp2_prompt_v2",
        "notes": "Prompt v2 adds strict NA rule + unit discipline. Same retrieval as exp1. Targets hallucinations.",
        "config": PipelineConfig(
            prompt_version="v2",
            chunk_size=800,
            chunk_overlap=100,
            retrieval_k=5,
            query_set="default",
        ),
    },
    {
        "tag": "exp3_v3_retrieval",
        "notes": "Prompt v3 (few-shot + year disambiguation) + larger overlapping chunks + 4-query expansion, k=6. Targets wrong-year & recall.",
        "config": PipelineConfig(
            prompt_version="v3",
            chunk_size=1200,
            chunk_overlap=200,
            retrieval_k=6,
            query_set="expanded",
        ),
    },
]

HEADLINE_METRICS = [
    "overall_field_accuracy",
    "na_f1",
    "hallucination_rate",
    "numeric_match_rate",
    "reporting_year_accuracy",
]


def _resolve_pdf_path(pdf_name: str) -> str | None:
    path = os.path.join(PDF_DIR, pdf_name)
    return path if os.path.exists(path) else None


def run_experiment(exp: dict) -> dict:
    gt = load_ground_truth()
    store = QdrantStore()
    predictions: dict[str, dict] = {}
    cfg: PipelineConfig = exp["config"]

    for offset, pdf_name in enumerate(gt):
        path = _resolve_pdf_path(pdf_name)
        if not path:
            logger.warning("Skipping %s — file not present in %s", pdf_name, PDF_DIR)
            continue
        # Stable synthetic document id per (experiment, file) so Qdrant points
        # never collide across experiments.
        document_id = 10_000 + EXPERIMENTS.index(exp) * 1000 + offset
        output = run_extraction(
            document_id=document_id,
            pdf_name=pdf_name,
            pdf_bytes=read_pdf(path),
            cfg=cfg,
            experiment_tag=exp["tag"],
            store=store,
        )
        output.trace.send_to_langfuse()
        predictions[pdf_name] = output.parsed
        with session_scope() as db:
            db.add(output.trace.to_model(None))
        logger.info("[%s] %s -> %s", exp["tag"], pdf_name, output.parsed)

    with session_scope() as db:
        metrics = evaluate_predictions(db, exp["tag"], predictions)
        # Upsert the experiment row.
        existing = db.query(Experiment).filter_by(tag=exp["tag"]).first()
        config_json = {
            "prompt_version": cfg.prompt_version,
            "chunk_size": cfg.chunk_size,
            "chunk_overlap": cfg.chunk_overlap,
            "retrieval_k": cfg.retrieval_k,
            "query_set": cfg.query_set,
        }
        if existing:
            existing.config = config_json
            existing.metrics = metrics
            existing.notes = exp["notes"]
        else:
            db.add(
                Experiment(
                    tag=exp["tag"], config=config_json, metrics=metrics, notes=exp["notes"]
                )
            )
    return metrics


def write_trend_report(results: list[tuple[str, dict, str]]) -> None:
    os.makedirs(REPORT_DIR, exist_ok=True)

    # --- Markdown table ---
    lines = ["# Trend Report — Emission Extraction Experiments\n"]
    lines.append("One row per experiment, in chronological order.\n")
    header = "| Experiment | " + " | ".join(HEADLINE_METRICS) + " |"
    sep = "|" + "---|" * (len(HEADLINE_METRICS) + 1)
    lines += [header, sep]
    for tag, metrics, _notes in results:
        cells = [tag] + [str(metrics.get(m, "—")) for m in HEADLINE_METRICS]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("\n## What changed and why it helped\n")
    for tag, metrics, notes in results:
        lines.append(f"- **{tag}** — {notes}")
        lines.append(
            f"  - accuracy={metrics.get('overall_field_accuracy')}, "
            f"hallucination_rate={metrics.get('hallucination_rate')}, "
            f"na_f1={metrics.get('na_f1')}"
        )

    md_path = os.path.join(REPORT_DIR, "trend_report.md")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Wrote %s", md_path)

    # --- JSON dump ---
    with open(os.path.join(REPORT_DIR, "trend_report.json"), "w", encoding="utf-8") as fh:
        json.dump(
            [{"tag": t, "metrics": m, "notes": n} for t, m, n in results], fh, indent=2
        )

    # --- Chart ---
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        tags = [t for t, _, _ in results]
        fig, ax = plt.subplots(figsize=(9, 5))
        for metric in HEADLINE_METRICS:
            ys = [r[1].get(metric) or 0 for r in results]
            ax.plot(tags, ys, marker="o", label=metric)
        ax.set_title("Metric trend across experiments")
        ax.set_ylabel("score")
        ax.set_ylim(0, 1)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(REPORT_DIR, "trend_report.png"), dpi=120)
        logger.info("Wrote trend_report.png")
    except Exception as exc:  # pragma: no cover
        logger.warning("Chart generation skipped: %s", exc)


def main() -> None:
    init_db()
    results = []
    for exp in EXPERIMENTS:
        logger.info("=== Running %s ===", exp["tag"])
        metrics = run_experiment(exp)
        results.append((exp["tag"], metrics, exp["notes"]))
    write_trend_report(results)
    logger.info("Done. Reports in %s", REPORT_DIR)


if __name__ == "__main__":
    main()
