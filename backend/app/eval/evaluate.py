"""Evaluation orchestration: compare predictions to ground truth and persist.

Used by the experiment runner and by the /evaluate API endpoint.
"""
import json

from sqlalchemy import delete

from app.config import settings
from app.db.models import Evaluation
from app.eval import metrics


def load_ground_truth() -> dict:
    with open(settings.ground_truth_path, encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_predictions(
    db, experiment_tag: str, predictions: dict[str, dict]
) -> dict:
    """predictions: {pdf_name: parsed_output}. Returns aggregate metrics and
    persists per-document evaluation rows."""
    gt = load_ground_truth()

    # Replace any prior eval rows for this tag (re-running is idempotent).
    db.execute(delete(Evaluation).where(Evaluation.experiment_tag == experiment_tag))

    per_doc_scores = []
    for pdf_name, truth in gt.items():
        pred = predictions.get(pdf_name)
        if pred is None:
            continue
        field_scores = metrics.score_document(pred, truth)
        per_doc_scores.append(field_scores)
        db.add(
            Evaluation(
                experiment_tag=experiment_tag,
                pdf_name=pdf_name,
                predicted=pred,
                ground_truth=truth,
                field_scores=field_scores,
            )
        )

    agg = metrics.aggregate(per_doc_scores)
    db.commit()
    return agg
