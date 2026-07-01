"""Evaluation metrics (candidate-designed).

WHY THESE METRICS — reasoning is part of the assignment, so it lives here:

For ESG emission extraction the costly mistakes are not small formatting
differences; they are (a) inventing a number that isn't in the report, and
(b) reporting the wrong figure entirely. So we measure on three axes:

1. NA-handling (precision/recall/F1 for the "NA" class)
   The dataset deliberately includes missing Scope 3 and fully-empty reports.
   A model that hallucinates a plausible number is worse than useless. We track
   how often the model correctly says NA, and crucially the HALLUCINATION RATE
   (predicted a number where truth is NA) as its own headline number.

2. Numeric correctness (tolerance match + MAPE)
   For fields where truth is a real number, we normalise both sides to tCO2e
   and count a hit when within ±5% (tolerance absorbs rounding/formatting). We
   also report MAPE over matched fields to see how far off the misses are.

3. Field accuracy (overall correctness)
   A single per-field "correct" that combines the two above: NA<->NA is correct,
   number-within-tolerance is correct, everything else is wrong. Aggregated to
   an overall accuracy that is the headline trend-line number.

We also track reporting-year accuracy, because picking the wrong year in a
multi-year table is a distinct, important failure mode.
"""
from app.eval.normalize import is_na, to_tonnes

TOLERANCE = 0.05  # ±5% counts as a numeric match
SCOPES = ("scope1", "scope2", "scope3")


def _field_correct(pred: dict, truth: dict) -> tuple[bool, str, float | None]:
    """Return (is_correct, category, abs_pct_error_or_None) for one scope field.

    category ∈ {na_correct, na_hallucination, na_missed, num_correct, num_wrong}
    """
    pred_na = is_na(pred.get("value"))
    truth_na = is_na(truth.get("value"))

    if truth_na and pred_na:
        return True, "na_correct", None
    if truth_na and not pred_na:
        return False, "na_hallucination", None  # invented a number
    if not truth_na and pred_na:
        return False, "na_missed", None  # missed a real number

    # Both are numbers — compare magnitudes in tCO2e.
    p = to_tonnes(pred.get("value"), pred.get("unit"))
    t = to_tonnes(truth.get("value"), truth.get("unit"))
    if p is None or t is None or t == 0:
        return False, "num_wrong", None
    pct_err = abs(p - t) / abs(t)
    if pct_err <= TOLERANCE:
        return True, "num_correct", pct_err
    return False, "num_wrong", pct_err


def score_document(predicted: dict, truth: dict) -> dict:
    """Per-document field scores + the categories needed for aggregation."""
    fields = {}
    for scope in SCOPES:
        correct, category, pct = _field_correct(
            predicted.get(scope, {}), truth.get(scope, {})
        )
        fields[scope] = {"correct": correct, "category": category, "pct_error": pct}

    year_correct = str(predicted.get("reporting_year", "NA")).strip() == str(
        truth.get("reporting_year", "NA")
    ).strip()
    fields["reporting_year"] = {"correct": year_correct}
    return fields


def aggregate(per_doc_scores: list[dict]) -> dict:
    """Aggregate per-document field scores into experiment-level metrics."""
    total_fields = 0
    correct_fields = 0
    cats = {
        "na_correct": 0,
        "na_hallucination": 0,
        "na_missed": 0,
        "num_correct": 0,
        "num_wrong": 0,
    }
    pct_errors: list[float] = []
    year_correct = 0
    n_docs = len(per_doc_scores)
    # For NA-class precision/recall.
    truth_na = pred_na = na_true_positive = 0

    for doc in per_doc_scores:
        for scope in SCOPES:
            f = doc[scope]
            total_fields += 1
            correct_fields += int(f["correct"])
            cats[f["category"]] += 1
            if f["pct_error"] is not None:
                pct_errors.append(f["pct_error"])

            cat = f["category"]
            # truth is NA in {na_correct, na_hallucination}; pred is NA in
            # {na_correct, na_missed}.
            if cat in ("na_correct", "na_hallucination"):
                truth_na += 1
            if cat in ("na_correct", "na_missed"):
                pred_na += 1
            if cat == "na_correct":
                na_true_positive += 1
        year_correct += int(doc["reporting_year"]["correct"])

    def ratio(n, d):
        return round(n / d, 4) if d else 0.0

    na_precision = ratio(na_true_positive, pred_na)
    na_recall = ratio(na_true_positive, truth_na)
    na_f1 = (
        round(2 * na_precision * na_recall / (na_precision + na_recall), 4)
        if (na_precision + na_recall)
        else 0.0
    )

    return {
        "n_documents": n_docs,
        "n_fields": total_fields,
        "overall_field_accuracy": ratio(correct_fields, total_fields),
        "reporting_year_accuracy": ratio(year_correct, n_docs),
        "hallucination_rate": ratio(cats["na_hallucination"], total_fields),
        "na_precision": na_precision,
        "na_recall": na_recall,
        "na_f1": na_f1,
        "numeric_match_rate": ratio(
            cats["num_correct"], cats["num_correct"] + cats["num_wrong"]
        ),
        "mape": round(sum(pct_errors) / len(pct_errors), 4) if pct_errors else None,
        "category_counts": cats,
    }
