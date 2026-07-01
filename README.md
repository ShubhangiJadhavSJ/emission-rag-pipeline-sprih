# Emission Data Extraction — End-to-End RAG Pipeline with Evaluation

Extracts **Scope 1 / 2 / 3** greenhouse-gas emission figures from ESG /
sustainability report PDFs, shows them in a table, traces every extraction run,
persists everything to a database, and lets you design metrics and iterate the
RAG pipeline across experiments with a trend report.

---

## 1. Run it (two steps)

**Prerequisites:** Docker + Docker Compose, and one LLM API key
(Anthropic by default).

```bash
# 1) configure
cp .env.example .env
#    then edit .env and set GROQ_API_KEY=...   

# 2) bring up the whole stack
docker compose up --build
```


| Service     | URL                              | What                         |
|-------------|----------------------------------|------------------------------|
| Frontend    | http://localhost:8501            | Streamlit table + upload     |
| Backend API | http://localhost:8000/docs       | FastAPI (OpenAPI docs)       |
| Qdrant      | http://localhost:6333/dashboard  | Vector store                 |
| MariaDB     | localhost:3306                   | Relational store             |

### Use it
1. Open the frontend, go to **Add Row**, upload an ESG PDF, click **Process**.
   The row shows `processing…` then populates Scope 1/2/3 (with units) or `NA`.
2. **Traces** tab → inspect the full trace for any upload.
3. **Evaluation** tab → score uploads against ground truth.
4. **Trend Report** tab → see metrics across experiments (after step below).

### Reproduce the experiments / trend report
Put the provided PDFs in `data/pdfs/` and label them in
`data/ground_truth/ground_truth.json`, then:

```bash
docker compose exec backend python -m app.eval.experiments
```

This runs the 3 experiments, writes per-experiment metrics to the DB, and emits
`data/reports/trend_report.md`, `.json`, and `.png`. The Trend Report tab reads
the same data.

---

## 2. Architecture

```
┌──────────────┐    upload PDF     ┌──────────────────────────────────────┐
│  Streamlit   │ ────────────────▶ │              FastAPI                  │
│  frontend    │ ◀──────────────── │  (ingestion · RAG · eval · tracing)   │
└──────────────┘   table / traces  └───────┬───────────────┬───────────────┘
                                           │               │
                          blob storage ◀───┤               ├───▶ Qdrant (vectors)
                       (raw PDFs, local)   │               │
                                           ▼               ▼
                                       MariaDB        Langfuse (optional)
                              documents · traces ·
                              evaluations · experiments
```

**Separation of concerns** (each in its own module — `backend/app/`):

| Concern        | Module                          | Responsibility                              |
|----------------|---------------------------------|---------------------------------------------|
| API            | `main.py`, `schemas.py`         | HTTP routes, request/response models        |
| Config         | `config.py`                     | All env-driven settings in one place        |
| DB             | `db/`                           | ORM models, session, auto-init/seed         |
| Storage        | `storage/blob.py`               | Local blob storage for raw PDFs             |
| Ingestion      | `ingestion/`                    | PDF text + OCR, chunking, embeddings        |
| Vector store   | `vectorstore/qdrant_store.py`   | Index + scoped retrieval                    |
| LLM            | `llm/client.py`                 | Provider-agnostic completion + cost         |
| RAG            | `rag/`                          | Prompts (v1–v3), extractor, pipeline        |
| Observability  | `tracing/tracer.py`             | One trace per upload (DB + optional Langfuse)|
| Evaluation     | `eval/`                         | Metrics, evaluator, experiment runner       |

### Request flow (one upload)
`upload → save blob → create row (processing) → background task:` 
`extract text (OCR fallback) → chunk → embed → index in Qdrant → retrieve → LLM extract → parse JSON → write trace + update row (done/error)`.

The upload returns immediately; the frontend polls until the row leaves
`processing`.

---

## 3. Observability — one trace per upload

Every upload writes exactly one row to the `traces` table containing: retrieval
queries, retrieved chunks (with scores), the exact prompt, model, input/output
tokens, **estimated USD cost**, latency, OCR-used flag, raw model output, parsed
output, and status/error. This is viewable in the **Traces** tab — enough to
debug a failed extraction without re-running it.

---

## 5. Evaluation design

> lives in code (`backend/app/eval/metrics.py`) and is summarised here.

**Ground truth** is a hand-labelled JSON answer key
(`data/ground_truth/ground_truth.json`); the labelling protocol is in
`data/ground_truth/README.md`. Predictions and labels are normalised to a
common magnitude in **tCO2e** before comparison (`eval/normalize.py`), so
`4.2 ktCO2e` and `4,200 tCO2e` compare equal.

**Metrics and why they were chosen** — the expensive errors in ESG extraction
are *hallucinating a number that isn't there* and *reporting the wrong figure*,
not minor formatting differences. So:

| Metric                       | Why it matters                                                        |
|------------------------------|-----------------------------------------------------------------------|
| `overall_field_accuracy`     | Headline correctness across all scope fields (NA↔NA or number-in-tol).|
| `hallucination_rate`         | Fraction of fields where we invented a number but truth is NA. **The most important safety metric.** |
| `na_precision/recall/f1`     | Quality of NA detection — the dataset is built to test this.          |
| `numeric_match_rate`         | Of fields that are real numbers, fraction within ±5% (tCO2e).         |
| `mape`                       | How far off the numeric misses are.                                   |
| `reporting_year_accuracy`    | Did we pick the right year in multi-year tables? A distinct failure mode. |

Per-document scores and aggregate metrics are persisted to the `evaluations`
and `experiments` tables.

---

## 6. Prompt iteration & experiments

Three experiments change one lever at a time and are tagged so traces and
metrics line up. Defined in `backend/app/eval/experiments.py`; prompts versioned
in `backend/app/rag/prompts.py`; full narrative in **[PROMPT_ITERATION.md](PROMPT_ITERATION.md)**.

| Experiment            | Lever changed                                              | Targets                  |
|-----------------------|------------------------------------------------------------|--------------------------|
| `exp1_baseline`       | Prompt v1, 800-char chunks, single query, k=5              | establish a floor        |
| `exp2_prompt_v2`      | Prompt v2: strict NA rule + unit discipline                | hallucinations           |
| `exp3_v3_retrieval`   | Prompt v3 (few-shot + year rule) + bigger chunks + 4-query expansion, k=6 | wrong-year & recall |

The **trend report** (`data/reports/`) shows the headline metrics moving across
these runs; see **[data/reports/trend_report.md](data/reports/trend_report.md)**
for the template/example.

---
