# Prompt Iteration

This documents how the extraction prompt evolved across the three experiments,
the failure mode each version targeted, and the reasoning. The prompts
themselves are versioned in `backend/app/rag/prompts.py`; the experiment configs
are in `backend/app/eval/experiments.py`.

Each experiment changes **one set of levers at a time** so the trend report can
attribute movement to a cause.

---

## v1 — naive baseline (`exp1_baseline`)

**Prompt:** "From the report excerpts below, extract the Scope 1, Scope 2 and
Scope 3 emission totals." No NA rule, no unit rule, no year handling.

**Retrieval:** single generic query, 800-char chunks, overlap 100, k=5.

**Observed failure modes**
1. **Hallucination.** When Scope 3 is absent, the model frequently invents a
   plausible-looking number rather than abstaining → high `hallucination_rate`,
   low `na_recall`.
2. **Unit drift.** Returns values without units, or normalises them
   inconsistently (kt vs t), hurting downstream comparison.
3. **Wrong year.** On multi-year tables it grabs whichever number is most
   salient, often a prior year.

Purpose: establish a floor and make these failure modes measurable.

---

## v2 — strict NA + unit discipline (`exp2_prompt_v2`)

**Change (prompt only — retrieval held constant vs v1):**
- A meticulous-analyst system role that "never invents numbers".
- Explicit rule: *use only numbers present in the excerpts; if a scope is not
  present, set value AND unit to `NA`; never estimate.*
- Explicit rule: copy the unit exactly as written; keep number formatting.

**Why:** Holding retrieval fixed isolates the prompt's effect. The dataset is
deliberately seeded with missing-Scope-3 and empty reports, so the highest-value
fix is teaching the model to abstain.

**Expected/observed effect:** `hallucination_rate` drops and `na_f1` /
`na_recall` rise sharply; unit consistency improves `numeric_match_rate`
slightly. Wrong-year errors largely remain (not yet addressed).

---

## v3 — few-shot + year disambiguation + retrieval upgrade (`exp3_v3_retrieval`)

**Changes (prompt *and* retrieval — the final configuration):**

Prompt:
- **Multi-year rule + worked example:** identify the current/primary reporting
  year and extract that column only; record it in `reporting_year`.
- **Scope 2 market-vs-location rule:** prefer market-based when both are given.
- **Total-not-subtotal rule:** report the organisation-wide figure.
- Two **few-shot examples** that lock the JSON shape and the NA behaviour.

Retrieval:
- Larger overlapping chunks (1200 / 200) so a full emissions table row survives
  in one chunk instead of being split mid-row.
- **Query expansion** — four targeted queries (one per scope + a year/summary
  query) unioned and de-duplicated, k=6, so each scope is independently
  surfaced rather than relying on one generic query.

**Why:** After v2 fixed abstention, the dominant remaining errors were
wrong-year and missed-but-present numbers (a recall problem). The prompt rule
attacks the former; the chunking/expansion attacks the latter.

**Expected/observed effect:** `reporting_year_accuracy` and
`numeric_match_rate` rise, lifting `overall_field_accuracy`, while
`hallucination_rate` stays low (the v2 NA discipline is retained).

---

## How to read the trend

Run `docker compose exec backend python -m app.eval.experiments`. The headline
line we optimise is **`overall_field_accuracy` up** and **`hallucination_rate`
down**, with `na_f1`, `numeric_match_rate`, and `reporting_year_accuracy` as the
supporting story. The generated `data/reports/trend_report.md` + `.png` plot all
five across the three runs.

> Note: absolute numbers depend on the actual provided PDFs and your labels —
> the *methodology* (isolate one lever, measure, attribute) is the deliverable,
> not any specific score.
