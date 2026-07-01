"""Streamlit frontend for the Emission Data Extraction pipeline.

Implements the reference UI:
  - a table with PDF Name | Scope 1 | Scope 2 | Scope 3 (with units, NA where
    not found),
  - an "Add Row" upload control that triggers extraction and shows a
    processing... state until the row populates,
  - graceful error states,
plus tabs for per-upload traces (observability), evaluation, and the trend
report across experiments.
"""
import os
import time

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Emission Data Extraction", layout="wide")


def api_get(path: str, **params):
    r = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path: str, **kwargs):
    r = requests.post(f"{BACKEND_URL}{path}", timeout=120, **kwargs)
    r.raise_for_status()
    return r.json()


def fmt_cell(value, unit, status):
    if status == "processing":
        return "processing…"
    if status == "error":
        return "error"
    if not value or str(value).upper() == "NA":
        return "NA"
    return f"{value} {unit}" if unit and unit.upper() != "NA" else str(value)


# -----------------------------------------------------------------------------
st.title("🌍 Emission Data Extraction")
st.caption("Scope 1 / 2 / 3 extraction from ESG report PDFs — RAG pipeline with evaluation")

tab_table, tab_trace, tab_eval, tab_trend = st.tabs(
    ["📋 Extraction Table", "🔍 Traces", "✅ Evaluation", "📈 Trend Report"]
)

# =============================================================================
# Tab 1 — the main table + upload flow
# =============================================================================
with tab_table:
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()

    try:
        docs = api_get("/api/documents", experiment_tag="live")
    except Exception as exc:
        st.error(f"Cannot reach backend at {BACKEND_URL}: {exc}")
        docs = []

    if docs:
        rows = []
        for d in docs:
            rows.append(
                {
                    "PDF Name": d["pdf_name"],
                    "Scope 1": fmt_cell(d["scope1_value"], d["scope1_unit"], d["status"]),
                    "Scope 2": fmt_cell(d["scope2_value"], d["scope2_unit"], d["status"]),
                    "Scope 3": fmt_cell(d["scope3_value"], d["scope3_unit"], d["status"]),
                    "Year": d.get("reporting_year") or "—",
                    "Status": d["status"],
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Surface row-level errors clearly.
        for d in docs:
            if d["status"] == "error":
                st.warning(f"⚠️ {d['pdf_name']}: {d.get('error') or 'extraction failed'}")
    else:
        st.info("No documents yet. Use **Add Row** below to upload a PDF.")

    st.divider()
    st.subheader("➕ Add Row")
    st.caption("Click to upload a PDF and extract emissions.")
    uploaded = st.file_uploader("Upload an ESG report PDF", type=["pdf"], key="uploader")
    if uploaded is not None:
        if st.button("Process PDF", type="primary"):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                api_post("/api/documents/upload", files=files, data={"experiment_tag": "live"})
                st.success(f"Uploaded {uploaded.name}. Extracting…")
                # Poll until the row leaves the processing state.
                placeholder = st.empty()
                for _ in range(40):
                    time.sleep(2)
                    latest = api_get("/api/documents", experiment_tag="live")
                    match = next((x for x in latest if x["pdf_name"] == uploaded.name), None)
                    if match and match["status"] != "processing":
                        break
                    placeholder.info("processing…")
                placeholder.empty()
                st.rerun()
            except Exception as exc:
                st.error(f"Upload failed: {exc}")

# =============================================================================
# Tab 2 — Traces (observability)
# =============================================================================
with tab_trace:
    st.subheader("Per-upload traces")
    st.caption("Each upload emits one trace: queries, retrieved chunks, prompt, model, tokens, cost, latency, parsed output.")
    try:
        docs = api_get("/api/documents", experiment_tag="live")
    except Exception:
        docs = []
    traced = [d for d in docs if d.get("trace_id")]
    if not traced:
        st.info("No traces yet — process a PDF first.")
    else:
        label = st.selectbox(
            "Select an upload",
            options=traced,
            format_func=lambda d: f"{d['pdf_name']} ({d['status']})",
        )
        if label:
            trace = api_get(f"/api/documents/{label['id']}/trace")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Model", trace["model"])
            c2.metric("Tokens (in/out)", f"{trace['input_tokens']}/{trace['output_tokens']}")
            c3.metric("Cost (USD)", f"${trace['cost_usd']:.5f}")
            c4.metric("Latency", f"{trace['latency_ms']} ms")
            st.write(f"**Prompt version:** {trace['prompt_version']} · "
                     f"**k:** {trace['retrieval_k']} · "
                     f"**chunk:** {trace['chunk_size']}/{trace['chunk_overlap']} · "
                     f"**status:** {trace['status']}")
            if trace.get("error"):
                st.error(f"Error: {trace['error']}")
            with st.expander("Retrieval queries"):
                st.json(trace.get("queries"))
            with st.expander("Retrieved chunks"):
                for ch in trace.get("retrieved_chunks") or []:
                    st.markdown(f"**chunk {ch['chunk_index']}** (score={ch['score']:.3f})")
                    st.text(ch["text"][:1500])
            with st.expander("Prompt sent to LLM"):
                st.text(trace.get("prompt") or "")
            with st.expander("Raw model output"):
                st.text(trace.get("raw_output") or "")
            with st.expander("Parsed output", expanded=True):
                st.json(trace.get("parsed_output"))

# =============================================================================
# Tab 3 — Evaluation
# =============================================================================
with tab_eval:
    st.subheader("Evaluate live uploads against ground truth")
    st.caption("Scores only uploads whose file name matches a ground-truth label.")
    if st.button("Run evaluation"):
        try:
            res = api_post("/api/evaluate", json={"experiment_tag": "live"})
            st.json(res["metrics"])
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")
    with st.expander("Ground truth labels"):
        try:
            st.json(api_get("/api/ground-truth"))
        except Exception as exc:
            st.warning(f"No ground truth available: {exc}")

# =============================================================================
# Tab 4 — Trend report across experiments
# =============================================================================
with tab_trend:
    st.subheader("Metric trend across experiments")
    st.caption("Populated by: docker compose exec backend python -m app.eval.experiments")
    try:
        exps = api_get("/api/experiments")
    except Exception:
        exps = []
    if not exps:
        st.info("No experiments recorded yet. Run the experiment runner.")
    else:
        table = []
        for e in exps:
            row = {"experiment": e["tag"], **{k: v for k, v in e["metrics"].items() if not isinstance(v, dict)}}
            table.append(row)
        df = pd.DataFrame(table).set_index("experiment")
        st.dataframe(df, use_container_width=True)
        chart_cols = [c for c in ["overall_field_accuracy", "na_f1", "numeric_match_rate", "reporting_year_accuracy", "hallucination_rate"] if c in df.columns]
        if chart_cols:
            st.line_chart(df[chart_cols])
        for e in exps:
            st.markdown(f"**{e['tag']}** — {e.get('notes', '')}")
