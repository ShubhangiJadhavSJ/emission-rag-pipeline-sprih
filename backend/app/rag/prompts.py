"""Prompt versions for the extraction step.

The three versions are the heart of the "prompt iteration" deliverable. Each
one is a deliberate response to a failure mode observed in the previous one —
see PROMPT_ITERATION.md for the full narrative. They are kept here, versioned
and side by side, so an experiment can pin an exact prompt.

All versions instruct the model to return STRICT JSON of the shape:

    {
      "reporting_year": "2024" | "NA",
      "scope1": {"value": "12,500", "unit": "tCO2e"} | {"value": "NA", "unit": "NA"},
      "scope2": {...},
      "scope3": {...}
    }
"""

JSON_SHAPE = """Return ONLY a JSON object, no prose, with exactly this shape:
{
  "reporting_year": "<year or NA>",
  "scope1": {"value": "<number as written, or NA>", "unit": "<unit or NA>"},
  "scope2": {"value": "<number as written, or NA>", "unit": "<unit or NA>"},
  "scope3": {"value": "<number as written, or NA>", "unit": "<unit or NA>"}
}"""


# -----------------------------------------------------------------------------
# v1 — naive baseline.
# Just asks for the numbers. No NA discipline, no year handling, no units rule.
# Establishes a floor and surfaces hallucination / wrong-year failure modes.
# -----------------------------------------------------------------------------
V1_SYSTEM = "You extract greenhouse gas emission figures from ESG reports."

V1_USER = """From the report excerpts below, extract the Scope 1, Scope 2 and
Scope 3 emission totals.

{context}

{json_shape}"""


# -----------------------------------------------------------------------------
# v2 — structured, with explicit NA + unit rules.
# Fixes hallucination (return NA, never guess) and inconsistent units.
# -----------------------------------------------------------------------------
V2_SYSTEM = (
    "You are a meticulous ESG data analyst. You extract Scope 1/2/3 greenhouse "
    "gas emission totals from sustainability reports. You never invent numbers."
)

V2_USER = """Extract the organisation-wide Scope 1, Scope 2 and Scope 3 GHG
emission totals from the excerpts below.

Rules:
- Use ONLY numbers explicitly present in the excerpts. Never estimate or infer.
- If a scope is not present in the excerpts, set its value AND unit to "NA".
- Copy the unit exactly as written (e.g. tCO2e, ktCO2e, mtCO2e).
- Keep the number formatting as written (e.g. "12,500", "4.2").

{context}

{json_shape}"""


# -----------------------------------------------------------------------------
# v3 — few-shot + reporting-year disambiguation + market/location guidance.
# Fixes the multi-year-table failure (pick the CURRENT reporting year) and the
# Scope 2 market-vs-location ambiguity. Adds worked examples to lock formatting.
# -----------------------------------------------------------------------------
V3_SYSTEM = (
    "You are a meticulous ESG data analyst extracting Scope 1/2/3 greenhouse "
    "gas emission totals from sustainability reports. You are precise about the "
    "reporting year and never fabricate figures."
)

V3_USER = """Extract the organisation-wide Scope 1, Scope 2 and Scope 3 GHG
emission totals from the report excerpts below.

Rules:
1. Use ONLY figures explicitly present in the excerpts. If a scope is absent,
   return value "NA" and unit "NA". Returning NA is correct and expected when
   the data is not there — do NOT guess.
2. MULTI-YEAR TABLES: reports often show several years side by side. Identify
   the CURRENT reporting year (the latest/primary year the report is about) and
   extract that year's column only. Put that year in "reporting_year".
3. SCOPE 2: if both market-based and location-based totals are given, prefer
   the market-based figure.
4. Report the total/gross organisation-wide figure, not a single site or
   category subtotal.
5. Copy units and number formatting exactly as written.

Worked examples:
- Excerpt "Scope 1: 12,500 tCO2e (2024) | 11,000 (2023)" with 2024 current ->
  scope1 = {{"value": "12,500", "unit": "tCO2e"}}, reporting_year = "2024".
- Excerpt mentions Scope 1 and 2 but never Scope 3 ->
  scope3 = {{"value": "NA", "unit": "NA"}}.

{context}

{json_shape}"""


_PROMPTS = {
    "v1": (V1_SYSTEM, V1_USER),
    "v2": (V2_SYSTEM, V2_USER),
    "v3": (V3_SYSTEM, V3_USER),
}


def build_prompt(version: str, context: str) -> tuple[str, str]:
    """Return (system, user) for a prompt version."""
    if version not in _PROMPTS:
        raise ValueError(f"Unknown prompt version: {version}")
    system, user_tpl = _PROMPTS[version]
    user = user_tpl.format(context=context, json_shape=JSON_SHAPE)
    return system, user


# Retrieval queries. v3 expands the query set to better surface each scope and
# the reporting year; v1/v2 use a single generic query.
RETRIEVAL_QUERIES = {
    "default": ["total Scope 1 Scope 2 Scope 3 greenhouse gas emissions tCO2e"],
    "expanded": [
        "Scope 1 direct greenhouse gas emissions total tCO2e",
        "Scope 2 indirect emissions electricity market-based location-based tCO2e",
        "Scope 3 value chain emissions total tCO2e",
        "GHG emissions summary table reporting year current year",
    ],
}
