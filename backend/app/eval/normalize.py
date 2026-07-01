"""Value normalisation shared by extraction parsing and evaluation.

Turns messy human-written figures ("12,500", "4.2 kt", "145000 tCO₂e") into a
canonical numeric magnitude in tCO2e so predictions and ground truth can be
compared on equal footing.

Unit scale is case-INSENSITIVE. "MT", "Mt", "mt", "metric tonne", "tonne", "t",
"tCO2e", "CO2e" are all treated as tonnes (×1). "kt" is kilotonnes (×1000).
The worded forms "million"/"mega" map to ×1e6 (rarely needed, kept as a safety
net). Because "Mt" is no longer auto-treated as megatonne, any genuinely
megatonne ground-truth value should be written in absolute tonnes.
"""
import re

NA_TOKENS = {"na", "n/a", "none", "not reported", "not disclosed", "", "-", "—"}


def is_na(value: str | None) -> bool:
    if value is None:
        return True
    return str(value).strip().lower() in NA_TOKENS


def _scale_factor(unit_raw: str, value_raw: str) -> float:
    """Return the tonnes multiplier for a (unit, value) pair, case-insensitively."""
    for text in (unit_raw, value_raw):
        if not text:
            continue
        low = text.lower()
        if "million" in low or "mega" in low:
            return 1_000_000.0
        if re.search(r"\bk\s*t", low) or "kilo" in low or "thousand tonn" in low:
            return 1_000.0
        if re.search(r"(mt|metric\s*ton|tonne|tco2|tco₂|co2e|co₂e|\bt\b)", low):
            return 1.0
    return 1.0  # default: assume the number is already in tonnes


def to_tonnes(value: str | None, unit: str | None) -> float | None:
    """Convert (value, unit) to a float magnitude in tCO2e, or None if NA/invalid."""
    if is_na(value):
        return None
    raw = str(value)
    # Strip thousands separators and any stray unit text glued to the number.
    num_match = re.search(r"[-+]?\d[\d,\.]*", raw.replace(" ", ""))
    if not num_match:
        return None
    num_str = num_match.group(0).replace(",", "")
    try:
        num = float(num_str)
    except ValueError:
        return None

    return num * _scale_factor((unit or "").strip(), raw)
