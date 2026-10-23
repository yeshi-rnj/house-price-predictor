"""Validation of prediction requests against the trained model's metadata."""
from __future__ import annotations

import math

from .config import (
    FEATURE_LABELS,
    FIELD_LIMITS,
    INTEGER_FIELDS,
    MAX_EXTRA_BATHS,
    MIN_SQFT_PER_BHK,
    NUMERIC_FEATURES,
)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_payload(payload: dict, metadata: dict) -> tuple[dict, dict, list]:
    """Return (clean_values, errors, warnings).

    errors maps field name -> message; clean_values is only meaningful when
    errors is empty.
    """
    errors: dict[str, str] = {}
    clean: dict = {}
    warnings: list[str] = []

    # Categorical fields: must be a known category (case-insensitive match).
    for name in metadata["features"]["categorical"]:
        label = FEATURE_LABELS[name]
        raw = payload.get(name)
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            errors[name] = f"{label} is required."
            continue
        if not isinstance(raw, str):
            errors[name] = f"{label} must be text."
            continue
        lookup = {c.lower(): c for c in metadata["categories"][name]}
        match = lookup.get(raw.strip().lower())
        if match is None:
            errors[name] = f"'{raw.strip()[:60]}' is not a {label.lower()} the model was trained on."
        else:
            clean[name] = match

    # Numeric fields: required, finite, within limits, whole numbers where needed.
    for name in NUMERIC_FEATURES:
        label = FEATURE_LABELS[name]
        lo, hi = FIELD_LIMITS[name]
        raw = payload.get(name)
        if raw is None or raw == "":
            errors[name] = f"{label} is required."
            continue
        if not _is_number(raw) or not math.isfinite(raw):
            errors[name] = f"{label} must be a number."
            continue
        if name in INTEGER_FIELDS and float(raw) != int(raw):
            errors[name] = f"{label} must be a whole number."
            continue
        if not lo <= raw <= hi:
            errors[name] = f"{label} must be between {lo:,} and {hi:,}."
            continue
        clean[name] = int(raw) if name in INTEGER_FIELDS else float(raw)

    if errors:
        return clean, errors, warnings

    # Plausibility warnings (not errors): the training data excluded these cases.
    if clean["total_sqft"] / clean["bhk"] < MIN_SQFT_PER_BHK:
        warnings.append(
            f"Less than {MIN_SQFT_PER_BHK} sq ft per bedroom. Listings like this were "
            "excluded from the training data, so the estimate may be unreliable."
        )
    if clean["bath"] > clean["bhk"] + MAX_EXTRA_BATHS:
        warnings.append(
            "Unusually many bathrooms for this number of bedrooms. Listings like this "
            "were excluded from the training data."
        )
    ranges = metadata.get("numeric_ranges", {})
    sqft_range = ranges.get("total_sqft")
    if sqft_range and not sqft_range["min"] <= clean["total_sqft"] <= sqft_range["max"]:
        warnings.append(
            f"Area is outside the range seen in training "
            f"({sqft_range['min']:,.0f} to {sqft_range['max']:,.0f} sq ft)."
        )
    return clean, errors, warnings
