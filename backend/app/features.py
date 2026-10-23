"""Parsing and cleaning of the raw housing CSV.

The cleaning logic lives here (not in train.py) so it can be unit-tested and
reused by the exploration script.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd

from .config import (
    FIELD_LIMITS,
    MAX_EXTRA_BATHS,
    MIN_SQFT_PER_BHK,
    TARGET,
)

RAW_REQUIRED_COLUMNS = ["area_type", "location", "size", "total_sqft", "bath", "balcony", "price"]


def parse_sqft(value) -> float:
    """Convert raw area text to a float.

    "1200" -> 1200.0, "2100 - 2850" -> 2475.0 (midpoint of a range).
    Values with units such as "34.46Sq. Meter" return NaN, because converting
    them silently could hide mistakes.
    """
    if value is None:
        return math.nan
    text = str(value).strip()
    try:
        return float(text)
    except ValueError:
        pass
    parts = [p.strip() for p in text.split(" - ")]
    if len(parts) == 2:
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return math.nan
    return math.nan


def parse_bhk(value) -> float:
    """Extract the leading integer from text like "2 BHK" or "4 Bedroom"."""
    if value is None:
        return math.nan
    match = re.match(r"^\s*(\d+)", str(value))
    return float(match.group(1)) if match else math.nan


def _clean_text(value):
    return value.strip() if isinstance(value, str) and value.strip() else np.nan


def clean_raw_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return (clean_df, report). Every dropped row is counted in the report."""
    missing_cols = [c for c in RAW_REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(
            f"CSV is missing expected columns: {missing_cols}. "
            f"Found columns: {list(raw.columns)}"
        )

    report: dict = {"rows_raw": int(len(raw)), "dropped": {}}

    deduped = raw.drop_duplicates()
    report["dropped"]["exact_duplicates"] = int(len(raw) - len(deduped))

    df = pd.DataFrame(
        {
            "location": deduped["location"].map(_clean_text).astype(object),
            "area_type": deduped["area_type"].map(_clean_text).astype(object),
            "total_sqft": deduped["total_sqft"].map(parse_sqft),
            "bhk": deduped["size"].map(parse_bhk),
            "bath": pd.to_numeric(deduped["bath"], errors="coerce"),
            "balcony": pd.to_numeric(deduped["balcony"], errors="coerce"),
            TARGET: pd.to_numeric(deduped["price"], errors="coerce"),
        }
    ).reset_index(drop=True)

    def keep(frame: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
        report["dropped"][label] = int((~mask).sum())
        return frame[mask].reset_index(drop=True)

    required = ["location", "area_type", "total_sqft", "bhk", "bath", TARGET]
    df = keep(df, df[required].notna().all(axis=1), "missing_or_unparseable_required_value")
    df = keep(df, np.isfinite(df[["total_sqft", "bhk", "bath", TARGET]]).all(axis=1), "non_finite_value")
    df = keep(df, df[TARGET] > 0, "non_positive_price")

    lo, hi = FIELD_LIMITS["total_sqft"]
    df = keep(df, df["total_sqft"].between(lo, hi), "area_outside_limits")
    lo, hi = FIELD_LIMITS["bhk"]
    df = keep(df, df["bhk"].between(lo, hi), "bedrooms_outside_limits")
    lo, hi = FIELD_LIMITS["bath"]
    df = keep(df, df["bath"].between(lo, hi), "bathrooms_outside_limits")
    lo, hi = FIELD_LIMITS["balcony"]
    df = keep(df, df["balcony"].isna() | df["balcony"].between(lo, hi), "balconies_outside_limits")

    df = keep(df, df["total_sqft"] / df["bhk"] >= MIN_SQFT_PER_BHK, "too_little_area_per_bedroom")
    df = keep(df, df["bath"] <= df["bhk"] + MAX_EXTRA_BATHS, "too_many_bathrooms_for_bedrooms")

    report["rows_clean"] = int(len(df))
    report["balcony_missing_kept_for_imputation"] = int(df["balcony"].isna().sum())
    return df, report
