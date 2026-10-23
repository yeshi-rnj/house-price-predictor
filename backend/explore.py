"""Exploratory data analysis for the raw CSV. Prints findings; changes nothing.

Usage: python explore.py --data data/raw/Bengaluru_House_Data.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from app.config import DEFAULT_DATA_PATH, LAKH, TARGET
from app.features import RAW_REQUIRED_COLUMNS, clean_raw_dataframe, parse_sqft


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    args = parser.parse_args()
    if not args.data.is_file():
        print(f"Dataset not found at {args.data}. See README for the manual download.", file=sys.stderr)
        return 1

    raw = pd.read_csv(args.data)
    section("Shape and columns")
    print(raw.shape)
    print(raw.dtypes)

    section("Missing values per column")
    print(raw.isna().sum().sort_values(ascending=False))

    section("Duplicates")
    print(f"exact duplicate rows: {raw.duplicated().sum()}")

    missing = [c for c in RAW_REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        print(f"\nExpected columns not found: {missing}. Update app/features.py before continuing.")
        return 1

    section("Categorical columns")
    print(raw["area_type"].value_counts(dropna=False))
    print(f"distinct locations: {raw['location'].nunique()}")
    print(raw["location"].value_counts().head(10))
    print(f"locations with fewer than 10 rows: {(raw['location'].value_counts() < 10).sum()}")

    section("total_sqft that is not a plain number")
    parsed_plain = pd.to_numeric(raw["total_sqft"], errors="coerce")
    odd = raw.loc[parsed_plain.isna(), "total_sqft"]
    print(f"{len(odd)} values; unparseable even after range handling: {odd.map(parse_sqft).isna().sum()}")
    print(odd.head(10).to_string())

    section("Target (price, lakh INR) summary")
    print(raw["price"].describe())

    clean, report = clean_raw_dataframe(raw)
    section("Cleaning report (rows dropped per rule)")
    print(f"raw={report['rows_raw']}  clean={report['rows_clean']}")
    for reason, count in report["dropped"].items():
        print(f"  {count:>6}  {reason}")

    section("Price per sq ft in cleaned data (rupees)")
    ppsf = clean[TARGET] * LAKH / clean["total_sqft"]
    print(ppsf.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]))

    section("Leakage check")
    print("Target: price. Features used: location, area_type, total_sqft, bhk, bath, balcony.")
    print("price_per_sqft is derived from the target, so it is NOT used as a feature.")
    print("'society' (mostly missing, near-unique labels) and 'availability' (dates) are excluded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
