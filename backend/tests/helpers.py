"""Test helpers.

SYNTHETIC DATA WARNING: the data generated here exists only so the tests can
build a tiny model in a temporary folder. It is not real housing data and is
never used for the model shipped with the app.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import DATASET_INFO, TARGET
from train import run_training


def synthetic_clean_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locations = np.array(["Alpha Nagar", "Beta Layout", "Gamma Colony", "Delta Town"])
    area_types = np.array(["Super built-up  Area", "Built-up  Area", "Plot  Area"])
    loc_factor = {"Alpha Nagar": 1.4, "Beta Layout": 1.0, "Gamma Colony": 0.8, "Delta Town": 1.2}
    location = rng.choice(locations, n)
    sqft = rng.uniform(500, 3000, n).round()
    bhk = np.clip((sqft / 600).round(), 1, 6).astype(float)
    bath = np.clip(bhk + rng.integers(-1, 2, n), 1, 8).astype(float)
    balcony = rng.integers(0, 4, n).astype(float)
    price = sqft * 0.05 * np.array([loc_factor[x] for x in location]) + rng.normal(0, 5, n)
    return pd.DataFrame(
        {
            "location": location,
            "area_type": rng.choice(area_types, n),
            "total_sqft": sqft,
            "bhk": bhk,
            "bath": bath,
            "balcony": balcony,
            TARGET: np.clip(price, 10, None),
        }
    )


def build_test_model(model_dir) -> dict:
    return run_training(
        synthetic_clean_frame(),
        model_dir,
        {**DATASET_INFO, "file_name": "synthetic_for_tests.csv"},
        verbose=False,
    )
