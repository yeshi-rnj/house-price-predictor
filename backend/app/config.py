"""Central configuration: paths, feature schema, limits and dataset info.

Both the training script and the API import from here, so the feature list
cannot silently drift between training and inference.
"""
from __future__ import annotations

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = BACKEND_DIR / "models"
DEFAULT_DATA_PATH = BACKEND_DIR / "data" / "raw" / "Bengaluru_House_Data.csv"

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
METRICS_FILE = "metrics.json"
MODEL_REQUIREMENTS_FILE = "model-requirements.txt"

RANDOM_SEED = 42
TEST_SIZE = 0.2

# --- Feature schema -------------------------------------------------------
CATEGORICAL_FEATURES = ["location", "area_type"]
NUMERIC_FEATURES = ["total_sqft", "bhk", "bath", "balcony"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "price_lakh"  # 1 lakh = 100,000 rupees
LAKH = 100_000

FEATURE_LABELS = {
    "location": "Location",
    "area_type": "Area measured as",
    "total_sqft": "Area (sq ft)",
    "bhk": "Bedrooms",
    "bath": "Bathrooms",
    "balcony": "Balconies",
}

# Inclusive limits enforced by the API and used for cleaning the training data.
FIELD_LIMITS = {
    "total_sqft": (200, 30000),
    "bhk": (1, 10),
    "bath": (1, 10),
    "balcony": (0, 5),
}
INTEGER_FIELDS = {"bhk", "bath", "balcony"}

# --- Cleaning heuristics (decisions to revisit after looking at the data) ---
MIN_SQFT_PER_BHK = 300  # drop listings with implausibly little space per bedroom
MAX_EXTRA_BATHS = 2     # drop listings with more than bhk + 2 bathrooms
TRIM_QUANTILES = (0.01, 0.99)  # price-per-sqft trimming, TRAINING split only

# --- Dataset provenance ---------------------------------------------------
DATASET_INFO = {
    "name": "Bengaluru House Price Data",
    "source_url": "https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data",
    "license_note": (
        "Check the license shown on the Kaggle dataset page. The data file is "
        "not redistributed in this repository."
    ),
    "price_unit": "lakh INR (1 lakh = 100,000 rupees)",
    "coverage": "Bengaluru, India",
}

DISCLAIMER = (
    "This is a statistical estimate from a model trained on listing data. "
    "It is not a professional property valuation and should not be used as "
    "the sole basis for any financial decision."
)
