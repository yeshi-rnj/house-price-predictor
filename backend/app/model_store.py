"""Loads the trained pipeline once and exposes it to the API.

Kept separate from the route handlers. Only load model files you created
yourself: joblib files can execute code when loaded.
"""
from __future__ import annotations

import json
import logging
from importlib.metadata import version
from pathlib import Path

import joblib
import pandas as pd

from .config import METADATA_FILE, MODEL_FILE

log = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when a prediction is requested but no model is loaded."""


class ModelStore:
    def __init__(self, model_dir: Path | str):
        self.model_dir = Path(model_dir)
        self._pipeline = None
        self._metadata: dict | None = None
        self.load_error: str | None = None
        self.warnings: list[str] = []
        self.load()

    def load(self) -> None:
        self._pipeline, self._metadata, self.load_error = None, None, None
        self.warnings = []
        model_path = self.model_dir / MODEL_FILE
        meta_path = self.model_dir / METADATA_FILE
        if not model_path.is_file() or not meta_path.is_file():
            self.load_error = "Model files not found. Train a model first: python train.py"
            log.warning(self.load_error)
            return
        try:
            self._metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            self._pipeline = joblib.load(model_path)
            self._check_versions()
        except Exception:  # noqa: BLE001 - report any load problem without crashing the app
            log.exception("Failed to load model artifacts")
            self._pipeline, self._metadata = None, None
            self.load_error = "Model files could not be loaded. Re-run: python train.py"

    def _check_versions(self) -> None:
        """Warn if the serving environment differs from the training environment."""
        trained = (self._metadata or {}).get("library_versions", {})
        for package in ("scikit-learn", "numpy", "pandas"):
            expected = trained.get(package)
            if expected and version(package) != expected:
                msg = f"{package} {version(package)} is installed but the model was trained with {expected}."
                log.warning(msg)
                self.warnings.append(msg)

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None and self._metadata is not None

    @property
    def metadata(self) -> dict:
        if not self.is_loaded:
            raise ModelNotAvailableError(self.load_error or "Model not loaded")
        return self._metadata  # type: ignore[return-value]

    def predict_lakh(self, values: dict) -> float:
        """Predict price in lakh INR from already-validated feature values."""
        meta = self.metadata
        columns = meta["features"]["all"]
        frame = pd.DataFrame([{c: values[c] for c in columns}], columns=columns)
        return float(self._pipeline.predict(frame)[0])  # type: ignore[union-attr]
