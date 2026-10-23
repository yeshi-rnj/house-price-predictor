"""HTTP routes. Handlers stay thin: parse, validate, delegate, respond."""
from __future__ import annotations

from flask import Blueprint, current_app, jsonify, request

from .model_store import ModelNotAvailableError, ModelStore
from .predictor import UnreliablePredictionError, describe_fields, predict_response
from .validation import validate_payload

api = Blueprint("api", __name__, url_prefix="/api")


def _store() -> ModelStore:
    return current_app.extensions["model_store"]


def _unavailable(store: ModelStore):
    return (
        jsonify({"error": "model_unavailable", "message": store.load_error or "Model not loaded."}),
        503,
    )


@api.get("/health")
def health():
    store = _store()
    if not store.is_loaded:
        return jsonify({"status": "degraded", "model_loaded": False, "message": store.load_error}), 503
    body = {"status": "ok", "model_loaded": True}
    if store.warnings:
        body["warnings"] = store.warnings
    return jsonify(body)


@api.get("/metadata")
def metadata():
    store = _store()
    if not store.is_loaded:
        return _unavailable(store)
    meta = store.metadata
    return jsonify(
        {
            "fields": describe_fields(meta),
            "currency": "INR",
            "model": {
                "selected": meta["selected_model"],
                "selected_label": meta["selected_model_label"],
                "trained_at": meta["trained_at"],
                "metrics": meta["metrics"],
                "metric_unit": "lakh INR",
                "rows": meta["data"],
                "permutation_importance": meta.get("permutation_importance", []),
                "baseline_beaten": meta.get("baseline_beaten"),
            },
            "dataset": meta["dataset"],
            "library_versions": meta.get("library_versions", {}),
        }
    )


@api.post("/predict")
def predict():
    store = _store()
    if not store.is_loaded:
        return _unavailable(store)

    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid_request", "message": "Send a JSON object as the request body."}), 400

    try:
        values, errors, warnings = validate_payload(payload, store.metadata)
        if errors:
            return jsonify({"error": "validation_failed", "message": "Some inputs are invalid.", "errors": errors}), 422
        return jsonify(predict_response(store, values, warnings))
    except UnreliablePredictionError as exc:
        return jsonify({"error": "unreliable_prediction", "message": str(exc)}), 422
    except ModelNotAvailableError:
        return _unavailable(store)
