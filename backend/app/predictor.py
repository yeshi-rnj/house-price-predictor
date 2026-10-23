"""Turns validated inputs into the API response. Only reports measured facts."""
from __future__ import annotations

from .config import DISCLAIMER, FEATURE_LABELS, FIELD_LIMITS, INTEGER_FIELDS, LAKH, NUMERIC_FEATURES
from .formatting import format_inr, inr_in_words
from .model_store import ModelStore


class UnreliablePredictionError(ValueError):
    """The model produced a result that cannot be a real price."""


def predict_response(store: ModelStore, values: dict, warnings: list[str]) -> dict:
    meta = store.metadata
    lakh = store.predict_lakh(values)
    rupees = round(lakh * LAKH, -3)  # nearest Rs 1,000: avoids false precision
    if rupees <= 0:
        raise UnreliablePredictionError(
            "The model could not produce a reliable estimate for these details."
        )

    selected = meta["selected_model"]
    test_metrics = meta["metrics"][selected]["test"]
    importance = meta.get("permutation_importance", [])[:3]
    top_inputs = [FEATURE_LABELS.get(i["feature"], i["feature"]) for i in importance]

    summary = (
        f"This estimate comes from a {meta['selected_model_label']} trained on "
        f"{meta['data']['rows_train']:,} listings from the {meta['dataset']['name']} dataset. "
        f"On {meta['data']['rows_test']:,} listings it had not seen during training, its average "
        f"absolute error was about \u20b9{test_metrics['mae']:.1f} lakh. Errors for individual "
        "properties can be much larger."
    )
    return {
        "predicted_price": int(rupees),
        "currency": "INR",
        "formatted_price": format_inr(rupees),
        "price_in_words": inr_in_words(rupees),
        "price_per_sqft": int(round(rupees / values["total_sqft"])),
        "inputs": values,
        "explanation": {
            "summary": summary,
            "model": meta["selected_model_label"],
            "typical_error_lakh": test_metrics["mae"],
            "most_influential_inputs": top_inputs,
            "influence_note": (
                "Ranked by how much the model's test error grew when that input was "
                "shuffled. This describes the model overall, not this one property."
            ),
        },
        "warnings": warnings,
        "disclaimer": DISCLAIMER,
    }


def describe_fields(meta: dict) -> list[dict]:
    """Form-field descriptions for the frontend, built from the trained model."""
    fields = []
    for name in meta["features"]["categorical"]:
        fields.append(
            {
                "name": name,
                "label": FEATURE_LABELS[name],
                "type": "category",
                "required": True,
                "options": meta["categories"][name],
            }
        )
    for name in NUMERIC_FEATURES:
        lo, hi = FIELD_LIMITS[name]
        fields.append(
            {
                "name": name,
                "label": FEATURE_LABELS[name],
                "type": "number",
                "required": True,
                "min": lo,
                "max": hi,
                "integer": name in INTEGER_FIELDS,
            }
        )
    return fields
