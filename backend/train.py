"""Train, compare and save the house-price model.

Usage (from the backend folder, venv active):
    python train.py --data data/raw/Bengaluru_House_Data.csv

Pipeline: clean -> split -> trim outliers (training split only) -> cross-validate and
fit candidates -> choose by CV -> report once on the untouched test split -> save.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config import (
    CATEGORICAL_FEATURES,
    DATASET_INFO,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_DIR,
    FEATURES,
    LAKH,
    METADATA_FILE,
    METRICS_FILE,
    MODEL_FILE,
    MODEL_REQUIREMENTS_FILE,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
    TRIM_QUANTILES,
)
from app.features import clean_raw_dataframe

MODEL_LABELS = {
    "baseline": "Baseline (always predicts the training mean)",
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting (log-price target)",
}

CV_FOLDS = 5


def build_preprocessor() -> ColumnTransformer:
    """Same object is saved inside the pipeline, so inference matches training."""
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # Locations with fewer than 10 training rows are pooled together;
            # unseen categories are mapped to that pool instead of crashing.
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10, sparse_output=False)),
        ]
    )
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer(
        [("cat", categorical, CATEGORICAL_FEATURES), ("num", numeric, NUMERIC_FEATURES)],
        sparse_threshold=0,
    )


def build_candidates(seed: int) -> dict[str, Pipeline]:
    models = {
        "baseline": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        # min_samples_leaf=3 and 150 trees keep the saved file reasonably small.
        "random_forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=3, random_state=seed, n_jobs=-1
        ),
        # House prices are right-skewed, so this model learns log(1 + price).
        "gradient_boosting": TransformedTargetRegressor(
            regressor=HistGradientBoostingRegressor(
                learning_rate=0.08, max_iter=400, early_stopping=True, random_state=seed
            ),
            func=np.log1p,
            inverse_func=np.expm1,
        ),
    }
    return {name: Pipeline([("prep", build_preprocessor()), ("model", m)]) for name, m in models.items()}


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    pred = pipeline.predict(X)
    return {
        "mae": round(float(mean_absolute_error(y, pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 4),
        "r2": round(float(r2_score(y, pred)), 4),
    }


def trim_training_outliers(train_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop extreme price-per-sqft rows from the TRAINING split only.

    price_per_sqft uses the target, so it is never a feature, and the test
    split is never trimmed (so test results reflect realistic data).
    """
    ppsf = train_df[TARGET] * LAKH / train_df["total_sqft"]
    lo, hi = ppsf.quantile(TRIM_QUANTILES[0]), ppsf.quantile(TRIM_QUANTILES[1])
    kept = train_df[(ppsf >= lo) & (ppsf <= hi)].reset_index(drop=True)
    return kept, {
        "price_per_sqft_lower": round(float(lo), 2),
        "price_per_sqft_upper": round(float(hi), 2),
        "rows_removed": int(len(train_df) - len(kept)),
    }


def run_training(
    clean_df: pd.DataFrame,
    model_dir: Path | str,
    dataset_info: dict,
    cleaning_report: dict | None = None,
    seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    verbose: bool = True,
) -> dict:
    """Fit candidates on clean_df, save the best pipeline plus metadata. Returns metadata."""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(clean_df, test_size=test_size, random_state=seed)
    train_df, trim_info = trim_training_outliers(train_df.reset_index(drop=True))
    test_df = test_df.reset_index(drop=True)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    metrics: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=seed)
    for name, pipeline in build_candidates(seed).items():
        # Cross-validation uses the TRAINING split only; the test split is not touched here.
        cv_rmse = -cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="neg_root_mean_squared_error")
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        metrics[name] = {
            "cv": {"rmse_mean": round(float(cv_rmse.mean()), 4), "rmse_std": round(float(cv_rmse.std()), 4), "folds": CV_FOLDS},
            "train": evaluate(pipeline, X_train, y_train),
            "test": evaluate(pipeline, X_test, y_test),
        }
        if verbose:
            t, c = metrics[name]["test"], metrics[name]["cv"]
            print(
                f"{name:<18} cv RMSE={c['rmse_mean']:.3f}+/-{c['rmse_std']:.3f} | "
                f"test MAE={t['mae']:.3f} RMSE={t['rmse']:.3f} R2={t['r2']:.3f}"
            )

    # Select with cross-validation on the training split, so the test set stays an
    # honest, single-use final check rather than part of model selection.
    contenders = {k: v for k, v in metrics.items() if k != "baseline"}
    selected = min(contenders, key=lambda k: contenders[k]["cv"]["rmse_mean"])
    baseline_beaten = metrics[selected]["test"]["rmse"] < metrics["baseline"]["test"]["rmse"]
    if verbose:
        print(f"Selected (lowest cross-validated RMSE): {selected}")
        if not baseline_beaten:
            print("WARNING: the selected model does not beat the baseline. Do not trust it.")

    best = fitted[selected]
    imp = permutation_importance(
        best, X_test, y_test, scoring="neg_root_mean_squared_error", n_repeats=5, random_state=seed
    )
    importance = sorted(
        (
            {"feature": f, "rmse_increase_lakh": round(float(m), 4), "std": round(float(s), 4)}
            for f, m, s in zip(FEATURES, imp.importances_mean, imp.importances_std)
        ),
        key=lambda d: d["rmse_increase_lakh"],
        reverse=True,
    )

    joblib.dump(best, model_dir / MODEL_FILE, compress=3)
    model_size_mb = round((model_dir / MODEL_FILE).stat().st_size / 1_048_576, 2)
    versions = {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "joblib")}
    # Exact pins matching the saved model, used by the deploy build so server and model agree.
    (model_dir / MODEL_REQUIREMENTS_FILE).write_text(
        "".join(f"{p}=={v}\n" for p, v in versions.items()), encoding="utf-8"
    )
    if verbose:
        print(f"Model file size: {model_size_mb} MB (GitHub rejects files over 100 MB)")

    cat_prep = best.named_steps["prep"].named_transformers_["cat"].named_steps["onehot"]
    categories = {
        name: sorted(str(c) for c in cats) for name, cats in zip(CATEGORICAL_FEATURES, cat_prep.categories_)
    }
    numeric_ranges = {
        c: {"min": float(train_df[c].min()), "max": float(train_df[c].max())} for c in NUMERIC_FEATURES
    }
    metadata = {
        "schema_version": 1,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": seed,
        "test_size": test_size,
        "dataset": dataset_info,
        "target": {"name": TARGET, "unit": "lakh INR"},
        "features": {"all": FEATURES, "categorical": CATEGORICAL_FEATURES, "numeric": NUMERIC_FEATURES},
        "categories": categories,
        "numeric_ranges": numeric_ranges,
        "data": {
            "rows_after_cleaning": int(len(clean_df)),
            "rows_train": int(len(train_df)),
            "rows_test": int(len(test_df)),
            "training_outlier_trim": trim_info,
        },
        "cleaning_report": cleaning_report or {},
        "selected_model": selected,
        "selected_model_label": MODEL_LABELS[selected],
        "baseline_beaten": bool(baseline_beaten),
        "metrics": metrics,
        "permutation_importance": importance,
        "model_size_mb": model_size_mb,
        "selection_rule": f"lowest {CV_FOLDS}-fold cross-validated RMSE on the training split",
        "library_versions": {"python": platform.python_version(), **versions},
    }
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (model_dir / METRICS_FILE).write_text(
        json.dumps({"selected_model": selected, "unit": "lakh INR", "metrics": metrics}, indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(f"Saved model and metadata to {model_dir}")
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH, help="path to the raw CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_DIR, help="output directory")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    args = parser.parse_args(argv)

    if not args.data.is_file():
        print(
            f"Dataset not found at {args.data}\nDownload it manually (see README) and place it there.",
            file=sys.stderr,
        )
        return 1

    raw = pd.read_csv(args.data)
    clean_df, report = clean_raw_dataframe(raw)
    print(f"Rows: raw={report['rows_raw']}, clean={report['rows_clean']}")
    for reason, count in report["dropped"].items():
        print(f"  dropped {count:>5}  {reason}")

    info = {**DATASET_INFO, "file_name": args.data.name}
    run_training(clean_df, args.out, info, report, seed=args.seed, test_size=args.test_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
