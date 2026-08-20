#!/usr/bin/env python
"""
scripts/smoke_test_models.py
----------------------------
CI smoke test — loads all three production model bundles and runs one
prediction through each. Exits with code 1 if any model fails.

Usage:
    python scripts/smoke_test_models.py
"""

import sys
from pathlib import Path

# Force UTF-8 output on Windows (avoids cp1252 encoding errors with arrow chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure repo root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.config import MODEL_BUNDLES
from api.predictor import (
    LogisticRegressionPredictor,
    RandomForestPredictor,
    XGBoostPredictor,
)
from api.schemas import ModelName, StudentInput
from feature_engineering import make_sample_student

PREDICTOR_TYPES = {
    "logistic_regression": LogisticRegressionPredictor,
    "random_forest": RandomForestPredictor,
    "xgboost": XGBoostPredictor,
}

# Built from the canonical schema rather than a literal kwargs list, so adding
# or renaming a raw feature in feature_engineering.py cannot leave this smoke
# test silently constructing an outdated payload. .item() unwraps the numpy
# scalars a DataFrame row yields, which Pydantic will not coerce.
_sample_row = {
    name: (value.item() if hasattr(value, "item") else value)
    for name, value in make_sample_student().iloc[0].items()
}
SAMPLE_INPUT = StudentInput(model=ModelName.random_forest, **_sample_row)

PASSED = []
FAILED = []

print("\n========================================")
print("  Smoke Test — Production Model Bundles")
print("========================================\n")

for model_key, bundle in MODEL_BUNDLES.items():
    predictor_cls = PREDICTOR_TYPES.get(model_key)
    if not predictor_cls:
        print(f"  [SKIP] {model_key}: no predictor class registered")
        continue

    model_path: Path = bundle["model"]
    if not model_path.exists():
        print(f"  [SKIP] {model_key}: artifact not found at {model_path}")
        continue

    try:
        predictor = predictor_cls.load(
            preprocessor_path=bundle["preprocessor"],
            model_path=model_path,
            manifest_path=bundle.get("manifest"),
        )
        # Swap model field to match the current model key
        sample = SAMPLE_INPUT.model_copy(update={"model": ModelName(model_key)})
        result = predictor.predict(sample)
        assert result.placement_status in (0, 1), "placement_status must be 0 or 1"
        assert 0.0 <= result.probability_placed <= 1.0, "probability out of range"
        print(
            f"  [PASS] {model_key:25s} -> "
            f"placed={result.placement_status}  "
            f"prob={result.probability_placed:.4f}"
        )
        PASSED.append(model_key)
    except Exception as exc:
        print(f"  [FAIL] {model_key}: {exc}")
        FAILED.append(model_key)

print(f"\nResult: {len(PASSED)} passed, {len(FAILED)} failed\n")

if FAILED:
    print("The following models FAILED the smoke test:")
    for m in FAILED:
        print(f"  - {m}")
    sys.exit(1)

print("All models passed the smoke test.")
sys.exit(0)
