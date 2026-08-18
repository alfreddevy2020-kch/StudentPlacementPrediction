"""
scripts/package_model.py
------------------------
Utility to package model and preprocessor artifacts into the production
artifacts directory with SHA-256 integrity hashing and manifest generation.

Usage:
    python scripts/package_model.py \\
        --model-name logistic_regression \\
        --model-version 2026.08.18-lr.1 \\
        --preprocessor part2/models/preprocessor.joblib \\
        --model part2/models/logistic_regression_best.joblib \\
        --overwrite
"""

import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

import joblib


def compute_sha256(filepath: Path) -> str:
    """Compute the SHA-256 checksum of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_model_class_name(model_path: Path) -> str:
    """Attempt to load the model and determine its class name."""
    try:
        model = joblib.load(model_path)
        return type(model).__name__
    except Exception:
        return "UnknownClassifier"


def package_model(
    model_name: str,
    model_version: str,
    preprocessor_src: Path,
    model_src: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Package model and preprocessor into destination folder and generate manifest."""
    if not preprocessor_src.exists():
        raise FileNotFoundError(f"Source preprocessor not found: {preprocessor_src}")
    if not model_src.exists():
        raise FileNotFoundError(f"Source model not found: {model_src}")

    target_dir = output_dir / model_name
    target_dir.mkdir(parents=True, exist_ok=True)

    dest_preprocessor = target_dir / "preprocessor.joblib"
    dest_model = target_dir / "model.joblib"
    dest_manifest = target_dir / "manifest.json"

    if not overwrite:
        existing = [p.name for p in (dest_preprocessor, dest_model, dest_manifest) if p.exists()]
        if existing:
            raise FileExistsError(
                f"Target files already exist in {target_dir}: {existing}. "
                "Use --overwrite to replace them."
            )

    # Copy artifacts
    shutil.copy2(preprocessor_src, dest_preprocessor)
    shutil.copy2(model_src, dest_model)

    # Compute checksums
    prep_sha256 = compute_sha256(dest_preprocessor)
    model_sha256 = compute_sha256(dest_model)
    model_class = get_model_class_name(dest_model)

    created_at_utc = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    manifest_data = {
        "schema_version": 1,
        "model_name": model_name,
        "model_version": model_version,
        "model_class": model_class,
        "created_at_utc": created_at_utc,
        "artifacts": {
            "preprocessor": {
                "filename": "preprocessor.joblib",
                "sha256": prep_sha256,
            },
            "model": {
                "filename": "model.joblib",
                "sha256": model_sha256,
            },
        },
    }

    with open(dest_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"\n[OK] Model successfully packaged: {model_name}")
    print(f"     Directory    : {target_dir}")
    print(f"     Model Class  : {model_class}")
    print(f"     Version      : {model_version}")
    print(f"     Preprocessor : {prep_sha256}")
    print(f"     Model Hash   : {model_sha256}")

    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Package model artifacts for production.")
    parser.add_argument("--model-name", required=True, help="Model name identifier (e.g. random_forest)")
    parser.add_argument("--model-version", required=True, help="Model version string (e.g. 2026.08.18-rf.1)")
    parser.add_argument("--preprocessor", required=True, type=Path, help="Path to source preprocessor.joblib")
    parser.add_argument("--model", required=True, type=Path, help="Path to source model file (.joblib)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/production"),
        help="Root directory for production artifacts (default: artifacts/production)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files in target folder")

    args = parser.parse_args()

    try:
        package_model(
            model_name=args.model_name,
            model_version=args.model_version,
            preprocessor_src=args.preprocessor,
            model_src=args.model,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"\n[FAIL] Packaging error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
