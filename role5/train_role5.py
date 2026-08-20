"""Generate reproducible Role 5 cohort-analysis artifacts on demand."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from feature_engineering import load_raw_dataset

from .reporting import run_role5_analysis

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"


def save_role5_analysis(output_dir: Path, bootstrap_iterations: int = 200) -> None:
    """Run and save cohort-level Role 5 artifacts; never alters classifier bundles."""
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis = run_role5_analysis(load_raw_dataset(), bootstrap_iterations=bootstrap_iterations)
    analysis.archetype_assignments.to_csv(output_dir / "archetype_assignments.csv", index=False)
    analysis.archetype_profile.to_csv(output_dir / "archetype_profile.csv", index=False)
    analysis.clustering.k_search.to_csv(output_dir / "cluster_k_search.csv", index=False)
    effects = analysis.observational_effects
    effects.cate_by_archetype.to_csv(output_dir / "cate_by_archetype.csv", index=False)
    effects.association_scores.to_csv(output_dir / "association_scores.csv", index=False)
    effects.cate_scores.to_csv(output_dir / "observational_cate_scores.csv", index=False)
    effects.diagnostics.balance_before.to_csv(output_dir / "balance_before.csv", index=False)
    effects.diagnostics.balance_after.to_csv(output_dir / "balance_after.csv", index=False)
    summary = {
        "selected_k": analysis.clustering.selected_k,
        "silhouette_score": analysis.clustering.silhouette_score,
        "bootstrap_ari_mean": analysis.clustering.bootstrap_ari_mean,
        "aggregate_ate": effects.aggregate_ate,
        "ate_ci_lower": effects.ate_ci_lower,
        "ate_ci_upper": effects.ate_ci_upper,
        "diagnostic_status": effects.diagnostics.status,
        "diagnostic_warnings": list(effects.diagnostics.warnings),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Parse CLI options and persist results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    args = parser.parse_args()
    save_role5_analysis(args.output_dir, args.bootstrap_iterations)


if __name__ == "__main__":
    main()
