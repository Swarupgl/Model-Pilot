"""
scripts/run_pipeline.py

Command-line entry point for ModelPilot Phase 1.

Usage examples (run from the project root with the venv activated):

  # Evaluate on the built-in Iris dataset (default)
  python scripts/run_pipeline.py

  # Use a different built-in dataset
  python scripts/run_pipeline.py --dataset breast_cancer
  python scripts/run_pipeline.py --dataset diabetes

  # Use a CSV file
  python scripts/run_pipeline.py --csv path/to/data.csv --target my_target_col

  # Control the number of candidates and CV folds
  python scripts/run_pipeline.py --dataset iris --max-candidates 4 --cv-folds 3

This script must be run from the project root so that configs/ and outputs/ are
resolved correctly relative to the working directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure src/ is on the import path when running as a script
# This is only needed when running directly; when installed as a package
# (via pip install -e .) the src/ layout is handled by pyproject.toml.
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root / "src") not in sys.path:
    sys.path.insert(0, str(_project_root / "src"))

from modelpilot.data.loader import load_builtin, load_csv
from modelpilot.evaluation.model_evaluator import (
    evaluate_multiple,
    get_available_algorithms,
)
from modelpilot.experiments.result_store import save_experiment

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("modelpilot.pipeline")


# Main pipeline function

def run_pipeline(
    dataset_name: str | None = "iris",
    csv_path: str | None = None,
    target_column: str | None = None,
    max_candidates: int = 5,
    cv_folds: int = 5,
    random_seed: int = 42,
) -> dict:
    """
    Run the Phase 1 end-to-end pipeline.

    Returns the experiment record dict so tests can assert on it.
    """
separator = "─" * 60

# Step 1: Load data
    print(f"\n{separator}")
    print(" ModelPilot — Phase 1 Pipeline")
    print(separator)

    if csv_path:
        print(f"\n📂 Loading CSV: {csv_path}")
        dataset = load_csv(csv_path, target_column=target_column)
    else:
        name = dataset_name or "iris"
        print(f"\n📦 Loading built-in dataset: '{name}'")
        dataset = load_builtin(name, target_column=target_column)

    print(f"   Rows:        {dataset['n_rows']}")
    print(f"   Features:    {dataset['n_features']}")
    print(f"   Target:      '{dataset['target']}'")
    print(f"   Task type:   {dataset['task_type']}")

    if dataset["warnings"]:
        print("\n⚠️  Loader warnings:")
        for w in dataset["warnings"]:
            print(f"   • {w}")

# Step 2: Get candidate algorithms
    print(f"\n{separator}")
    all_algorithms = get_available_algorithms(
        task_type=dataset["task_type"],
        algorithms_config_path=_project_root / "configs" / "algorithms.yaml",
    )

    # Limit to budget — always include DummyClassifier/DummyRegressor as baseline
    baseline = all_algorithms[0]  # First in list is always the baseline
    candidates = [baseline]
    for algo in all_algorithms[1:]:
        if len(candidates) >= max_candidates:
            break
        candidates.append(algo)

    print(f"\n🔬 Candidates to evaluate ({len(candidates)} of {len(all_algorithms)} available):")
    for i, name in enumerate(candidates):
        marker = "  [baseline]" if i == 0 else ""
        print(f"   {i+1}. {name}{marker}")

# Step 3: Evaluate candidates
    print(f"\n{separator}")
    print(f"\n⚙️  Running {cv_folds}-fold cross-validation (seed={random_seed}) ...")
    print()

    results = evaluate_multiple(
        algorithm_names=candidates,
        X=dataset["X"],
        y=dataset["y"],
        task_type=dataset["task_type"],
        cv_folds=cv_folds,
        random_seed=random_seed,
        algorithms_config_path=_project_root / "configs" / "algorithms.yaml",
    )

# Step 4: Print results table
    print(f"\n{separator}")
    print("\n📊 Results (sorted by primary metric, best first):\n")

    primary_metric = results[0]["primary_metric"] if results else "score"
    header = f"  {'Rank':<5} {'Algorithm':<30} {primary_metric:<22} {'Std':<10} {'Time(s)':<10} {'Status'}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for rank, r in enumerate(results, start=1):
        if r["status"] == "ok":
            score_str = f"{r['primary_score']:.4f}"
            std_str = f"± {r['primary_std']:.4f}"
            time_str = f"{r['train_time_s']:.2f}s"
            status_str = "✅"
        else:
            score_str = "FAILED"
            std_str = ""
            time_str = ""
            status_str = f"❌ {r['error'][:40]}"

        print(f"  {rank:<5} {r['algorithm']:<30} {score_str:<22} {std_str:<10} {time_str:<10} {status_str}")

# Step 5: Show secondary metrics for best model
    best = next((r for r in results if r["status"] == "ok"), None)
    if best and best["secondary_metrics"]:
        print(f"\n🏆 Best model: {best['algorithm']}")
        print("   Secondary metrics:")
        for k, v in best["secondary_metrics"].items():
            print(f"   • {k}: {v:.4f}")

# Step 6: Save results
    print(f"\n{separator}")
    config_used = {
        "cv_folds": cv_folds,
        "random_seed": random_seed,
        "max_candidates": max_candidates,
    }
    saved_path = save_experiment(
        dataset_source=dataset["source"],
        task_type=dataset["task_type"],
        target_column=dataset["target"],
        evaluation_results=results,
        experiment_config=config_used,
        output_dir=_project_root / "outputs" / "experiments",
    )
    print(f"\n💾 Experiment saved: {saved_path}")
    print(f"\n{separator}\n")

    return {
        "dataset_source": dataset["source"],
        "task_type": dataset["task_type"],
        "results": results,
        "saved_path": str(saved_path),
    }


# CLI argument parsing

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ModelPilot Phase 1 — run baseline ML evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--dataset",
        default="iris",
        choices=["iris", "breast_cancer", "wine", "diabetes", "california"],
        help="Built-in dataset name (default: iris)",
    )
    source_group.add_argument(
        "--csv",
        metavar="PATH",
        help="Path to a CSV file",
    )
    parser.add_argument(
        "--target",
        metavar="COLUMN",
        help="Target column name (required when using --csv; optional for built-ins)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Maximum number of algorithms to evaluate (default: 5)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outcome = run_pipeline(
        dataset_name=args.dataset if not args.csv else None,
        csv_path=args.csv,
        target_column=args.target,
        max_candidates=args.max_candidates,
        cv_folds=args.cv_folds,
        random_seed=args.seed,
    )
    # Exit with a non-zero code if every candidate failed
    all_failed = all(r["status"] != "ok" for r in outcome["results"])
    sys.exit(1 if all_failed else 0)
