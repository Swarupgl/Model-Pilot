# ModelPilot

**Autonomous, Confidence-Aware ML Algorithm Selection and Optimization using Neuro-SAN**

*Cognizant Hackathon Project*

ModelPilot investigates whether historical machine-learning experiments can predict which algorithms are likely to perform well on a new dataset — reducing unnecessary model training while maintaining competitive predictive performance.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Workspace and compatibility audit | ✅ Complete |
| Phase 1 | Minimal working ML application | ✅ Complete |
| Phase 2 | Dataset diagnosis and meta-features | 🔜 Next |
| Phase 3 | Experiment database and algorithm recommendation | 🔮 Planned |
| Phase 4 | Meta-learning performance predictor | 🔮 Planned |
| Phase 5 | Selective evaluation with budget control | 🔮 Planned |
| Phase 6 | Neuro-SAN agent integration + OpenRouter | 🔮 Planned |
| Phase 7 | Streamlit frontend | 🔮 Planned |

---

## Quick Start

### Prerequisites

- Python 3.13 or 3.14 (Windows)
- PowerShell

### 1. Clone and set up

```powershell
# From the project root:
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Run the pipeline (built-in Iris dataset)

```powershell
python scripts/run_pipeline.py
```

### 3. Try other datasets and options

```powershell
# 3-class classification
python scripts/run_pipeline.py --dataset wine

# Binary classification
python scripts/run_pipeline.py --dataset breast_cancer

# Regression
python scripts/run_pipeline.py --dataset diabetes

# Your own CSV file
python scripts/run_pipeline.py --csv path/to/data.csv --target my_column

# Control evaluation budget and folds
python scripts/run_pipeline.py --dataset iris --max-candidates 4 --cv-folds 3
```

### 4. Run the test suite

```powershell
python -m pytest tests/ -v
```

Expected output: all tests green.

---

## Project Structure

```
model pilot/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example                  ← Copy to .env for API keys (Phase 6)
├── configs/
│   ├── algorithms.yaml           ← Algorithm registry (add/remove algorithms here)
│   └── experiment.yaml           ← Evaluation protocol settings
├── data/
│   └── sample/                   ← Place your CSV files here
├── backend/                      ← Neuro-SAN integration (Phase 6)
│   ├── run.py
│   ├── registries/
│   └── coded_tools/
├── frontend/                     ← Streamlit UI (Phase 7)
│   └── run.py
├── src/
│   └── modelpilot/
│       ├── data/
│       │   └── loader.py         ← CSV and built-in dataset loading
│       ├── evaluation/
│       │   └── model_evaluator.py ← Fair CV evaluation with Pipelines
│       └── experiments/
│           └── result_store.py   ← JSON experiment persistence
├── scripts/
│   └── run_pipeline.py           ← CLI entry point
├── tests/
│   ├── conftest.py
│   ├── test_loader.py
│   ├── test_evaluator.py
│   └── test_result_store.py
└── outputs/
    └── experiments/              ← Saved experiment JSON files (gitignored)
```

---

## Research Hypothesis

> "Can confidence-aware, cross-dataset algorithm performance prediction and adaptive experiment selection reduce model-selection computation while maintaining competitive predictive performance on previously unseen datasets?"

This is a hypothesis under investigation, not an established result.

---

## Supported Algorithms

Defined in `configs/algorithms.yaml`. No code changes needed to add/remove algorithms.

**Classification:** DummyClassifier (baseline), LogisticRegression, KNeighborsClassifier, SVC, DecisionTreeClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, GaussianNB

**Regression:** DummyRegressor (baseline), LinearRegression, Ridge, KNeighborsRegressor, SVR, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor

---

## Experiment Output Format

Each run saves a JSON file in `outputs/experiments/`:

```json
{
  "experiment_id": "uuid",
  "timestamp_utc": "ISO 8601",
  "dataset_source": "builtin:iris",
  "task_type": "classification",
  "target_column": "target",
  "config": { "cv_folds": 5, "random_seed": 42 },
  "n_candidates_evaluated": 5,
  "results": [ ... ],
  "best_algorithm": {
    "algorithm": "RandomForestClassifier",
    "primary_metric": "balanced_accuracy",
    "primary_score": 0.96,
    "primary_std": 0.02
  }
}
```

---

## Key Design Decisions

1. **No data leakage**: All preprocessing (imputation, scaling) is wrapped in sklearn Pipelines and fitted only on training folds.
2. **Config-driven algorithms**: `configs/algorithms.yaml` is the single source of truth — no algorithm names in code.
3. **Deterministic by default**: Random seed 42 everywhere. Override with `--seed`.
4. **Loader never modifies data**: Missing values, outliers, and duplicates are reported but never automatically removed.
5. **Failed evaluations are recorded**: Status `"failed"` with the error message — never silently dropped.

---

## Acknowledgements

- [Cognizant Neuro-SAN](https://github.com/cognizant-ai-lab/neuro-san) — agent framework
- [Reference project](https://github.com/kaushik-cognizant/neuro-san-project-example) — structural patterns
- [OpenML](https://www.openml.org/) — inspiration for meta-learning benchmarks
- [FLAML Zero-Shot AutoML](https://microsoft.github.io/FLAML/docs/Use-Cases/Zero-Shot-AutoML/) — related work
