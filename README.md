# ModelPilot

**Autonomous, Confidence-Aware ML Algorithm Selection and Optimization using Neuro-SAN**

*Cognizant Hackathon Project*

ModelPilot investigates whether historical machine-learning experiments can predict which algorithms are likely to perform well on a new dataset — reducing unnecessary model training while maintaining competitive predictive performance.

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Workspace and compatibility audit | Completed |
| Phase 1 | Minimal working ML application | Completed |
| Phase 2 | Dataset diagnosis and meta-features | Completed |
| Phase 3 | Experiment database and algorithm recommendation | Completed |
| Phase 4 | Meta-learning performance predictor | Completed |
| Phase 5 | Selective evaluation with budget control | Completed |
| Phase 6 | Neuro-SAN agent integration + OpenRouter | Completed |
| Phase 7 | Streamlit frontend and demonstration | Completed |

---

## Architecture Overview

ModelPilot follows a modular, dual-engine architecture:
1. **Deterministic ML Engine (`src/modelpilot/`)**: Functions independently of any LLM for dataset diagnostics, meta-feature extraction, statistical performance prediction, and cross-validation pipelines.
2. **Autonomous Coordinator Agent (`backend/`)**: Built on Cognizant's **Neuro-SAN** multi-agent framework and OpenRouter (`openai/gpt-4o-mini`). The coordinator reasons over user goals, calls coded tools, and provides evidence-backed explanations.
3. **Interactive UI (`frontend/`)**: Streamlit dashboard combining diagnostics, predictions with uncertainty bounds, selective evaluation, live agent chat, and experiment provenance.

```
ModelPilot/
├── configs/
│   ├── algorithms.yaml           # Algorithm registry (17 supported algorithms)
│   └── experiment.yaml           # Protocol & budget settings
├── backend/
│   ├── run.py                    # Neuro-SAN runner with OpenRouter
│   ├── registries/
│   │   ├── manifest.hocon        # Agent registry
│   │   ├── modelpilot.hocon      # ModelPilot agent network definition
│   │   └── llm_config.hocon      # OpenRouter configuration
│   └── coded_tools/
│       └── modelpilot/
│           ├── dataset_diagnostics_tool.py
│           ├── algorithm_recommendation_tool.py
│           ├── model_evaluation_tool.py
│           └── performance_prediction_tool.py
├── frontend/
│   └── run.py                    # Streamlit interactive interface
├── src/
│   └── modelpilot/
│       ├── data/loader.py        # Dataset loading & task inference
│       ├── diagnosis/dataset_diagnostics.py
│       ├── meta_learning/
│       │   ├── meta_feature_extractor.py
│       │   └── performance_predictor.py
│       ├── selection/algorithm_selector.py
│       ├── evaluation/model_evaluator.py
│       ├── optimization/budget_optimizer.py
│       └── experiments/
│           ├── experiment_store.py  # SQLite database
│           └── result_store.py      # JSON persistence
├── scripts/
│   └── run_pipeline.py           # CLI pipeline entry point
└── tests/                        # 72 automated tests (100% pass)
```

---

## Quick Start (PowerShell on Windows)

### 1. Environment Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Credentials

Create `.env` in the project root:
```ini
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_API_BASE=https://openrouter.ai/api/v1
```

### 3. Run Automated Tests (72 tests)

```powershell
python -m pytest tests/ -v
```

### 4. Run CLI Pipeline (Headless ML Engine)

```powershell
# Built-in classification benchmark
python scripts/run_pipeline.py --dataset iris

# Built-in regression benchmark
python scripts/run_pipeline.py --dataset diabetes

# Custom CSV
python scripts/run_pipeline.py --csv path/to/dataset.csv --target target_column
```

### 5. Run Neuro-SAN Agent (Headless Agent Coordinator)

```powershell
python backend/run.py
```

### 6. Launch Streamlit Web Dashboard

```powershell
streamlit run frontend/run.py
```

---

## Core Capabilities

### 1. Dataset Diagnostics
- Non-destructive profiling: shape, memory usage, sparsity, missingness ratios.
- Duplicate row and constant/quasi-constant feature detection.
- Multicollinearity check (Pearson $|r| \ge 0.90$).
- Target leakage detection and class imbalance ratio / entropy analysis.

### 2. Standardized Meta-Feature Extraction
- Reproducible 16-dimensional numerical vector per dataset.
- Captures structural, statistical, variance, correlation, and distribution properties.
- SHA-256 fingerprinting for dataset version tracking and provenance.

### 3. Confidence-Aware Meta-Learning Predictor
- Predicts validation scores before training using historical dataset results.
- Tree-variance uncertainty quantification ($\pm 1.96\sigma$).
- Out-of-Distribution (OOD) distance detector preventing unfounded extrapolation on unfamiliar datasets.

### 4. Budget-Constrained Selective Evaluation
- Upper Confidence Bound (UCB) candidate acquisition balancing exploitation and exploration.
- Measurable compute savings (typically $40\%-60\%$ compute reduction vs. brute-force search).
- Zero data leakage: preprocessing fitted strictly within cross-validation folds.

---

## Research Benchmark Baselines

ModelPilot evaluates algorithms against:
1. **Dummy Baseline**: Uniform / most frequent / mean strategy.
2. **Transparent Heuristic Selector**: Rule-based scoring with explicit domain justifications.
3. **Historical Similarity Recommendation**: Meta-feature distance matching to nearest historical datasets.
4. **Supervised Meta-Predictor**: Random Forest over dataset meta-features and algorithm identity with Leave-One-Dataset-Out cross-validation.
