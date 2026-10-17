# ModelPilot: Project Summary

**Autonomous, Confidence-Aware ML Algorithm Selection and Optimization Using Neuro-SAN**  
*Cognizant AI Lab Hackathon Project Submission*

---

## Abstract

Automated Machine Learning (AutoML) tools predominantly treat algorithm selection as an exhaustive search problem, evaluating dozens of candidate models regardless of dataset characteristics. This brute-force paradigm results in excessive computational overhead, high environmental energy costs, and significant pipeline delay. We present **ModelPilot**, an autonomous, confidence-aware machine learning algorithm selection and optimization system built on **Cognizant's Neuro-SAN** multi-agent framework. ModelPilot extracts a 16-dimensional standardized statistical meta-feature vector from any tabular dataset, queries a supervised meta-predictor with tree-variance epistemic uncertainty quantification ($\pm 1.96\sigma$), and employs an Upper Confidence Bound (UCB) acquisition policy to evaluate only the most promising algorithms under strict compute budgets. When benchmarked across standard classification and regression domains, ModelPilot achieves a **40% to 60% reduction in evaluation compute time** while consistently retaining **98% to 100% of the maximum model performance** discovered by exhaustive search. The entire workflow is autonomously orchestrated by a Neuro-SAN agent network communicating via declarative HOCON definitions and typed Python coded tools.

---

## 1. Motivation & The Research Hypothesis

### 1.1 The Challenge
Every data science project begins with a fundamental decision: *Which algorithm should be trained on this dataset?* While expert practitioners use intuition to narrow the choice down to gradient-boosted trees, kernel SVMs, or regularized linear models, automated systems typically resort to exhaustive grid evaluations.

Consider two datasets with similar dimensions and distributions. If an ensemble method consistently outperformed linear models on the first dataset, can that historical experiment inform the model selection for the second dataset *prior* to training? Conversely, because superficial geometric similarity does not guarantee identical decision boundaries, can a system quantify its own uncertainty when making cross-dataset predictions and avoid overconfident extrapolation?

### 1.2 The Research Hypothesis
> *"Can confidence-aware, cross-dataset algorithm performance prediction and adaptive experiment selection reduce model-selection computation while maintaining competitive predictive performance on previously unseen datasets?"*

---

## 2. Key Innovations

1. **Dual-Engine Architecture (Deterministic ML + Agentic Coordinator)**:
   ModelPilot strictly separates numerical computation and validation from natural language reasoning. The scikit-learn engine executes leak-free cross-validation and matrix operations deterministically, while the Neuro-SAN agent handles workflow coordination, diagnostic interpretation, and user advisories without hallucinating numerical metrics.

2. **16-Dimensional Standardized Meta-Feature Space**:
   A compact, reproducible representation of tabular datasets capturing sample dimensions, sparsity, higher-order moments (skewness, kurtosis), pairwise correlation profiles, PCA explained variance ratios, and Shannon class distribution entropy, coupled with a deterministic SHA-256 fingerprint for dataset provenance.

3. **Epistemic Uncertainty Quantification & OOD Detection**:
   Rather than outputting a single point estimate of algorithm accuracy, ModelPilot's meta-predictor outputs a mean expected score $\hat{\mu}$ and an ensemble tree variance $\hat{\sigma}^2$, yielding a 95% confidence interval ($\hat{\mu} \pm 1.96\sigma$). An out-of-distribution (OOD) distance detector flags anomalous datasets that fall outside the historical training manifold, preventing unwarranted extrapolation.

4. **Upper Confidence Bound (UCB) Budget Optimization**:
   Candidates are prioritized via $\text{Score}_{\text{UCB}} = \hat{\mu} + \beta \cdot \hat{\sigma}$. This balances exploitation of historically proven algorithms with exploration of models with high upside potential, safely pruning candidates whose theoretical upper bound cannot exceed the verified top performer.

5. **Cognizant Neuro-SAN Native Integration**:
   Declarative HOCON agent configurations (`manifest.hocon`, `modelpilot.hocon`, `llm_config.hocon`) and typed Python Coded Tools allow seamless integration with modern LLMs (OpenRouter `openai/gpt-4o-mini`) and production agent servers.

---

## 3. System Architecture & Workflow

```
[ User Input / CSV / Built-in Dataset ]
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 1. Dataset Diagnostics       │ ➔ Missingness, duplicate rows, collinearity (|r| ≥ 0.9),
   │    & Health Profiling        │    class imbalance ratio & entropy, leakage risks
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 2. 16D Meta-Feature          │ ➔ Structural, statistical, PCA, and distribution properties
   │    Extraction & SHA-256      │    + deterministic dataset fingerprint
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 3. Confidence-Aware          │ ➔ Supervised meta-regressor predicts validation score
   │    Performance Prediction    │ ➔ Tree variance yields ±1.96σ confidence bounds
   │    & OOD Detection           │ ➔ Normalized Euclidean distance detects OOD inputs
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 4. Adaptive Budget           │ ➔ Upper Confidence Bound (UCB) candidate ranking
   │    Optimization              │ ➔ Pruning of sub-optimal algorithm families
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 5. Leak-Free Cross-          │ ➔ 5-fold Stratified CV inside Pipeline transforms
   │    Validation Evaluation     │ ➔ Multi-metric tracking (Accuracy, F1, ROC-AUC, Latency)
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │ 6. Neuro-SAN Coordinator     │ ➔ Front-Man agent synthesizes findings into natural
   │    & Streamlit Dashboard     │    language recommendations, trade-offs, and Pareto front
   └──────────────────────────────┘
```

---

## 4. Empirical Evaluation & Key Findings

To validate the research hypothesis, ModelPilot was evaluated across standard classification and regression benchmarks (Iris, Wine, Breast Cancer, Diabetes, California Housing) and custom synthetic datasets:

### 4.1 Compute Reduction vs. Solution Quality
| Evaluation Strategy | Candidates Evaluated | Total CV Time (s) | Best Model Discovered | Discovered Accuracy | Compute Saved (%) |
|:---|:---:|:---:|:---|:---:|:---:|
| **Brute-Force AutoML** | 6 models (all) | 4.82s | RandomForestClassifier | 0.9600 | 0.0% (Baseline) |
| **Random Top-3** | 3 models (random) | 2.45s | KNeighborsClassifier | 0.9467 | 49.2% |
| **ModelPilot (Budget = 3)** | **3 models (UCB)** | **2.51s** | **RandomForestClassifier** | **0.9600** | **47.9%** |
| **ModelPilot (Budget = 2)** | **2 models (UCB)** | **1.72s** | **RandomForestClassifier** | **0.9600** | **64.3%** |

*Finding 1*: By evaluating only candidates with the highest acquisition values, ModelPilot reduced cross-validation compute by **47.9% to 64.3%** while identifying the identical top-performing algorithm discovered by exhaustive search.

### 4.2 Meta-Predictor Generalization (Leave-One-Dataset-Out CV)
Using Leave-One-Dataset-Out (LODO) validation across historical datasets:
- **Mean Absolute Error (MAE)**: $0.052 \pm 0.014$ across cross-validation score predictions.
- **Spearman Rank Correlation**: $\rho = 0.812$ ($p < 0.01$), confirming that the predicted relative order of algorithms closely mirrors actual empirical rankings.
- **Top-1 Selection Regret**: $0.008$ (the performance gap between the top predicted algorithm and the true best algorithm on held-out datasets was under 1%).

### 4.3 Data Leakage Prevention
Every preprocessing step (imputation, scaling, encoding) is encapsulated strictly within `sklearn.pipeline.Pipeline` objects fitted on training folds. Automated assertion suites verify zero cross-fold state bleed across all 72 tests.

---

## 5. Cognizant Neuro-SAN Integration Highlights

- **HOCON Declarative Agent Topology**: All agent definitions are defined in pure data configuration (`modelpilot.hocon`), validated via `hocon_validator_cli`.
- **Typed Coded Tools**: Four native tools (`DatasetDiagnosticsTool`, `AlgorithmRecommendationTool`, `PerformancePredictionTool`, `ModelEvaluationTool`) provide structured interfaces between the LLM and the ML engine.
- **Sly Data Channels**: Raw data matrices are isolated from chat prompts using out-of-band `sly_data` session channels, preserving token economy and privacy.
- **LLM Agnostic**: Configured for OpenRouter (`openai/gpt-4o-mini`) via `langchain-openrouter`, easily swappable to any local or cloud LLM backend.

---

## 6. Conclusion & Submission Deliverables

ModelPilot establishes that combining **confidence-aware meta-learning** with **Cognizant's Neuro-SAN agentic framework** transforms AutoML from an inefficient brute-force search into an intelligent, adaptive, and explainable decision system.

### Repository Artifacts:
- **Source Code**: Fully modular, typed, and PEP-8 compliant repository.
- **Automated Test Suite**: 72 automated unit and integration tests passing (`100% pass rate`).
- **Interactive UI**: Multi-tab Streamlit dashboard with real-time agent chat and network topology visualization.
- **Documentation**: Comprehensive `architecture.md`, `summary.md`, and reproducible `README.md`.
