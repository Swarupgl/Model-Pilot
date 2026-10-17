"""ModelPilot Interactive Streamlit Application.

Provides a unified interface for dataset diagnosis, meta-learning performance
predictions, selective model evaluation, and Neuro-SAN agent coordination.
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

_frontend_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_frontend_dir)
_backend_dir = os.path.join(_project_root, "backend")
_src_dir = os.path.join(_project_root, "src")
_venv_site = os.path.join(_project_root, ".venv", "Lib", "site-packages")

for p in [_venv_site, _project_root, _backend_dir, _src_dir]:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from modelpilot.data.loader import load_builtin, load_csv
from modelpilot.diagnosis.dataset_diagnostics import diagnose_dataset
from modelpilot.experiments.experiment_store import ExperimentStore
from modelpilot.meta_learning.meta_feature_extractor import extract_meta_features
from modelpilot.meta_learning.performance_predictor import PerformancePredictor
from modelpilot.optimization.budget_optimizer import SelectiveEvaluator
from modelpilot.selection.algorithm_selector import (
    rank_algorithms_heuristic,
    recommend_algorithms_from_history,
)

from backend.run import invoke_agent

st.set_page_config(
    page_title="ModelPilot — Autonomous ML Selection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 ModelPilot")
st.caption(
    "Autonomous, Confidence-Aware ML Algorithm Selection and Selective Optimization using Neuro-SAN"
)

# ── Sidebar: Dataset & Budget Configuration ──────────────────────────────────
st.sidebar.header("1. Dataset Configuration")
data_source_type = st.sidebar.radio(
    "Data Source",
    ["Built-in Benchmark", "Upload Custom CSV"],
)

dataset_name = None
uploaded_file = None

if data_source_type == "Built-in Benchmark":
    dataset_name = st.sidebar.selectbox(
        "Choose Benchmark Dataset",
        ["iris", "breast_cancer", "wine", "diabetes", "california"],
    )
    loaded = load_builtin(dataset_name)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        temp_csv_path = Path("data/sample") / uploaded_file.name
        temp_csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(temp_csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loaded = load_csv(temp_csv_path)
    else:
        st.info("Please upload a CSV file or select a built-in benchmark dataset.")
        st.stop()

df = loaded["df"]
available_cols = list(df.columns)
target_col = st.sidebar.selectbox(
    "Target Column",
    available_cols,
    index=available_cols.index(loaded["target"]) if loaded["target"] in available_cols else len(available_cols) - 1,
)

task_type = loaded["task_type"]
task_type_override = st.sidebar.radio(
    "Task Type",
    ["classification", "regression"],
    index=0 if task_type == "classification" else 1,
)

st.sidebar.header("2. Evaluation Budget")
max_candidates = st.sidebar.slider("Max Candidates to Evaluate", min_value=1, max_value=8, value=3)
cv_folds = st.sidebar.slider("Cross-Validation Folds", min_value=2, max_value=10, value=5)

# Session state initialization
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_context" not in st.session_state:
    st.session_state.chat_context = None
if "last_eval_results" not in st.session_state:
    st.session_state.last_eval_results = None

# Extract Diagnostics & Meta-Features
diag = diagnose_dataset(df, target_column=target_col, task_type=task_type_override)
meta = extract_meta_features(df, target_column=target_col, task_type=task_type_override)
store = ExperimentStore()

# ── Main Tabs ────────────────────────────────────────────────────────────────
tab_diag, tab_meta, tab_eval, tab_agent, tab_history = st.tabs([
    "📊 Dataset Health & Diagnostics",
    "🧠 Meta-Learning & Recommendations",
    "🔬 Selective Model Evaluation",
    "🤖 Neuro-SAN Agent Coordinator",
    "📜 History & Provenance",
])

# ── Tab 1: Dataset Health & Diagnostics ──────────────────────────────────────
with tab_diag:
    st.subheader("Dataset Profiling & Quality Health Check")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Rows", f"{diag['dataset_shape']['n_rows']:,}")
    c2.metric("Features", f"{diag['dataset_shape']['n_features']:,}")
    c3.metric("Missing Cells", f"{diag['missing_values']['missing_ratio']*100:.1f}%")
    c4.metric("Duplicate Rows", f"{diag['duplicate_rows']['count']:,}")
    if task_type_override == "classification":
        c5.metric("Imbalance Ratio", f"{diag['target_diagnostics'].get('imbalance_ratio', 1.0):.1f}:1")
    else:
        c5.metric("Target Std", f"{diag['target_diagnostics'].get('std', 0.0):.2f}")

    if diag["advisories"]:
        st.markdown("#### Diagnostic Advisories")
        for adv in diag["advisories"]:
            sev = adv["severity"]
            icon = "⚠️" if sev == "warning" else "ℹ️"
            st.warning(f"{icon} **[{adv['category'].upper()}]** {adv['message']}")
    else:
        st.success("✅ No quality anomalies, high collinearity, or class imbalance risks detected.")

    with st.expander("Preview Raw Dataset Sample", expanded=False):
        st.dataframe(df.head(25), use_container_width=True)

# ── Tab 2: Meta-Learning & Recommendations ──────────────────────────────────
with tab_meta:
    st.subheader("Predictive Meta-Learning & Candidate Recommendations")
    st.caption(f"Dataset Fingerprint: `{meta['dataset_fingerprint']}` | Meta-Extractor Version: `{meta['extractor_version']}`")

    # Meta-learning performance predictor
    historical_records = store.get_meta_dataset(task_type=task_type_override)
    has_history = len(historical_records) >= 3

    col_meta1, col_meta2 = st.columns([3, 2])

    with col_meta1:
        st.markdown("#### Predicted Algorithm Performance (Before Training)")
        if has_history:
            predictor = PerformancePredictor()
            predictor.fit(historical_records)
            preds = predictor.predict(meta)
            pred_df = pd.DataFrame(preds)

            if preds[0]["is_out_of_distribution"]:
                st.error("⚠️ Out-of-Distribution Warning: This dataset is significantly distant from historical training benchmarks. Uncertainty is elevated.")

            # Display predictions table
            display_cols = ["algorithm", "predicted_score", "uncertainty_std", "confidence", "confidence_interval_95"]
            st.dataframe(pred_df[display_cols], use_container_width=True)
        else:
            st.info("ℹ️ Fewer than 3 historical benchmark runs available in local store. Displaying transparent rule-based heuristics.")
            heuristic_ranks = rank_algorithms_heuristic(task_type_override, diag)
            h_df = pd.DataFrame(heuristic_ranks)
            st.dataframe(h_df[["algorithm", "heuristic_score", "tags", "rationale"]], use_container_width=True)

    with col_meta2:
        st.markdown("#### Meta-Feature Vector Overview")
        st.json(meta["features"])

# ── Tab 3: Selective Model Evaluation ────────────────────────────────────────
with tab_eval:
    st.subheader("Selective Model Evaluation within Budget")
    st.markdown(
        f"Evaluating up to **{max_candidates} candidates** using **{cv_folds}-fold cross-validation** "
        f"with strict preprocessing pipeline isolation to prevent data leakage."
    )

    if st.button("🚀 Run Selective Model Evaluation", type="primary"):
        with st.spinner("Executing budget-constrained evaluation pipeline..."):
            # Prioritize candidates using UCB
            if has_history:
                raw_cands = preds
            else:
                raw_cands = rank_algorithms_heuristic(task_type_override, diag)

            evaluator = SelectiveEvaluator(max_candidates=max_candidates)
            prioritized = evaluator.prioritize_candidates(raw_cands)

            X = df.drop(columns=[target_col])
            y = df[target_col]

            eval_res = evaluator.run_selective_evaluation(
                prioritized_candidates=prioritized,
                X=X,
                y=y,
                task_type=task_type_override,
                cv_folds=cv_folds,
            )

            st.session_state.last_eval_results = eval_res

            # Record in SQLite store
            import uuid
            from datetime import datetime, timezone
            store.record_experiment(
                experiment_id=str(uuid.uuid4()),
                timestamp_utc=datetime.now(tz=timezone.utc).isoformat(),
                dataset_name=dataset_name or (uploaded_file.name if uploaded_file else "custom_dataset"),
                dataset_fingerprint=meta["dataset_fingerprint"],
                task_type=task_type_override,
                target_column=target_col,
                n_rows=len(df),
                n_features=len(X.columns),
                meta_features=meta,
                config={"max_candidates": max_candidates, "cv_folds": cv_folds},
                evaluations=eval_res["evaluations"],
            )

    if st.session_state.last_eval_results:
        res = st.session_state.last_eval_results

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Evaluations Executed", f"{res['candidates_evaluated']}")
        m2.metric("Evaluations Skipped", f"{res['candidates_skipped']}")
        m3.metric("Compute Saved", f"{res['estimated_compute_saved_ratio']*100:.1f}%")
        m4.metric("Total Wall Time", f"{res['total_elapsed_s']:.2f}s")

        if res["best_candidate"]:
            best = res["best_candidate"]
            st.success(
                f"🏆 **Top Performer:** `{best['algorithm']}` — "
                f"{best['primary_metric']}: **{best['primary_score']:.4f}** (± {best['primary_std']:.4f})"
            )

        # Detailed results table
        eval_table = []
        for e in res["evaluations"]:
            eval_table.append({
                "Algorithm": e["algorithm"],
                "Primary Metric": e["primary_metric"],
                "Measured Score": e["primary_score"],
                "Score Std": e["primary_std"],
                "Prior Predicted Score": e.get("prior_predicted_score", "-"),
                "Runtime (s)": e["train_time_s"],
                "Status": e["status"],
            })
        st.dataframe(pd.DataFrame(eval_table), use_container_width=True)

# ── Tab 4: Neuro-SAN Agent Coordinator ───────────────────────────────────────
with tab_agent:
    st.subheader("Autonomous Neuro-SAN Multi-Agent Coordinator")
    st.caption("Powered by Cognizant Neuro-SAN and OpenRouter (`openai/gpt-4o-mini`)")

    with st.expander("🔍 Neuro-SAN Multi-Agent Topology & Tool Architecture", expanded=True):
        st.markdown("""
```mermaid
graph TD
    User["👤 User / Streamlit Interface"] --> Coordinator["🧠 ModelPilotCoordinator<br/>(Cognizant Neuro-SAN Agent)"]
    Coordinator --> Tool1["🔬 DatasetDiagnosticsTool<br/>(Coded Tool)"]
    Coordinator --> Tool2["💡 AlgorithmRecommendationTool<br/>(Coded Tool)"]
    Coordinator --> Tool3["📊 PerformancePredictionTool<br/>(Coded Tool)"]
    Coordinator --> Tool4["⚙️ ModelEvaluationTool<br/>(Coded Tool)"]
    
    Tool1 -.-> Output1["Dataset Health, Multicollinearity, Class Imbalance"]
    Tool2 -.-> Output2["Heuristic & Historical Similarity Recommendations"]
    Tool3 -.-> Output3["Cross-Dataset Prediction & Uncertainty Bounds"]
    Tool4 -.-> Output4["Leakage-Free Cross-Validation & SQLite Store"]

    classDef agent fill:#e0f2fe,stroke:#0284c7,stroke-width:2px,color:#0369a1;
    classDef tool fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#b45309;
    classDef output fill:#f8fafc,stroke:#94a3b8,stroke-width:1px,color:#334155;
    class Coordinator agent;
    class Tool1,Tool2,Tool3,Tool4 tool;
    class Output1,Output2,Output3,Output4 output;
```
""")
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        col_t1.success("🔬 **Diagnostics Tool**: Active")
        col_t2.success("💡 **Recommendation Tool**: Active")
        col_t3.success("📊 **Meta-Predictor Tool**: Active")
        col_t4.success("⚙️ **Evaluator Tool**: Active")

    # Render previous conversation
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask ModelPilot (e.g., 'Diagnose the dataset and tell me which model to pick')...")
    if user_query:
        # Display user message
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Stream response from Neuro-SAN agent
        with st.chat_message("assistant"):
            with st.spinner("ModelPilot is coordinating tools and reasoning..."):
                response = invoke_agent(
                    agent_name="modelpilot",
                    user_text=user_query,
                    chat_context=st.session_state.chat_context,
                )

                if "response" in response and "text" in response["response"]:
                    ans_text = response["response"]["text"]
                    st.markdown(ans_text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": ans_text})
                    if "chat_context" in response["response"]:
                        st.session_state.chat_context = response["response"]["chat_context"]
                else:
                    st.write(response)

# ── Tab 5: History & Provenance ──────────────────────────────────────────────
with tab_history:
    st.subheader("Historical Experiment Records (SQLite)")
    exps = store.get_all_experiments()

    if exps:
        exp_df = pd.DataFrame(exps)
        st.dataframe(
            exp_df[["experiment_id", "timestamp_utc", "dataset_name", "task_type", "n_rows", "n_features"]],
            use_container_width=True,
        )

        st.download_button(
            "📥 Export Historical Database as JSON",
            data=json.dumps(exps, indent=2),
            file_name="modelpilot_experiments.json",
            mime="application/json",
        )
    else:
        st.info("No recorded experiments found yet. Run evaluations to populate history.")
