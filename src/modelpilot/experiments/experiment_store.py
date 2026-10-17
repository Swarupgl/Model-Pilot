"""Historical experiment store using SQLite for ModelPilot.

Provides structured persistence and querying of historical dataset experiments,
meta-features, algorithm evaluations, and run provenance.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("outputs/experiments/modelpilot_history.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    timestamp_utc TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    task_type TEXT NOT NULL,
    target_column TEXT NOT NULL,
    n_rows INTEGER NOT NULL,
    n_features INTEGER NOT NULL,
    meta_features_json TEXT,
    config_json TEXT
);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    dataset_fingerprint TEXT NOT NULL,
    task_type TEXT NOT NULL,
    algorithm TEXT NOT NULL,
    params_json TEXT,
    primary_metric TEXT NOT NULL,
    primary_score REAL,
    primary_std REAL,
    cv_scores_json TEXT,
    secondary_metrics_json TEXT,
    train_time_s REAL,
    status TEXT NOT NULL,
    error TEXT,
    FOREIGN KEY(experiment_id) REFERENCES experiments(experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_algo ON evaluations(algorithm);
CREATE INDEX IF NOT EXISTS idx_eval_fingerprint ON evaluations(dataset_fingerprint);
CREATE INDEX IF NOT EXISTS idx_eval_task ON evaluations(task_type);
"""


class ExperimentStore:
    """Manages SQLite-based historical experiment records and queries."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    def record_experiment(
        self,
        experiment_id: str,
        timestamp_utc: str,
        dataset_name: str,
        dataset_fingerprint: str,
        task_type: str,
        target_column: str,
        n_rows: int,
        n_features: int,
        meta_features: Dict[str, Any],
        config: Dict[str, Any],
        evaluations: List[Dict[str, Any]],
    ) -> None:
        """Record an experiment and its candidate evaluations."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, timestamp_utc, dataset_name, dataset_fingerprint,
                    task_type, target_column, n_rows, n_features,
                    meta_features_json, config_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    timestamp_utc,
                    dataset_name,
                    dataset_fingerprint,
                    task_type,
                    target_column,
                    n_rows,
                    n_features,
                    json.dumps(meta_features),
                    json.dumps(config),
                ),
            )

            for eval_res in evaluations:
                conn.execute(
                    """
                    INSERT INTO evaluations (
                        experiment_id, dataset_fingerprint, task_type, algorithm,
                        params_json, primary_metric, primary_score, primary_std,
                        cv_scores_json, secondary_metrics_json, train_time_s,
                        status, error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        experiment_id,
                        dataset_fingerprint,
                        task_type,
                        eval_res.get("algorithm", "unknown"),
                        json.dumps(eval_res.get("params_used", {})),
                        eval_res.get("primary_metric", ""),
                        eval_res.get("primary_score"),
                        eval_res.get("primary_std"),
                        json.dumps(eval_res.get("cv_scores", [])),
                        json.dumps(eval_res.get("secondary_metrics", {})),
                        eval_res.get("train_time_s", 0.0),
                        eval_res.get("status", "ok"),
                        eval_res.get("error", ""),
                    ),
                )
            conn.commit()

    def get_all_experiments(self) -> List[Dict[str, Any]]:
        """Retrieve all recorded experiment summaries."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY timestamp_utc DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def get_evaluations_for_algorithm(
        self, algorithm: str, task_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve all successful evaluations for a specific algorithm."""
        query = "SELECT * FROM evaluations WHERE algorithm = ? AND status = 'ok'"
        params: List[Any] = [algorithm]
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        query += " ORDER BY primary_score DESC"

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_dataset_experiments(self, dataset_fingerprint: str) -> List[Dict[str, Any]]:
        """Retrieve evaluations performed on a specific dataset fingerprint."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM evaluations
                WHERE dataset_fingerprint = ?
                ORDER BY primary_score DESC
                """,
                (dataset_fingerprint,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_meta_dataset(
        self, task_type: str = "classification"
    ) -> List[Dict[str, Any]]:
        """Retrieve paired meta-feature vectors and evaluation scores across all recorded datasets.

        Useful for meta-learning performance predictors.
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT 
                    e.dataset_name,
                    e.dataset_fingerprint,
                    e.meta_features_json,
                    ev.algorithm,
                    ev.primary_metric,
                    ev.primary_score,
                    ev.train_time_s
                FROM evaluations ev
                JOIN experiments e ON ev.experiment_id = e.experiment_id
                WHERE ev.task_type = ? AND ev.status = 'ok'
                """,
                (task_type,),
            ).fetchall()

            result = []
            for r in rows:
                meta = json.loads(r["meta_features_json"]) if r["meta_features_json"] else {}
                result.append({
                    "dataset_name": r["dataset_name"],
                    "dataset_fingerprint": r["dataset_fingerprint"],
                    "meta_features": meta,
                    "algorithm": r["algorithm"],
                    "primary_metric": r["primary_metric"],
                    "primary_score": r["primary_score"],
                    "train_time_s": r["train_time_s"],
                })
            return result
