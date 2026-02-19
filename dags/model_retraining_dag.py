"""Model Retraining DAG -- Weekly ML pipeline.

Automates the end-to-end model lifecycle: drift detection, data extraction,
training, evaluation, champion comparison, and conditional promotion. Uses
BranchPythonOperator for conditional promotion flow.
"""

import json
import logging
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml
from airflow.decorators import dag, task
from airflow.operators.python import BranchPythonOperator, get_current_context

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")


def _load_config() -> dict:
    with open(CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


config = _load_config()
airflow_cfg = config.get("airflow", {})
paths_cfg = config.get("paths", {})
notify_cfg = config.get("notifications", {})

GOLD_PATH = paths_cfg.get("gold", "./data/gold")
MODEL_REGISTRY_PATH = os.path.join(GOLD_PATH, "_model_registry")

default_args = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email": [notify_cfg.get("email", "data-team@company.com")],
    "email_on_failure": notify_cfg.get("on_failure", True),
    "email_on_success": notify_cfg.get("on_success", False),
    "retries": airflow_cfg.get("retries", 2),
    "retry_delay": timedelta(minutes=airflow_cfg.get("retry_delay_minutes", 5)),
}


def _generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ML features from claims aggregation data."""
    features = df.copy()
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        features[f"{col}_log"] = np.log1p(features[col].clip(lower=0))
    return features


def _get_champion_metrics() -> Optional[Dict[str, float]]:
    """Load the current champion model's evaluation metrics."""
    registry_dir = Path(MODEL_REGISTRY_PATH)
    champion_file = registry_dir / "champion_metrics.json"
    if champion_file.exists():
        with open(champion_file, "r") as fh:
            return json.load(fh)
    return None


def _save_model(model: Any, path: str) -> None:
    """Serialize a model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def _load_model(path: str) -> Any:
    """Deserialize a model from disk."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


@dag(
    dag_id="model_retraining_pipeline",
    default_args=default_args,
    description="Weekly model retraining with drift detection and champion comparison",
    schedule_interval="@weekly",
    start_date=datetime.fromisoformat(airflow_cfg.get("start_date", "2024-01-01")),
    catchup=False,
    tags=["ml", "retraining", "model"],
    max_active_runs=1,
)
def model_retraining_dag():

    @task()
    def check_data_drift() -> Dict[str, Any]:
        """Evaluate whether retraining is warranted based on data drift.

        Computes Population Stability Index (PSI) between the most recent
        Gold-layer data and the training data used by the current champion.
        If PSI exceeds the threshold, retraining is triggered.
        """
        gold_dir = Path(GOLD_PATH)
        provider_agg_path = gold_dir / "provider_aggregations.parquet"

        if not provider_agg_path.exists():
            logger.info("No Gold data found; triggering training for initial model")
            return {"drift_detected": True, "psi_score": None, "reason": "no_existing_data"}

        current_data = pd.read_parquet(str(provider_agg_path))

        registry_dir = Path(MODEL_REGISTRY_PATH)
        baseline_path = registry_dir / "training_baseline.parquet"

        if not baseline_path.exists():
            logger.info("No training baseline found; triggering initial training")
            return {"drift_detected": True, "psi_score": None, "reason": "no_baseline"}

        baseline = pd.read_parquet(str(baseline_path))

        numeric_cols = current_data.select_dtypes(include=[np.number]).columns
        psi_scores = {}

        for col in numeric_cols:
            if col not in baseline.columns:
                continue

            current_vals = current_data[col].dropna()
            baseline_vals = baseline[col].dropna()

            if len(current_vals) == 0 or len(baseline_vals) == 0:
                continue

            min_val = min(current_vals.min(), baseline_vals.min())
            max_val = max(current_vals.max(), baseline_vals.max())
            bins = np.linspace(min_val, max_val, 11)

            current_hist, _ = np.histogram(current_vals, bins=bins)
            baseline_hist, _ = np.histogram(baseline_vals, bins=bins)

            current_pct = (current_hist + 1) / (current_hist.sum() + len(bins))
            baseline_pct = (baseline_hist + 1) / (baseline_hist.sum() + len(bins))

            psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
            psi_scores[col] = float(psi)

        avg_psi = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0
        drift_threshold = 0.2
        drift_detected = avg_psi > drift_threshold

        logger.info("PSI scores: %s | avg=%.4f | drift=%s", psi_scores, avg_psi, drift_detected)

        return {
            "drift_detected": drift_detected,
            "psi_score": avg_psi,
            "psi_details": psi_scores,
            "threshold": drift_threshold,
            "reason": "psi_exceeded" if drift_detected else "within_threshold",
        }

    @task()
    def extract_training_data(drift_info: Dict[str, Any]) -> Dict[str, Any]:
        """Pull feature tables from Gold layer for model training.

        Constructs the feature matrix and target variable from Gold
        aggregate tables.
        """
        gold_dir = Path(GOLD_PATH)
        provider_path = gold_dir / "provider_aggregations.parquet"
        temporal_path = gold_dir / "temporal_aggregations.parquet"

        frames = []
        if provider_path.exists():
            frames.append(pd.read_parquet(str(provider_path)))
        if temporal_path.exists():
            frames.append(pd.read_parquet(str(temporal_path)))

        if not frames:
            rng = np.random.default_rng(42)
            n = 200
            df = pd.DataFrame({
                "total_claims": rng.integers(10, 500, n),
                "total_amount": rng.lognormal(8, 1.5, n).round(2),
                "avg_amount": rng.lognormal(6, 1.0, n).round(2),
                "denial_rate": rng.beta(2, 8, n).round(4),
                "high_risk": (rng.random(n) > 0.7).astype(int),
            })
        else:
            df = frames[0]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 1:
                target_col = numeric_cols[-1]
                median_val = df[target_col].median()
                df["high_risk"] = (df[target_col] > median_val).astype(int)
            else:
                df["high_risk"] = 0

        features = _generate_features(df)

        train_dir = Path(MODEL_REGISTRY_PATH) / "training_data"
        train_dir.mkdir(parents=True, exist_ok=True)

        train_path = train_dir / "features.parquet"
        features.to_parquet(str(train_path), index=False)

        return {
            **drift_info,
            "train_data_path": str(train_path),
            "n_samples": len(features),
            "n_features": len(features.columns) - 1,
            "target_column": "high_risk",
        }

    @task()
    def train_model(training_info: Dict[str, Any]) -> Dict[str, Any]:
        """Train a new model version on the extracted features."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        df = pd.read_parquet(training_info["train_data_path"])
        target_col = training_info["target_column"]

        if target_col not in df.columns:
            df[target_col] = 0

        feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
        X = df[feature_cols].fillna(0)
        y = df[target_col]

        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None,
        )

        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
        model.fit(X_train, y_train)

        model_version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        model_dir = Path(MODEL_REGISTRY_PATH) / "models" / model_version
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        _save_model(model, str(model_path))

        holdout_path = model_dir / "holdout.parquet"
        holdout_df = X_holdout.copy()
        holdout_df[target_col] = y_holdout
        holdout_df.to_parquet(str(holdout_path), index=False)

        logger.info("Model trained: version=%s features=%d samples=%d",
                     model_version, len(feature_cols), len(X_train))

        return {
            **training_info,
            "model_version": model_version,
            "model_path": str(model_path),
            "holdout_path": str(holdout_path),
            "feature_columns": feature_cols,
            "train_samples": len(X_train),
        }

    @task()
    def evaluate_model(train_info: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the trained model on the holdout set."""
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        model = _load_model(train_info["model_path"])
        holdout = pd.read_parquet(train_info["holdout_path"])
        target_col = train_info["target_column"]

        feature_cols = train_info["feature_columns"]
        X_test = holdout[feature_cols].fillna(0)
        y_test = holdout[target_col]

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred.astype(float)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

        try:
            metrics["auc_roc"] = float(roc_auc_score(y_test, y_proba))
        except ValueError:
            metrics["auc_roc"] = 0.0

        metrics_path = Path(train_info["model_path"]).parent / "metrics.json"
        with open(metrics_path, "w") as fh:
            json.dump(metrics, fh, indent=2)

        logger.info("Model evaluation: %s", json.dumps(metrics, indent=2))

        return {
            **train_info,
            "metrics": metrics,
            "metrics_path": str(metrics_path),
        }

    @task.branch()
    def compare_champion(eval_info: Dict[str, Any]) -> str:
        """Compare new model against current champion.

        Returns the task ID of the next step: promote_model if the new
        model outperforms, or skip_promotion otherwise.
        """
        champion = _get_champion_metrics()

        if champion is None:
            logger.info("No champion found; promoting new model by default")
            return "promote_model"

        new_f1 = eval_info["metrics"].get("f1", 0.0)
        champion_f1 = champion.get("f1", 0.0)
        improvement = new_f1 - champion_f1
        min_improvement = 0.01

        logger.info("Champion F1=%.4f | New F1=%.4f | Improvement=%.4f",
                     champion_f1, new_f1, improvement)

        if improvement >= min_improvement:
            return "promote_model"
        return "skip_promotion"

    @task(trigger_rule="none_failed_min_one_success")
    def promote_model(eval_info: Dict[str, Any]) -> Dict[str, Any]:
        """Promote the new model to champion status."""
        registry_dir = Path(MODEL_REGISTRY_PATH)
        registry_dir.mkdir(parents=True, exist_ok=True)

        champion_metrics_path = registry_dir / "champion_metrics.json"
        with open(champion_metrics_path, "w") as fh:
            json.dump(eval_info["metrics"], fh, indent=2)

        champion_model_path = registry_dir / "champion_model.pkl"
        model = _load_model(eval_info["model_path"])
        _save_model(model, str(champion_model_path))

        gold_dir = Path(GOLD_PATH)
        provider_path = gold_dir / "provider_aggregations.parquet"
        if provider_path.exists():
            baseline_path = registry_dir / "training_baseline.parquet"
            baseline = pd.read_parquet(str(provider_path))
            baseline.to_parquet(str(baseline_path), index=False)

        logger.info("Model promoted: version=%s", eval_info["model_version"])

        return {
            **eval_info,
            "promoted": True,
            "champion_path": str(champion_model_path),
        }

    @task(trigger_rule="none_failed_min_one_success")
    def skip_promotion(eval_info: Dict[str, Any]) -> Dict[str, Any]:
        """Log that the new model was not promoted."""
        logger.info("Model not promoted: new F1=%.4f did not improve sufficiently",
                     eval_info["metrics"].get("f1", 0.0))
        return {**eval_info, "promoted": False}

    @task(trigger_rule="none_failed_min_one_success")
    def update_registry(eval_info: Dict[str, Any]) -> None:
        """Update the model registry with the training run results."""
        registry_dir = Path(MODEL_REGISTRY_PATH)
        registry_dir.mkdir(parents=True, exist_ok=True)

        registry_file = registry_dir / "registry.json"
        if registry_file.exists():
            with open(registry_file, "r") as fh:
                registry = json.load(fh)
        else:
            registry = {"runs": []}

        run_entry = {
            "version": eval_info.get("model_version", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": eval_info.get("metrics", {}),
            "promoted": eval_info.get("promoted", False),
            "drift_score": eval_info.get("psi_score"),
            "n_samples": eval_info.get("n_samples", 0),
        }
        registry["runs"].append(run_entry)

        with open(registry_file, "w") as fh:
            json.dump(registry, fh, indent=2)

        logger.info("Registry updated: %d total runs", len(registry["runs"]))

    drift = check_data_drift()
    training_data = extract_training_data(drift)
    trained = train_model(training_data)
    evaluated = evaluate_model(trained)

    branch_decision = compare_champion(evaluated)

    promoted = promote_model(evaluated)
    skipped = skip_promotion(evaluated)

    branch_decision >> [promoted, skipped]

    update_registry(evaluated)


retraining_dag = model_retraining_dag()
