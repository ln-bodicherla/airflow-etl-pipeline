"""Claims ETL DAG -- Daily medallion architecture pipeline.

Orchestrates the extraction, validation, and transformation of claims data
through Bronze, Silver, and Gold layers. Uses the Airflow TaskFlow API for
clean task definitions and automatic XCom data passing.
"""

import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")


def load_config() -> dict:
    with open(CONFIG_PATH, "r") as fh:
        return yaml.safe_load(fh)


config = load_config()
airflow_cfg = config.get("airflow", {})
paths_cfg = config.get("paths", {})
quality_cfg = config.get("quality", {})
notify_cfg = config.get("notifications", {})

default_args = {
    "owner": "data-engineering",
    "depends_on_past": False,
    "email": [notify_cfg.get("email", "data-team@company.com")],
    "email_on_failure": notify_cfg.get("on_failure", True),
    "email_on_success": notify_cfg.get("on_success", False),
    "retries": airflow_cfg.get("retries", 2),
    "retry_delay": timedelta(minutes=airflow_cfg.get("retry_delay_minutes", 5)),
}

RAW_PATH = paths_cfg.get("raw", "./data/raw")
BRONZE_PATH = paths_cfg.get("bronze", "./data/bronze")
SILVER_PATH = paths_cfg.get("silver", "./data/silver")
GOLD_PATH = paths_cfg.get("gold", "./data/gold")


def _generate_batch_id() -> str:
    return f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _row_hash(row: pd.Series) -> str:
    content = "|".join(str(v) for v in row.values)
    return hashlib.sha256(content.encode()).hexdigest()


def _generate_sample_claims(n_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic claims data for demonstration purposes."""
    rng = np.random.default_rng(seed=42)

    claim_ids = [f"CLM-{i:08d}" for i in range(n_rows)]
    patient_ids = [f"PAT-{rng.integers(1000, 9999):04d}" for _ in range(n_rows)]
    provider_ids = [f"PRV-{rng.integers(100, 999):03d}" for _ in range(n_rows)]

    base_date = datetime(2024, 1, 1)
    service_dates = [
        (base_date + timedelta(days=int(rng.integers(0, 365)))).strftime("%Y-%m-%d")
        for _ in range(n_rows)
    ]

    diagnosis_codes = [f"{rng.choice(['E11', 'I10', 'J06', 'M54', 'K21'])}.{rng.integers(0, 9)}" for _ in range(n_rows)]
    procedure_codes = [f"{rng.integers(10000, 99999)}" for _ in range(n_rows)]
    claim_amounts = np.round(rng.lognormal(mean=6.0, sigma=1.2, size=n_rows), 2)
    paid_amounts = np.round(claim_amounts * rng.uniform(0.5, 1.0, size=n_rows), 2)
    statuses = rng.choice(["approved", "denied", "pending", "review"], size=n_rows, p=[0.6, 0.15, 0.15, 0.1])

    null_mask = rng.random(n_rows) < 0.03
    diagnosis_codes_with_nulls = [None if null_mask[i] else v for i, v in enumerate(diagnosis_codes)]

    return pd.DataFrame({
        "claim_id": claim_ids,
        "patient_id": patient_ids,
        "provider_id": provider_ids,
        "service_date": service_dates,
        "diagnosis_code": diagnosis_codes_with_nulls,
        "procedure_code": procedure_codes,
        "claim_amount": claim_amounts,
        "paid_amount": paid_amounts,
        "status": statuses,
    })


@dag(
    dag_id="claims_etl_pipeline",
    default_args=default_args,
    description="Daily claims ETL: Raw -> Bronze -> Silver -> Gold",
    schedule_interval=airflow_cfg.get("schedule_interval", "@daily"),
    start_date=datetime.fromisoformat(airflow_cfg.get("start_date", "2024-01-01")),
    catchup=False,
    tags=["etl", "claims", "medallion"],
    max_active_runs=1,
)
def claims_etl_dag():

    @task()
    def extract_raw_claims() -> Dict[str, Any]:
        """Extract claims from source systems.

        In production this would connect to a database, S3, or API. For
        demonstration purposes, either reads from CSV files in the raw
        directory or generates synthetic data.
        """
        context = get_current_context()
        execution_date = context["ds"]
        batch_id = _generate_batch_id()

        raw_dir = Path(RAW_PATH)
        raw_dir.mkdir(parents=True, exist_ok=True)

        csv_files = sorted(raw_dir.glob("*.csv"))
        if csv_files:
            frames = [pd.read_csv(f) for f in csv_files]
            df = pd.concat(frames, ignore_index=True)
            logger.info("Loaded %d rows from %d CSV files", len(df), len(csv_files))
        else:
            df = _generate_sample_claims(500)
            logger.info("Generated %d synthetic claims rows", len(df))

        extract_path = raw_dir / f"claims_extract_{execution_date}.parquet"
        df.to_parquet(str(extract_path), index=False)

        return {
            "batch_id": batch_id,
            "execution_date": execution_date,
            "extract_path": str(extract_path),
            "row_count": len(df),
            "columns": list(df.columns),
        }

    @task()
    def validate_raw_data(extract_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run validation checks on raw extracted data."""
        from plugins.quality_checks import ClaimsDataValidator

        df = pd.read_parquet(extract_info["extract_path"])
        validator = ClaimsDataValidator(config_path=CONFIG_PATH)

        schema_result = validator.validate_schema(df)
        completeness_result = validator.validate_completeness(df)

        is_valid = schema_result["passed"] and completeness_result["passed"]

        if not is_valid:
            logger.warning("Raw data validation failed: schema=%s completeness=%s",
                           schema_result["passed"], completeness_result["passed"])

        return {
            **extract_info,
            "validation_passed": is_valid,
            "schema_check": schema_result,
            "completeness_check": completeness_result,
        }

    @task()
    def bronze_ingestion(validated_info: Dict[str, Any]) -> Dict[str, Any]:
        """Ingest raw data into Bronze layer as-is with metadata columns.

        Adds ingestion timestamp, source file, batch ID, and row hash
        for lineage tracking.
        """
        df = pd.read_parquet(validated_info["extract_path"])

        df["_ingestion_ts"] = datetime.utcnow().isoformat()
        df["_source"] = validated_info["extract_path"]
        df["_batch_id"] = validated_info["batch_id"]
        df["_row_hash"] = df.apply(_row_hash, axis=1)

        bronze_dir = Path(BRONZE_PATH)
        bronze_dir.mkdir(parents=True, exist_ok=True)
        partition_date = validated_info["execution_date"]
        bronze_path = bronze_dir / f"claims_{partition_date}.parquet"
        df.to_parquet(str(bronze_path), index=False)

        logger.info("Bronze ingestion: %d rows -> %s", len(df), bronze_path)

        return {
            **validated_info,
            "bronze_path": str(bronze_path),
            "bronze_row_count": len(df),
        }

    @task()
    def silver_transformation(bronze_info: Dict[str, Any]) -> Dict[str, Any]:
        """Transform Bronze data into Silver layer.

        Applies: data cleaning, type casting, deduplication, standardized
        codes, and SCD Type-2 merge logic for provider/patient dimensions.
        """
        from plugins.transformations import ClaimsTransformer

        df = pd.read_parquet(bronze_info["bronze_path"])
        transformer = ClaimsTransformer()

        df = transformer.clean_claims_data(df)
        df = transformer.standardize_codes(df)
        df = transformer.enrich_data(df)

        silver_dir = Path(SILVER_PATH)
        silver_dir.mkdir(parents=True, exist_ok=True)

        claims_path = silver_dir / "claims_fact.parquet"
        if claims_path.exists():
            existing = pd.read_parquet(str(claims_path))
            df = pd.concat([existing, df], ignore_index=True)
            df = df.drop_duplicates(subset=["claim_id"], keep="last")

        df.to_parquet(str(claims_path), index=False)

        provider_dim = df[["provider_id"]].drop_duplicates()
        provider_dim["provider_name"] = provider_dim["provider_id"].apply(
            lambda x: f"Provider {x}"
        )
        provider_dim_path = silver_dir / "provider_dim.parquet"
        provider_dim_scd = transformer.apply_scd_type2(
            new_data=provider_dim,
            existing_path=str(provider_dim_path),
            key_column="provider_id",
        )
        provider_dim_scd.to_parquet(str(provider_dim_path), index=False)

        patient_dim = df[["patient_id"]].drop_duplicates()
        patient_dim["patient_name"] = patient_dim["patient_id"].apply(
            lambda x: f"Patient {x}"
        )
        patient_dim_path = silver_dir / "patient_dim.parquet"
        patient_dim_scd = transformer.apply_scd_type2(
            new_data=patient_dim,
            existing_path=str(patient_dim_path),
            key_column="patient_id",
        )
        patient_dim_scd.to_parquet(str(patient_dim_path), index=False)

        logger.info("Silver transformation complete: %d claims rows", len(df))

        return {
            **bronze_info,
            "silver_claims_path": str(claims_path),
            "silver_row_count": len(df),
            "provider_dim_count": len(provider_dim_scd),
            "patient_dim_count": len(patient_dim_scd),
        }

    @task()
    def gold_aggregation(silver_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build Gold-layer aggregate tables from Silver data.

        Creates provider-level, patient-level, and temporal aggregation
        tables suitable for analytics dashboards and ML feature stores.
        """
        from plugins.transformations import ClaimsTransformer

        df = pd.read_parquet(silver_info["silver_claims_path"])
        transformer = ClaimsTransformer()
        aggregations = transformer.compute_aggregations(df)

        gold_dir = Path(GOLD_PATH)
        gold_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}
        for name, agg_df in aggregations.items():
            path = gold_dir / f"{name}.parquet"
            agg_df.to_parquet(str(path), index=False)
            output_paths[name] = str(path)
            logger.info("Gold table '%s': %d rows -> %s", name, len(agg_df), path)

        return {
            **silver_info,
            "gold_paths": output_paths,
            "gold_tables": list(output_paths.keys()),
        }

    @task()
    def data_quality_check(gold_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Gold-layer tables against quality gates.

        Ensures aggregate tables meet minimum completeness, null rate,
        and row count thresholds before marking the pipeline run as
        successful.
        """
        from plugins.quality_checks import ClaimsDataValidator
        validator = ClaimsDataValidator(config_path=CONFIG_PATH)

        results = {}
        all_passed = True

        for table_name, path in gold_info["gold_paths"].items():
            df = pd.read_parquet(path)
            completeness = validator.validate_completeness(df)
            schema = validator.validate_schema(df)

            passed = completeness["passed"] and (len(df) > 0)
            results[table_name] = {
                "row_count": len(df),
                "completeness": completeness,
                "schema": schema,
                "passed": passed,
            }
            if not passed:
                all_passed = False
                logger.warning("Quality gate failed for %s", table_name)

        return {
            **gold_info,
            "quality_results": results,
            "all_quality_passed": all_passed,
        }

    @task()
    def notify_completion(quality_info: Dict[str, Any]) -> None:
        """Send pipeline completion notification.

        In production this would send an email, Slack message, or PagerDuty
        alert. For now, logs a structured summary.
        """
        context = get_current_context()
        status = "SUCCESS" if quality_info["all_quality_passed"] else "QUALITY_FAILURE"

        summary = {
            "dag_id": context["dag"].dag_id,
            "execution_date": quality_info["execution_date"],
            "batch_id": quality_info["batch_id"],
            "status": status,
            "raw_rows": quality_info["row_count"],
            "bronze_rows": quality_info["bronze_row_count"],
            "silver_rows": quality_info["silver_row_count"],
            "gold_tables": quality_info["gold_tables"],
            "quality_passed": quality_info["all_quality_passed"],
        }

        logger.info("Pipeline complete: %s", json.dumps(summary, indent=2))

        notification_dir = Path(GOLD_PATH) / "_notifications"
        notification_dir.mkdir(parents=True, exist_ok=True)
        notification_path = notification_dir / f"run_{quality_info['execution_date']}.json"
        with open(notification_path, "w") as fh:
            json.dump(summary, fh, indent=2)

    extracted = extract_raw_claims()
    validated = validate_raw_data(extracted)
    bronze = bronze_ingestion(validated)
    silver = silver_transformation(bronze)
    gold = gold_aggregation(silver)
    quality = data_quality_check(gold)
    notify_completion(quality)


claims_dag = claims_etl_dag()
