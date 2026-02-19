"""Data transformation utilities for the claims ETL pipeline.

Provides the ClaimsTransformer class with methods for data cleaning, code
standardization, SCD Type-2 dimension management, aggregation computation,
and data enrichment.
"""

import hashlib
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ICD10_PATTERN = re.compile(r"^[A-Z]\d{2}(\.\d{1,4})?$")
CPT_PATTERN = re.compile(r"^\d{5}$")


class ClaimsTransformer:
    """Transforms claims data through cleaning, standardization, and enrichment.

    Designed for the Silver layer of the medallion architecture. Handles
    null imputation, outlier capping, deduplication, code normalization,
    SCD Type-2 dimension logic, and aggregate computation for the Gold layer.
    """

    def __init__(
        self,
        claim_amount_cap: float = 500_000.0,
        outlier_std_threshold: float = 4.0,
    ) -> None:
        """Initialize the transformer.

        Args:
            claim_amount_cap: Maximum claim amount before capping.
            outlier_std_threshold: Number of standard deviations beyond
                which numeric values are considered outliers.
        """
        self.claim_amount_cap = claim_amount_cap
        self.outlier_std_threshold = outlier_std_threshold

    def clean_claims_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean raw claims data.

        Operations performed:
        - Drop exact duplicate rows
        - Deduplicate by claim_id (keep latest)
        - Cast service_date to datetime
        - Fill missing status with "pending"
        - Cap extreme claim amounts
        - Ensure paid_amount <= claim_amount
        - Remove rows with negative amounts

        Args:
            df: Raw claims DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        original_len = len(df)
        df = df.copy()

        df = df.drop_duplicates()
        dup_removed = original_len - len(df)
        if dup_removed > 0:
            logger.info("Removed %d exact duplicates", dup_removed)

        if "claim_id" in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=["claim_id"], keep="last")
            logger.info("Deduplicated by claim_id: %d -> %d", before, len(df))

        if "service_date" in df.columns:
            df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
            invalid_dates = df["service_date"].isnull().sum()
            if invalid_dates > 0:
                logger.warning("Dropped %d rows with invalid service_date", invalid_dates)
                df = df.dropna(subset=["service_date"])

        if "status" in df.columns:
            df["status"] = df["status"].fillna("pending").str.lower().str.strip()

        if "claim_amount" in df.columns:
            df["claim_amount"] = df["claim_amount"].clip(upper=self.claim_amount_cap)
            df = df[df["claim_amount"] >= 0]

        if "paid_amount" in df.columns and "claim_amount" in df.columns:
            df["paid_amount"] = df["paid_amount"].clip(lower=0)
            df["paid_amount"] = df[["paid_amount", "claim_amount"]].min(axis=1)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mean_val = df[col].mean()
            std_val = df[col].std()
            if std_val > 0:
                upper = mean_val + self.outlier_std_threshold * std_val
                lower = mean_val - self.outlier_std_threshold * std_val
                outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
                if outlier_count > 0:
                    df[col] = df[col].clip(lower=lower, upper=upper)
                    logger.info("Capped %d outliers in column '%s'", outlier_count, col)

        logger.info("Cleaning complete: %d -> %d rows", original_len, len(df))
        return df.reset_index(drop=True)

    def standardize_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize diagnosis and procedure codes.

        - Strips whitespace and converts to uppercase
        - Validates ICD-10 format for diagnosis_code
        - Validates CPT format for procedure_code
        - Invalid codes are set to None

        Args:
            df: DataFrame with code columns.

        Returns:
            DataFrame with standardized codes.
        """
        df = df.copy()

        if "diagnosis_code" in df.columns:
            df["diagnosis_code"] = df["diagnosis_code"].apply(self._clean_diagnosis_code)
            invalid_dx = df["diagnosis_code"].isnull().sum()
            logger.info("Diagnosis codes: %d null/invalid out of %d", invalid_dx, len(df))

        if "procedure_code" in df.columns:
            df["procedure_code"] = df["procedure_code"].apply(self._clean_procedure_code)
            invalid_px = df["procedure_code"].isnull().sum()
            logger.info("Procedure codes: %d null/invalid out of %d", invalid_px, len(df))

        return df

    @staticmethod
    def _clean_diagnosis_code(code: Any) -> Optional[str]:
        if pd.isna(code):
            return None
        cleaned = str(code).strip().upper()
        if ICD10_PATTERN.match(cleaned):
            return cleaned
        if re.match(r"^[A-Z]\d{2}\d{1,4}$", cleaned):
            return cleaned[:3] + "." + cleaned[3:]
        return None

    @staticmethod
    def _clean_procedure_code(code: Any) -> Optional[str]:
        if pd.isna(code):
            return None
        cleaned = str(code).strip()
        cleaned = re.sub(r"[^0-9]", "", cleaned)
        if len(cleaned) == 5:
            return cleaned
        return None

    def apply_scd_type2(
        self,
        new_data: pd.DataFrame,
        existing_path: str,
        key_column: str,
        tracked_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply Slowly Changing Dimension Type 2 logic.

        Compares new data against the existing dimension table. For rows
        whose tracked attributes have changed, the old row is closed
        (effective_end_date set, is_current = False) and a new row is
        inserted. New keys get a fresh row.

        Args:
            new_data: Incoming dimension data.
            existing_path: Path to the existing dimension Parquet file.
            key_column: Business key column name.
            tracked_columns: Columns whose changes trigger a new version.
                If None, all non-key non-metadata columns are tracked.

        Returns:
            Updated dimension DataFrame with SCD-2 history.
        """
        now = datetime.utcnow().isoformat()

        scd_cols = {"effective_start_date", "effective_end_date", "is_current", "_row_hash"}
        meta_cols = {"_ingestion_ts", "_source", "_batch_id"} | scd_cols

        if tracked_columns is None:
            tracked_columns = [
                c for c in new_data.columns
                if c != key_column and c not in meta_cols
            ]

        if os.path.exists(existing_path):
            existing = pd.read_parquet(existing_path)
        else:
            result = new_data.copy()
            result["effective_start_date"] = now
            result["effective_end_date"] = None
            result["is_current"] = True
            result["_row_hash"] = result.apply(
                lambda r: hashlib.sha256(
                    "|".join(str(r[c]) for c in tracked_columns if c in r.index).encode()
                ).hexdigest(),
                axis=1,
            )
            logger.info("SCD-2: Created new dimension with %d rows", len(result))
            return result

        current = existing[existing.get("is_current", pd.Series(True, index=existing.index)) == True].copy()

        new_hashes = {}
        for _, row in new_data.iterrows():
            key = row[key_column]
            hash_input = "|".join(str(row.get(c, "")) for c in tracked_columns)
            new_hashes[key] = hashlib.sha256(hash_input.encode()).hexdigest()

        changed_keys = []
        new_keys = []

        for key, new_hash in new_hashes.items():
            match = current[current[key_column] == key]
            if len(match) == 0:
                new_keys.append(key)
            else:
                old_hash = match.iloc[0].get("_row_hash", "")
                if old_hash != new_hash:
                    changed_keys.append(key)

        rows_to_close = existing[
            (existing[key_column].isin(changed_keys))
            & (existing.get("is_current", pd.Series(True, index=existing.index)) == True)
        ].index
        existing.loc[rows_to_close, "effective_end_date"] = now
        existing.loc[rows_to_close, "is_current"] = False

        new_rows = []
        for key in changed_keys + new_keys:
            row_data = new_data[new_data[key_column] == key].iloc[0].to_dict()
            row_data["effective_start_date"] = now
            row_data["effective_end_date"] = None
            row_data["is_current"] = True
            row_data["_row_hash"] = new_hashes[key]
            new_rows.append(row_data)

        if new_rows:
            new_rows_df = pd.DataFrame(new_rows)
            result = pd.concat([existing, new_rows_df], ignore_index=True)
        else:
            result = existing

        logger.info(
            "SCD-2 merge: %d changed, %d new, %d total rows",
            len(changed_keys),
            len(new_keys),
            len(result),
        )
        return result

    def compute_aggregations(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute Gold-layer aggregate tables.

        Produces three aggregate tables:
        - provider_aggregations: Per-provider claim metrics
        - patient_aggregations: Per-patient claim metrics
        - temporal_aggregations: Monthly claim volume and amounts

        Args:
            df: Silver-layer claims fact DataFrame.

        Returns:
            Dict mapping table name to aggregated DataFrame.
        """
        aggregations = {}

        if "provider_id" in df.columns:
            provider_agg = df.groupby("provider_id").agg(
                total_claims=("claim_id", "count"),
                total_claim_amount=("claim_amount", "sum"),
                avg_claim_amount=("claim_amount", "mean"),
                max_claim_amount=("claim_amount", "max"),
                total_paid_amount=("paid_amount", "sum"),
                avg_paid_amount=("paid_amount", "mean"),
                unique_patients=("patient_id", "nunique"),
            ).reset_index()

            if "status" in df.columns:
                denial_rate = (
                    df.groupby("provider_id")["status"]
                    .apply(lambda s: (s == "denied").mean())
                    .reset_index()
                    .rename(columns={"status": "denial_rate"})
                )
                provider_agg = provider_agg.merge(denial_rate, on="provider_id", how="left")

            for col in ["total_claim_amount", "avg_claim_amount", "total_paid_amount", "avg_paid_amount"]:
                if col in provider_agg.columns:
                    provider_agg[col] = provider_agg[col].round(2)

            aggregations["provider_aggregations"] = provider_agg
            logger.info("Provider aggregations: %d rows", len(provider_agg))

        if "patient_id" in df.columns:
            patient_agg = df.groupby("patient_id").agg(
                total_claims=("claim_id", "count"),
                total_claim_amount=("claim_amount", "sum"),
                avg_claim_amount=("claim_amount", "mean"),
                unique_providers=("provider_id", "nunique"),
                unique_diagnoses=("diagnosis_code", "nunique"),
            ).reset_index()

            if "service_date" in df.columns:
                date_range = df.groupby("patient_id")["service_date"].agg(
                    first_service_date="min",
                    last_service_date="max",
                ).reset_index()
                patient_agg = patient_agg.merge(date_range, on="patient_id", how="left")

            for col in ["total_claim_amount", "avg_claim_amount"]:
                if col in patient_agg.columns:
                    patient_agg[col] = patient_agg[col].round(2)

            aggregations["patient_aggregations"] = patient_agg
            logger.info("Patient aggregations: %d rows", len(patient_agg))

        if "service_date" in df.columns:
            df_temp = df.copy()
            if not pd.api.types.is_datetime64_any_dtype(df_temp["service_date"]):
                df_temp["service_date"] = pd.to_datetime(df_temp["service_date"], errors="coerce")

            df_temp["year_month"] = df_temp["service_date"].dt.to_period("M").astype(str)
            temporal_agg = df_temp.groupby("year_month").agg(
                claim_count=("claim_id", "count"),
                total_amount=("claim_amount", "sum"),
                avg_amount=("claim_amount", "mean"),
                unique_patients=("patient_id", "nunique"),
                unique_providers=("provider_id", "nunique"),
            ).reset_index()

            for col in ["total_amount", "avg_amount"]:
                if col in temporal_agg.columns:
                    temporal_agg[col] = temporal_agg[col].round(2)

            aggregations["temporal_aggregations"] = temporal_agg
            logger.info("Temporal aggregations: %d rows", len(temporal_agg))

        return aggregations

    def enrich_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns and computed fields.

        Enrichments include:
        - payment_ratio: paid_amount / claim_amount
        - is_high_value: claim_amount above the 90th percentile
        - day_of_week: from service_date
        - diagnosis_category: first three characters of diagnosis_code
        - processing_flag: derived from status

        Args:
            df: Claims DataFrame.

        Returns:
            Enriched DataFrame with additional columns.
        """
        df = df.copy()

        if "claim_amount" in df.columns and "paid_amount" in df.columns:
            df["payment_ratio"] = (
                df["paid_amount"] / df["claim_amount"].replace(0, np.nan)
            ).round(4)

        if "claim_amount" in df.columns:
            threshold_90 = df["claim_amount"].quantile(0.90)
            df["is_high_value"] = df["claim_amount"] >= threshold_90

        if "service_date" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["service_date"]):
                df["service_date"] = pd.to_datetime(df["service_date"], errors="coerce")
            df["service_day_of_week"] = df["service_date"].dt.day_name()
            df["service_month"] = df["service_date"].dt.month

        if "diagnosis_code" in df.columns:
            df["diagnosis_category"] = df["diagnosis_code"].apply(
                lambda x: str(x)[:3] if pd.notna(x) else None
            )

        if "status" in df.columns:
            status_map = {
                "approved": "complete",
                "denied": "complete",
                "pending": "in_progress",
                "review": "in_progress",
            }
            df["processing_flag"] = df["status"].map(status_map).fillna("unknown")

        logger.info("Data enrichment complete: %d columns", len(df.columns))
        return df
