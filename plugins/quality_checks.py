"""Data quality validation for claims ETL pipeline.

Provides the ClaimsDataValidator class with schema validation,
completeness checks, distribution analysis, and referential integrity
verification. Produces structured validation reports.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

EXPECTED_CLAIMS_COLUMNS: Dict[str, str] = {
    "claim_id": "object",
    "patient_id": "object",
    "provider_id": "object",
    "service_date": "object",
    "diagnosis_code": "object",
    "procedure_code": "object",
    "claim_amount": "float64",
    "paid_amount": "float64",
    "status": "object",
}

VALID_STATUSES: Set[str] = {"approved", "denied", "pending", "review"}


class ClaimsDataValidator:
    """Validates claims data quality at various pipeline stages.

    Attributes:
        max_null_rate: Maximum acceptable fraction of nulls per column.
        min_completeness: Minimum acceptable fraction of non-null values
            across all columns of a row.
        max_duplicate_rate: Maximum acceptable fraction of duplicate rows.
    """

    def __init__(self, config_path: Optional[str] = None) -> None:
        config = self._load_config(config_path)
        quality_cfg = config.get("quality", {})
        self.max_null_rate = quality_cfg.get("max_null_rate", 0.05)
        self.min_completeness = quality_cfg.get("min_completeness", 0.95)
        self.max_duplicate_rate = quality_cfg.get("max_duplicate_rate", 0.01)

    @staticmethod
    def _load_config(config_path: Optional[str] = None) -> dict:
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as fh:
                return yaml.safe_load(fh) or {}
        return {}

    def validate_schema(
        self,
        df: pd.DataFrame,
        expected_columns: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Validate column presence and data types.

        Args:
            df: DataFrame to validate.
            expected_columns: Dict mapping column name to expected dtype
                string. Defaults to EXPECTED_CLAIMS_COLUMNS.

        Returns:
            Validation result dict with ``passed``, ``missing_columns``,
            ``extra_columns``, and ``type_mismatches``.
        """
        expected = expected_columns or EXPECTED_CLAIMS_COLUMNS

        actual_cols = set(df.columns)
        expected_cols = set(expected.keys())

        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols

        type_mismatches = {}
        for col, expected_dtype in expected.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not self._dtype_compatible(actual_dtype, expected_dtype):
                    type_mismatches[col] = {
                        "expected": expected_dtype,
                        "actual": actual_dtype,
                    }

        passed = len(missing) == 0 and len(type_mismatches) == 0

        result = {
            "passed": passed,
            "missing_columns": list(missing),
            "extra_columns": list(extra),
            "type_mismatches": type_mismatches,
            "column_count": len(df.columns),
            "expected_count": len(expected),
        }

        if not passed:
            logger.warning("Schema validation failed: %s", result)
        else:
            logger.info("Schema validation passed (%d columns)", len(df.columns))

        return result

    @staticmethod
    def _dtype_compatible(actual: str, expected: str) -> bool:
        """Check if an actual dtype is compatible with the expected one."""
        if actual == expected:
            return True
        compatible_groups = {
            "float64": {"float32", "float64", "Float64"},
            "int64": {"int32", "int64", "Int64"},
            "object": {"string", "object", "str"},
        }
        return actual in compatible_groups.get(expected, set())

    def validate_completeness(
        self,
        df: pd.DataFrame,
        critical_columns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Validate null rates and row-level completeness.

        Args:
            df: DataFrame to validate.
            critical_columns: Columns that must have zero nulls.

        Returns:
            Validation result dict with per-column null rates and
            overall completeness score.
        """
        if len(df) == 0:
            return {
                "passed": False,
                "reason": "empty_dataframe",
                "row_count": 0,
                "null_rates": {},
                "completeness_score": 0.0,
            }

        null_rates = {}
        violations = []

        for col in df.columns:
            rate = float(df[col].isnull().mean())
            null_rates[col] = round(rate, 6)
            if rate > self.max_null_rate:
                violations.append(col)

        non_null_frac = df.notna().mean(axis=1)
        overall_completeness = float(non_null_frac.mean())

        critical_check = True
        critical_failures = []
        if critical_columns:
            for col in critical_columns:
                if col in df.columns and df[col].isnull().any():
                    critical_check = False
                    critical_failures.append(col)

        passed = (
            len(violations) == 0
            and overall_completeness >= self.min_completeness
            and critical_check
        )

        result = {
            "passed": passed,
            "null_rates": null_rates,
            "violation_columns": violations,
            "completeness_score": round(overall_completeness, 6),
            "min_completeness_threshold": self.min_completeness,
            "max_null_rate_threshold": self.max_null_rate,
            "critical_failures": critical_failures,
            "row_count": len(df),
        }

        if passed:
            logger.info("Completeness validation passed: score=%.4f", overall_completeness)
        else:
            logger.warning("Completeness validation failed: %s", result)

        return result

    def validate_distributions(
        self,
        df: pd.DataFrame,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, Any]:
        """Validate statistical distribution of numeric columns.

        Args:
            df: DataFrame to validate.
            bounds: Dict mapping column names to (min, max) acceptable
                bounds. If not provided, uses reasonable defaults for
                claims data.

        Returns:
            Validation result dict with per-column statistics and
            out-of-bounds flags.
        """
        default_bounds = {
            "claim_amount": (0.0, 1_000_000.0),
            "paid_amount": (0.0, 1_000_000.0),
        }
        effective_bounds = bounds or default_bounds

        stats = {}
        out_of_bounds = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            col_stats = {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "median": float(df[col].median()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75)),
            }
            stats[col] = col_stats

            if col in effective_bounds:
                lo, hi = effective_bounds[col]
                oob_count = int(((df[col] < lo) | (df[col] > hi)).sum())
                if oob_count > 0:
                    out_of_bounds[col] = {
                        "count": oob_count,
                        "fraction": round(oob_count / len(df), 6),
                        "bounds": [lo, hi],
                    }

        passed = len(out_of_bounds) == 0

        result = {
            "passed": passed,
            "statistics": stats,
            "out_of_bounds": out_of_bounds,
        }

        if passed:
            logger.info("Distribution validation passed for %d numeric columns", len(stats))
        else:
            logger.warning("Distribution validation failed: %s", list(out_of_bounds.keys()))

        return result

    def validate_referential_integrity(
        self,
        fact_df: pd.DataFrame,
        dim_df: pd.DataFrame,
        fk_column: str,
        pk_column: str,
    ) -> Dict[str, Any]:
        """Validate foreign key references between fact and dimension tables.

        Args:
            fact_df: Fact table DataFrame.
            dim_df: Dimension table DataFrame.
            fk_column: Foreign key column name in the fact table.
            pk_column: Primary key column name in the dimension table.

        Returns:
            Validation result with orphan count and sample orphan values.
        """
        fact_keys = set(fact_df[fk_column].dropna().unique())
        dim_keys = set(dim_df[pk_column].dropna().unique())

        orphans = fact_keys - dim_keys
        orphan_rate = len(orphans) / max(len(fact_keys), 1)

        passed = len(orphans) == 0

        result = {
            "passed": passed,
            "fk_column": fk_column,
            "pk_column": pk_column,
            "total_fk_values": len(fact_keys),
            "total_pk_values": len(dim_keys),
            "orphan_count": len(orphans),
            "orphan_rate": round(orphan_rate, 6),
            "sample_orphans": list(orphans)[:10],
        }

        if passed:
            logger.info("Referential integrity passed: %s -> %s", fk_column, pk_column)
        else:
            logger.warning("Referential integrity failed: %d orphans", len(orphans))

        return result

    def generate_report(
        self,
        df: pd.DataFrame,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a comprehensive HTML validation report.

        Runs all validation checks and renders the results as an HTML
        document suitable for review and audit trails.

        Args:
            df: DataFrame to validate.
            output_path: File path for the HTML report. If None, returns
                HTML as a string without writing to disk.

        Returns:
            HTML report content as a string.
        """
        schema_result = self.validate_schema(df)
        completeness_result = self.validate_completeness(df)
        distribution_result = self.validate_distributions(df)

        timestamp = datetime.utcnow().isoformat()
        overall_passed = (
            schema_result["passed"]
            and completeness_result["passed"]
            and distribution_result["passed"]
        )

        status_color = "#2ecc71" if overall_passed else "#e74c3c"
        status_text = "PASSED" if overall_passed else "FAILED"

        html_parts = [
            "<!DOCTYPE html>",
            "<html><head><title>Data Quality Report</title>",
            "<style>",
            "body { font-family: Arial, sans-serif; margin: 40px; }",
            "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
            "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
            "th { background-color: #f5f5f5; }",
            ".pass { color: #2ecc71; } .fail { color: #e74c3c; }",
            "</style></head><body>",
            f"<h1>Data Quality Report</h1>",
            f"<p>Generated: {timestamp}</p>",
            f"<p>Overall Status: <strong style='color:{status_color}'>{status_text}</strong></p>",
            f"<p>Row Count: {len(df)} | Columns: {len(df.columns)}</p>",
            "<h2>Schema Validation</h2>",
            f"<p class='{'pass' if schema_result['passed'] else 'fail'}'>",
            f"{'PASSED' if schema_result['passed'] else 'FAILED'}</p>",
        ]

        if schema_result["missing_columns"]:
            html_parts.append(f"<p>Missing: {', '.join(schema_result['missing_columns'])}</p>")

        html_parts.append("<h2>Completeness</h2>")
        html_parts.append(f"<p>Score: {completeness_result['completeness_score']:.4f}</p>")
        html_parts.append("<table><tr><th>Column</th><th>Null Rate</th></tr>")

        for col, rate in completeness_result["null_rates"].items():
            css_class = "fail" if rate > self.max_null_rate else "pass"
            html_parts.append(f"<tr><td>{col}</td><td class='{css_class}'>{rate:.4f}</td></tr>")

        html_parts.append("</table>")
        html_parts.append("<h2>Distribution Checks</h2>")

        if distribution_result["statistics"]:
            html_parts.append("<table><tr><th>Column</th><th>Mean</th><th>Std</th>"
                              "<th>Min</th><th>Max</th></tr>")
            for col, stats in distribution_result["statistics"].items():
                html_parts.append(
                    f"<tr><td>{col}</td><td>{stats['mean']:.2f}</td>"
                    f"<td>{stats['std']:.2f}</td><td>{stats['min']:.2f}</td>"
                    f"<td>{stats['max']:.2f}</td></tr>"
                )
            html_parts.append("</table>")

        html_parts.append("</body></html>")
        html = "\n".join(html_parts)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as fh:
                fh.write(html)
            logger.info("Quality report written to %s", output_path)

        return html
