# Apache Airflow ETL Pipeline

Production-grade data orchestration pipeline using Apache Airflow. Implements the Bronze/Silver/Gold medallion architecture for claims data processing with data quality validation, SCD Type-2 dimension management, and operational monitoring.

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  Raw Source  │───>│   Bronze    │───>│    Silver    │───>│    Gold     │
│  (CSV/API)  │    │  (as-is +   │    │  (cleaned,   │    │ (aggregates,│
│             │    │  metadata)  │    │  dedup, SCD) │    │  features)  │
└─────────────┘    └─────────────┘    └──────────────┘    └─────────────┘
                         │                   │                   │
                    ┌────┴────┐        ┌─────┴────┐        ┌────┴─────┐
                    │Validate │        │Transform │        │ Quality  │
                    │  Schema │        │  & Merge │        │  Gates   │
                    └─────────┘        └──────────┘        └──────────┘
```

### Medallion Layers

- **Bronze**: Raw data ingested as-is with added metadata (ingestion timestamp, source, batch ID). No transformations.
- **Silver**: Cleaned, deduplicated, and type-cast data. SCD Type-2 logic applied for dimension changes. Referential integrity enforced.
- **Gold**: Business-level aggregated tables and feature sets ready for analytics and ML consumption.

## Project Structure

```
airflow-etl-pipeline/
├── dags/
│   ├── claims_etl_dag.py          # Main daily ETL DAG
│   └── model_retraining_dag.py    # Weekly model retraining DAG
├── plugins/
│   ├── quality_checks.py          # Great Expectations validation
│   └── transformations.py         # Data transformation utilities
├── scripts/
│   └── init_airflow.sh            # Airflow initialization script
├── configs/
│   └── config.yaml                # Pipeline configuration
├── data/
│   ├── raw/
│   ├── bronze/
│   ├── silver/
│   └── gold/
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/yourusername/airflow-etl-pipeline.git
cd airflow-etl-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Initialize Airflow
bash scripts/init_airflow.sh
```

## DAGs

### Claims ETL DAG (`claims_etl_dag`)

Runs daily. Processes raw claims data through the full medallion architecture:

1. **extract_raw_claims** -- Reads source data (CSV files or simulated API calls)
2. **validate_raw_data** -- Runs Great Expectations validation suite on raw data
3. **bronze_ingestion** -- Writes raw data to Bronze layer with metadata
4. **silver_transformation** -- Cleans, deduplicates, casts types, applies SCD-2
5. **gold_aggregation** -- Builds provider-level, patient-level, and temporal aggregates
6. **data_quality_check** -- Validates Gold layer against quality gates
7. **notify_completion** -- Sends pipeline completion notification

### Model Retraining DAG (`model_retraining_dag`)

Runs weekly. Retrains ML models using data from the Gold layer:

1. **check_data_drift** -- Statistical tests for distribution shift
2. **extract_training_data** -- Pulls feature tables from Gold
3. **train_model** -- Trains a new model version
4. **evaluate_model** -- Evaluates on holdout set
5. **compare_champion** -- Compares new model vs. current champion
6. **promote_model** -- Promotes the new model if it outperforms (conditional)
7. **update_registry** -- Updates the model registry

## Configuration

All pipeline parameters are centralized in `configs/config.yaml`:

| Section | Key | Description |
|---|---|---|
| `airflow.schedule_interval` | Cron or preset schedule | `@daily` |
| `paths.bronze` | Bronze layer path | `./data/bronze` |
| `quality.max_null_rate` | Maximum null fraction per column | `0.05` |
| `quality.min_completeness` | Minimum row completeness | `0.95` |
| `notifications.on_failure` | Alert on DAG failure | `true` |

## Data Quality

Quality is enforced at multiple checkpoints using Great Expectations:

- **Schema validation**: Column presence, data types, allowed values
- **Completeness**: Null rate thresholds per column
- **Distribution checks**: Statistical bounds on numeric columns
- **Referential integrity**: Foreign key consistency between tables
- **Business rules**: Claim amount ranges, valid date sequences

## SCD Type-2

The Silver layer implements Slowly Changing Dimension Type 2 for tracking historical changes to dimension attributes (provider info, patient demographics). Each dimension row carries:

- `effective_start_date`
- `effective_end_date`
- `is_current` flag

## Requirements

- Python 3.9+
- Apache Airflow 2.8+
- (Optional) Spark 3.5+ for large-scale processing

## License

MIT License
