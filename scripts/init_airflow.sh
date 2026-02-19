#!/usr/bin/env bash
# Initialize Apache Airflow for the claims ETL pipeline.
#
# This script:
#   1. Sets up the Airflow home directory
#   2. Initializes the metadata database
#   3. Creates an admin user
#   4. Configures Airflow variables and connections
#   5. Creates required data directories
#
# Usage:
#   bash scripts/init_airflow.sh
#
# Environment variables:
#   AIRFLOW_HOME     - Airflow home directory (default: ./airflow_home)
#   AIRFLOW_ADMIN    - Admin username (default: admin)
#   AIRFLOW_PASSWORD - Admin password (default: admin)
#   AIRFLOW_EMAIL    - Admin email (default: admin@example.com)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

export AIRFLOW_HOME="${AIRFLOW_HOME:-${PROJECT_DIR}/airflow_home}"
AIRFLOW_ADMIN="${AIRFLOW_ADMIN:-admin}"
AIRFLOW_PASSWORD="${AIRFLOW_PASSWORD:-admin}"
AIRFLOW_EMAIL="${AIRFLOW_EMAIL:-admin@example.com}"

echo "=== Airflow ETL Pipeline Initialization ==="
echo "Project directory: ${PROJECT_DIR}"
echo "Airflow home:      ${AIRFLOW_HOME}"
echo ""

# -------------------------------------------------------------------
# 1. Create Airflow home and required directories
# -------------------------------------------------------------------
echo "[1/6] Creating directories ..."

mkdir -p "${AIRFLOW_HOME}"
mkdir -p "${AIRFLOW_HOME}/dags"
mkdir -p "${AIRFLOW_HOME}/plugins"
mkdir -p "${AIRFLOW_HOME}/logs"

mkdir -p "${PROJECT_DIR}/data/raw"
mkdir -p "${PROJECT_DIR}/data/bronze"
mkdir -p "${PROJECT_DIR}/data/silver"
mkdir -p "${PROJECT_DIR}/data/gold"

# Symlink DAGs and plugins into Airflow home
ln -sf "${PROJECT_DIR}/dags/"* "${AIRFLOW_HOME}/dags/" 2>/dev/null || true
ln -sf "${PROJECT_DIR}/plugins/"* "${AIRFLOW_HOME}/plugins/" 2>/dev/null || true

echo "  Directories created and symlinked."

# -------------------------------------------------------------------
# 2. Set Airflow configuration overrides
# -------------------------------------------------------------------
echo "[2/6] Configuring Airflow environment ..."

export AIRFLOW__CORE__DAGS_FOLDER="${AIRFLOW_HOME}/dags"
export AIRFLOW__CORE__PLUGINS_FOLDER="${AIRFLOW_HOME}/plugins"
export AIRFLOW__CORE__LOAD_EXAMPLES="False"
export AIRFLOW__CORE__EXECUTOR="SequentialExecutor"
export AIRFLOW__DATABASE__SQL_ALCHEMY_CONN="sqlite:///${AIRFLOW_HOME}/airflow.db"
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT="8080"
export AIRFLOW__WEBSERVER__SECRET_KEY="$(python3 -c 'import secrets; print(secrets.token_hex(32))')"

echo "  Environment configured."

# -------------------------------------------------------------------
# 3. Initialize the Airflow metadata database
# -------------------------------------------------------------------
echo "[3/6] Initializing Airflow database ..."

airflow db init 2>&1 | tail -5

echo "  Database initialized."

# -------------------------------------------------------------------
# 4. Create the admin user
# -------------------------------------------------------------------
echo "[4/6] Creating admin user ..."

airflow users create \
    --username "${AIRFLOW_ADMIN}" \
    --password "${AIRFLOW_PASSWORD}" \
    --firstname "Admin" \
    --lastname "User" \
    --role "Admin" \
    --email "${AIRFLOW_EMAIL}" \
    2>&1 | tail -3

echo "  Admin user created: ${AIRFLOW_ADMIN}"

# -------------------------------------------------------------------
# 5. Set Airflow variables
# -------------------------------------------------------------------
echo "[5/6] Setting Airflow variables ..."

airflow variables set project_dir "${PROJECT_DIR}" 2>/dev/null || true
airflow variables set data_raw_path "${PROJECT_DIR}/data/raw" 2>/dev/null || true
airflow variables set data_bronze_path "${PROJECT_DIR}/data/bronze" 2>/dev/null || true
airflow variables set data_silver_path "${PROJECT_DIR}/data/silver" 2>/dev/null || true
airflow variables set data_gold_path "${PROJECT_DIR}/data/gold" 2>/dev/null || true
airflow variables set config_path "${PROJECT_DIR}/configs/config.yaml" 2>/dev/null || true
airflow variables set notification_email "data-team@company.com" 2>/dev/null || true

echo "  Variables set."

# -------------------------------------------------------------------
# 6. Verify setup
# -------------------------------------------------------------------
echo "[6/6] Verifying installation ..."

echo ""
echo "  Airflow version:  $(airflow version 2>/dev/null || echo 'not installed')"
echo "  Database:         ${AIRFLOW_HOME}/airflow.db"
echo "  DAGs folder:      ${AIRFLOW__CORE__DAGS_FOLDER}"
echo "  Plugins folder:   ${AIRFLOW__CORE__PLUGINS_FOLDER}"
echo ""

DAG_COUNT=$(airflow dags list 2>/dev/null | grep -c "claims_etl\|model_retraining" || echo "0")
echo "  Detected DAGs:    ${DAG_COUNT}"
echo ""

echo "=== Initialization Complete ==="
echo ""
echo "To start the Airflow webserver:"
echo "  export AIRFLOW_HOME=${AIRFLOW_HOME}"
echo "  airflow webserver --port 8080 &"
echo "  airflow scheduler &"
echo ""
echo "Then open http://localhost:8080 and log in with:"
echo "  Username: ${AIRFLOW_ADMIN}"
echo "  Password: ${AIRFLOW_PASSWORD}"
echo ""
