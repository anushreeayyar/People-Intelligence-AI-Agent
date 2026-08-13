"""Central configuration for the People Intelligence platform."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WAREHOUSE = DATA_DIR / "warehouse.duckdb"
SEMANTIC_DIR = ROOT / "semantic"
METRICS_FILE = SEMANTIC_DIR / "metrics.yaml"
AUDIT_LOG = DATA_DIR / "audit_log.jsonl"

# ---------------------------------------------------------------- data shape
AS_OF_DATE = "2026-06-30"          # the "today" of the synthetic world
HISTORY_MONTHS = 24
N_EMPLOYEES = 2500
RANDOM_SEED = 20260630

BUSINESS_UNITS = [
    "Commerce", "Operations", "Engineering", "Customer Success",
    "Marketing", "Corporate",
]

# ------------------------------------------------------------------ modeling
#: Minimum group size before a result may be shown. Protects re-identification.
MIN_AGGREGATION_GROUP = 5

#: Columns that may never be returned at row level by the agent, for any role.
RESTRICTED_COLUMNS = {
    "employee_id", "full_name", "email", "base_salary", "bonus_target_pct",
    "total_comp", "date_of_birth", "home_zip", "performance_rating",
    "manager_name", "termination_reason_note",
}

#: Direct identifiers - never surfaced even in aggregate form.
PII_COLUMNS = {"full_name", "email", "date_of_birth", "home_zip", "manager_name"}


# --------------------------------------------------------------------- model
def anthropic_api_key() -> str | None:
    return os.environ.get("ANTHROPIC_API_KEY") or None


MODEL = os.environ.get("PI_MODEL", "claude-sonnet-5")
MAX_AGENT_STEPS = int(os.environ.get("PI_MAX_AGENT_STEPS", "8"))
