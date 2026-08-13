"""
Data quality engine.

Every check is declarative: an id, the rule in words, the SQL that finds the
offending rows, a severity, and what the product does about it. Hard-severity
failures are quarantined by the governed views, so they are excluded from every
metric consistently rather than being handled differently in each dashboard.

The completeness score published on the Data & Governance page is computed here,
not asserted.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from pi import warehouse


@dataclass
class Check:
    id: str
    dataset: str
    rule: str
    severity: str          # 'hard' (quarantine) | 'soft' (flag only)
    treatment: str
    sql: str
    field: str | None = None


CHECKS: list[Check] = [
    Check(
        id="EMP001", dataset="employees", field="employee_id",
        rule="Employee id must be unique",
        severity="hard",
        treatment="All copies of a duplicated id are quarantined from v_employees.",
        sql="""SELECT COUNT(*) AS failed FROM (
                 SELECT employee_id FROM employees
                 GROUP BY 1 HAVING COUNT(*) > 1)""",
    ),
    Check(
        id="EMP002", dataset="employees", field="department",
        rule="Department must be populated",
        severity="soft",
        treatment="Imputed to 'Unassigned' and flagged; the record still counts in headcount.",
        sql="SELECT COUNT(*) AS failed FROM employees WHERE department IS NULL",
    ),
    Check(
        id="EMP003", dataset="employees", field="termination_date",
        rule="A terminated employee must have a termination date",
        severity="hard",
        treatment="Quarantined - otherwise the exit is invisible to attrition and the "
                  "employee counts as active forever.",
        sql="""SELECT COUNT(*) AS failed FROM employees
               WHERE termination_type IS NOT NULL AND termination_date IS NULL""",
    ),
    Check(
        id="EMP004", dataset="employees", field="termination_date",
        rule="Termination date must not precede hire date",
        severity="hard",
        treatment="Quarantined as a logically impossible employment period.",
        sql="""SELECT COUNT(*) AS failed FROM employees
               WHERE termination_date IS NOT NULL AND termination_date < hire_date""",
    ),
    Check(
        id="EMP005", dataset="employees", field="hire_date",
        rule="Hire date must not be in the future relative to the reporting date",
        severity="hard",
        treatment="Quarantined; future starters belong in the pipeline, not in headcount.",
        sql="""SELECT COUNT(*) AS failed FROM employees
               WHERE hire_date > (SELECT as_of_date FROM meta_asof)""",
    ),
    Check(
        id="EMP006", dataset="employees", field="job_level",
        rule="Job level must be populated",
        severity="soft",
        treatment="Flagged; excluded from level-based breakdowns only.",
        sql="SELECT COUNT(*) AS failed FROM employees WHERE job_level IS NULL",
    ),
    Check(
        id="REC001", dataset="candidates", field="candidate_id",
        rule="Candidate id must be unique",
        severity="hard",
        treatment="De-duplicated to the earliest application in v_candidates.",
        sql="""SELECT COUNT(*) AS failed FROM (
                 SELECT candidate_id FROM candidates
                 GROUP BY 1 HAVING COUNT(*) > 1)""",
    ),
    Check(
        id="REC002", dataset="requisitions", field="time_to_fill_days",
        rule="Time to fill must be within a plausible range (0-365 days)",
        severity="hard",
        treatment="Flagged as an outlier and excluded from time-to-fill averages.",
        sql="""SELECT COUNT(*) AS failed FROM requisitions
               WHERE time_to_fill_days IS NOT NULL
                 AND (time_to_fill_days > 365 OR time_to_fill_days < 0)""",
    ),
    Check(
        id="REC003", dataset="requisitions", field="offer_accepted_date",
        rule="A filled requisition must carry an accepted offer date",
        severity="hard",
        treatment="Excluded from time-to-fill; the requisition still counts as filled.",
        sql="""SELECT COUNT(*) AS failed FROM requisitions
               WHERE status = 'Filled' AND offer_accepted_date IS NULL""",
    ),
    Check(
        id="REC004", dataset="requisitions", field="opened_date",
        rule="Requisition close date must not precede open date",
        severity="hard",
        treatment="Quarantined from recruiting cycle-time metrics.",
        sql="""SELECT COUNT(*) AS failed FROM requisitions
               WHERE closed_date IS NOT NULL AND closed_date < opened_date""",
    ),
    Check(
        id="EX001", dataset="survey_responses", field="engagement_score",
        rule="Engagement score must be present and within the 1-5 scale",
        severity="hard",
        treatment="Excluded from v_survey so that averages are not silently biased.",
        sql="""SELECT COUNT(*) AS failed FROM survey_responses
               WHERE engagement_score IS NULL
                  OR engagement_score < 1 OR engagement_score > 5""",
    ),
    Check(
        id="EX002", dataset="survey_responses", field="comment_text",
        rule="Themed comments must carry a sentiment label",
        severity="soft",
        treatment="Flagged; the response still contributes to scores.",
        sql="""SELECT COUNT(*) AS failed FROM survey_responses
               WHERE comment_text IS NOT NULL AND sentiment IS NULL""",
    ),
    Check(
        id="MOB001", dataset="internal_moves", field="employee_id",
        rule="Internal moves must reference an employee that survives quarantine",
        severity="soft",
        treatment="Orphan moves are dropped from mobility metrics.",
        sql="""SELECT COUNT(*) AS failed FROM internal_moves m
               WHERE NOT EXISTS (SELECT 1 FROM v_employees e
                                 WHERE e.employee_id = m.employee_id)""",
    ),
]

DATASET_ROWCOUNT = {
    "employees": "SELECT COUNT(*) FROM employees",
    "candidates": "SELECT COUNT(*) FROM candidates",
    "requisitions": "SELECT COUNT(*) FROM requisitions",
    "survey_responses": "SELECT COUNT(*) FROM survey_responses",
    "internal_moves": "SELECT COUNT(*) FROM internal_moves",
}


def run_checks() -> pd.DataFrame:
    counts = {k: int(warehouse.query(v).iloc[0, 0]) for k, v in DATASET_ROWCOUNT.items()}
    rows = []
    for c in CHECKS:
        failed = int(warehouse.query(c.sql).iloc[0, 0])
        total = counts.get(c.dataset, 0)
        rows.append({
            "check_id": c.id,
            "dataset": c.dataset,
            "field": c.field or "",
            "rule": c.rule,
            "severity": c.severity,
            "records_checked": total,
            "records_failed": failed,
            "pass_rate_pct": round(100.0 * (total - failed) / total, 2) if total else 100.0,
            "status": "Pass" if failed == 0 else ("Quarantined" if c.severity == "hard" else "Flagged"),
            "treatment": c.treatment,
        })
    return pd.DataFrame(rows)


def scorecard() -> dict:
    df = run_checks()
    total_checked = int(df["records_checked"].sum())
    total_failed = int(df["records_failed"].sum())
    completeness = round(100.0 * (total_checked - total_failed) / total_checked, 2) if total_checked else 100.0

    excl = warehouse.query(
        "SELECT exclusion_reason, COUNT(*) AS records "
        "FROM v_employee_exclusions GROUP BY 1 ORDER BY 2 DESC")
    raw_emp = int(warehouse.query("SELECT COUNT(*) FROM employees").iloc[0, 0])
    clean_emp = int(warehouse.query("SELECT COUNT(*) FROM v_employees").iloc[0, 0])

    return {
        "completeness_pct": completeness,
        "checks_run": int(len(df)),
        "checks_passed": int((df["records_failed"] == 0).sum()),
        "records_checked": total_checked,
        "records_failed": total_failed,
        "hard_failures": int(df.loc[df["severity"] == "hard", "records_failed"].sum()),
        "soft_failures": int(df.loc[df["severity"] == "soft", "records_failed"].sum()),
        "employees_raw": raw_emp,
        "employees_in_scope": clean_emp,
        "employees_excluded": raw_emp - clean_emp,
        "exclusions": excl,
        "detail": df,
    }


def banner() -> str:
    """One-line quality statement shown alongside any answer."""
    s = scorecard()
    return (f"Data quality {s['completeness_pct']}% complete across "
            f"{s['records_checked']:,} records checked - "
            f"{s['employees_excluded']} employee records excluded from all metrics "
            f"({s['checks_passed']}/{s['checks_run']} checks fully clean).")


def lineage() -> pd.DataFrame:
    """Where each governed view comes from and what it enforces."""
    return pd.DataFrame([
        {"view": "v_employee_exclusions", "sources": "employees",
         "enforces": "Identifies every record failing a hard employee rule, once, in one place."},
        {"view": "v_employees", "sources": "employees, v_employee_exclusions",
         "enforces": "Quarantines failed records; imputes department; derives tenure and active status."},
        {"view": "v_headcount_monthly", "sources": "v_employees, v_month_spine",
         "enforces": "Employee-month fact used by every point-in-time and average headcount figure."},
        {"view": "v_movement_monthly", "sources": "v_employees",
         "enforces": "Hire and exit events on a common grain, so joiners and leavers reconcile."},
        {"view": "v_requisitions", "sources": "requisitions",
         "enforces": "Derives requisition age and flags implausible time-to-fill."},
        {"view": "v_candidates", "sources": "candidates",
         "enforces": "De-duplicates repeat applications by candidate id."},
        {"view": "v_survey", "sources": "survey_responses",
         "enforces": "Drops responses with a missing engagement score."},
        {"view": "v_internal_moves", "sources": "internal_moves, v_employees",
         "enforces": "Keeps only moves whose employee survives quarantine."},
    ])
