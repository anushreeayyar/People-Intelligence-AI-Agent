"""The metrics must be internally consistent, or none of the rest matters."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pi import analysis, semantic_layer as sl, warehouse   # noqa: E402
from pi.governance import AccessContext                    # noqa: E402
from pi.quality import checks                              # noqa: E402

CTX = AccessContext(role_key="hr_leader", persona="Enterprise", user="t")
ALL_METRICS = sorted(sl.catalog())


# --------------------------------------------------------------- dictionary
@pytest.mark.parametrize("metric_id", ALL_METRICS)
def test_every_metric_is_fully_documented(metric_id: str):
    s = sl.get(metric_id)
    assert len(s.definition) > 60, "a definition should actually define something"
    assert s.formula and s.source_system and s.owner and s.refresh
    assert s.source_objects and s.dimensions
    assert s.direction in ("higher_is_better", "lower_is_better", "context")
    assert s.sql or s.computed_by


@pytest.mark.parametrize("metric_id", ALL_METRICS)
def test_every_metric_executes_on_every_declared_dimension(metric_id: str):
    spec = sl.get(metric_id)
    for dim in spec.dimensions:
        r = sl.run(metric_id, dim, None, CTX)
        assert "dimension" in r.data.columns or r.data.empty
        assert "value" in r.data.columns or r.data.empty


@pytest.mark.parametrize("metric_id", ALL_METRICS)
def test_every_metric_returns_provenance(metric_id: str):
    p = sl.run(metric_id, "business_unit", None, CTX).provenance()
    for field in ("metric", "definition", "formula", "source_system",
                  "source_objects", "owner", "sql", "row_policy"):
        assert p[field], f"{metric_id} missing {field}"


def test_metrics_only_read_governed_views():
    for spec in sl.catalog().values():
        if not spec.sql:
            continue
        for raw in ("FROM employees", "FROM candidates", "FROM requisitions",
                    "FROM survey_responses", "FROM internal_moves"):
            assert raw not in spec.sql, f"{spec.id} reads a raw table"


# ------------------------------------------------------------ reconciliation
def test_headcount_matches_the_warehouse():
    got = int(sl.run("headcount", "All", None, CTX).data["value"].iloc[0])
    expected = int(warehouse.query(
        "SELECT COUNT(DISTINCT employee_id) FROM v_employees WHERE is_active").iloc[0, 0])
    assert got == expected


def test_headcount_by_unit_sums_to_the_total():
    total = int(sl.run("headcount", "All", None, CTX).data["value"].iloc[0])
    by_bu = int(sl.run("headcount", "business_unit", None, CTX).data["value"].sum())
    assert total == by_bu


def test_headcount_trend_last_month_matches_point_in_time():
    trend = sl.run("headcount_trend", "All", None, CTX).data
    last = trend[trend["period"] == trend["period"].max()]["value"].sum()
    now = sl.run("headcount", "All", None, CTX).data["value"].iloc[0]
    assert abs(int(last) - int(now)) <= 1


def test_voluntary_attrition_never_exceeds_total_attrition():
    v = sl.run("voluntary_attrition_rate", "business_unit", None, CTX).data
    t = sl.run("total_attrition_rate", "business_unit", None, CTX).data
    merged = v.merge(t, on="dimension", suffixes=("_vol", "_tot"))
    assert (merged["value_vol"] <= merged["value_tot"] + 1e-6).all()


def test_funnel_is_monotonically_narrowing():
    f = sl.run("funnel_conversion", "All", None, CTX).data.sort_values("stage_index")
    assert f["value"].is_monotonic_decreasing


def test_funnel_conversions_stay_within_bounds():
    f = sl.run("funnel_conversion", "All", None, CTX).data
    steps = f["step_conversion_pct"].dropna()
    assert (steps >= 0).all() and (steps <= 100).all()


def test_offer_acceptance_within_bounds():
    d = sl.run("offer_acceptance_rate", "business_unit", None, CTX).data
    assert (d["value"] >= 0).all() and (d["value"] <= 100).all()
    assert (d["offers_accepted"] <= d["offers_decided"]).all()


def test_engagement_scores_are_on_the_declared_scale():
    d = sl.run("engagement_score", "business_unit", None, CTX).data
    assert (d["value"] >= 1).all() and (d["value"] <= 5).all()


def test_enps_is_within_range():
    d = sl.run("enps", "business_unit", None, CTX).data
    assert (d["value"] >= -100).all() and (d["value"] <= 100).all()


def test_time_to_fill_excludes_outliers():
    d = sl.run("time_to_fill", "business_unit", None, CTX).data
    assert (d["mean_days"] <= 365).all()


def test_retention_risk_is_bounded_and_ranked():
    d = sl.run("retention_risk_index", "business_unit", None, CTX).data
    assert (d["value"] >= 0).all() and (d["value"] <= 100).all()
    assert d["value"].is_monotonic_decreasing


def test_results_are_deterministic():
    a = sl.run("voluntary_attrition_rate", "business_unit", None, CTX).data
    b = sl.run("voluntary_attrition_rate", "business_unit", None, CTX).data
    assert a.equals(b)


# ------------------------------------------------------------- data quality
def test_quality_scorecard_is_computed_not_asserted():
    s = checks.scorecard()
    assert 0 <= s["completeness_pct"] <= 100
    assert s["records_checked"] > 0
    assert s["employees_excluded"] == s["employees_raw"] - s["employees_in_scope"]


def test_hard_failures_are_actually_quarantined():
    """Every hard employee rule must produce zero failures inside v_employees."""
    for check in checks.CHECKS:
        if check.severity != "hard" or check.dataset != "employees":
            continue
        governed = check.sql.replace("FROM employees", "FROM v_employees")
        failed = int(warehouse.query(governed).iloc[0, 0])
        assert failed == 0, f"{check.id} still fails inside the governed view"


def test_seeded_defects_are_detected():
    df = checks.run_checks()
    detected = df[df["records_failed"] > 0]["check_id"].tolist()
    for expected in ["EMP001", "EMP002", "EMP003", "EMP004", "EMP005",
                     "REC001", "REC002", "EX001"]:
        assert expected in detected, f"{expected} should have caught seeded defects"


# ------------------------------------------------------------------ analysis
def test_scenario_is_monotonic_in_attrition():
    low = analysis.headcount_scenario(1.0, 12, 1.0, None, CTX)
    high = analysis.headcount_scenario(5.0, 12, 1.0, None, CTX)
    assert high["scenario_end_headcount"] < low["scenario_end_headcount"]


def test_zero_change_scenario_matches_baseline():
    out = analysis.headcount_scenario(0.0, 12, 1.0, None, CTX)
    assert out["scenario_end_headcount"] == out["baseline_end_headcount"]


def test_scenario_states_its_assumptions():
    out = analysis.headcount_scenario(3.0, 12, 1.0, None, CTX)
    assert len(out["assumptions"]) >= 4


def test_workforce_gaps_covers_every_unit_in_scope():
    gaps = analysis.workforce_gaps(CTX)
    hc = sl.run("headcount", "business_unit", None, CTX).data
    assert set(gaps["dimension"]) == set(hc["dimension"])
