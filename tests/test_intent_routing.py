"""
Intent routing accuracy.

The router is the reference implementation of question -> metric selection. If
it drifts, the deterministic mode silently starts answering a different question
from the one that was asked, which is worse than failing outright.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pi.agent.agent import PeopleAgent               # noqa: E402
from pi.agent.router import classify                 # noqa: E402
from pi.governance import AccessContext              # noqa: E402

CTX = AccessContext(role_key="hr_leader", persona="Enterprise", user="t")

# (question, expected intent key, expected metric id or None)
CASES = [
    # Retention
    ("Which business units have the highest voluntary attrition?", "attrition",
     "voluntary_attrition_rate"),
    ("What is our turnover by department?", "attrition", "voluntary_attrition_rate"),
    ("What employee groups have increasing retention risk?", "retention_risk",
     "retention_risk_index"),
    ("How has attrition changed over the last 12 months?", "attrition_trend",
     "attrition_trend"),
    ("Why are people leaving?", "exit_reasons", "exit_reason_mix"),
    ("What is our internal mobility rate?", "internal_mobility", "internal_mobility_rate"),
    # Recruiting
    ("Where are candidates dropping out of the funnel?", "funnel", "funnel_conversion"),
    ("Where is the bottleneck in hiring?", "funnel", "funnel_conversion"),
    ("Which recruiting sources have the strongest conversion?", "source_effectiveness",
     "source_effectiveness"),
    ("Which job families have the longest time-to-fill?", "time_to_fill", "time_to_fill"),
    ("How many open requisitions do we have?", "requisitions", "open_requisitions"),
    ("What is our offer acceptance rate?", "offer_acceptance", "offer_acceptance_rate"),
    # Workforce
    ("Which functions are growing fastest?", "growth", "headcount_growth_rate"),
    ("Where are we seeing workforce gaps?", "workforce_gaps", None),
    ("What happens to projected headcount if attrition increases by 3%?", "scenario", None),
    ("What is our headcount?", "headcount", "headcount"),
    ("How has headcount trended over the last two years?", "headcount_trend",
     "headcount_trend"),
    # Employee experience
    ("What are the biggest themes in employee feedback?", "themes", "feedback_themes"),
    ("Which themes are associated with lower engagement?", "theme_engagement",
     "theme_engagement_link"),
    ("What changed in employee sentiment this quarter?", "engagement_trend",
     "engagement_trend"),
    ("How does engagement compare across business units?", "engagement",
     "engagement_score"),
    ("What is our eNPS?", "enps", "enps"),
    # Governance
    ("How is voluntary attrition defined?", "metric_definition", None),
    ("How reliable is this data?", "data_quality", None),
    ("Which business units should HR focus on this quarter?", "focus_areas", None),
]


@pytest.mark.parametrize("question,intent_key,metric", CASES,
                         ids=[c[0][:42] for c in CASES])
def test_intent_and_metric_selection(question: str, intent_key: str, metric: str | None):
    intent = classify(question)
    assert intent.key == intent_key, f"{question!r} routed to {intent.key}"
    if metric is not None:
        assert intent.metric == metric, f"{question!r} selected {intent.metric}"


DIMENSION_CASES = [
    ("What is attrition by department?", "department"),
    ("Show attrition by job level", "job_level"),
    ("Which job families have the longest time to fill?", "job_family"),
    ("Attrition by location", "location"),
    ("Attrition by tenure band", "tenure_band"),
    ("Attrition by business unit", "business_unit"),
]


@pytest.mark.parametrize("question,dimension", DIMENSION_CASES)
def test_dimension_detection(question: str, dimension: str):
    assert classify(question).dimension == dimension


def test_business_unit_is_detected_as_a_filter():
    intent = classify("What is attrition in Operations?")
    assert intent.filters.get("business_unit") == "Operations"


def test_scenario_parses_the_magnitude():
    assert classify("what if attrition increases by 5 percentage points").extras[
        "attrition_change_pts"] == 5.0


def test_scenario_parses_the_horizon():
    assert classify("what if attrition rises 2% over 24 months").extras["months"] == 24


# ------------------------------------------------------ end-to-end answers
ANSWERABLE = [c[0] for c in CASES]


@pytest.mark.parametrize("question", ANSWERABLE, ids=[q[:42] for q in ANSWERABLE])
def test_every_sample_question_produces_a_grounded_answer(question: str):
    agent = PeopleAgent(CTX, force_deterministic=True)
    ans = agent.ask(question)
    assert not ans.refused
    assert len(ans.markdown) > 120, "answer is too thin to be useful"
    assert ans.belt.calls, "an answer must be backed by at least one tool call"
    assert all(c.ok for c in ans.belt.calls), \
        [c.error for c in ans.belt.calls if not c.ok]


def test_answers_carry_sql_provenance():
    agent = PeopleAgent(CTX, force_deterministic=True)
    ans = agent.ask("Which business units have the highest voluntary attrition?")
    ev = ans.evidence()
    assert ev["sql"], "no SQL recorded"
    assert "SELECT" in ev["sql"][0]["sql"].upper()
    assert ev["provenance"][0]["definition"]
    assert ev["data_quality"]


def test_the_same_question_answers_identically_twice():
    agent = PeopleAgent(CTX, force_deterministic=True)
    q = "Which business units have the highest voluntary attrition?"
    assert agent.ask(q).markdown == agent.ask(q).markdown


def test_hrbp_answer_is_scoped_to_their_units():
    hrbp = AccessContext(role_key="hrbp", persona="Technology HRBP")
    ans = PeopleAgent(hrbp, force_deterministic=True).ask(
        "Which business units have the highest voluntary attrition?")
    for other in ("Operations", "Commerce", "Marketing", "Corporate"):
        assert other not in ans.markdown, f"{other} leaked into an HRBP answer"
