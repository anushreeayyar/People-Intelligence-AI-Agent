"""
Deterministic intent router.

Two jobs.

  1. It is the fallback engine. With no ANTHROPIC_API_KEY set, the product still
     answers every supported question - same tools, same governed SQL, same
     provenance - just with a templated narrative instead of a written one. A
     People Intelligence product that goes dark when a vendor key is missing is
     not a product.

  2. It is the reference implementation of intent -> metric selection. When the
     model is driving, this router is what its choices are evaluated against in
     tests/test_intent_routing.py.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from pi import analysis, charts, semantic_layer as sl
from pi.agent.tools import ToolBelt
from pi.governance import AccessContext

BUSINESS_UNITS = ["Commerce", "Operations", "Engineering", "Customer Success",
                  "Marketing", "Corporate"]

DIMENSION_CUES: list[tuple[str, str]] = [
    (r"\bby department\b|\bdepartment(s)?\b|\bteam(s)?\b", "department"),
    (r"\bjob famil(y|ies)\b|\brole type|\bdiscipline", "job_family"),
    (r"\bjob level|\bseniority|\blevel(s)?\b|\bmanager(-| )level|\bgrade", "job_level"),
    (r"\blocation(s)?\b|\bsite(s)?\b|\boffice(s)?\b|\bgeograph", "location"),
    (r"\btenure\b|\blength of service\b|\bnew (joiners|hires) vs", "tenure_band"),
    (r"\bsource(s)?\b|\bchannel(s)?\b", "source"),
    (r"\bremote\b|\bhybrid\b|\bwork model\b|\bonsite\b", "work_model"),
    (r"\bbusiness unit(s)?\b|\bfunction(s)?\b|\bunit(s)?\b|\bdivision", "business_unit"),
]


@dataclass
class Intent:
    key: str
    label: str
    metric: str | None
    dimension: str = "business_unit"
    chart: str = "bar"
    confidence: float = 0.0
    filters: dict = field(default_factory=dict)
    extras: dict = field(default_factory=dict)


# (intent key, human label, regex, metric, default dimension, chart, weight)
RULES: list[tuple[str, str, str, str | None, str, str, float]] = [
    ("metric_definition", "Metric definition request",
     r"\b(what (do you mean|is meant) by|how (do|is) .{0,25}(defined|calculated|computed)|"
     r"definition of|what counts as|how do you define|what (does|do) .{0,30}\bmean\b|"
     r"how (is|are) .{0,30}\bmeasured\b|what is the (definition|formula))\b",
     None, "All", "none", 3.4),
    ("data_quality", "Data quality enquiry",
     r"\b(data quality|how (reliable|accurate|complete)|can i trust|missing data|"
     r"duplicate|completeness)\b", None, "All", "none", 3.0),
    ("scenario", "Workforce planning scenario",
     r"\bwhat happens if\b|\bwhat if\b|\bscenario\b|\bif attrition (increase|rise|go(es)? up)\b|"
     r"\bproject(ed|ion)?\b.{0,30}\bheadcount\b|\bsimulate\b", None, "All", "line", 3.2),
    ("focus_areas", "Where should HR focus",
     r"\b(where|which|what).{0,30}\b(should|do) (hr|we|i|the business)\b.{0,25}\b(focus|prioriti|"
     r"attention|worry)\b|\btop (workforce )?(risks|issues|priorities)\b|"
     r"\bwhat needs attention\b", None, "business_unit", "bar", 3.2),
    ("workforce_gaps", "Workforce gaps",
     r"\bworkforce gap|\bcapacity gap|\bunder(-| )?staff|\bshort(-| )?staff|"
     r"\bwhere are we (seeing|facing) (gaps|shortfalls)|\bgaps\b", None, "business_unit", "bar", 2.6),

    ("attrition_trend", "Attrition over time",
     r"\battrition\b.{0,40}\b(over time|trend|trended|trending|changed|change|"
     r"last \w+ (months?|years?)|past \w+ (months?|years?))\b"
     r"|\bhow has (attrition|turnover) (changed|trended|moved)\b",
     "attrition_trend", "business_unit", "line", 2.8),
    ("retention_risk", "Retention risk",
     r"\bretention risk\b|\bflight risk\b|\bat risk of leaving\b|\brisk of attrition\b|"
     r"\b(increasing|rising|growing) (retention )?risk\b|\bwhich groups are at risk\b",
     "retention_risk_index", "business_unit", "bar", 2.9),
    ("exit_reasons", "Exit reasons",
     r"\bwhy (are|do) (people|employees|they) (leav|quit|resign)|\bexit reason|"
     r"\breasons for leaving\b|\bwhat.{0,20}driving (attrition|exits)\b",
     "exit_reason_mix", "All", "bar", 2.7),
    ("internal_mobility", "Internal mobility",
     r"\binternal mobility\b|\binternal move|\bpromotion rate\b|\bmoving internally\b",
     "internal_mobility_rate", "business_unit", "bar", 2.7),
    ("attrition", "Attrition levels",
     r"\battrition\b|\bturnover\b|\bleavers?\b|\bexits?\b|\bquit(ting)?\b|\bresignation",
     "voluntary_attrition_rate", "business_unit", "bar", 2.0),

    ("funnel", "Recruiting funnel",
     r"\bfunnel\b|\bdrop(ping)? (out|off)\b|\bbottleneck\b|\bstage(s)? .{0,20}(lose|losing)\b|"
     r"\bconversion\b.{0,20}\bstage\b|\bwhere are (we losing|candidates dropping)\b",
     "funnel_conversion", "All", "funnel", 2.9),
    ("source_effectiveness", "Source effectiveness",
     r"\bsource(s)?\b.{0,30}\b(convert|conversion|effective|strongest|best|quality)\b|"
     r"\bwhich (channel|source)\b|\bbest source of hire\b|\bsource of hire\b",
     "source_effectiveness", "All", "bar", 2.9),
    ("time_to_fill", "Time to fill",
     r"\btime(-| )to(-| )fill\b|\bhow long .{0,25}(to )?(fill|hire)\b|\blongest to fill\b|"
     r"\bdays to fill\b|\bfill(ing)? (roles|positions|reqs)\b",
     "time_to_fill", "business_unit", "bar", 2.9),
    ("requisitions", "Open requisitions",
     r"\bopen (req|requisition|role|position|vacanc)|\brequisition (ageing|aging|age)\b|"
     r"\bhow many (roles|reqs|vacancies)\b|\bvacanc(y|ies)\b|\baging req",
     "open_requisitions", "business_unit", "bar", 2.7),
    ("offer_acceptance", "Offer acceptance",
     r"\boffer acceptance\b|\bdeclin(e|ing) offers\b|\baccept(ance)? rate\b",
     "offer_acceptance_rate", "business_unit", "bar", 2.9),
    ("hires", "Hiring volume",
     r"\bhow many (people )?(have we |did we )?(hire|hired)\b|\bhiring volume\b|\bnew hires\b",
     "hires", "business_unit", "bar", 2.4),

    ("theme_engagement", "Themes linked to engagement",
     r"\btheme(s)?\b.{0,40}\b(lower|low|poor|worse) engagement\b|"
     r"\bwhat.{0,25}(drives|drive|associated with).{0,20}engagement\b|"
     r"\bengagement\b.{0,25}\btheme",
     "theme_engagement_link", "All", "diverging", 3.0),
    ("themes", "Feedback themes",
     r"\btheme(s)?\b|\bopen(-| )text\b|\bcomments?\b|\bverbatim|\bwhat are (people|employees) saying\b|"
     r"\bfeedback\b", "feedback_themes", "All", "bar", 2.6),
    ("engagement_trend", "Engagement over time",
     r"\b(engagement|sentiment|morale)\b.{0,40}\b(trend|over time|changed|change|this quarter|"
     r"declin|drop|fall|improv)\b|\bwhat changed in (employee )?(sentiment|engagement)\b",
     "engagement_trend", "business_unit", "line", 2.9),
    ("enps", "eNPS", r"\benps\b|\bnet promoter\b|\brecommend .{0,20}place to work\b",
     "enps", "business_unit", "bar", 3.0),
    ("engagement", "Engagement levels",
     r"\bengagement\b|\bsentiment\b|\bmorale\b|\bsurvey\b|\bhow do (people|employees) feel\b",
     "engagement_score", "business_unit", "bar", 2.0),

    ("growth", "Growth",
     r"\bgrow(ing|th)\b|\bfastest(-| )growing\b|\bexpand(ing)?\b|\bshrink(ing)?\b|\bdeclin(e|ing) headcount\b",
     "headcount_growth_rate", "business_unit", "bar", 2.8),
    ("headcount_trend", "Headcount over time",
     r"\bheadcount\b.{0,30}\b(trend|trended|trending|over time|history|"
     r"last \w+ (months?|years?)|past \w+ (months?|years?))\b",
     "headcount_trend", "business_unit", "line", 2.8),
    ("headcount", "Headcount",
     r"\bheadcount\b|\bhow many (people|employees|staff)\b|\bworkforce size\b|\bfte\b",
     "headcount", "business_unit", "bar", 2.0),
]


# --------------------------------------------------------------------------
def detect_dimension(q: str, metric_id: str | None, default: str) -> str:
    for pattern, dim in DIMENSION_CUES:
        if re.search(pattern, q, re.IGNORECASE):
            if metric_id is None:
                return dim
            spec = sl.catalog().get(metric_id)
            if spec and dim in spec.dimensions:
                return dim
    return default


def detect_filters(q: str) -> dict:
    hits = [bu for bu in BUSINESS_UNITS if re.search(rf"\b{re.escape(bu)}\b", q, re.IGNORECASE)]
    return {"business_unit": hits if len(hits) > 1 else hits[0]} if hits else {}


STOPWORDS = {"rate", "index", "score", "the", "of", "and", "trend", "12m", "rolling",
             "yoy", "to", "link"}


def detect_metric_reference(q: str) -> str | None:
    """Find which defined metric a question is talking *about*.

    Scores on how much of the metric's own label the question contains, which
    beats free-text search: "how is voluntary attrition defined" should resolve
    to Voluntary Attrition Rate, not to every metric whose definition happens
    to mention attrition.
    """
    ql = " " + re.sub(r"[^a-z0-9 ]", " ", q.lower()) + " "
    best, best_score = None, 0.0
    for spec in sl.catalog().values():
        label_tokens = [t for t in re.sub(r"[^a-z0-9 ]", " ", spec.label.lower()).split()
                        if t and t not in STOPWORDS]
        id_tokens = [t for t in spec.id.split("_") if t not in STOPWORDS]
        tokens = label_tokens or id_tokens
        if not tokens:
            continue
        hit = sum(1 for t in set(tokens) if f" {t} " in ql)
        if not hit:
            continue
        score = hit / len(set(tokens)) + 0.35 * hit
        if spec.label.lower() in ql or spec.id.replace("_", " ") in ql:
            score += 2.0
        if score > best_score:
            best, best_score = spec.id, score
    if best:
        return best
    hits = sl.search(q)
    return hits[0].id if hits else None


def classify(question: str) -> Intent:
    q = question.lower()
    best: Intent | None = None
    for key, label, pattern, metric, dim, chart, weight in RULES:
        m = re.search(pattern, q, re.IGNORECASE)
        if not m:
            continue
        score = weight + 0.15 * len(m.group(0).split())
        if best is None or score > best.confidence:
            best = Intent(key=key, label=label, metric=metric, dimension=dim,
                          chart=chart, confidence=round(score, 2))
    if best is None:
        metric = detect_metric_reference(question)
        best = Intent("unmatched", "Unrecognised - best-effort metric match",
                      metric, "business_unit", "bar", 0.4)
    if best.metric:
        best.dimension = detect_dimension(q, best.metric, best.dimension)
    best.filters = detect_filters(question)
    if best.key == "scenario":
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:percentage points?|pts?|%|percent)", q)
        best.extras["attrition_change_pts"] = float(m.group(1)) if m else 3.0
        m2 = re.search(r"(\d+)\s*(month|year)", q)
        if m2:
            best.extras["months"] = int(m2.group(1)) * (12 if m2.group(2) == "year" else 1)
    return best


# --------------------------------------------------------------------------
def _bullets(findings) -> str:
    return "\n".join(f"- **{f.headline}.** {f.detail}" for f in findings)


def _table(result, limit: int = 8) -> str:
    df = result.data
    if df.empty:
        return ""
    if "period" in df.columns:
        latest = sorted(df["period"].unique())[-1]
        df = df[df["period"] == latest].copy()
        df["period"] = str(latest)[:10]
    df = df.head(limit)
    cols = [c for c in df.columns if c not in ("stage_index",)][:5]
    head = "| " + " | ".join(c.replace("_", " ").title() for c in cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    rows = []
    for r in df[cols].itertuples(index=False):
        cells = []
        for c, v in zip(cols, r):
            if c == "value":
                cells.append(result.metric.fmt(v))
            elif isinstance(v, float):
                cells.append("n/a" if pd.isna(v) else f"{v:,.1f}".rstrip("0").rstrip("."))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep, *rows])


ACTIONS = {
    "attrition": "Pull the exit-reason mix and manager-level cut for the worst unit, then "
                 "test whether the exits concentrate in a specific tenure band before "
                 "designing an intervention.",
    "attrition_trend": "Confirm whether the movement is a step change or a drift. A step "
                       "usually points to a single event - a reorganisation, a competitor "
                       "hiring wave, a leadership change - and is easier to act on.",
    "retention_risk": "Take the top-ranked group to its leadership team with the three "
                      "component inputs, not the composite score. The components are what "
                      "they can actually change.",
    "exit_reasons": "Compare the top stated reason against the engagement themes for the "
                    "same population. Where they agree, you have a real cause; where they "
                    "diverge, the exit interview is probably being polite.",
    "funnel": "Take the tightest step to the hiring managers involved. Onsite-to-offer "
              "conversion is usually a calibration problem rather than a supply problem.",
    "source_effectiveness": "Rebalance recruiter time towards the highest-converting sources "
                            "and set a volume cap on the sources that consume screening "
                            "capacity without producing hires.",
    "time_to_fill": "Read this alongside requisition ageing. Time to fill only measures the "
                    "roles you managed to fill, so a slow unit can look fast if its hardest "
                    "roles are still open.",
    "requisitions": "Triage anything past 90 days: re-scope, re-band, or close it. Ageing "
                    "requisitions distort capacity planning because the business still "
                    "assumes the person is coming.",
    "offer_acceptance": "Where acceptance is below benchmark, check offer-to-decision elapsed "
                        "time and the compensation band before assuming a pay problem.",
    "themes": "Theme volume tells you what people talk about; theme sentiment tells you what "
              "hurts. Act on the intersection of high volume and high negative share.",
    "theme_engagement": "The themes furthest below the average are where engagement is being "
                        "lost. Prioritise those over the loudest theme.",
    "engagement_trend": "Check whether the movement is broad or concentrated in one unit. "
                        "Enterprise averages routinely hide a single unit in trouble.",
    "engagement": "Pair the score with the workload and manager sub-scores; the composite on "
                  "its own rarely tells you what to change.",
    "growth": "Compare growth against time to fill and attrition for the same unit. Fast growth "
              "on a slow hiring machine is a plan that will not land.",
    "headcount": "Use this as the denominator sanity check for every rate metric on this page.",
    "headcount_trend": "Look at the slope, not the level. A flat line hiding heavy churn is a "
                       "different problem from a genuinely stable population.",
    "internal_mobility": "Low internal mobility alongside high attrition usually means people "
                         "are finding their next role outside rather than inside.",
    "hires": "Compare hires against exits for the same unit to see whether recruiting is "
             "growing the team or running to stand still.",
    "workforce_gaps": "Start with the highest gap score. That is where demand, attrition and "
                      "slow hiring are compounding rather than occurring separately.",
}


# --------------------------------------------------------------------------
def answer(question: str, ctx: AccessContext, belt: ToolBelt | None = None) -> dict:
    """Execute the routed plan and produce a grounded, templated narrative."""
    belt = belt or ToolBelt(ctx)
    intent = classify(question)
    parts: list[str] = []

    # ---------------------------------------------------------- definitions
    if intent.key == "metric_definition":
        mid = detect_metric_reference(question)
        if not mid:
            return {"intent": intent, "belt": belt, "markdown":
                    "I could not tell which metric you mean. The dictionary covers: "
                    + ", ".join(s.label for s in sl.catalog().values()) + "."}
        d = belt.dispatch("explain_metric", {"metric": mid})
        parts.append(f"**{d['label']}** — {d['definition']}")
        parts.append(f"\n**Formula.** `{d['formula']}`")
        parts.append(f"\n**Source.** {d['source_system']} via "
                     f"{', '.join(d['source_objects'])}, owned by {d['owner']}, "
                     f"refreshed {d['refresh'].lower()}.")
        if d["benchmark"] is not None:
            parts.append(f"\n**Benchmark.** {d['benchmark']}")
        if d["caveats"]:
            parts.append("\n**Read it with these caveats.**\n"
                         + "\n".join(f"- {c}" for c in d["caveats"]))
        return {"intent": intent, "belt": belt, "markdown": "\n".join(parts)}

    # -------------------------------------------------------- data quality
    if intent.key == "data_quality":
        d = belt.dispatch("check_data_quality", {})
        parts.append(f"**{d['banner']}**\n")
        parts.append(f"{d['checks_passed']} of {d['checks_run']} checks are fully clean. "
                     f"{d['employees_excluded_from_all_metrics']} employee records are "
                     f"quarantined from every metric, leaving "
                     f"{d['employee_records_in_scope']:,} in scope.\n")
        if d["failing_checks"]:
            parts.append("| Check | Rule | Severity | Failed | Treatment |")
            parts.append("|---|---|---|---|---|")
            for c in d["failing_checks"]:
                parts.append(f"| {c['check_id']} | {c['rule']} | {c['severity']} | "
                             f"{c['records_failed']} | {c['treatment']} |")
        return {"intent": intent, "belt": belt, "markdown": "\n".join(parts)}

    # ------------------------------------------------------------ scenario
    if intent.key == "scenario":
        args = {"attrition_change_pts": intent.extras.get("attrition_change_pts", 3.0),
                "months": intent.extras.get("months", 12)}
        if intent.filters.get("business_unit") and isinstance(intent.filters["business_unit"], str):
            args["business_unit"] = intent.filters["business_unit"]
        d = belt.dispatch("workforce_scenario", args)
        if "error" in d:
            return {"intent": intent, "belt": belt, "markdown": d["error"]}
        parts.append(
            f"If voluntary attrition rises {args['attrition_change_pts']:.1f} points from "
            f"{d['current_voluntary_attrition_pct']}% to {d['scenario_attrition_pct']}% and "
            f"hiring continues at today's run rate, headcount for {d['scope']} lands at "
            f"**{d['scenario_end_headcount']:,}** after {d['months']} months against a "
            f"baseline of **{d['baseline_end_headcount']:,}** — a gap of "
            f"**{abs(d['headcount_gap']):,} people**.\n")
        parts.append(f"That is roughly {d['additional_exits']:,} additional exits to absorb. "
                     f"Holding the baseline would need about "
                     f"{d['additional_hires_needed_to_hold_flat']:,} extra hires over the "
                     f"same period, on top of the current plan.\n")
        parts.append("**Assumptions.**\n" + "\n".join(f"- {a}" for a in d["assumptions"]))
        rid = d.get("result_id")
        if rid and rid in belt.frames:
            belt.figures.append((
                f"Projected headcount: baseline vs +{args['attrition_change_pts']:.1f}pt attrition",
                charts.projection(belt.frames[rid],
                                  f"Projected headcount over {d['months']} months"),
            ))
        return {"intent": intent, "belt": belt, "markdown": "\n".join(parts)}

    # -------------------------------------------------------- focus / gaps
    if intent.key in ("focus_areas", "workforce_gaps"):
        gaps = belt.dispatch("workforce_gaps", {})
        rows = sorted(gaps["rows"], key=lambda r: -r["gap_score"])[:3]
        att = belt.dispatch("query_retention_data",
                            {"metric": "voluntary_attrition_rate", "dimension": "business_unit"})
        eng = belt.dispatch("query_employee_experience",
                            {"metric": "engagement_score", "dimension": "business_unit"})
        eng_map = {r["dimension"]: r for r in eng.get("rows", [])}
        belt.dispatch("create_visualization", {"result_id": att["result_id"],
                                               "chart_type": "bar"})
        parts.append(f"**{len(rows)} areas need attention**, ranked by a blended gap score "
                     f"across vacancy rate, voluntary attrition and time to fill.\n")
        for i, r in enumerate(rows, 1):
            e = eng_map.get(r["dimension"], {})
            bits = [f"voluntary attrition **{r['attrition_pct']:.1f}%**",
                    f"vacancy rate **{r['vacancy_rate_pct']:.1f}%** "
                    f"({int(r['open_reqs'])} open, {int(r['critical_requisitions'])} critical)"]
            if r.get("time_to_fill_days") == r.get("time_to_fill_days"):
                bits.append(f"time to fill **{r['time_to_fill_days']:.0f} days**")
            if e.get("value"):
                bits.append(f"engagement **{e['value']:.2f}**")
            parts.append(f"**{i}. {r['dimension']}** — " + "; ".join(bits) + ".")
            if r["net_flow"] < 0:
                parts.append(f"   Hiring is not keeping pace: {int(r['hires_12m'])} hires "
                             f"against {int(r['voluntary_exits'])} voluntary exits over "
                             f"twelve months.")
            parts.append("")
        parts.append("**What to do next.** " + ACTIONS["workforce_gaps"])
        return {"intent": intent, "belt": belt, "markdown": "\n".join(parts)}

    # ------------------------------------------------------ standard metric
    if not intent.metric:
        return {"intent": intent, "belt": belt, "markdown":
                "I could not map that to a defined metric. Try one of: attrition by "
                "business unit, recruiting funnel drop-off, time to fill by job family, "
                "engagement themes, or a headcount scenario."}

    spec = sl.get(intent.metric)
    tool = {"Workforce": "query_workforce_data", "Retention": "query_retention_data",
            "Recruiting": "query_recruiting_data",
            "Employee Experience": "query_employee_experience"}[spec.domain]
    args = {"metric": intent.metric, "dimension": intent.dimension}
    if intent.filters:
        args["filters"] = intent.filters
    res = belt.dispatch(tool, args)
    if "access_denied" in res:
        return {"intent": intent, "belt": belt, "markdown": res["access_denied"]}
    if "error" in res:
        return {"intent": intent, "belt": belt, "markdown":
                f"That metric could not be run as asked: {res['error']}"}

    result = belt.results[res["result_id"]]
    an = belt.dispatch("analyze_result", {"result_id": res["result_id"]})
    belt.dispatch("create_visualization",
                  {"result_id": res["result_id"], "chart_type": intent.chart})

    findings = analysis.describe_distribution(result) + analysis.biggest_movers(result)
    parts.append(f"**{spec.label}** — {spec.definition.split('.')[0]}.\n")
    if intent.key == "funnel":
        b = analysis.funnel_bottleneck(result)
        parts.append(f"- **{b.headline}.** {b.detail}")
    parts.append(_bullets(findings[:3]))
    table = _table(result)
    if table:
        parts.append("\n" + table)
    parts.append("\n**What to do next.** " + ACTIONS.get(intent.key, ACTIONS.get("headcount", "")))
    if spec.caveats:
        parts.append(f"\n*Caveat: {spec.caveats[0]}*")
    return {"intent": intent, "belt": belt, "markdown": "\n".join(parts), "analysis": an}
