"""
The agent's toolbelt.

Each tool is a narrow, governed capability. The model chooses which to call and
with what arguments; it never receives a database handle, never sees a row of
employee data, and cannot compute a metric itself. Tool results are returned as
compact JSON so the model reasons over numbers it did not invent.

Every call is logged, and the ToolBelt keeps the resulting DataFrames so the
application can render exactly the data the model was shown.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from pi import analysis, charts, semantic_layer as sl
from pi.agent import sql_guard
from pi.governance import AccessContext, audit, check_dimension_allowed
from pi.quality import checks

DOMAIN_TOOLS = {
    "query_workforce_data": "Workforce",
    "query_retention_data": "Retention",
    "query_recruiting_data": "Recruiting",
    "query_employee_experience": "Employee Experience",
}


def _metric_ids(domain: str) -> list[str]:
    return [m.id for m in sl.catalog().values() if m.domain == domain]


# --------------------------------------------------------------------------
@dataclass
class ToolCall:
    name: str
    arguments: dict
    ok: bool
    summary: str
    result_id: str | None = None
    error: str | None = None


@dataclass
class ToolBelt:
    """Stateful execution surface for one conversation turn."""
    ctx: AccessContext
    results: dict[str, Any] = field(default_factory=dict)      # result_id -> MetricResult
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    figures: list[tuple[str, Any]] = field(default_factory=list)
    calls: list[ToolCall] = field(default_factory=list)
    _seq: int = 0

    # ----------------------------------------------------------- plumbing
    def _next_id(self) -> str:
        self._seq += 1
        return f"R{self._seq}"

    def _store(self, result) -> str:
        rid = self._next_id()
        self.results[rid] = result
        self.frames[rid] = result.data
        return rid

    def dispatch(self, name: str, args: dict) -> dict:
        fn = HANDLERS.get(name)
        if fn is None:
            return {"error": f"Unknown tool '{name}'."}
        try:
            out = fn(self, **args)
            ok, err = True, None
        except PermissionError as exc:
            out, ok, err = {"access_denied": str(exc)}, False, str(exc)
        except (KeyError, ValueError, NotImplementedError) as exc:
            out, ok, err = {"error": str(exc)}, False, str(exc)
        except Exception as exc:                                    # noqa: BLE001
            out, ok, err = {"error": f"{type(exc).__name__}: {exc}"}, False, str(exc)

        self.calls.append(ToolCall(
            name=name, arguments=args, ok=ok,
            summary=_summarise(out), result_id=out.get("result_id") if isinstance(out, dict) else None,
            error=err,
        ))
        audit("tool_call", self.ctx, tool=name, arguments=args, ok=ok, error=err)
        return out

    # ------------------------------------------------------- introspection
    def sql_log(self) -> list[dict]:
        return [{"result_id": rid, "metric": r.metric.label, "sql": r.sql}
                for rid, r in self.results.items()]

    def provenance(self) -> list[dict]:
        return [r.provenance() for r in self.results.values()]


def _summarise(out: Any, limit: int = 400) -> str:
    text = json.dumps(out, default=str) if not isinstance(out, str) else out
    return text[:limit] + (" ..." if len(text) > limit else "")


# --------------------------------------------------------------------------
# Tool implementations
# --------------------------------------------------------------------------
def _run_metric(belt: ToolBelt, domain: str | None, metric: str,
                dimension: str = "business_unit",
                filters: dict | None = None, top_n: int = 15) -> dict:
    spec = sl.get(metric)
    if domain and spec.domain != domain:
        raise ValueError(
            f"'{metric}' belongs to the {spec.domain} domain. Use the "
            f"{[k for k, v in DOMAIN_TOOLS.items() if v == spec.domain][0]} tool, "
            f"or calculate_metric for any domain."
        )
    if spec.domain not in belt.ctx.role.allowed_domains:
        raise PermissionError(f"The {spec.domain} domain is not available to your role.")
    check_dimension_allowed(dimension, belt.ctx)

    result = sl.run(metric, dimension, filters, belt.ctx)
    rid = belt._store(result)
    return {
        "result_id": rid,
        "metric": spec.label,
        "metric_id": spec.id,
        "definition": spec.definition,
        "unit": spec.unit,
        "dimension": dimension,
        "filters_applied": result.filters or "none",
        "row_policy": result.row_policy,
        "rows": result.to_records(top_n),
        "row_count": int(len(result.data)),
        "notes": result.notes,
        "sql_executed": result.sql,
    }


def query_workforce_data(belt: ToolBelt, metric: str, dimension: str = "business_unit",
                         filters: dict | None = None, top_n: int = 15) -> dict:
    return _run_metric(belt, "Workforce", metric, dimension, filters, top_n)


def query_retention_data(belt: ToolBelt, metric: str, dimension: str = "business_unit",
                         filters: dict | None = None, top_n: int = 15) -> dict:
    return _run_metric(belt, "Retention", metric, dimension, filters, top_n)


def query_recruiting_data(belt: ToolBelt, metric: str, dimension: str = "business_unit",
                          filters: dict | None = None, top_n: int = 15) -> dict:
    return _run_metric(belt, "Recruiting", metric, dimension, filters, top_n)


def query_employee_experience(belt: ToolBelt, metric: str, dimension: str = "business_unit",
                              filters: dict | None = None, top_n: int = 15) -> dict:
    return _run_metric(belt, "Employee Experience", metric, dimension, filters, top_n)


def calculate_metric(belt: ToolBelt, metric: str, dimension: str = "business_unit",
                     filters: dict | None = None, top_n: int = 15) -> dict:
    return _run_metric(belt, None, metric, dimension, filters, top_n)


def list_metrics(belt: ToolBelt, domain: str | None = None, search: str | None = None) -> dict:
    specs = list(sl.catalog().values())
    if domain:
        specs = [s for s in specs if s.domain.lower() == domain.lower()]
    if search:
        hits = {s.id for s in sl.search(search)}
        specs = [s for s in specs if s.id in hits]
    return {"metrics": [
        {"metric_id": s.id, "label": s.label, "domain": s.domain, "unit": s.unit,
         "dimensions": s.dimensions, "definition": s.definition[:180]}
        for s in specs
    ]}


def explain_metric(belt: ToolBelt, metric: str) -> dict:
    return sl.explain(metric)


def analyze_result(belt: ToolBelt, result_id: str, compare_periods_back: int = 12) -> dict:
    if result_id not in belt.results:
        raise KeyError(f"No such result '{result_id}'. Run a query tool first.")
    result = belt.results[result_id]
    findings = analysis.describe_distribution(result)
    movers = analysis.biggest_movers(result, compare_periods_back)
    bottleneck = None
    if result.metric.id == "funnel_conversion":
        bottleneck = analysis.funnel_bottleneck(result)
    return {
        "result_id": result_id,
        "metric": result.metric.label,
        "findings": [{"headline": f.headline, "detail": f.detail, "severity": f.severity}
                     for f in findings + movers],
        "bottleneck": ({"headline": bottleneck.headline, "detail": bottleneck.detail}
                       if bottleneck else None),
        "caveats": result.metric.caveats,
    }


def create_visualization(belt: ToolBelt, result_id: str, chart_type: str = "bar",
                         title: str | None = None) -> dict:
    if result_id not in belt.results:
        raise KeyError(f"No such result '{result_id}'. Run a query tool first.")
    result = belt.results[result_id]
    if chart_type not in ("bar", "line", "funnel", "diverging"):
        raise ValueError("chart_type must be one of: bar, line, funnel, diverging.")
    if result.data.empty:
        return {"chart": "not created - the result set is empty"}
    fig = charts.build(chart_type, result, title)
    label = title or f"{result.metric.label} by {result.dimension.replace('_', ' ')}"
    belt.figures.append((label, fig))
    return {"chart_created": True, "chart_type": chart_type, "title": label,
            "series_points": int(len(result.data)),
            "note": "The chart is rendered in the interface; describe it, do not restate every value."}


def check_data_quality(belt: ToolBelt, dataset: str | None = None) -> dict:
    score = checks.scorecard()
    detail = score["detail"]
    if dataset:
        detail = detail[detail["dataset"] == dataset]
    return {
        "completeness_pct": score["completeness_pct"],
        "checks_run": score["checks_run"],
        "checks_passed": score["checks_passed"],
        "employees_excluded_from_all_metrics": score["employees_excluded"],
        "employee_records_in_scope": score["employees_in_scope"],
        "banner": checks.banner(),
        "failing_checks": detail[detail["records_failed"] > 0][
            ["check_id", "dataset", "rule", "severity", "records_failed", "treatment"]
        ].to_dict(orient="records"),
    }


def run_validated_sql(belt: ToolBelt, sql: str, purpose: str = "") -> dict:
    """Escape hatch for questions the dictionary does not cover. Analyst role only."""
    if not belt.ctx.role.can_run_adhoc_sql:
        raise PermissionError(
            "Ad-hoc SQL is limited to the People Analytics role. Every other role "
            "is served from the metric dictionary, so that definitions stay standard. "
            "Use list_metrics to find the right defined metric."
        )
    check = sql_guard.validate(sql)
    audit("sql_validation", belt.ctx, ok=check.ok, errors=check.errors,
          purpose=purpose, sql=sql)
    if not check.ok:
        return {"validation": "failed", "errors": check.errors,
                "guidance": "Rewrite the query against the v_ views, aggregate the "
                            "result, and avoid restricted columns."}
    from pi import warehouse
    df = warehouse.query(check.sql)
    rid = belt._next_id()
    belt.frames[rid] = df
    return {
        "validation": "passed",
        "warnings": check.warnings,
        "objects_read": sorted(set(check.tables)),
        "sql_executed": check.sql,
        "result_id": rid,
        "row_count": int(len(df)),
        "rows": df.head(30).to_dict(orient="records"),
    }


def workforce_scenario(belt: ToolBelt, attrition_change_pts: float, months: int = 12,
                       business_unit: str | None = None,
                       hiring_fill_rate: float = 1.0) -> dict:
    out = analysis.headcount_scenario(attrition_change_pts, months, hiring_fill_rate,
                                      business_unit, belt.ctx)
    proj = out.pop("projection", None)
    if proj is not None:
        rid = belt._next_id()
        belt.frames[rid] = proj
        out["result_id"] = rid
        out["projection_sample"] = proj.iloc[[0, len(proj) // 2, -1]].to_dict(orient="records")
    return out


def workforce_gaps(belt: ToolBelt) -> dict:
    df = analysis.workforce_gaps(belt.ctx)
    rid = belt._next_id()
    belt.frames[rid] = df
    return {"result_id": rid, "method": (
        "Gap score blends vacancy rate (40%), voluntary attrition (35%) and "
        "time to fill (25%), each as a percentile rank across units in scope."),
        "rows": df.to_dict(orient="records")}


def generate_executive_summary(belt: ToolBelt, topic: str, audience: str = "HR Leader",
                               max_points: int = 4) -> dict:
    """Assemble a leader-ready narrative from results already retrieved this turn.

    Deliberately mechanical: it can only summarise numbers that were actually
    queried, so an executive summary cannot contain a figure that was never
    computed.
    """
    if not belt.results:
        raise ValueError("Run at least one query tool before requesting a summary.")
    lines, evidence = [], []
    for rid, res in list(belt.results.items())[:max_points]:
        for f in analysis.describe_distribution(res)[:1]:
            lines.append(f"{f.headline}. {f.detail}")
            evidence.append({"result_id": rid, "metric": res.metric.label,
                             "source": res.metric.source_system})
    return {
        "topic": topic,
        "audience": audience,
        "as_of": sl._raw()["as_of_date"],
        "points": lines,
        "evidence": evidence,
        "data_quality": checks.banner(),
        "instruction": ("Rewrite these points as a short narrative for the stated "
                        "audience. Lead with the decision or action, keep every "
                        "figure exactly as given, and add no new numbers."),
    }


HANDLERS: dict[str, Callable[..., dict]] = {
    "list_metrics": list_metrics,
    "explain_metric": explain_metric,
    "query_workforce_data": query_workforce_data,
    "query_retention_data": query_retention_data,
    "query_recruiting_data": query_recruiting_data,
    "query_employee_experience": query_employee_experience,
    "calculate_metric": calculate_metric,
    "analyze_result": analyze_result,
    "create_visualization": create_visualization,
    "check_data_quality": check_data_quality,
    "run_validated_sql": run_validated_sql,
    "workforce_scenario": workforce_scenario,
    "workforce_gaps": workforce_gaps,
    "generate_executive_summary": generate_executive_summary,
}


# --------------------------------------------------------------------------
# Schemas handed to the model
# --------------------------------------------------------------------------
def _metric_enum(domain: str | None = None) -> list[str]:
    return sorted(_metric_ids(domain)) if domain else sorted(sl.catalog())


def _query_schema(domain: str, description: str) -> dict:
    return {
        "name": [k for k, v in DOMAIN_TOOLS.items() if v == domain][0],
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": _metric_enum(domain),
                           "description": "Metric id from the People Metrics Dictionary."},
                "dimension": {"type": "string",
                              "description": "Grouping dimension, or 'All' for a single total. "
                                             "Must be listed for the chosen metric."},
                "filters": {"type": "object",
                            "description": "Optional equality filters, e.g. "
                                           "{\"business_unit\": \"Operations\"} or "
                                           "{\"job_level\": [\"Manager\", \"Senior Manager\"]}. "
                                           "Only declared dimensions are accepted."},
                "top_n": {"type": "integer", "description": "Rows to return (default 15)."},
            },
            "required": ["metric"],
        },
    }


def schemas(ctx: AccessContext) -> list[dict]:
    tools = [
        {
            "name": "list_metrics",
            "description": ("Browse the People Metrics Dictionary. Use this first when you "
                            "are unsure which standard metric answers the question."),
            "input_schema": {"type": "object", "properties": {
                "domain": {"type": "string",
                           "enum": ["Workforce", "Retention", "Recruiting", "Employee Experience"]},
                "search": {"type": "string", "description": "Free-text search over labels and definitions."},
            }},
        },
        {
            "name": "explain_metric",
            "description": ("Return the full dictionary entry for a metric: definition, "
                            "formula, source system, owner, refresh cadence, caveats and the "
                            "exact SQL. Use whenever the user asks what a metric means, or "
                            "when a caveat materially changes how the answer should be read."),
            "input_schema": {"type": "object", "properties": {
                "metric": {"type": "string", "enum": _metric_enum()}}, "required": ["metric"]},
        },
        _query_schema("Workforce",
                      "Headcount, headcount trend and growth from the HRIS. Use for questions "
                      "about size, shape and growth of the workforce."),
        _query_schema("Retention",
                      "Attrition, attrition trend, exit reasons, internal mobility and the "
                      "retention risk index. Use for anything about leavers or retention risk."),
        _query_schema("Recruiting",
                      "Time to fill, open and ageing requisitions, funnel conversion, source "
                      "effectiveness, offer acceptance and hires, from the ATS."),
        _query_schema("Employee Experience",
                      "Engagement score and trend, eNPS, open-text feedback themes and the "
                      "association between themes and engagement."),
        {
            "name": "calculate_metric",
            "description": ("Run any metric in the dictionary regardless of domain. Use when a "
                            "question spans domains, for example comparing attrition with "
                            "engagement."),
            "input_schema": {"type": "object", "properties": {
                "metric": {"type": "string", "enum": _metric_enum()},
                "dimension": {"type": "string"},
                "filters": {"type": "object"},
                "top_n": {"type": "integer"},
            }, "required": ["metric"]},
        },
        {
            "name": "analyze_result",
            "description": ("Compute the analysis over a result you already retrieved: ranking, "
                            "spread, benchmark gaps, concentration and period-on-period movement. "
                            "Call this before writing your answer rather than eyeballing the rows."),
            "input_schema": {"type": "object", "properties": {
                "result_id": {"type": "string"},
                "compare_periods_back": {"type": "integer",
                                         "description": "Periods to look back for trend metrics (default 12)."},
            }, "required": ["result_id"]},
        },
        {
            "name": "create_visualization",
            "description": ("Render a chart from a result. Use 'line' for anything with a period "
                            "column, 'funnel' for funnel_conversion, 'diverging' for signed "
                            "differences, otherwise 'bar'. Create at most two charts per answer."),
            "input_schema": {"type": "object", "properties": {
                "result_id": {"type": "string"},
                "chart_type": {"type": "string", "enum": ["bar", "line", "funnel", "diverging"]},
                "title": {"type": "string"},
            }, "required": ["result_id"]},
        },
        {
            "name": "check_data_quality",
            "description": ("Current data-quality scorecard: completeness, failing checks and how "
                            "many records are excluded from metrics. Call when the user asks how "
                            "reliable the numbers are, or when a result looks anomalous."),
            "input_schema": {"type": "object", "properties": {
                "dataset": {"type": "string",
                            "enum": ["employees", "candidates", "requisitions",
                                     "survey_responses", "internal_moves"]}}},
        },
        {
            "name": "workforce_scenario",
            "description": ("Project headcount forward under a change in the voluntary attrition "
                            "run rate. Use for 'what if attrition rises by N points' questions. "
                            "Returns the projection and the assumptions behind it."),
            "input_schema": {"type": "object", "properties": {
                "attrition_change_pts": {"type": "number",
                                         "description": "Change in annual attrition in percentage points, e.g. 3."},
                "months": {"type": "integer", "description": "Projection horizon, default 12."},
                "business_unit": {"type": "string"},
                "hiring_fill_rate": {"type": "number",
                                     "description": "Fraction of the current hiring run rate that continues, default 1.0."},
            }, "required": ["attrition_change_pts"]},
        },
        {
            "name": "workforce_gaps",
            "description": ("Composite view of where demand outstrips supply: vacancy rate, "
                            "attrition, time to fill, replacement ratio and a blended gap score "
                            "per business unit. Use for 'where are our workforce gaps'."),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "generate_executive_summary",
            "description": ("Turn the results retrieved in this turn into leader-ready points. "
                            "Only call after you have the underlying numbers."),
            "input_schema": {"type": "object", "properties": {
                "topic": {"type": "string"},
                "audience": {"type": "string",
                             "enum": ["HRBP", "HR Leader", "Executive", "Talent Acquisition"]},
                "max_points": {"type": "integer"},
            }, "required": ["topic"]},
        },
    ]
    if ctx.role.can_run_adhoc_sql:
        tools.append({
            "name": "run_validated_sql",
            "description": ("Read-only SQL against the governed v_ views, for questions the "
                            "dictionary genuinely cannot answer. Every statement is validated "
                            "before execution: single SELECT, governed views only, no restricted "
                            "columns, must aggregate, bounded LIMIT. Prefer a defined metric "
                            "whenever one exists."),
            "input_schema": {"type": "object", "properties": {
                "sql": {"type": "string"},
                "purpose": {"type": "string",
                            "description": "Why the dictionary could not answer this."},
            }, "required": ["sql"]},
        })
    return tools
