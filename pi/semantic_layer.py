"""
The semantic layer: turns a metric id + dimension + filters into governed SQL,
runs it, and hands back both the numbers and the provenance behind them.

Design rule for the whole product: no component may compute a People metric by
any route other than this module. The dashboards call it, the daily brief calls
it, and the AI agent calls it through a tool. That is what makes every number on
screen reproducible and auditable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yaml

from pi import config, warehouse
from pi.governance import AccessContext, apply_row_policy, suppress_small_groups

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# --------------------------------------------------------------------------
@dataclass
class MetricSpec:
    id: str
    label: str
    domain: str
    definition: str
    formula: str
    source_system: str
    source_objects: list[str]
    grain: str
    unit: str
    format: str
    direction: str
    owner: str
    refresh: str
    caveats: list[str]
    dimensions: list[str]
    sql: str | None = None
    computed_by: str | None = None
    benchmark: float | None = None

    def fmt(self, value: Any) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        try:
            return self.format.format(value)
        except (ValueError, TypeError):
            return str(value)


@dataclass
class MetricResult:
    metric: MetricSpec
    data: pd.DataFrame
    sql: str
    dimension: str
    filters: dict[str, Any]
    row_policy: str
    suppressed_groups: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return self.data.empty

    def headline(self) -> str:
        if self.empty:
            return "No rows returned for this metric under the current filters."
        top = self.data.iloc[0]
        return f"{top.get('dimension', 'All')}: {self.metric.fmt(top.get('value'))}"

    def to_records(self, limit: int = 40) -> list[dict]:
        return self.data.head(limit).to_dict(orient="records")

    def provenance(self) -> dict:
        return {
            "metric": self.metric.label,
            "definition": " ".join(self.metric.definition.split()),
            "formula": self.metric.formula,
            "source_system": self.metric.source_system,
            "source_objects": self.metric.source_objects,
            "owner": self.metric.owner,
            "refresh": self.metric.refresh,
            "grain": self.metric.grain,
            "dimension": self.dimension,
            "filters": self.filters or "none",
            "row_policy": self.row_policy,
            "sql": self.sql,
            "caveats": self.metric.caveats,
            "suppressed_groups": self.suppressed_groups,
            "rows_returned": int(len(self.data)),
        }


# --------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _raw() -> dict:
    with open(config.METRICS_FILE) as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def catalog() -> dict[str, MetricSpec]:
    raw = _raw()
    out: dict[str, MetricSpec] = {}
    for mid, m in raw["metrics"].items():
        out[mid] = MetricSpec(
            id=mid,
            label=m["label"],
            domain=m["domain"],
            definition=" ".join(m["definition"].split()),
            formula=m["formula"],
            source_system=m["source_system"],
            source_objects=m["source_objects"],
            grain=m["grain"],
            unit=m["unit"],
            format=m["format"],
            direction=m["direction"],
            owner=m["owner"],
            refresh=m["refresh"],
            caveats=m.get("caveats", []),
            dimensions=m.get("dimensions", []),
            sql=m.get("sql"),
            computed_by=m.get("computed_by"),
            benchmark=m.get("benchmark"),
        )
    return out


def dimensions() -> dict[str, dict]:
    return _raw()["dimensions"]


def get(metric_id: str) -> MetricSpec:
    try:
        return catalog()[metric_id]
    except KeyError:
        raise KeyError(
            f"'{metric_id}' is not a defined metric. Defined metrics: "
            + ", ".join(sorted(catalog()))
        )


def by_domain() -> dict[str, list[MetricSpec]]:
    out: dict[str, list[MetricSpec]] = {}
    for spec in catalog().values():
        out.setdefault(spec.domain, []).append(spec)
    return out


# --------------------------------------------------------------------------
def _literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    text = str(value)
    if len(text) > 120:
        raise ValueError("Filter value too long.")
    if any(ch in text for ch in (";", "--", "/*", "*/", "\n", "\r")):
        raise ValueError(f"Illegal characters in filter value: {value!r}")
    return "'" + text.replace("'", "''") + "'"


def build_filter_clause(filters: dict[str, Any] | None,
                        allowed: list[str]) -> str:
    """Compile a validated filter dict into an AND-prefixed SQL fragment.

    Column names are checked against the metric's own declared dimension list,
    so a filter can never introduce a column the metric was not designed for,
    and values are always emitted as quoted literals.
    """
    if not filters:
        return ""
    parts = []
    for col, val in filters.items():
        if col not in allowed:
            raise ValueError(
                f"'{col}' is not a filterable dimension for this metric. "
                f"Allowed: {', '.join(allowed)}"
            )
        if not IDENT.match(col):
            raise ValueError(f"Invalid column name: {col!r}")
        if isinstance(val, (list, tuple, set)):
            vals = ", ".join(_literal(v) for v in val)
            parts.append(f"AND {col} IN ({vals})")
        else:
            parts.append(f"AND {col} = {_literal(val)}")
    return " " + " ".join(parts)


def resolve(metric_id: str,
            dimension: str = "business_unit",
            filters: dict[str, Any] | None = None,
            ctx: AccessContext | None = None) -> tuple[str, dict, str]:
    """Return (sql, effective_filters, row_policy_description)."""
    spec = get(metric_id)
    ctx = ctx or AccessContext.default()

    if dimension not in ("All", "none", None) and dimension not in spec.dimensions:
        raise ValueError(
            f"'{spec.label}' cannot be broken down by '{dimension}'. "
            f"Available: {', '.join(spec.dimensions)}"
        )
    dim_sql = "'All'" if dimension in ("All", "none", None) else dimension

    effective, policy_note = apply_row_policy(dict(filters or {}), ctx, spec)
    clause = build_filter_clause(effective, spec.dimensions)

    if spec.sql is None:
        return "", effective, policy_note
    sql = spec.sql.replace("{dimension}", dim_sql).replace("{filters}", clause)
    return sql, effective, policy_note


def run(metric_id: str,
        dimension: str = "business_unit",
        filters: dict[str, Any] | None = None,
        ctx: AccessContext | None = None) -> MetricResult:
    spec = get(metric_id)
    ctx = ctx or AccessContext.default()

    if spec.computed_by:
        return _run_computed(spec, dimension, filters, ctx)

    sql, effective, policy = resolve(metric_id, dimension, filters, ctx)
    df = warehouse.query(sql)
    df, suppressed = suppress_small_groups(df, spec, ctx)
    notes = []
    if suppressed:
        notes.append(
            f"{suppressed} group(s) suppressed for falling below the minimum "
            f"aggregation size of {config.MIN_AGGREGATION_GROUP}."
        )
    return MetricResult(spec, df, sql, dimension, effective, policy, suppressed, notes)


# --------------------------------------------------------------------------
def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not sd or np.isnan(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def _run_computed(spec: MetricSpec, dimension: str,
                  filters: dict | None, ctx: AccessContext) -> MetricResult:
    """Composite metrics that are assembled from other governed metrics.

    They still never touch raw SQL of their own: every input is itself a
    dictionary-defined metric, so the lineage stays intact.
    """
    if spec.id != "retention_risk_index":
        raise NotImplementedError(spec.id)

    dim = dimension if dimension in spec.dimensions else "business_unit"
    att = run("voluntary_attrition_rate", dim, filters, ctx).data
    trend_dim = dim if dim in get("attrition_trend").dimensions else "business_unit"
    trend = (run("attrition_trend", trend_dim, filters, ctx).data
             if trend_dim == dim else pd.DataFrame())
    eng_dim = dim if dim in get("engagement_score").dimensions else "business_unit"
    eng = run("engagement_score", eng_dim, filters, ctx).data

    if att.empty:
        return MetricResult(spec, pd.DataFrame(), "-- composite metric", dim,
                            filters or {}, "n/a", 0,
                            ["No population met the minimum group size."])

    # year-on-year change in the rolling attrition rate
    delta = pd.DataFrame(columns=["dimension", "attrition_delta"])
    if not trend.empty:
        t = trend.copy()
        t["period"] = pd.to_datetime(t["period"])
        last = t["period"].max()
        prior = last - pd.DateOffset(months=12)
        cur = t[t["period"] == last][["dimension", "value"]].rename(columns={"value": "now"})
        old = (t[t["period"] <= prior].sort_values("period")
                 .groupby("dimension").tail(1)[["dimension", "value"]]
                 .rename(columns={"value": "prior"}))
        delta = cur.merge(old, on="dimension", how="left")
        delta["attrition_delta"] = delta["now"] - delta["prior"]
        delta = delta[["dimension", "attrition_delta"]]

    df = att[["dimension", "value", "voluntary_exits", "avg_headcount"]].rename(
        columns={"value": "attrition_pct"})
    df = df.merge(delta, on="dimension", how="left")
    if not eng.empty and eng_dim == dim:
        df = df.merge(eng[["dimension", "value", "workload_score", "responses"]].rename(
            columns={"value": "engagement", "responses": "survey_responses"}),
            on="dimension", how="left")
    for col in ("engagement", "workload_score", "attrition_delta"):
        if col not in df:
            df[col] = np.nan
    df["attrition_delta"] = df["attrition_delta"].fillna(0.0)
    df["engagement"] = df["engagement"].fillna(df["engagement"].mean())
    df["workload_score"] = df["workload_score"].fillna(df["workload_score"].mean())

    raw = (0.40 * _zscore(df["attrition_pct"])
           + 0.25 * _zscore(df["attrition_delta"])
           + 0.20 * _zscore(-df["engagement"].fillna(0))
           + 0.15 * _zscore(-df["workload_score"].fillna(0)))
    lo, hi = raw.min(), raw.max()
    df["value"] = 50.0 if hi == lo else np.round(100 * (raw - lo) / (hi - lo), 0)
    df["risk_band"] = pd.cut(df["value"], [-0.1, 40, 70, 100],
                             labels=["Monitor", "Elevated", "Priority"])
    df = df.sort_values("value", ascending=False).reset_index(drop=True)

    explain = (
        "-- Composite metric assembled in the semantic layer from three governed metrics.\n"
        "-- inputs : voluntary_attrition_rate, attrition_trend, engagement_score\n"
        "-- weights: attrition 40%, YoY change 25%, engagement (inverted) 20%,\n"
        "--          workload (inverted) 15%; standardised then rescaled 0-100.\n"
        "-- Each input is itself defined in semantic/metrics.yaml and executed as SQL."
    )
    return MetricResult(spec, df, explain, dim, filters or {}, "inherited from inputs", 0,
                        ["Composite prioritisation index, not a probability of leaving."])


# --------------------------------------------------------------------------
def explain(metric_id: str) -> dict:
    """Full dictionary entry for a metric - powers explain_metric()."""
    s = get(metric_id)
    return {
        "metric_id": s.id,
        "label": s.label,
        "domain": s.domain,
        "definition": s.definition,
        "formula": s.formula,
        "source_system": s.source_system,
        "source_objects": s.source_objects,
        "grain": s.grain,
        "unit": s.unit,
        "direction": s.direction,
        "benchmark": s.benchmark,
        "owner": s.owner,
        "refresh": s.refresh,
        "available_dimensions": s.dimensions,
        "caveats": s.caveats,
        "sql_definition": (s.sql or "").strip() or "Composite - see computed_by",
    }


def search(text: str) -> list[MetricSpec]:
    t = text.lower()
    hits = []
    for s in catalog().values():
        blob = f"{s.id} {s.label} {s.domain} {s.definition}".lower()
        score = sum(w in blob for w in t.split() if len(w) > 2)
        if score:
            hits.append((score, s))
    return [s for _, s in sorted(hits, key=lambda x: -x[0])]
