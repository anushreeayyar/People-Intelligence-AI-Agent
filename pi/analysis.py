"""
Deterministic analysis on top of metric results.

Everything here is arithmetic over a DataFrame that the semantic layer already
produced. It is what turns "here are six numbers" into "here is what is unusual
about these six numbers" - concentration, spread, movement, benchmark gaps -
without any model involvement, so the same question always yields the same
finding.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pi import semantic_layer as sl
from pi.governance import AccessContext


# --------------------------------------------------------------------------
@dataclass
class Finding:
    headline: str
    detail: str
    severity: str          # 'info' | 'watch' | 'attention'
    evidence: dict


def describe_distribution(result) -> list[Finding]:
    """Rank, spread and benchmark comparison for a single-dimension metric."""
    df = result.data
    m = result.metric
    if df.empty or "value" not in df.columns:
        return [Finding("No data", "No rows met the filter and suppression rules.",
                        "info", {})]

    # For a time series, "distribution" means the distribution in the latest
    # period - not across every period-dimension pair, which is meaningless.
    period_note = ""
    if "period" in df.columns:
        latest = sorted(df["period"].unique())[-1]
        df = df[df["period"] == latest]
        period_note = f" as at {str(latest)[:10]}"

    lower_better = m.direction == "lower_is_better"
    d = df.dropna(subset=["value"]).sort_values("value", ascending=lower_better)
    if d.empty:
        return [Finding("No data", "All values were null.", "info", {})]

    best, worst = d.iloc[0], d.iloc[-1]
    median = float(d["value"].median())
    out: list[Finding] = []

    gap = abs(float(worst["value"]) - float(best["value"]))
    out.append(Finding(
        headline=f"{worst['dimension']} sits at {m.fmt(worst['value'])}{period_note}",
        detail=(f"That is the {'highest' if lower_better else 'lowest'} of "
                f"{len(d)} groups, against a median of {m.fmt(median)} and "
                f"{best['dimension']} at {m.fmt(best['value'])} - a spread of "
                f"{m.fmt(gap)}."),
        severity="attention" if lower_better else "watch",
        evidence={"worst": worst.to_dict(), "best": best.to_dict(), "median": median},
    ))

    if m.benchmark is not None:
        off = d[d["value"] > m.benchmark] if lower_better else d[d["value"] < m.benchmark]
        if len(off):
            out.append(Finding(
                headline=f"{len(off)} of {len(d)} groups are outside the "
                         f"{m.fmt(m.benchmark)} benchmark",
                detail=", ".join(f"{r.dimension} {m.fmt(r.value)}" for r in off.itertuples()),
                severity="attention" if len(off) > len(d) / 2 else "watch",
                evidence={"benchmark": m.benchmark, "groups": off["dimension"].tolist()},
            ))

    # concentration: does one group account for a disproportionate share of a count?
    if m.unit == "count" and len(d) > 2:
        total = float(d["value"].sum())
        share = float(worst["value"]) / total if total else 0
        if share > 1.6 / len(d):
            out.append(Finding(
                headline=f"{worst['dimension']} accounts for {share:.0%} of the total",
                detail=f"{m.fmt(worst['value'])} of {m.fmt(total)} across {len(d)} groups.",
                severity="watch",
                evidence={"share": share, "total": total},
            ))
    return out


def trend_delta(result, periods_back: int = 12) -> pd.DataFrame:
    """Latest value vs the value N periods earlier, per dimension."""
    df = result.data
    if df.empty or "period" not in df.columns:
        return pd.DataFrame()
    d = df.copy()
    d["period"] = pd.to_datetime(d["period"], errors="coerce") if not \
        d["period"].astype(str).str.match(r"^\d{4}Q\d$").all() else d["period"]
    order = sorted(d["period"].unique())
    if len(order) < 2:
        return pd.DataFrame()
    last = order[-1]
    prior = order[max(0, len(order) - 1 - periods_back)]
    cur = d[d["period"] == last][["dimension", "value"]].rename(columns={"value": "current"})
    old = d[d["period"] == prior][["dimension", "value"]].rename(columns={"value": "prior"})
    out = cur.merge(old, on="dimension", how="left")
    out["change"] = out["current"] - out["prior"]
    out["pct_change"] = np.where(out["prior"].fillna(0) != 0,
                                 100 * out["change"] / out["prior"], np.nan)
    out["from_period"] = str(prior)[:10]
    out["to_period"] = str(last)[:10]
    return out.sort_values("change", ascending=False).reset_index(drop=True)


def biggest_movers(result, periods_back: int = 12, top: int = 3) -> list[Finding]:
    d = trend_delta(result, periods_back)
    if d.empty:
        return []
    m = result.metric
    lower_better = m.direction == "lower_is_better"
    d = d.dropna(subset=["change"])
    worsening = d.sort_values("change", ascending=not lower_better).head(top)
    out = []
    for r in worsening.itertuples():
        direction = "up" if r.change > 0 else "down"
        bad = (r.change > 0) == lower_better
        out.append(Finding(
            headline=f"{r.dimension} {direction} {m.fmt(abs(r.change))} since {r.from_period}",
            detail=(f"{m.fmt(r.prior)} to {m.fmt(r.current)}"
                    + (f" ({r.pct_change:+.1f}% relative)" if pd.notna(r.pct_change) else "")),
            severity="attention" if bad and abs(r.change) > 0 else "info",
            evidence={"dimension": r.dimension, "prior": r.prior, "current": r.current,
                      "change": r.change, "from": r.from_period, "to": r.to_period},
        ))
    return out


# --------------------------------------------------------------------------
def headcount_scenario(attrition_delta_pts: float,
                       months: int = 12,
                       hiring_fill_rate: float = 1.0,
                       business_unit: str | None = None,
                       ctx: AccessContext | None = None) -> dict:
    """Project headcount forward under a change in the attrition run rate.

    Deliberately simple and fully stated: starting headcount and the current
    attrition and hiring run rates come from the semantic layer, then the
    projection is monthly compounding. The assumptions are returned with the
    result so a reader can disagree with them explicitly.
    """
    ctx = ctx or AccessContext.default()
    filters = {"business_unit": business_unit} if business_unit else None

    hc = sl.run("headcount", "business_unit", filters, ctx).data
    att = sl.run("voluntary_attrition_rate", "business_unit", filters, ctx).data
    hires = sl.run("hires", "business_unit", filters, ctx).data

    start_hc = float(hc["value"].sum())
    if start_hc == 0:
        return {"error": "No headcount in scope."}
    weighted_att = float((att["value"] * att["avg_headcount"]).sum()
                         / max(att["avg_headcount"].sum(), 1)) if not att.empty else 0.0
    annual_hires = float(hires["value"].sum()) if not hires.empty else 0.0

    base_monthly_exit = weighted_att / 100 / 12
    scen_monthly_exit = (weighted_att + attrition_delta_pts) / 100 / 12
    monthly_hires = annual_hires / 12 * hiring_fill_rate

    rows = []
    base, scen = start_hc, start_hc
    for m in range(1, months + 1):
        base = base - base * base_monthly_exit + monthly_hires
        scen = scen - scen * scen_monthly_exit + monthly_hires
        rows.append({"month": m, "baseline": round(base, 1), "scenario": round(scen, 1),
                     "gap": round(scen - base, 1)})
    proj = pd.DataFrame(rows)
    end_gap = float(proj.iloc[-1]["gap"])
    extra_exits = round(start_hc * (attrition_delta_pts / 100) * (months / 12))

    return {
        "starting_headcount": int(start_hc),
        "current_voluntary_attrition_pct": round(weighted_att, 1),
        "scenario_attrition_pct": round(weighted_att + attrition_delta_pts, 1),
        "months": months,
        "baseline_end_headcount": int(round(proj.iloc[-1]["baseline"])),
        "scenario_end_headcount": int(round(proj.iloc[-1]["scenario"])),
        "headcount_gap": int(round(end_gap)),
        "additional_exits": int(extra_exits),
        "additional_hires_needed_to_hold_flat": int(round(abs(end_gap))),
        "assumptions": [
            f"Starting headcount {int(start_hc):,} as at the reporting date.",
            f"Current weighted voluntary attrition {weighted_att:.1f}% annualised.",
            f"Hiring continues at the trailing-12-month run rate of "
            f"{annual_hires:,.0f} hires per year, scaled by a fill rate of {hiring_fill_rate:.0%}.",
            "Involuntary exits, internal transfers and seasonality are held constant.",
            "Monthly compounding; no attempt to model attrition's dependence on tenure mix.",
        ],
        "projection": proj,
        "scope": business_unit or "all units in scope",
    }


# --------------------------------------------------------------------------
def workforce_gaps(ctx: AccessContext | None = None) -> pd.DataFrame:
    """Where demand (open reqs) is high relative to supply (headcount) and the
    unit is simultaneously losing people faster than it hires."""
    ctx = ctx or AccessContext.default()
    hc = sl.run("headcount", "business_unit", None, ctx).data.rename(columns={"value": "headcount"})
    reqs = sl.run("open_requisitions", "business_unit", None, ctx).data.rename(
        columns={"value": "open_reqs"})
    att = sl.run("voluntary_attrition_rate", "business_unit", None, ctx).data.rename(
        columns={"value": "attrition_pct"})
    hires = sl.run("hires", "business_unit", None, ctx).data.rename(columns={"value": "hires_12m"})
    ttf = sl.run("time_to_fill", "business_unit", None, ctx).data.rename(
        columns={"value": "time_to_fill_days"})

    df = (hc[["dimension", "headcount"]]
          .merge(reqs[["dimension", "open_reqs", "critical_requisitions", "aged_over_90d"]],
                 on="dimension", how="left")
          .merge(att[["dimension", "attrition_pct", "voluntary_exits"]], on="dimension", how="left")
          .merge(hires[["dimension", "hires_12m"]], on="dimension", how="left")
          .merge(ttf[["dimension", "time_to_fill_days"]], on="dimension", how="left"))
    df = df.fillna({"open_reqs": 0, "critical_requisitions": 0, "aged_over_90d": 0,
                    "voluntary_exits": 0, "hires_12m": 0})
    df["vacancy_rate_pct"] = (100 * df["open_reqs"] / df["headcount"]).round(1)
    df["net_flow"] = df["hires_12m"] - df["voluntary_exits"]
    df["replacement_ratio"] = (df["hires_12m"] / df["voluntary_exits"].replace(0, np.nan)).round(2)
    df["gap_score"] = (
        df["vacancy_rate_pct"].rank(pct=True) * 0.4
        + df["attrition_pct"].rank(pct=True) * 0.35
        + df["time_to_fill_days"].rank(pct=True).fillna(0.5) * 0.25
    ).round(3)
    return df.sort_values("gap_score", ascending=False).reset_index(drop=True)


def funnel_bottleneck(result) -> Finding:
    """Where the recruiting funnel loses the most people relative to normal."""
    df = result.data.sort_values("stage_index")
    if len(df) < 2:
        return Finding("Insufficient funnel data", "", "info", {})
    steps = df.dropna(subset=["step_conversion_pct"])
    if steps.empty:
        return Finding("Insufficient funnel data", "", "info", {})
    worst = steps.loc[steps["step_conversion_pct"].idxmin()]
    prev = df[df["stage_index"] == worst["stage_index"] - 1].iloc[0]
    lost = int(prev["value"] - worst["value"])
    return Finding(
        headline=f"{prev['dimension']} to {worst['dimension']} is the tightest step "
                 f"at {worst['step_conversion_pct']:.1f}%",
        detail=f"{lost:,} candidates do not progress past {prev['dimension']}. "
               f"Cumulative pass-through to this stage is {worst['cumulative_pct']:.2f}% "
               f"of all applicants.",
        severity="attention",
        evidence={"from_stage": prev["dimension"], "to_stage": worst["dimension"],
                  "conversion_pct": float(worst["step_conversion_pct"]), "candidates_lost": lost},
    )
