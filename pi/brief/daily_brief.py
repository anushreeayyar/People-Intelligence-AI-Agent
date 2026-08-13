"""
People Intelligence Daily Brief.

A signal detector, not a report generator. It re-runs a fixed set of governed
metrics, compares each against its own recent history, and emits only the
movements that clear a materiality threshold. On a quiet day it says so.

Each signal carries three things a dashboard tile does not: what moved, why it
matters commercially, and the specific next step for the HRBP who owns it.

Run headless:  python automation/run_daily_brief.py
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from pi import config, semantic_layer as sl
from pi.governance import AccessContext
from pi.quality import checks

BRIEF_DIR = config.DATA_DIR / "briefs"

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}
SEVERITY_MARK = {"high": "🔴", "medium": "🟠", "low": "🟡"}


@dataclass
class Signal:
    signal_id: str
    domain: str
    entity: str
    headline: str
    metric: str
    current: float
    prior: float
    change: float
    unit: str
    severity: str
    why_it_matters: str
    recommended_action: str
    evidence: dict = field(default_factory=dict)

    @property
    def marker(self) -> str:
        return SEVERITY_MARK[self.severity]


# --------------------------------------------------------------------------
# Thresholds. Deliberately explicit and in one place, because "what counts as
# a signal" is a business decision, not an implementation detail.
# --------------------------------------------------------------------------
THRESHOLDS = {
    "attrition_pts": {"low": 1.0, "medium": 2.0, "high": 3.0},
    "time_to_fill_days": {"low": 5, "medium": 10, "high": 15},
    "engagement_points": {"low": 0.10, "medium": 0.20, "high": 0.30},
    "headcount_pct": {"low": 2.0, "medium": 4.0, "high": 6.0},
    "critical_aged_reqs": {"low": 5, "medium": 10, "high": 15},
    "negative_sentiment_pts": {"low": 4.0, "medium": 8.0, "high": 12.0},
}


def _severity(kind: str, magnitude: float) -> str | None:
    t = THRESHOLDS[kind]
    a = abs(magnitude)
    if a >= t["high"]:
        return "high"
    if a >= t["medium"]:
        return "medium"
    if a >= t["low"]:
        return "low"
    return None


# --------------------------------------------------------------------------
def _attrition_signals(ctx: AccessContext, lookback_months: int = 3) -> list[Signal]:
    res = sl.run("attrition_trend", "business_unit", None, ctx)
    df = res.data
    if df.empty:
        return []
    periods = sorted(df["period"].unique())
    if len(periods) < lookback_months + 1:
        return []
    now, prior = periods[-1], periods[-1 - lookback_months]
    cur = df[df["period"] == now].set_index("dimension")["value"]
    old = df[df["period"] == prior].set_index("dimension")["value"]
    out = []
    for bu in cur.index:
        if bu not in old.index:
            continue
        delta = float(cur[bu] - old[bu])
        sev = _severity("attrition_pts", delta)
        if sev is None or delta <= 0:
            continue
        out.append(Signal(
            signal_id=f"ATT-{bu.replace(' ', '')[:6].upper()}",
            domain="Retention",
            entity=bu,
            headline=f"{bu} voluntary attrition up {delta:.1f} points to {cur[bu]:.1f}%",
            metric="attrition_trend",
            current=round(float(cur[bu]), 1), prior=round(float(old[bu]), 1),
            change=round(delta, 1), unit="pp", severity=sev,
            why_it_matters=(
                f"On a population of this size, a {delta:.1f} point rise in the rolling "
                f"annual rate is roughly {int(round(delta / 100 * _headcount(ctx, bu)))} "
                "additional exits a year, each carrying replacement cost, ramp time and "
                "lost institutional knowledge. Rolling rates move slowly, so a move this "
                "size over one quarter is a run-rate change rather than noise."),
            recommended_action=(
                f"Cut {bu} attrition by job level and tenure band to find where it "
                "concentrates, then read the exit-reason mix for that segment. If it "
                "concentrates at manager level, treat it as a span-and-support problem "
                "before treating it as a pay problem."),
            evidence={"from_period": str(prior)[:10], "to_period": str(now)[:10]},
        ))
    return out


def _headcount(ctx: AccessContext, bu: str) -> float:
    df = sl.run("headcount", "business_unit", {"business_unit": bu}, ctx).data
    return float(df["value"].sum()) if not df.empty else 0.0


def _time_to_fill_signals(ctx: AccessContext) -> list[Signal]:
    res = sl.run("time_to_fill_trend", "business_unit", None, ctx)
    df = res.data
    if df.empty:
        return []
    periods = sorted(df["period"].unique())
    if len(periods) < 2:
        return []
    now, prior = periods[-1], periods[-2]
    cur = df[df["period"] == now].set_index("dimension")
    old = df[df["period"] == prior].set_index("dimension")
    out = []
    for bu in cur.index:
        if bu not in old.index:
            continue
        delta = float(cur.loc[bu, "value"] - old.loc[bu, "value"])
        sev = _severity("time_to_fill_days", delta)
        if sev is None or delta <= 0:
            continue
        out.append(Signal(
            signal_id=f"TTF-{bu.replace(' ', '')[:6].upper()}",
            domain="Recruiting",
            entity=bu,
            headline=f"{bu} time to fill up {delta:.0f} days to "
                     f"{cur.loc[bu, 'value']:.0f} days",
            metric="time_to_fill_trend",
            current=float(cur.loc[bu, "value"]), prior=float(old.loc[bu, "value"]),
            change=round(delta, 0), unit="days", severity=sev,
            why_it_matters=(
                "Every extra day a requisition stays open is a day of unabsorbed work "
                "for the team carrying the vacancy, and it compounds: slower hiring "
                "raises workload, workload drives attrition, and attrition opens more "
                "requisitions."),
            recommended_action=(
                f"Check whether the slowdown sits in candidate supply or in internal "
                f"decision time for {bu}. If onsite-to-offer conversion is stable but "
                "elapsed time has grown, it is a scheduling and decision-making problem "
                "the hiring managers can fix this week."),
            evidence={"from_period": str(prior)[:10], "to_period": str(now)[:10],
                      "filled_requisitions": int(cur.loc[bu, "filled_requisitions"])},
        ))
    return out


def _engagement_signals(ctx: AccessContext) -> list[Signal]:
    res = sl.run("engagement_trend", "business_unit", None, ctx)
    df = res.data
    if df.empty:
        return []
    periods = sorted(df["period"].unique())
    if len(periods) < 2:
        return []
    now, prior = periods[-1], periods[-2]
    cur = df[df["period"] == now].set_index("dimension")
    old = df[df["period"] == prior].set_index("dimension")
    out = []
    for bu in cur.index:
        if bu not in old.index:
            continue
        delta = float(cur.loc[bu, "value"] - old.loc[bu, "value"])
        sev = _severity("engagement_points", delta)
        if sev is None or delta >= 0:
            continue
        pct = 100 * delta / float(old.loc[bu, "value"])
        theme = _dominant_negative_theme(ctx, bu)
        out.append(Signal(
            signal_id=f"ENG-{bu.replace(' ', '')[:6].upper()}",
            domain="Employee Experience",
            entity=bu,
            headline=f"{bu} engagement down {abs(delta):.2f} points ({pct:+.1f}%) "
                     f"to {cur.loc[bu, 'value']:.2f}",
            metric="engagement_trend",
            current=float(cur.loc[bu, "value"]), prior=float(old.loc[bu, "value"]),
            change=round(delta, 2), unit="score", severity=sev,
            why_it_matters=(
                f"Engagement is a leading indicator: it moves before attrition does. "
                + (f"Negative comments in {bu} concentrate on {theme}, which is where "
                   "an intervention would have to land to move the score."
                   if theme else "")),
            recommended_action=(
                f"Take the {bu} theme breakdown to that leadership team and pick one "
                "theme to act on before the next survey cycle. Acting visibly on one "
                "theme moves the score further than acknowledging all of them."),
            evidence={"from_period": str(prior), "to_period": str(now),
                      "responses": int(cur.loc[bu, "responses"]),
                      "dominant_negative_theme": theme},
        ))
    return out


def _dominant_negative_theme(ctx: AccessContext, bu: str) -> str | None:
    df = sl.run("feedback_themes", "business_unit", {"business_unit": bu}, ctx).data
    if df.empty or "negative" not in df.columns:
        return None
    top = df.sort_values("negative", ascending=False).iloc[0]
    return str(top["dimension"])


def _requisition_signals(ctx: AccessContext) -> list[Signal]:
    df = sl.run("open_requisitions", "business_unit", None, ctx).data
    out = []
    for r in df.itertuples():
        aged = int(getattr(r, "aged_over_90d", 0) or 0)
        crit = int(getattr(r, "critical_requisitions", 0) or 0)
        sev = _severity("critical_aged_reqs", aged)
        if sev is None:
            continue
        out.append(Signal(
            signal_id=f"REQ-{r.dimension.replace(' ', '')[:6].upper()}",
            domain="Recruiting",
            entity=r.dimension,
            headline=f"{r.dimension} has {aged} requisitions open beyond 90 days, "
                     f"out of {int(r.value)} open ({crit} of them business critical)",
            metric="open_requisitions",
            current=float(aged), prior=float("nan"), change=float(aged),
            unit="requisitions", severity=sev,
            why_it_matters=(
                "Requisitions past 90 days distort capacity planning, because the "
                "business keeps assuming the person is arriving. They also consume "
                "recruiter attention that would convert faster elsewhere."),
            recommended_action=(
                f"Run a triage session on the {r.dimension} aged requisitions: re-scope, "
                "re-band, or close. Anything a hiring manager will not defend in that "
                "session should not stay open."),
            evidence={"open_requisitions": int(r.value),
                      "avg_age_days": float(getattr(r, "avg_age_days", 0) or 0)},
        ))
    return out


def _sentiment_signals(ctx: AccessContext) -> list[Signal]:
    """Themes where the negative share is unusually concentrated."""
    df = sl.run("feedback_themes", "All", None, ctx).data
    if df.empty:
        return []
    overall_neg = 100 * df["negative"].sum() / df["value"].sum()
    out = []
    for r in df.sort_values("negative_pct", ascending=False).head(2).itertuples():
        delta = float(r.negative_pct - overall_neg)
        sev = _severity("negative_sentiment_pts", delta)
        if sev is None or delta <= 0:
            continue
        out.append(Signal(
            signal_id=f"SEN-{r.dimension.split()[0][:6].upper()}",
            domain="Employee Experience",
            entity=r.dimension,
            headline=f"'{r.dimension}' comments are {r.negative_pct:.0f}% negative, "
                     f"{delta:.0f} points above the enterprise average",
            metric="feedback_themes",
            current=float(r.negative_pct), prior=round(overall_neg, 1),
            change=round(delta, 1), unit="pp", severity=sev,
            why_it_matters=(
                f"Respondents raising this theme average {r.avg_engagement:.2f} on "
                "engagement, so it is not simply a vocal minority - it travels with "
                "measurably lower engagement."),
            recommended_action=(
                f"Cross the '{r.dimension}' theme against attrition for the same "
                "population. Where both are elevated, treat it as the retention "
                "hypothesis to test first."),
            evidence={"comments": int(r.value), "negative": int(r.negative),
                      "avg_engagement": float(r.avg_engagement)},
        ))
    return out


# --------------------------------------------------------------------------
def detect_signals(ctx: AccessContext | None = None) -> list[Signal]:
    ctx = ctx or AccessContext(role_key="hr_leader", persona="Enterprise",
                               user="daily_brief")
    signals: list[Signal] = []
    for fn in (_attrition_signals, _time_to_fill_signals, _engagement_signals,
               _requisition_signals, _sentiment_signals):
        try:
            signals.extend(fn(ctx))
        except Exception as exc:                                # noqa: BLE001
            signals.append(Signal(
                signal_id="ERR", domain="System", entity=fn.__name__,
                headline=f"Signal check failed: {type(exc).__name__}",
                metric="-", current=0, prior=0, change=0, unit="", severity="low",
                why_it_matters="A detector did not run, so this brief may be incomplete.",
                recommended_action="Check the scheduled job logs.",
                evidence={"error": str(exc)[:200]},
            ))
    signals.sort(key=lambda s: (SEVERITY_ORDER[s.severity], -abs(s.change)))
    return signals


def build_brief(ctx: AccessContext | None = None, max_signals: int = 6) -> dict:
    ctx = ctx or AccessContext(role_key="hr_leader", persona="Enterprise",
                               user="daily_brief")
    signals = detect_signals(ctx)[:max_signals]
    quality = checks.scorecard()
    hc = sl.run("headcount", "All", None, ctx).data
    att = sl.run("voluntary_attrition_rate", "All", None, ctx).data

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reporting_date": sl._raw()["as_of_date"],
        "signal_count": len(signals),
        "headline": (f"{len(signals)} workforce signal{'s' if len(signals) != 1 else ''} "
                     "detected" if signals else
                     "No workforce signals cleared the materiality thresholds"),
        "context": {
            "headcount": int(hc["value"].sum()) if not hc.empty else 0,
            "voluntary_attrition_pct": float(att["value"].iloc[0]) if not att.empty else 0.0,
            "data_quality_pct": quality["completeness_pct"],
        },
        "signals": [asdict(s) for s in signals],
        "thresholds": THRESHOLDS,
        "scope": ctx.describe(),
    }


def to_markdown(brief: dict) -> str:
    lines = [
        "# People Intelligence Daily Brief",
        f"*Reporting date {brief['reporting_date']} · generated "
        f"{brief['generated_at'][:16].replace('T', ' ')} UTC*",
        "",
        f"**{brief['headline']}.** Headcount {brief['context']['headcount']:,} · "
        f"voluntary attrition {brief['context']['voluntary_attrition_pct']:.1f}% · "
        f"data quality {brief['context']['data_quality_pct']}%",
        "",
    ]
    if not brief["signals"]:
        lines.append("Nothing moved beyond the materiality thresholds since the last run. "
                     "Thresholds are published at the end of this brief.")
    for s in brief["signals"]:
        lines += [
            f"### {SEVERITY_MARK[s['severity']]} {s['headline']}",
            f"*{s['domain']} · {s['entity']} · signal {s['signal_id']} · "
            f"severity {s['severity']}*",
            "",
            f"**Why it matters.** {s['why_it_matters']}",
            "",
            f"**Recommended action.** {s['recommended_action']}",
            "",
        ]
    lines += [
        "---",
        "**Materiality thresholds.** " + "; ".join(
            f"{k.replace('_', ' ')}: low {v['low']}, medium {v['medium']}, high {v['high']}"
            for k, v in brief["thresholds"].items()),
        "",
        f"*Scope: {brief['scope']}. All figures produced by the People Metrics "
        "Dictionary against governed views; synthetic data only.*",
    ]
    return "\n".join(lines)


def save(brief: dict, directory: Path | None = None) -> tuple[Path, Path]:
    directory = directory or BRIEF_DIR
    directory.mkdir(parents=True, exist_ok=True)
    stamp = brief["generated_at"][:10]
    json_path = directory / f"brief-{stamp}.json"
    md_path = directory / f"brief-{stamp}.md"
    json_path.write_text(json.dumps(brief, indent=2, default=str))
    md_path.write_text(to_markdown(brief))
    return json_path, md_path


def history(directory: Path | None = None) -> pd.DataFrame:
    directory = directory or BRIEF_DIR
    if not directory.exists():
        return pd.DataFrame(columns=["date", "signals", "headline"])
    rows = []
    for f in sorted(directory.glob("brief-*.json")):
        try:
            b = json.loads(f.read_text())
            rows.append({"date": b["generated_at"][:10], "signals": b["signal_count"],
                         "headline": b["headline"], "file": f.name})
        except (json.JSONDecodeError, KeyError):
            continue
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)
