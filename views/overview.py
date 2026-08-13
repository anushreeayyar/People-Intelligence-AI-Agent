"""Executive Overview - the landing page."""
from __future__ import annotations

import numpy as np
import streamlit as st

from pi import charts, semantic_layer as sl, ui
from pi.brief import daily_brief
from pi.quality import checks

ctx = st.session_state.ctx

ui.page_header(
    "People Intelligence",
    "Enterprise workforce position as at "
    f"{sl._raw()['as_of_date']}. Every figure is produced by the People Metrics "
    "Dictionary against governed data, and every figure can be traced back to its query.",
)
ui.scope_notice(ctx)


@st.cache_data(ttl=600, show_spinner=False)
def load(role: str, persona: str):
    hc = sl.run("headcount", "All", None, ctx)
    hc_bu = sl.run("headcount", "business_unit", None, ctx)
    hc_trend = sl.run("headcount_trend", "business_unit", None, ctx)
    att = sl.run("voluntary_attrition_rate", "business_unit", None, ctx)
    att_trend = sl.run("attrition_trend", "business_unit", None, ctx)
    reqs = sl.run("open_requisitions", "business_unit", None, ctx)
    ttf = sl.run("time_to_fill", "business_unit", None, ctx)
    funnel = sl.run("funnel_conversion", "All", None, ctx)
    eng = sl.run("engagement_trend", "business_unit", None, ctx)
    risk = sl.run("retention_risk_index", "business_unit", None, ctx)
    return hc, hc_bu, hc_trend, att, att_trend, reqs, ttf, funnel, eng, risk


hc, hc_bu, hc_trend, att, att_trend, reqs, ttf, funnel, eng, risk = load(
    ctx.role_key, ctx.persona)


# --------------------------------------------------------------------- KPIs
def _trend_delta(result, months_back: int = 12):
    df = result.data
    if df.empty or "period" not in df.columns:
        return None, None
    periods = sorted(df["period"].unique())
    if len(periods) < 2:
        return None, None
    now = df[df["period"] == periods[-1]]
    prior_p = periods[max(0, len(periods) - 1 - months_back)]
    prior = df[df["period"] == prior_p]
    return now, prior


headcount_total = int(hc.data["value"].iloc[0]) if not hc.data.empty else 0
hc_now, hc_prior = _trend_delta(hc_trend, 12)
hc_growth = None
if hc_now is not None and hc_prior is not None and hc_prior["value"].sum():
    hc_growth = 100 * (hc_now["value"].sum() - hc_prior["value"].sum()) / hc_prior["value"].sum()

att_weighted = float((att.data["value"] * att.data["avg_headcount"]).sum()
                     / max(att.data["avg_headcount"].sum(), 1)) if not att.data.empty else 0.0
att_now, att_prior = _trend_delta(att_trend, 12)
att_delta = None
if att_now is not None and att_prior is not None:
    att_delta = att_now["value"].mean() - att_prior["value"].mean()

open_reqs = int(reqs.data["value"].sum()) if not reqs.data.empty else 0
critical = int(reqs.data["critical_requisitions"].sum()) if not reqs.data.empty else 0
aged = int(reqs.data["aged_over_90d"].sum()) if not reqs.data.empty else 0
median_ttf = float(ttf.data["value"].median()) if not ttf.data.empty else float("nan")

eng_now, eng_prior = _trend_delta(eng, 1)
eng_score = float(eng_now["value"].mean()) if eng_now is not None else float("nan")
eng_delta = (eng_score - float(eng_prior["value"].mean())) if eng_prior is not None else None

offer_pass = (float(funnel.data.iloc[-1]["cumulative_pct"])
              if not funnel.data.empty else float("nan"))
top_risk = risk.data.iloc[0] if not risk.data.empty else None

c = st.columns(6)
with c[0]:
    ui.kpi("Headcount", f"{headcount_total:,}",
           f"{hc_growth:+.1f}% YoY" if hc_growth is not None else None,
           delta_good=(hc_growth or 0) >= 0, foot="Active employees, HRIS")
with c[1]:
    ui.kpi("Voluntary attrition", f"{att_weighted:.1f}%",
           f"{att_delta:+.1f} pts YoY" if att_delta is not None else None,
           delta_good=(att_delta or 0) < 0, foot="Trailing 12 months")
with c[2]:
    ui.kpi("Open requisitions", f"{open_reqs:,}",
           f"{critical} critical", delta_good=None,
           foot=f"{aged} open beyond 90 days")
with c[3]:
    ui.kpi("Median time to fill", f"{median_ttf:.0f} days",
           "benchmark 45 days", delta_good=median_ttf <= 45,
           foot="Filled reqs, last 12 months")
with c[4]:
    ui.kpi("Engagement", f"{eng_score:.2f}",
           f"{eng_delta:+.2f} vs prior quarter" if eng_delta is not None else None,
           delta_good=(eng_delta or 0) >= 0, foot="Latest listening survey")
with c[5]:
    ui.kpi("Highest retention risk",
           top_risk["dimension"] if top_risk is not None else "n/a",
           f"index {top_risk['value']:.0f}" if top_risk is not None else None,
           delta_good=False, foot="Composite 0-100")

ui.quality_strip()

# ------------------------------------------------------------- daily brief
st.markdown("## People Intelligence Daily Brief")


@st.cache_data(ttl=900, show_spinner=False)
def brief(role: str, persona: str):
    return daily_brief.build_brief(ctx)


b = brief(ctx.role_key, ctx.persona)
left, right = st.columns([3, 1])
with left:
    st.markdown(f"**{b['headline']}.** Signals are emitted only when a movement clears "
                "the published materiality threshold, so a quiet day reads as a quiet day.")
with right:
    st.download_button("Export brief (Markdown)",
                       daily_brief.to_markdown(b).encode(),
                       file_name=f"people-intelligence-brief-{b['generated_at'][:10]}.md",
                       use_container_width=True)

if not b["signals"]:
    st.success("Nothing cleared the materiality thresholds in this run.")
for s in b["signals"][:4]:
    ui.signal_card(s)

with st.expander("Materiality thresholds and detection logic"):
    st.markdown(
        "The brief re-runs a fixed set of governed metrics, compares each against its "
        "own recent history, and emits a signal only where the movement clears the "
        "threshold below. Thresholds are a business decision and live in one place "
        "(`pi/brief/daily_brief.py`), not scattered through the detectors."
    )
    st.dataframe(
        {"Signal type": list(b["thresholds"]),
         "Low": [v["low"] for v in b["thresholds"].values()],
         "Medium": [v["medium"] for v in b["thresholds"].values()],
         "High": [v["high"] for v in b["thresholds"].values()]},
        use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ charts
st.markdown("## Where the enterprise stands")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(charts.line(hc_trend, "Headcount by business unit"),
                    use_container_width=True)
    ui.explain_panel(hc_trend, "hc_trend")
with c2:
    st.plotly_chart(charts.bar(att, "Voluntary attrition by business unit"),
                    use_container_width=True)
    ui.explain_panel(att, "att")

c3, c4 = st.columns(2)
with c3:
    st.plotly_chart(charts.funnel(funnel, "Recruiting funnel, all open and closed reqs"),
                    use_container_width=True)
    ui.explain_panel(funnel, "funnel")
with c4:
    st.plotly_chart(charts.line(eng, "Engagement by business unit"),
                    use_container_width=True)
    ui.explain_panel(eng, "eng")

# ----------------------------------------------------------- risk register
st.markdown("## Retention risk register")
st.caption("Composite prioritisation index. It ranks groups for attention; it is not a "
           "probability that any individual will leave, and it is never computed at "
           "individual grain.")
if risk.data.empty:
    st.info("No group met the minimum size for a risk score under the current scope.")
else:
    cols = [c for c in ["dimension", "value", "risk_band", "attrition_pct",
                        "attrition_delta", "engagement", "workload_score",
                        "voluntary_exits", "avg_headcount"] if c in risk.data.columns]
    show = risk.data[cols].rename(columns={
        "dimension": "Group", "value": "Risk index", "risk_band": "Band",
        "attrition_pct": "Voluntary attrition %", "attrition_delta": "YoY change (pts)",
        "engagement": "Engagement", "workload_score": "Workload score",
        "voluntary_exits": "Exits (12m)", "avg_headcount": "Avg headcount"})
    ui.dataframe(show.round(2))
    ui.download(risk.data, "retention-risk-register.csv", key="risk_dl")
    ui.explain_panel(risk, "risk")

st.markdown("---")
st.caption("Next: **Ask People Intelligence** for a natural-language question against "
           "the same governed layer, or **Data & Governance** for the metric "
           "dictionary, quality checks and audit log.")
