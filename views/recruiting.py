"""Recruiting Intelligence - pipeline health."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pi import analysis, charts, semantic_layer as sl, ui

ctx = st.session_state.ctx

ui.page_header(
    "Recruiting Intelligence",
    "Pipeline health from requisition open to accepted offer. Time to fill measures "
    "only the roles that got filled, so it is shown next to requisition ageing, which "
    "is where the unfilled tail lives.",
)
ui.scope_notice(ctx)

dim = st.selectbox("Break down by",
                   ui.dimension_options(ctx, ["business_unit", "department",
                                              "job_family", "job_level"]),
                   key="rec_dim", format_func=lambda d: sl.dimensions()[d]["label"])

ttf = sl.run("time_to_fill", dim, None, ctx)
reqs = sl.run("open_requisitions", dim, None, ctx)
funnel = sl.run("funnel_conversion", "All", None, ctx)
sources = sl.run("source_effectiveness", "All", None, ctx)
offers = sl.run("offer_acceptance_rate", dim, None, ctx)
aging = sl.run("requisition_aging",
               dim if dim in sl.get("requisition_aging").dimensions else "business_unit",
               None, ctx)

c = st.columns(5)
with c[0]:
    ui.kpi("Open requisitions", f"{int(reqs.data['value'].sum()):,}",
           f"{int(reqs.data['critical_requisitions'].sum())} business critical")
with c[1]:
    ui.kpi("Aged beyond 90 days", f"{int(reqs.data['aged_over_90d'].sum()):,}",
           "escalation trigger", delta_good=False)
with c[2]:
    med = float(ttf.data["value"].median()) if not ttf.data.empty else float("nan")
    ui.kpi("Median time to fill", f"{med:.0f} days", "benchmark 45",
           delta_good=med <= 45)
with c[3]:
    acc = (100 * offers.data["offers_accepted"].sum() / max(offers.data["offers_decided"].sum(), 1)
           if not offers.data.empty else float("nan"))
    ui.kpi("Offer acceptance", f"{acc:.1f}%", "benchmark 85%", delta_good=acc >= 85)
with c[4]:
    passthru = float(funnel.data.iloc[-1]["cumulative_pct"]) if not funnel.data.empty else 0
    ui.kpi("Applicant to hire", f"{passthru:.2f}%",
           f"{int(funnel.data.iloc[0]['value']):,} applicants")

ui.quality_strip()

tab1, tab2, tab3 = st.tabs(["Funnel & bottlenecks", "Speed & ageing", "Source effectiveness"])

# ------------------------------------------------------------------ tab 1
with tab1:
    c1, c2 = st.columns([2, 3])
    with c1:
        st.plotly_chart(charts.funnel(funnel), use_container_width=True)
    with c2:
        b = analysis.funnel_bottleneck(funnel)
        st.markdown(f"##### {b.headline}")
        st.markdown(b.detail)
        st.markdown("")
        show = funnel.data.rename(columns={
            "dimension": "Stage", "value": "Candidates",
            "step_conversion_pct": "Step conversion %",
            "cumulative_pct": "Cumulative % of applicants"})
        ui.dataframe(show[["Stage", "Candidates", "Step conversion %",
                           "Cumulative % of applicants"]])

    st.markdown("##### Funnel by business unit")
    rows = []
    unit_list = (ctx.business_units
                 or list(sl.run("headcount", "business_unit", None, ctx).data["dimension"]))
    for bu in unit_list:
        f = sl.run("funnel_conversion", "All", {"business_unit": bu}, ctx)
        if f.data.empty:
            continue
        r = {"Business unit": bu}
        for _, row in f.data.iterrows():
            r[row["dimension"]] = int(row["value"])
        applied = max(r.get("Applied", 0), 1)
        r["Onsite to offer %"] = round(
            100 * r.get("Offer Extended", 0) / max(r.get("Onsite Interview", 1), 1), 1)
        r["Applicant to hire %"] = round(100 * r.get("Offer Accepted", 0) / applied, 2)
        rows.append(r)
    if rows:
        ui.dataframe(pd.DataFrame(rows))
        st.caption("Onsite-to-offer is usually the most diagnostic step: it separates "
                   "a supply problem from a calibration problem.")
    ui.explain_panel(funnel, "rec_funnel")

# ------------------------------------------------------------------ tab 2
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.bar(ttf), use_container_width=True)
    with c2:
        trend_dim = dim if dim in sl.get("time_to_fill_trend").dimensions else "business_unit"
        ttf_trend = sl.run("time_to_fill_trend", trend_dim, None, ctx)
        st.plotly_chart(charts.line(ttf_trend, "Median time to fill by quarter"),
                        use_container_width=True)

    for f in analysis.describe_distribution(ttf)[:1] + analysis.biggest_movers(ttf_trend, 1)[:2]:
        st.markdown(f"- **{f.headline}.** {f.detail}")

    st.markdown("##### Requisition ageing")
    if not aging.data.empty:
        pivot = (aging.data.pivot_table(index="dimension", columns="age_bucket",
                                        values="value", aggfunc="sum", fill_value=0)
                 .reset_index().rename(columns={"dimension": "Group"}))
        ui.dataframe(pivot)
        st.plotly_chart(
            charts.stacked_bar(aging.data, x="value", y="dimension", color="age_bucket",
                               title="Open requisitions by age bucket"),
            use_container_width=True)
    st.markdown("##### Open requisition detail")
    ui.dataframe(reqs.data.rename(columns={
        "dimension": "Group", "value": "Open reqs",
        "critical_requisitions": "Critical", "avg_age_days": "Avg age (days)",
        "aged_over_90d": "Aged 90d+"}))
    ui.explain_panel(ttf, "rec_ttf")
    ui.download(reqs.data, "open-requisitions.csv", key="rec_dl1")

# ------------------------------------------------------------------ tab 3
with tab3:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(charts.bar(sources, "Applicant-to-hire conversion by source"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(
            charts.scatter_quadrant(sources.data, "applications", "value", "dimension",
                                    "Volume against conversion",
                                    "Applications", "Conversion to hire %"),
            use_container_width=True)
    st.caption("Top-left is the problem quadrant: high volume, low conversion. Those "
               "sources consume screening capacity without producing hires.")
    ui.dataframe(sources.data.rename(columns={
        "dimension": "Source", "applications": "Applications", "offers": "Offers",
        "hires": "Hires", "value": "Conversion to hire %"}))

    st.markdown("##### Offer acceptance")
    st.plotly_chart(charts.bar(offers, "Offer acceptance rate"), use_container_width=True)
    ui.explain_panel(sources, "rec_src")
    ui.download(sources.data, "source-effectiveness.csv", key="rec_dl2")
