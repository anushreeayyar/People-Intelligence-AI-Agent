"""Workforce Intelligence - size, shape, movement and planning."""
from __future__ import annotations

import streamlit as st

from pi import analysis, charts, semantic_layer as sl, ui

ctx = st.session_state.ctx

ui.page_header(
    "Workforce Intelligence",
    "Headcount, growth, movement and planning scenarios. Breakdowns are limited to the "
    "dimensions each metric declares in the dictionary, so a cut that would not be "
    "meaningful is simply not offered.",
)
ui.scope_notice(ctx)

dim_options = ui.dimension_options(ctx, ["business_unit", "department", "job_family",
                                         "job_level", "location", "tenure_band"])
dim = st.selectbox("Break down by", dim_options, key="wf_dim",
                   format_func=lambda d: sl.dimensions()[d]["label"])

tab1, tab2, tab3, tab4 = st.tabs(
    ["Headcount & growth", "Attrition", "Movement", "Planning scenarios"])

# ------------------------------------------------------------------ tab 1
with tab1:
    hc = sl.run("headcount", dim, None, ctx)
    trend = sl.run("headcount_trend", dim, None, ctx)
    growth = sl.run("headcount_growth_rate",
                    dim if dim in sl.get("headcount_growth_rate").dimensions
                    else "business_unit", None, ctx)

    c = st.columns(3)
    with c[0]:
        ui.kpi("Total headcount", f"{int(hc.data['value'].sum()):,}",
               foot=f"{len(hc.data)} groups in scope")
    with c[1]:
        fastest = growth.data.iloc[0] if not growth.data.empty else None
        ui.kpi("Fastest growing",
               fastest["dimension"] if fastest is not None else "n/a",
               f"{fastest['value']:+.1f}% YoY" if fastest is not None else None,
               delta_good=True)
    with c[2]:
        slowest = growth.data.iloc[-1] if not growth.data.empty else None
        ui.kpi("Slowest / contracting",
               slowest["dimension"] if slowest is not None else "n/a",
               f"{slowest['value']:+.1f}% YoY" if slowest is not None else None,
               delta_good=False)

    st.plotly_chart(charts.line(trend), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.bar(hc), use_container_width=True)
    with c2:
        st.plotly_chart(charts.bar(growth, "Year-on-year headcount growth"),
                        use_container_width=True)

    for f in analysis.describe_distribution(growth)[:2]:
        st.markdown(f"- **{f.headline}.** {f.detail}")
    ui.explain_panel(growth, "wf_growth")
    ui.download(trend.data, "headcount-trend.csv", key="wf_dl1")

# ------------------------------------------------------------------ tab 2
with tab2:
    att_dim = dim if dim in sl.get("voluntary_attrition_rate").dimensions else "business_unit"
    att = sl.run("voluntary_attrition_rate", att_dim, None, ctx)
    total = sl.run("total_attrition_rate", att_dim, None, ctx)
    reasons = sl.run("exit_reason_mix", "All", None, ctx)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(charts.bar(att), use_container_width=True)
    with c2:
        st.plotly_chart(charts.bar(reasons, "Stated exit reasons, last 12 months",
                                   highlight_top=False), use_container_width=True)

    trend_dim = dim if dim in sl.get("attrition_trend").dimensions else "business_unit"
    att_trend = sl.run("attrition_trend", trend_dim, None, ctx)
    st.plotly_chart(charts.line(att_trend, "Rolling 12-month voluntary attrition"),
                    use_container_width=True)

    for f in analysis.describe_distribution(att)[:2] + analysis.biggest_movers(att_trend)[:2]:
        st.markdown(f"- **{f.headline}.** {f.detail}")

    st.markdown("#### Voluntary against total attrition")
    merged = (att.data[["dimension", "value", "voluntary_exits", "avg_headcount"]]
              .rename(columns={"value": "Voluntary %"})
              .merge(total.data[["dimension", "value"]].rename(columns={"value": "Total %"}),
                     on="dimension", how="left"))
    merged["Involuntary %"] = (merged["Total %"] - merged["Voluntary %"]).round(1)
    ui.dataframe(merged.rename(columns={"dimension": "Group",
                                        "voluntary_exits": "Voluntary exits",
                                        "avg_headcount": "Avg headcount"}))
    ui.explain_panel(att, "wf_att")
    ui.download(att.data, "voluntary-attrition.csv", key="wf_dl2")

# ------------------------------------------------------------------ tab 3
with tab3:
    mob = sl.run("internal_mobility_rate",
                 dim if dim in sl.get("internal_mobility_rate").dimensions else "business_unit",
                 None, ctx)
    hires = sl.run("hires", dim if dim in sl.get("hires").dimensions else "business_unit",
                   None, ctx)
    att_b = sl.run("voluntary_attrition_rate", "business_unit", None, ctx)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.bar(mob, "Internal mobility rate"), use_container_width=True)
    with c2:
        st.plotly_chart(charts.bar(hires, "External hires, last 12 months"),
                        use_container_width=True)

    st.markdown("#### Hiring against exits")
    flow = (hires.data[["dimension", "value"]].rename(columns={"value": "Hires (12m)"})
            .merge(att_b.data[["dimension", "voluntary_exits"]]
                   .rename(columns={"voluntary_exits": "Voluntary exits (12m)"}),
                   on="dimension", how="outer").fillna(0))
    flow["Net flow"] = flow["Hires (12m)"] - flow["Voluntary exits (12m)"]
    flow["Replacement ratio"] = (flow["Hires (12m)"] /
                                 flow["Voluntary exits (12m)"].replace(0, float("nan"))).round(2)
    ui.dataframe(flow.rename(columns={"dimension": "Group"}))
    st.caption("A replacement ratio below 1.0 means the unit is shrinking through "
               "attrition faster than recruiting is replacing it, regardless of what "
               "the headline headcount says.")
    ui.explain_panel(mob, "wf_mob")

# ------------------------------------------------------------------ tab 4
with tab4:
    st.markdown("#### What happens if attrition moves?")
    st.caption("A stated-assumption projection, not a forecast. Change the inputs and "
               "the assumptions list updates with them.")
    c = st.columns(4)
    delta = c[0].slider("Change in voluntary attrition (pts)", -5.0, 10.0, 3.0, 0.5)
    months = c[1].slider("Horizon (months)", 3, 36, 12, 3)
    fill = c[2].slider("Hiring continues at", 0.0, 1.5, 1.0, 0.05,
                       format="%.2fx", help="Fraction of the current hiring run rate.")
    units = ctx.business_units or ["All units in scope"]
    unit = c[3].selectbox("Scope", ["All units in scope"] + list(
        sl.run("headcount", "business_unit", None, ctx).data["dimension"]))

    out = analysis.headcount_scenario(delta, months, fill,
                                      None if unit == "All units in scope" else unit, ctx)
    if "error" in out:
        st.warning(out["error"])
    else:
        k = st.columns(4)
        with k[0]:
            ui.kpi("Starting headcount", f"{out['starting_headcount']:,}")
        with k[1]:
            ui.kpi("Baseline in " + f"{months}m", f"{out['baseline_end_headcount']:,}",
                   f"at {out['current_voluntary_attrition_pct']}% attrition")
        with k[2]:
            ui.kpi("Scenario in " + f"{months}m", f"{out['scenario_end_headcount']:,}",
                   f"at {out['scenario_attrition_pct']}% attrition",
                   delta_good=out["headcount_gap"] >= 0)
        with k[3]:
            ui.kpi("Gap to baseline", f"{out['headcount_gap']:+,}",
                   f"~{out['additional_hires_needed_to_hold_flat']:,} extra hires to hold flat",
                   delta_good=out["headcount_gap"] >= 0)

        st.plotly_chart(charts.projection(out["projection"],
                                          f"Projected headcount, {out['scope']}"),
                        use_container_width=True)
        st.markdown("**Assumptions.**")
        for a in out["assumptions"]:
            st.markdown(f"- {a}")
        ui.download(out["projection"], "headcount-scenario.csv", key="wf_dl3")

    st.markdown("#### Where the gaps are")
    gaps = analysis.workforce_gaps(ctx)
    ui.dataframe(gaps.rename(columns={
        "dimension": "Business unit", "headcount": "Headcount", "open_reqs": "Open reqs",
        "critical_requisitions": "Critical", "aged_over_90d": "Aged 90d+",
        "attrition_pct": "Voluntary attrition %", "voluntary_exits": "Exits (12m)",
        "hires_12m": "Hires (12m)", "time_to_fill_days": "Time to fill (days)",
        "vacancy_rate_pct": "Vacancy rate %", "net_flow": "Net flow",
        "replacement_ratio": "Replacement ratio", "gap_score": "Gap score"}))
    st.caption("Gap score blends vacancy rate (40%), voluntary attrition (35%) and time "
               "to fill (25%) as percentile ranks across the units in scope.")
    ui.download(gaps, "workforce-gaps.csv", key="wf_dl4")
