"""Employee Experience - engagement, eNPS and open-text themes."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pi import analysis, charts, semantic_layer as sl, ui, warehouse

ctx = st.session_state.ctx

ui.page_header(
    "Employee Experience",
    "Engagement, eNPS and what people actually write in the comment box. Themes are "
    "assigned at ingestion by the listening platform, not inferred at query time, so "
    "the same comment is always counted the same way.",
)
ui.scope_notice(ctx)

dim = st.selectbox("Break down by",
                   ui.dimension_options(ctx, ["business_unit", "department",
                                              "job_level", "tenure_band"]),
                   key="ex_dim", format_func=lambda d: sl.dimensions()[d]["label"])

eng = sl.run("engagement_score", dim, None, ctx)
trend = sl.run("engagement_trend", dim, None, ctx)
enps = sl.run("enps", dim, None, ctx)
themes = sl.run("feedback_themes", "All", None, ctx)
link = sl.run("theme_engagement_link", "All", None, ctx)

periods = sorted(trend.data["period"].unique()) if not trend.data.empty else []
latest, prior = (periods[-1], periods[-2]) if len(periods) >= 2 else (None, None)

c = st.columns(5)
with c[0]:
    now = trend.data[trend.data["period"] == latest]["value"].mean() if latest else float("nan")
    was = trend.data[trend.data["period"] == prior]["value"].mean() if prior else None
    ui.kpi("Engagement", f"{now:.2f}",
           f"{now - was:+.2f} vs {prior}" if was is not None else None,
           delta_good=(now - was) >= 0 if was is not None else None,
           foot=f"Survey period {latest}" if latest else "")
with c[1]:
    e = enps.data["value"].mean() if not enps.data.empty else float("nan")
    ui.kpi("eNPS", f"{e:+.0f}", foot="Promoters minus detractors")
with c[2]:
    resp = int(eng.data["responses"].sum()) if not eng.data.empty else 0
    ui.kpi("Responses", f"{resp:,}", foot=f"Latest period, {len(eng.data)} groups")
with c[3]:
    if not themes.data.empty:
        neg = 100 * themes.data["negative"].sum() / themes.data["value"].sum()
        ui.kpi("Negative comments", f"{neg:.0f}%", foot="Share of all classified comments")
with c[4]:
    if not eng.data.empty:
        worst = eng.data.iloc[0]
        ui.kpi("Lowest scoring group", str(worst["dimension"]),
               f"{worst['value']:.2f}", delta_good=False,
               foot=f"{int(worst['responses'])} responses")

ui.quality_strip()

tab1, tab2, tab3 = st.tabs(["Scores & trend", "Themes", "Theme to engagement"])

# ------------------------------------------------------------------ tab 1
with tab1:
    st.plotly_chart(charts.line(trend, "Engagement by survey period"),
                    use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(charts.bar(eng, "Engagement, latest period"),
                        use_container_width=True)
    with c2:
        st.plotly_chart(charts.bar(enps, "eNPS, latest period"), use_container_width=True)

    for f in analysis.describe_distribution(eng)[:1] + analysis.biggest_movers(trend, 1)[:3]:
        st.markdown(f"- **{f.headline}.** {f.detail}")

    st.markdown("##### Sub-scores")
    if not eng.data.empty:
        show = eng.data.rename(columns={
            "dimension": "Group", "value": "Engagement", "responses": "Responses",
            "manager_score": "Manager", "workload_score": "Workload",
            "growth_score": "Growth"})
        ui.dataframe(show)
        st.caption("Workload consistently below the headline engagement score is the "
                   "pattern that precedes burnout-driven attrition.")
    ui.explain_panel(eng, "ex_eng")

# ------------------------------------------------------------------ tab 2
with tab2:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(charts.bar(themes, "Comment volume by theme", highlight_top=False),
                        use_container_width=True)
    with c2:
        if not themes.data.empty:
            st.plotly_chart(
                charts.scatter_quadrant(themes.data, "value", "negative_pct", "dimension",
                                        "Volume against negativity",
                                        "Comments", "Negative share %"),
                use_container_width=True)
    st.caption("Act on the intersection of high volume and high negative share. Volume "
               "alone tells you what people talk about; negativity tells you what hurts.")
    if not themes.data.empty:
        ui.dataframe(themes.data.rename(columns={
            "dimension": "Theme", "value": "Comments", "negative": "Negative",
            "positive": "Positive", "negative_pct": "Negative %",
            "avg_engagement": "Avg engagement"}))

    st.markdown("##### Sample comments")
    st.caption("Comments are shown in aggregate and are never attributable. No identifier, "
               "department below the suppression threshold, or free-text name is displayed.")
    theme_pick = st.selectbox("Theme", list(themes.data["dimension"]) if not themes.data.empty else [])
    if theme_pick:
        units = ctx.business_units
        clause = ""
        if units:
            quoted = ", ".join("'" + u.replace("'", "''") + "'" for u in units)
            clause = f" AND business_unit IN ({quoted})"
        sample = warehouse.query(
            "SELECT sentiment, comment_text, COUNT(*) AS occurrences "
            "FROM v_survey WHERE theme = ? AND survey_period = "
            "(SELECT MAX(survey_period) FROM v_survey)" + clause +
            " GROUP BY 1, 2 HAVING COUNT(*) >= 3 ORDER BY 3 DESC LIMIT 12",
            [theme_pick])
        ui.dataframe(sample.rename(columns={
            "sentiment": "Sentiment", "comment_text": "Comment",
            "occurrences": "Times raised"}))
        st.caption("Only comment patterns raised by at least three respondents are shown, "
                   "which is the aggregation threshold applied to free text.")
    ui.explain_panel(themes, "ex_themes")

# ------------------------------------------------------------------ tab 3
with tab3:
    if link.data.empty:
        st.info("Not enough responses in scope to link themes to engagement.")
    else:
        st.plotly_chart(
            charts.diverging_bar(link.data, "dimension", "value",
                                 "Engagement gap by theme, versus the overall average",
                                 "Difference from overall average engagement"),
            use_container_width=True)
        st.markdown("Themes to the left travel with **lower** engagement than average.")
        for f in analysis.describe_distribution(link)[:1]:
            st.markdown(f"- **{f.headline}.** {f.detail}")
        ui.dataframe(link.data.rename(columns={
            "dimension": "Theme", "value": "Gap vs average",
            "theme_engagement": "Avg engagement of respondents",
            "responses": "Responses"}))
        st.warning(
            "This is an association, not a cause. Respondents who are already "
            "disengaged may be more likely to comment at all, which inflates the gap "
            "for every theme.", icon=":material/warning:")
        ui.explain_panel(link, "ex_link")
        ui.download(link.data, "theme-engagement-link.csv", key="ex_dl")
