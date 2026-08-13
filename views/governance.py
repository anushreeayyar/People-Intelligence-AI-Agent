"""Data & Governance - the differentiator page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from pi import charts, config, semantic_layer as sl, ui, warehouse
from pi.agent import sql_guard
from pi.governance import (EXECUTIVE_DIMENSION_ALLOWLIST, HRBP_ASSIGNMENTS, ROLES,
                           SENSITIVE_PATTERNS, read_audit, screen_question)
from pi.quality import checks

ctx = st.session_state.ctx

ui.page_header(
    "Data & Governance",
    "What the numbers are made of, what was excluded and why, who can see what, and "
    "what the AI is not allowed to do. This page is the reason the rest of the "
    "application can be trusted.",
)

score = checks.scorecard()

c = st.columns(5)
with c[0]:
    ui.kpi("Data quality", f"{score['completeness_pct']}%",
           f"{score['checks_passed']}/{score['checks_run']} checks fully clean",
           delta_good=score["completeness_pct"] >= 99)
with c[1]:
    ui.kpi("Records checked", f"{score['records_checked']:,}",
           foot="Across five source datasets")
with c[2]:
    ui.kpi("Quarantined", f"{score['employees_excluded']:,}",
           "employee records excluded from every metric", delta_good=False)
with c[3]:
    ui.kpi("In scope", f"{score['employees_in_scope']:,}",
           foot="Employee records feeding all metrics")
with c[4]:
    ui.kpi("Metrics defined", f"{len(sl.catalog())}",
           f"dictionary v{sl._raw()['version']}")

tabs = st.tabs(["Metrics dictionary", "Data quality", "Lineage", "Access control",
                "AI guardrails", "Audit log"])

# --------------------------------------------------------------- dictionary
with tabs[0]:
    st.markdown("### People Metrics Dictionary")
    st.markdown(
        "One definition per metric, versioned in `semantic/metrics.yaml`. Dashboards, "
        "the daily brief and the AI agent all execute these definitions - none of them "
        "carries its own copy of the arithmetic. When somebody asks *what is our "
        "attrition*, this file is the answer."
    )
    rows = [{
        "Metric": s.label, "Domain": s.domain, "Definition": s.definition,
        "Calculation": s.formula, "Source": s.source_system,
        "Grain": s.grain, "Owner": s.owner, "Refresh": s.refresh,
        "Benchmark": s.benchmark if s.benchmark is not None else "",
        "Dimensions": ", ".join(s.dimensions),
    } for s in sl.catalog().values()]
    df = pd.DataFrame(rows)
    domain_filter = st.multiselect("Filter by domain", sorted(df["Domain"].unique()))
    view = df[df["Domain"].isin(domain_filter)] if domain_filter else df
    ui.dataframe(view, column_config={
        "Definition": st.column_config.TextColumn(width="large")})
    ui.download(df, "people-metrics-dictionary.csv", key="gov_dl1")

    st.markdown("#### Inspect a definition")
    pick = st.selectbox("Metric", list(sl.catalog()),
                        format_func=lambda m: sl.get(m).label)
    ui.metric_caption(pick)

# ------------------------------------------------------------- data quality
with tabs[1]:
    st.markdown("### Validate before you analyse")
    st.markdown(
        f"**{checks.banner()}** Hard failures are quarantined by the governed views, "
        "so an excluded record is excluded from every metric identically rather than "
        "being handled differently in each dashboard. Soft failures are flagged and "
        "kept."
    )
    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(charts.gauge(score["completeness_pct"], "Completeness"),
                        use_container_width=True)
    with c2:
        st.markdown("#### Employee records excluded from all metrics")
        ui.dataframe(score["exclusions"].rename(columns={
            "exclusion_reason": "Reason", "records": "Records"}))

    st.markdown("#### Check register")
    detail = score["detail"]
    ui.dataframe(detail.rename(columns={
        "check_id": "ID", "dataset": "Dataset", "field": "Field", "rule": "Rule",
        "severity": "Severity", "records_checked": "Checked",
        "records_failed": "Failed", "pass_rate_pct": "Pass rate %",
        "status": "Status", "treatment": "Treatment"}))
    ui.download(detail, "data-quality-checks.csv", key="gov_dl2")
    st.caption("Every rule is declarative and lives in `pi/quality/checks.py`: an id, "
               "the rule in words, the SQL that finds the offending rows, a severity, "
               "and what the product does about it.")

# ------------------------------------------------------------------ lineage
with tabs[2]:
    st.markdown("### Lineage")
    st.markdown(
        "Five raw tables become eight governed views. Every query in this product - "
        "including every query the agent generates - reads the views, never the raw "
        "tables. That is enforced by the SQL validator, not by convention."
    )
    st.code("""
  HRIS  ──► employees ────────┐
                              ├─► v_employee_exclusions ─► v_employees ─┬─► v_headcount_monthly
  HRIS  ──► internal_moves ───┘                                         └─► v_movement_monthly
                                                                             │
  ATS   ──► requisitions ─────────────────────► v_requisitions               │
  ATS   ──► candidates ───────────────────────► v_candidates                 │
                                                                             ▼
  Engagement ─► survey_responses ─────────────► v_survey        semantic/metrics.yaml
                                                                             │
                                                          ┌──────────────────┴──────────────────┐
                                                          ▼                                     ▼
                                                 Dashboards + Daily Brief              AI agent tools
    """, language="text")
    ui.dataframe(checks.lineage().rename(columns={
        "view": "Governed view", "sources": "Built from", "enforces": "What it enforces"}))

    st.markdown("#### Object schemas visible to the agent")
    st.caption("Restricted columns are stripped from the schema before it is shown to "
               "the model, so it cannot ask for what it may not have.")
    st.code(warehouse.schema_summary(), language="text")
    st.markdown(f"**Columns withheld from every role:** "
                f"`{'`, `'.join(sorted(config.RESTRICTED_COLUMNS))}`")

# ----------------------------------------------------------- access control
with tabs[3]:
    st.markdown("### Role-based access")
    st.markdown(
        "The row policy is applied when the query is built, not filtered out of the "
        "answer afterwards, so out-of-scope rows are never read into memory. Switching "
        "role in the sidebar changes what the dashboards *and* the agent can see."
    )
    ui.dataframe(pd.DataFrame([{
        "Role": r.label,
        "Row scope": "Assigned business units" if r.scope == "assigned_units" else "Enterprise",
        "Minimum group size": r.min_group_size,
        "Ad-hoc SQL": "Yes" if r.can_run_adhoc_sql else "No",
        "Individual records": "No",
        "Description": r.description,
    } for r in ROLES.values()]))

    st.markdown("#### HRBP supported populations")
    ui.dataframe(pd.DataFrame([{"HRBP": k, "Business units": ", ".join(v)}
                               for k, v in HRBP_ASSIGNMENTS.items()]))
    st.caption(f"Executive views are additionally restricted to "
               f"{', '.join(sorted(EXECUTIVE_DIMENSION_ALLOWLIST))} grain - seniority "
               "does not grant finer resolution, because finer resolution is where "
               "re-identification risk lives.")

    st.markdown("#### Try the boundary")
    st.caption("Currently viewing as " + ctx.describe())
    test_unit = st.selectbox("Request voluntary attrition for",
                             ["Commerce", "Operations", "Engineering",
                              "Customer Success", "Marketing", "Corporate"])
    if st.button("Run as current role"):
        try:
            r = sl.run("voluntary_attrition_rate", "business_unit",
                       {"business_unit": test_unit}, ctx)
            st.success(f"Allowed. {r.row_policy}")
            ui.dataframe(r.data)
        except PermissionError as exc:
            st.error(f"Blocked by row policy. {exc}")

# ------------------------------------------------------------- AI guardrails
with tabs[4]:
    st.markdown("### AI guardrails")
    st.markdown(
        "Four layers, in the order they fire. Rules live in code, not in the prompt, "
        "so they cannot be talked out of."
    )
    st.markdown(
        "1. **Input validation** — sensitive-topic and prompt-injection screening "
        "before the model sees the question.\n"
        "2. **Tool-only data access** — the model has no database handle. It selects "
        "metrics; the semantic layer computes them.\n"
        "3. **SQL validation** — any model-written SQL must be a single aggregating "
        "SELECT over governed views, with no restricted columns and a bounded LIMIT. "
        "The connection itself is read-only.\n"
        "4. **Output screening** — identifiers are redacted from the final text, and "
        "every question, tool call and refusal is written to the audit log."
    )

    st.markdown("#### Restricted request categories")
    ui.dataframe(pd.DataFrame([{
        "Category": cat.replace("_", " ").title(),
        "Response": msg[:150] + "…"} for cat, _, msg in SENSITIVE_PATTERNS]))

    st.markdown("#### Test the input filter")
    probe = st.text_input("Question", "Give me the salary of employee 10483")
    if probe:
        d = screen_question(probe)
        if d.allowed:
            st.success("Allowed — this question would proceed to metric selection.")
        else:
            st.error(f"**Blocked** · category `{d.category}`")
            st.markdown(f"<div class='refusal'>{d.message}</div>", unsafe_allow_html=True)

    st.markdown("#### Test the SQL validator")
    st.caption("Paste any SQL. The validator runs the same checks it applies to "
               "model-generated SQL.")
    default_sql = ("SELECT business_unit, full_name, base_salary\n"
                   "FROM employees WHERE base_salary > 100000")
    probe_sql = st.text_area("SQL", default_sql, height=110)
    if st.button("Validate SQL"):
        v = sql_guard.validate(probe_sql)
        if v.ok:
            st.success("Passed validation.")
            st.code(v.sql, language="sql")
            for w in v.warnings:
                st.caption("Note: " + w)
        else:
            st.error("Rejected.")
            for e in v.errors:
                st.markdown(f"- {e}")

# ---------------------------------------------------------------- audit log
with tabs[5]:
    st.markdown("### Audit log")
    st.markdown(
        "Append-only record of every question asked, tool called, SQL validated and "
        "refusal issued, with the role and scope in force at the time. This is what "
        "makes an AI answer defensible after the fact."
    )
    log = read_audit(400)
    if log.empty:
        st.info("No activity recorded yet in this session. Ask a question on the "
                "**Ask People Intelligence** page and it will appear here.")
    else:
        events = st.multiselect("Event type", sorted(log["event"].unique()))
        view = log[log["event"].isin(events)] if events else log
        cols = [c for c in ["timestamp", "user", "role", "persona", "event", "tool",
                            "question", "arguments", "ok", "category", "error", "mode"]
                if c in view.columns]
        ui.dataframe(view[cols].head(250))
        ui.download(view, "audit-log.csv", key="gov_dl3")
        st.caption(f"{len(log)} recent events. Written to `data/audit_log.jsonl`.")
