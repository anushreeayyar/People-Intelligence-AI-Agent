"""Shared Streamlit presentation helpers."""
from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from pi import semantic_layer as sl
from pi.governance import AccessContext
from pi.quality import checks

CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px;}
  h1 {font-size: 1.85rem !important; font-weight: 650 !important; letter-spacing: -0.02em;}
  h2 {font-size: 1.25rem !important; font-weight: 620 !important; margin-top: 1.6rem !important;}
  h3 {font-size: 1.02rem !important; font-weight: 600 !important;}
  .muted {color:#6b7280; font-size:0.82rem;}
  .page-sub {color:#4b5563; font-size:0.95rem; margin:-0.5rem 0 1.4rem 0;}
  .scope-box {background:#f3f6fa; border:1px solid #dde5ee; border-radius:8px;
              padding:0.65rem 0.75rem; font-size:0.82rem; line-height:1.5; margin-top:0.4rem;}
  .kpi {background:#fff; border:1px solid #e5e9f0; border-radius:10px;
        padding:0.9rem 1rem; height:100%;}
  .kpi .label {font-size:0.74rem; text-transform:uppercase; letter-spacing:0.06em;
               color:#6b7280; font-weight:600;}
  .kpi .value {font-size:1.65rem; font-weight:650; color:#12263f; line-height:1.25; margin-top:0.1rem;}
  .kpi .delta {font-size:0.8rem; font-weight:600;}
  .kpi .foot {font-size:0.74rem; color:#6b7280; margin-top:0.2rem;}
  .up-bad {color:#c73e1d;} .up-good {color:#2a7f62;} .flat {color:#6b7280;}
  .signal {border-left:4px solid #c73e1d; background:#fff; border:1px solid #e5e9f0;
           border-left-width:4px; border-radius:8px; padding:0.85rem 1rem; margin-bottom:0.7rem;}
  .signal.medium {border-left-color:#e08b2f;} .signal.low {border-left-color:#d9b40a;}
  .signal .head {font-weight:640; font-size:0.98rem; color:#12263f;}
  .signal .meta {font-size:0.74rem; color:#6b7280; text-transform:uppercase;
                 letter-spacing:0.05em; margin-bottom:0.3rem;}
  .signal .body {font-size:0.88rem; color:#374151; margin-top:0.4rem;}
  .pill {display:inline-block; padding:0.12rem 0.55rem; border-radius:999px;
         font-size:0.72rem; font-weight:600; background:#eef2f7; color:#33506e;
         margin-right:0.3rem;}
  .pill.warn {background:#fdf0e6; color:#a2521a;}
  .pill.good {background:#e7f4ed; color:#1f6b50;}
  .pill.bad  {background:#fbeae6; color:#a53119;}
  .refusal {background:#fbf7e8; border:1px solid #ecdfb0; border-radius:8px;
            padding:0.85rem 1rem; font-size:0.9rem;}
  div[data-testid="stMetricValue"] {font-size:1.5rem;}
  .stTabs [data-baseweb="tab"] {font-size:0.9rem;}
  footer, #MainMenu {visibility:hidden;}
</style>
"""


def css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_footer() -> None:
    st.divider()
    st.caption(
        f"Reporting date **{sl._raw()['as_of_date']}** · dictionary "
        f"v{sl._raw()['version']}"
    )
    st.caption("100% synthetic data. No real employee, candidate or "
               "compensation record is present in this application.")


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f"<div class='page-sub'>{subtitle}</div>", unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str | None = None,
        delta_good: bool | None = None, foot: str = "") -> None:
    cls = "flat"
    if delta_good is True:
        cls = "up-good"
    elif delta_good is False:
        cls = "up-bad"
    st.markdown(
        f"<div class='kpi'><div class='label'>{label}</div>"
        f"<div class='value'>{value}</div>"
        + (f"<div class='delta {cls}'>{delta}</div>" if delta else "")
        + (f"<div class='foot'>{foot}</div>" if foot else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def signal_card(sig: dict) -> None:
    st.markdown(
        f"<div class='signal {sig['severity']}'>"
        f"<div class='meta'>{sig['domain']} · {sig['entity']} · {sig['signal_id']}</div>"
        f"<div class='head'>{sig['headline']}</div>"
        f"<div class='body'><b>Why it matters.</b> {sig['why_it_matters']}</div>"
        f"<div class='body'><b>Recommended action.</b> {sig['recommended_action']}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def metric_caption(metric_id: str) -> None:
    spec = sl.get(metric_id)
    with st.expander(f"How **{spec.label}** is defined", expanded=False):
        st.markdown(f"{spec.definition}")
        st.markdown(f"**Formula.** `{spec.formula}`")
        st.markdown(f"**Source.** {spec.source_system} · {', '.join(spec.source_objects)} "
                    f"· owned by {spec.owner} · refreshed {spec.refresh.lower()}")
        if spec.caveats:
            st.markdown("**Caveats.**")
            for c in spec.caveats:
                st.markdown(f"- {c}")
        st.code((spec.sql or "Composite metric - assembled from other governed metrics.").strip(),
                language="sql")


def explain_panel(result, key: str) -> None:
    """The 'Explain My Answer' surface for a single metric result."""
    p = result.provenance()
    with st.expander("Explain this number", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Metric.** {p['metric']}")
            st.markdown(f"**Definition.** {p['definition']}")
            st.markdown(f"**Formula.** `{p['formula']}`")
        with c2:
            st.markdown(f"**Source.** {p['source_system']}")
            st.markdown(f"**Objects.** {', '.join(p['source_objects'])}")
            st.markdown(f"**Owner.** {p['owner']} · refreshed {p['refresh'].lower()}")
            st.markdown(f"**Grain.** {p['grain']}")
        st.markdown(f"**Filters applied.** `{p['filters']}`")
        st.markdown(f"**Access policy.** {p['row_policy']}")
        if p["suppressed_groups"]:
            st.markdown(f"**Suppression.** {p['suppressed_groups']} group(s) below the "
                        "minimum size were removed.")
        st.markdown(f"**Data quality.** {checks.banner()}")
        if p["caveats"]:
            st.markdown("**Caveats.** " + " ".join(f"{c}" for c in p["caveats"]))
        st.markdown("**Query executed.**")
        st.code(p["sql"], language="sql")


def dataframe(df: pd.DataFrame, **kwargs: Any) -> None:
    st.dataframe(df, use_container_width=True, hide_index=True, **kwargs)


def download(df: pd.DataFrame, filename: str, label: str = "Export CSV",
             key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=False).encode(), file_name=filename,
                       mime="text/csv", key=key)


def scope_notice(ctx: AccessContext) -> None:
    units = ctx.business_units
    if units is not None:
        st.info(
            f"Viewing as **{ctx.role.label} — {ctx.persona}**. Every figure on this "
            f"page is restricted to {', '.join(units)} by the row policy, and groups "
            f"smaller than {ctx.role.min_group_size} are suppressed.",
            icon=":material/lock:",
        )


def quality_strip() -> None:
    st.caption(checks.banner())


def dimension_options(ctx: AccessContext, candidates: list[str]) -> list[str]:
    """Filter a page's breakdown options by what the role is allowed to see.

    Executives are held at a coarser grain deliberately: seniority does not
    grant finer resolution, because finer resolution is where re-identification
    risk lives.
    """
    from pi.governance import EXECUTIVE_DIMENSION_ALLOWLIST
    if ctx.role_key == "executive":
        return [d for d in candidates if d in EXECUTIVE_DIMENSION_ALLOWLIST] or ["business_unit"]
    return candidates
