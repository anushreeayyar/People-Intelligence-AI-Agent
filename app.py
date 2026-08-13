"""
People Intelligence - Streamlit entry point.

Six pages, one governed data layer. Role selection in the sidebar changes what
the whole application can see, including the agent, because the row policy is
enforced in the semantic layer rather than in the UI.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pi import config                                        # noqa: E402
from pi.governance import HRBP_ASSIGNMENTS, ROLES, AccessContext  # noqa: E402
from pi.ui import css, sidebar_footer                        # noqa: E402

st.set_page_config(
    page_title="People Intelligence",
    page_icon=":material/insights:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not config.WAREHOUSE.exists():
    st.error(
        "**Warehouse not found.** Generate the synthetic dataset first:\n\n"
        "```bash\npython data/generate_data.py\n```"
    )
    st.stop()

css()

# --------------------------------------------------------------------------
# Access context - the single control that scopes the entire application
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### People Intelligence")
    st.caption("Synthetic HR data · governed metrics · auditable AI")

    st.markdown("#### Viewing as")
    role_key = st.selectbox(
        "Role", list(ROLES), index=0,
        format_func=lambda k: ROLES[k].label,
        help="Role determines row scope, minimum group size and whether ad-hoc SQL is available.",
    )
    persona = "Enterprise"
    if ROLES[role_key].scope == "assigned_units":
        persona = st.selectbox("Supported population", list(HRBP_ASSIGNMENTS))
    st.caption(ROLES[role_key].description)

if ("ctx" not in st.session_state
        or st.session_state.ctx.role_key != role_key
        or st.session_state.ctx.persona != persona):
    st.session_state.ctx = AccessContext(role_key=role_key, persona=persona,
                                         user="demo.hrbp@example.test")
    st.session_state.pop("chat", None)

ctx: AccessContext = st.session_state.ctx

with st.sidebar:
    units = ctx.business_units
    st.markdown(
        f"<div class='scope-box'><b>Scope</b><br>"
        f"{'All business units' if units is None else '<br>'.join(units)}<br>"
        f"<span class='muted'>Minimum group size {ctx.role.min_group_size} · "
        f"ad-hoc SQL {'enabled' if ctx.role.can_run_adhoc_sql else 'disabled'}</span></div>",
        unsafe_allow_html=True,
    )

pages = [
    st.Page("views/overview.py", title="Executive Overview", icon=":material/dashboard:", default=True),
    st.Page("views/ask.py", title="Ask People Intelligence", icon=":material/forum:"),
    st.Page("views/workforce.py", title="Workforce Intelligence", icon=":material/groups:"),
    st.Page("views/recruiting.py", title="Recruiting Intelligence", icon=":material/person_search:"),
    st.Page("views/experience.py", title="Employee Experience", icon=":material/sentiment_satisfied:"),
    st.Page("views/governance.py", title="Data & Governance", icon=":material/verified_user:"),
]

with st.sidebar:
    sidebar_footer()

st.navigation(pages).run()
