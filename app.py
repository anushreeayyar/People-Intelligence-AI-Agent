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

@st.cache_resource(show_spinner=False)
def ensure_warehouse() -> str:
    """Build the synthetic warehouse on first run if it is not already present.

    The .duckdb file is deliberately not committed - a binary database does not
    belong in version control - so a fresh deployment has no warehouse until one
    is generated. Rather than failing with an instruction the deployed app can't
    follow, it builds its own. Cached as a resource so this happens once per
    container, not once per session.
    """
    if config.WAREHOUSE.exists():
        return "present"
    import subprocess
    generator = Path(__file__).resolve().parent / "data" / "generate_data.py"
    result = subprocess.run(
        [sys.executable, str(generator)],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0 or not config.WAREHOUSE.exists():
        raise RuntimeError(result.stderr[-2000:] or "Generator produced no warehouse.")
    return "built"


try:
    with st.spinner("First run: building the synthetic warehouse. About fifteen seconds."):
        warehouse_state = ensure_warehouse()
except Exception as exc:                                       # noqa: BLE001
    st.error(
        "**Could not build the synthetic warehouse.** Run it locally with "
        "`python data/generate_data.py` and check the output below."
    )
    st.code(str(exc), language="text")
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
