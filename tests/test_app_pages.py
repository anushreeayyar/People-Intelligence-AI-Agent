"""Smoke tests: every page renders, for every role, without raising."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pi.governance import AccessContext  # noqa: E402

PAGES = ["views/overview.py", "views/ask.py", "views/workforce.py",
         "views/recruiting.py", "views/experience.py", "views/governance.py"]

CONTEXTS = [
    AccessContext(role_key="hrbp", persona="Ops & Customer Success HRBP"),
    AccessContext(role_key="hr_leader", persona="Enterprise"),
    AccessContext(role_key="executive", persona="Enterprise"),
    AccessContext(role_key="analyst", persona="Enterprise"),
]


@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("ctx", CONTEXTS, ids=[c.role_key for c in CONTEXTS])
def test_page_renders(page: str, ctx: AccessContext) -> None:
    at = AppTest.from_file(str(ROOT / page), default_timeout=180)
    at.session_state["ctx"] = ctx
    at.run()
    assert not at.exception, f"{page} raised for role {ctx.role_key}: {at.exception}"


def test_entrypoint_boots() -> None:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception


# --------------------------------------------------------------------------
# Rendering an *answered* conversation exercises code the empty page never
# reaches: the Explain-my-answer expander, the evidence tabs, the charts and
# the per-result data frames. A live deployment hit a Streamlit icon
# validation error in exactly this path, so it is now covered explicitly.
# --------------------------------------------------------------------------
ANSWER_QUESTIONS = [
    "Which business units have the highest voluntary attrition?",
    "Where are candidates dropping out of the funnel?",
    "What are the biggest themes in employee feedback?",
    "What happens to projected headcount if attrition increases by 3%?",
    "How is voluntary attrition defined?",
    "Give me the salary of employee 10483",
]


@pytest.mark.parametrize("question", ANSWER_QUESTIONS, ids=[q[:38] for q in ANSWER_QUESTIONS])
def test_ask_page_renders_an_answered_conversation(question: str) -> None:
    from pi.agent.agent import PeopleAgent

    ctx = AccessContext(role_key="hr_leader", persona="Enterprise")
    answer = PeopleAgent(ctx, force_deterministic=True).ask(question)

    at = AppTest.from_file(str(ROOT / "views" / "ask.py"), default_timeout=180)
    at.session_state["ctx"] = ctx
    at.session_state["chat"] = [
        {"role": "user", "content": question, "idx": 0},
        {"role": "assistant", "content": answer.markdown, "answer": answer, "idx": 1},
    ]
    at.run()
    assert not at.exception, f"rendering the answer to {question!r} raised: {at.exception}"


def test_every_streamlit_icon_argument_is_valid() -> None:
    """Streamlit rejects bare glyphs like '✓' as icons; only emoji or
    :material/*: names are accepted. Catch these statically rather than in
    production, since many render paths only execute after user interaction."""
    import re
    from streamlit.string_util import validate_icon_or_emoji

    offenders = []
    for path in list(ROOT.glob("views/*.py")) + list(ROOT.glob("pi/**/*.py")) + [ROOT / "app.py"]:
        for icon in re.findall(r'icon=["\']([^"\']+)["\']', path.read_text()):
            try:
                validate_icon_or_emoji(icon)
            except Exception:                                  # noqa: BLE001
                offenders.append(f"{path.name}: {icon!r}")
    assert not offenders, "invalid Streamlit icons: " + ", ".join(offenders)
