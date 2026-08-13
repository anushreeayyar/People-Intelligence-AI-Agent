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
