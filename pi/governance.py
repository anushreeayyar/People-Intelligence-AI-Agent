"""
Governance: who may see what, at what grain, and with what left on the record.

Four controls live here.

  1. Role-based row policy    - an HRBP sees their own units, nobody else's.
  2. Aggregation thresholds   - no group smaller than the minimum is ever shown.
  3. Sensitive-topic policy   - individual identification and individual
                                compensation are refused for every role,
                                including the most senior one.
  4. Audit logging            - every question, decision, tool call and refusal
                                is written to an append-only log.

The order matters. The policy is applied when the query is built, not filtered
out of the answer afterwards, so restricted rows are never read into memory in
the first place.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from pi import config


# --------------------------------------------------------------------------
# Roles
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Role:
    key: str
    label: str
    description: str
    scope: str                       # 'assigned_units' | 'enterprise'
    min_group_size: int
    allowed_domains: tuple[str, ...]
    can_view_individual_rows: bool = False
    can_run_adhoc_sql: bool = False


ROLES: dict[str, Role] = {
    "hrbp": Role(
        key="hrbp",
        label="HR Business Partner",
        description=("Aggregated workforce, recruiting and experience data for "
                     "assigned business units only."),
        scope="assigned_units",
        min_group_size=5,
        allowed_domains=("Workforce", "Retention", "Recruiting", "Employee Experience"),
    ),
    "hr_leader": Role(
        key="hr_leader",
        label="HR Leader",
        description=("Enterprise-wide aggregated data across every People domain, "
                     "including cross-unit comparison."),
        scope="enterprise",
        min_group_size=5,
        allowed_domains=("Workforce", "Retention", "Recruiting", "Employee Experience"),
    ),
    "executive": Role(
        key="executive",
        label="Executive",
        description=("Enterprise strategic metrics at business-unit grain. "
                     "Deliberately coarser: no department or team-level cuts."),
        scope="enterprise",
        min_group_size=25,
        allowed_domains=("Workforce", "Retention", "Recruiting", "Employee Experience"),
    ),
    "analyst": Role(
        key="analyst",
        label="People Analytics",
        description=("Full metric catalogue plus validated read-only ad-hoc SQL "
                     "against the governed views. Still no access to identifiers "
                     "or individual compensation."),
        scope="enterprise",
        min_group_size=5,
        allowed_domains=("Workforce", "Retention", "Recruiting", "Employee Experience"),
        can_run_adhoc_sql=True,
    ),
}

#: Which units each HRBP persona supports. In a real deployment this comes from
#: the HRIS supported-population table, not a constant.
HRBP_ASSIGNMENTS: dict[str, list[str]] = {
    "Ops & Customer Success HRBP": ["Operations", "Customer Success"],
    "Technology HRBP": ["Engineering"],
    "Commercial HRBP": ["Commerce", "Marketing"],
    "Corporate Functions HRBP": ["Corporate"],
}

#: Executives may only slice at these grains, however senior they are.
EXECUTIVE_DIMENSION_ALLOWLIST = {"business_unit", "job_level", "All"}


@dataclass
class AccessContext:
    role_key: str = "hrbp"
    persona: str = "Ops & Customer Success HRBP"
    user: str = "demo.user"
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @staticmethod
    def default() -> "AccessContext":
        return AccessContext()

    @property
    def role(self) -> Role:
        return ROLES[self.role_key]

    @property
    def business_units(self) -> list[str] | None:
        """None means enterprise-wide."""
        if self.role.scope == "enterprise":
            return None
        return HRBP_ASSIGNMENTS.get(self.persona, list(config.BUSINESS_UNITS))

    def describe(self) -> str:
        units = self.business_units
        scope = "all business units" if units is None else ", ".join(units)
        return (f"{self.role.label} ({self.persona}) - scope: {scope}; "
                f"minimum group size {self.role.min_group_size}")


# --------------------------------------------------------------------------
# 1 + 2. Row policy and aggregation thresholds
# --------------------------------------------------------------------------
def apply_row_policy(filters: dict[str, Any], ctx: AccessContext,
                     spec: Any = None) -> tuple[dict[str, Any], str]:
    """Intersect the caller's requested filters with what their role permits."""
    allowed_units = ctx.business_units
    if allowed_units is None:
        return filters, f"Enterprise scope ({ctx.role.label}); no row restriction applied."

    requested = filters.get("business_unit")
    if requested is None:
        filters["business_unit"] = allowed_units
        note = f"Row policy applied: business_unit restricted to {allowed_units}."
    else:
        req = [requested] if isinstance(requested, str) else list(requested)
        permitted = [u for u in req if u in allowed_units]
        if not permitted:
            raise PermissionError(
                f"Your access covers {', '.join(allowed_units)}. "
                f"You requested {', '.join(req)}, which sits outside your supported "
                "population. An HR Leader can run this cross-unit view."
            )
        filters["business_unit"] = permitted
        note = f"Row policy applied: business_unit narrowed to {permitted}."
    return filters, note


def check_dimension_allowed(dimension: str, ctx: AccessContext) -> None:
    if ctx.role_key == "executive" and dimension not in EXECUTIVE_DIMENSION_ALLOWLIST:
        raise PermissionError(
            f"The Executive view is deliberately limited to "
            f"{', '.join(sorted(EXECUTIVE_DIMENSION_ALLOWLIST))} grain. "
            f"'{dimension}' is available to HR Leader and HRBP roles."
        )


def suppress_small_groups(df: pd.DataFrame, spec: Any,
                          ctx: AccessContext) -> tuple[pd.DataFrame, int]:
    """Drop rows describing a population smaller than the role's threshold."""
    if df.empty:
        return df, 0
    threshold = ctx.role.min_group_size
    size_cols = [c for c in ("avg_headcount", "responses", "headcount_now",
                             "filled_requisitions", "applications", "offers_decided",
                             "exits", "internal_moves")
                 if c in df.columns]
    if not size_cols:
        if "value" in df.columns and getattr(spec, "unit", "") == "count":
            size_cols = ["value"]
        else:
            return df, 0
    mask = pd.Series(True, index=df.index)
    for c in size_cols[:1]:
        mask &= df[c].fillna(0) >= threshold
    removed = int((~mask).sum())
    return df[mask].reset_index(drop=True), removed


# --------------------------------------------------------------------------
# 3. Sensitive-topic policy
# --------------------------------------------------------------------------
SENSITIVE_PATTERNS: list[tuple[str, str, str]] = [
    (
        "individual_identification",
        r"\b(employee|worker|person|staff member)\s*(id|number|#)?\s*[:#]?\s*\d{3,}"
        r"|\bE\d{6}\b|\bwho (is|are) the (employees|people|individuals)\b"
        r"|\b(name|names|list) of (the )?(employees|people|individuals|leavers|joiners)\b"
        r"|\bwhich employees\b|\bwho left\b|\bwho is (likely|going) to (leave|quit|resign)\b"
        r"|\bflight risk (list|of|for) (employee|individual|person)",
        "I can't return information that identifies an individual employee. "
        "Every metric in this product is defined at group grain and suppressed "
        "below a minimum group size. I can show you the same question as an "
        "aggregate - for example attrition or retention risk by business unit, "
        "job level, tenure band or job family.",
    ),
    (
        "individual_compensation",
        r"\b(salary|pay|compensation|comp|bonus|earnings|wage)\b.{0,40}\b(of|for)\b.{0,30}"
        r"(employee|E\d{6}|\d{4,}|this person|him|her|them)"
        r"|\bhow much (does|do|did)\b.{0,30}\b(earn|make|get paid)\b"
        r"|\bwhat (is|was) .{0,25}\b(salary|pay|comp)\b.{0,20}\bfor\b",
        "I can't provide individually identifiable compensation information. "
        "I can provide aggregated compensation insight by job level, job family "
        "or business unit, where the group is large enough to prevent "
        "re-identification.",
    ),
    (
        "protected_characteristics",
        r"\b(race|ethnicity|religion|disability|sexual orientation|pregnan|"
        r"medical|health condition|immigration status|visa status)\b",
        "Analysis of protected or health-related characteristics is out of scope "
        "for this self-service tool. Requests of this kind are handled by the "
        "People Analytics team under a separate approval and privacy review "
        "process, and are reported only in approved formats.",
    ),
    (
        "performance_individual",
        r"\b(performance rating|review score|pip|underperform)\b.{0,30}\b(of|for)\b.{0,25}"
        r"(employee|E\d{6}|\d{4,}|this person)",
        "Individual performance records aren't available through this tool. "
        "I can show performance distribution at group level where the population "
        "is large enough.",
    ),
    (
        "disciplinary_or_exit_detail",
        r"\b(why did|reason (that|why))\b.{0,30}\b(E\d{6}|employee \d+)\b"
        r"|\bexit interview (notes|transcript|verbatim) for\b",
        "Individual exit records and interview notes are confidential. I can "
        "show the distribution of stated exit reasons for a group.",
    ),
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore (all |your |the )?(previous|prior|above) instructions",
    r"disregard (your|the) (system|previous) (prompt|instructions|rules)",
    r"you are (now|no longer)\b.{0,40}(unrestricted|dan|admin|root)",
    r"(reveal|print|show|output) (your|the) (system prompt|instructions|rules)",
    r"\bbypass\b.{0,25}\b(governance|policy|restriction|guardrail|access control)",
    r"\bpretend (you|to be)\b.{0,30}\b(admin|executive|unrestricted)",
    r"\bgrant (me|myself)\b.{0,20}\b(access|permission|admin)",
]


@dataclass
class PolicyDecision:
    allowed: bool
    category: str | None = None
    message: str | None = None

    def __bool__(self) -> bool:
        return self.allowed


def screen_question(question: str) -> PolicyDecision:
    """Input validation run before the model ever sees the question."""
    q = question.lower()

    for pattern in PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return PolicyDecision(
                False, "prompt_injection",
                "That request looks like an attempt to change how this assistant "
                "operates. Access rules and metric definitions are enforced in code "
                "rather than in the prompt, so they can't be changed from the chat "
                "box. Ask me a workforce question instead and I'll answer it.",
            )

    for category, pattern, message in SENSITIVE_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return PolicyDecision(False, category, message)

    return PolicyDecision(True)


def screen_output(text: str) -> str:
    """Last-line defence: redact anything that looks like an employee identifier."""
    return re.sub(r"\bE\d{6}\b", "[identifier redacted]", text)


# --------------------------------------------------------------------------
# 4. Audit log
# --------------------------------------------------------------------------
def audit(event: str, ctx: AccessContext, **payload: Any) -> None:
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session_id": ctx.session_id,
        "user": ctx.user,
        "role": ctx.role_key,
        "persona": ctx.persona,
        "event": event,
        **payload,
    }
    config.AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(config.AUDIT_LOG, "a") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def read_audit(limit: int = 250) -> pd.DataFrame:
    if not config.AUDIT_LOG.exists():
        return pd.DataFrame(columns=["timestamp", "user", "role", "event", "detail"])
    rows = []
    with open(config.AUDIT_LOG) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(rows[-limit:])
    return df.iloc[::-1].reset_index(drop=True) if not df.empty else df
