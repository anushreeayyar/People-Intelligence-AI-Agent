"""
SQL validation.

Any SQL that did not come from the metric dictionary - that is, anything the
model wrote itself - passes through here before it reaches the database. The
guard is a whitelist, not a blacklist: a statement is rejected unless it is
provably a single read-only SELECT over the governed views, returning no
restricted column, with a bounded row count.

Defence in depth. Even if every rule below were bypassed, the connection is
opened read-only, so the warehouse still cannot be modified.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pi import config, warehouse

#: The only objects generated SQL may read.
ALLOWED_OBJECTS = {
    "v_employees", "v_headcount_monthly", "v_movement_monthly", "v_requisitions",
    "v_candidates", "v_survey", "v_internal_moves", "v_month_spine",
    "v_employee_exclusions", "meta_asof",
}

#: Statement types and side-effecting keywords that are refused outright.
FORBIDDEN_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "create", "truncate",
    "attach", "detach", "copy", "export", "import", "install", "load",
    "pragma", "set ", "call ", "vacuum", "checkpoint", "grant", "revoke",
    "merge", "replace into", "begin", "commit", "rollback",
]

#: Functions that can reach outside the database file.
FORBIDDEN_FUNCTIONS = [
    "read_csv", "read_parquet", "read_json", "read_text", "read_blob",
    "glob", "sniff_csv", "parquet_scan", "csv_scan", "httpfs", "url",
    "system", "shell", "getenv", "duckdb_settings", "sha256",
]

MAX_LIMIT = 5000
DEFAULT_LIMIT = 500


@dataclass
class ValidationResult:
    ok: bool
    sql: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    tables: list[str] = field(default_factory=list)
    rewritten: bool = False

    def report(self) -> str:
        lines = ["PASS" if self.ok else "FAIL"]
        for e in self.errors:
            lines.append(f"  error:   {e}")
        for w in self.warnings:
            lines.append(f"  note:    {w}")
        if self.tables:
            lines.append(f"  reads:   {', '.join(sorted(set(self.tables)))}")
        return "\n".join(lines)


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", " ", sql)
    return sql


def _referenced_objects(sql: str) -> list[str]:
    """Objects appearing after FROM or JOIN, ignoring CTE names."""
    cte_names = set(re.findall(r"(?:with|,)\s+([A-Za-z_][A-Za-z0-9_]*)\s+as\s*\(",
                               sql, re.IGNORECASE))
    refs = re.findall(r"\b(?:from|join)\s+([A-Za-z_][A-Za-z0-9_\.]*)", sql, re.IGNORECASE)
    out = []
    for r in refs:
        name = r.split(".")[-1].lower()
        if name in {n.lower() for n in cte_names}:
            continue
        out.append(name)
    return out


def validate(sql: str, allow_adhoc: bool = True) -> ValidationResult:
    original = sql
    errors: list[str] = []
    warnings: list[str] = []
    rewritten = False

    if not allow_adhoc:
        return ValidationResult(False, original,
                                ["Ad-hoc SQL is not permitted for this role."])

    if not sql or not sql.strip():
        return ValidationResult(False, original, ["Empty statement."])

    body = _strip_comments(sql).strip().rstrip(";").strip()
    low = " " + body.lower() + " "

    # ---- 1. exactly one statement
    if ";" in body:
        errors.append("Multiple statements are not allowed; submit a single SELECT.")

    # ---- 2. must be a read
    if not re.match(r"^\s*(select|with)\b", body, re.IGNORECASE):
        errors.append("Statement must begin with SELECT or WITH.")

    # ---- 3. no side effects
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"(?<![A-Za-z_]){re.escape(kw.strip())}(?![A-Za-z_])", low):
            errors.append(f"Forbidden keyword '{kw.strip().upper()}' in statement.")
    for fn in FORBIDDEN_FUNCTIONS:
        if re.search(rf"(?<![A-Za-z_]){re.escape(fn)}\s*\(", low):
            errors.append(f"Forbidden function '{fn}()' in statement.")

    # ---- 4. only governed objects
    refs = _referenced_objects(body)
    illegal = [r for r in refs if r not in ALLOWED_OBJECTS]
    if illegal:
        errors.append(
            "Query references object(s) outside the governed layer: "
            f"{', '.join(sorted(set(illegal)))}. Allowed: {', '.join(sorted(ALLOWED_OBJECTS))}."
        )
    if any(r in ("employees", "candidates", "requisitions", "survey_responses",
                 "internal_moves") for r in refs):
        errors.append(
            "Raw tables are not queryable. Use the v_ views, which apply the "
            "data-quality quarantine so that every metric reconciles."
        )

    # ---- 5. no restricted columns in the projection
    #
    # COUNT(employee_id) and COUNT(DISTINCT employee_id) are the legitimate
    # exception: counting people is the entire point, and a count cannot
    # identify anybody. The identifier is stripped from the text before the
    # restricted-column scan so that only genuine exposures are caught.
    scan = re.sub(r"count\s*\(\s*(distinct\s+)?employee_id\s*\)", "count(1)", low)
    for col in config.RESTRICTED_COLUMNS:
        if re.search(rf"(?<![A-Za-z_]){col}(?![A-Za-z_])", scan):
            errors.append(
                f"Column '{col}' is restricted and cannot be selected, filtered or "
                "grouped by. Individual identifiers and individual compensation are "
                "out of scope for every role."
            )
    if re.search(r"select\s+\*|\.\*", low):
        errors.append(
            "SELECT * is not allowed - it would expose restricted columns. "
            "Name the columns you need."
        )

    # ---- 6. must aggregate, not enumerate people
    has_agg = bool(re.search(r"\b(count|sum|avg|median|min|max|percentile|stddev)\s*\(", low))
    has_group = " group by " in low
    if not (has_agg or has_group):
        errors.append(
            "Query must aggregate. Row-level extracts of employee, candidate or "
            "survey records are not permitted."
        )

    # ---- 7. bounded result set
    m = re.search(r"\blimit\s+(\d+)", low)
    if not m:
        body = f"{body}\nLIMIT {DEFAULT_LIMIT}"
        rewritten = True
        warnings.append(f"LIMIT {DEFAULT_LIMIT} appended automatically.")
    elif int(m.group(1)) > MAX_LIMIT:
        body = re.sub(r"\blimit\s+\d+", f"LIMIT {MAX_LIMIT}", body, flags=re.IGNORECASE)
        rewritten = True
        warnings.append(f"LIMIT reduced to the maximum of {MAX_LIMIT}.")

    if errors:
        return ValidationResult(False, original, errors, warnings, refs, rewritten)

    # ---- 8. the database must agree it is a valid, planable read
    try:
        warehouse.query(f"EXPLAIN {body}")
    except Exception as exc:                                    # noqa: BLE001
        first = str(exc).strip().splitlines()[0]
        return ValidationResult(False, original,
                                [f"Query failed to plan: {first}"], warnings, refs, rewritten)

    return ValidationResult(True, body, [], warnings, refs, rewritten)
