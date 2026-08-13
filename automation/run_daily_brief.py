#!/usr/bin/env python3
"""
Headless daily brief runner.

Designed to be driven by anything: cron, GitHub Actions, or the n8n workflow in
automation/n8n_workflow.json. Writes JSON and Markdown to data/briefs/, prints
the JSON to stdout so an orchestrator can pick it up, and exits non-zero only on
failure - never because the brief happened to be quiet.

    python automation/run_daily_brief.py                 # write + print
    python automation/run_daily_brief.py --format md     # markdown to stdout
    python automation/run_daily_brief.py --min-severity medium
    python automation/run_daily_brief.py --webhook https://hooks.example/…
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pi.brief import daily_brief                      # noqa: E402
from pi.governance import AccessContext               # noqa: E402

RANK = {"low": 2, "medium": 1, "high": 0}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the People Intelligence Daily Brief.")
    ap.add_argument("--format", choices=["json", "md"], default="json")
    ap.add_argument("--min-severity", choices=["low", "medium", "high"], default="low")
    ap.add_argument("--max-signals", type=int, default=6)
    ap.add_argument("--webhook", help="POST the brief JSON to this URL (Slack, Teams, n8n).")
    ap.add_argument("--no-save", action="store_true", help="Print only; do not write files.")
    args = ap.parse_args()

    ctx = AccessContext(role_key="hr_leader", persona="Enterprise", user="automation")
    brief = daily_brief.build_brief(ctx, max_signals=args.max_signals)
    brief["signals"] = [s for s in brief["signals"]
                        if RANK[s["severity"]] <= RANK[args.min_severity]]
    brief["signal_count"] = len(brief["signals"])
    if not brief["signals"]:
        brief["headline"] = ("No workforce signals cleared the "
                             f"{args.min_severity} materiality threshold")

    if not args.no_save:
        json_path, md_path = daily_brief.save(brief)
        print(f"# written: {json_path.name}, {md_path.name}", file=sys.stderr)

    if args.webhook:
        payload = json.dumps({
            "text": daily_brief.to_markdown(brief),
            "brief": brief,
        }).encode()
        req = urllib.request.Request(
            args.webhook, data=payload,
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                print(f"# webhook {resp.status}", file=sys.stderr)
        except (urllib.error.URLError, TimeoutError) as exc:
            print(f"# webhook failed: {exc}", file=sys.stderr)
            return 1

    if args.format == "md":
        print(daily_brief.to_markdown(brief))
    else:
        print(json.dumps(brief, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
