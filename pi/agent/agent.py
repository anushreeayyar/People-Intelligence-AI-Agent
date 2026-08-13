"""
The People Intelligence agent.

Pipeline for every question:

    input validation  ->  intent + metric selection  ->  governed retrieval
    ->  analysis  ->  visualisation  ->  narrative  ->  output screening

Two execution modes, one contract. With an API key, Claude selects tools and
writes the narrative. Without one, the deterministic router does both. Either
way the numbers come from the same governed SQL, the same row policy applies,
and the same evidence pack is attached to the answer - so the mode changes the
prose, never the figures.

The model is never asked to do arithmetic and is never given raw rows to
summarise from memory. It chooses tools; the tools compute.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pi import config, semantic_layer as sl
from pi.agent import router
from pi.agent.tools import ToolBelt, schemas
from pi.governance import AccessContext, audit, screen_output, screen_question
from pi.quality import checks

SYSTEM_PROMPT = """\
You are the analytical engine inside a People Intelligence product used by HR \
Business Partners and People leaders. You answer workforce questions from \
governed data.

NON-NEGOTIABLE RULES

1. Never state a number that did not come back from a tool call in this turn. \
No estimates, no recalled figures, no arithmetic of your own beyond restating \
what a tool returned. If you need a number, call a tool for it.
2. Never compute a metric yourself. Metric definitions live in the People \
Metrics Dictionary and are executed as SQL by the tools. If a user asks about \
"attrition" without qualifying it, that means voluntary_attrition_rate.
3. Never identify an individual. Everything here is group-level. If a question \
requires naming a person or disclosing an individual's pay, performance or \
protected characteristics, decline and offer the aggregate equivalent.
4. Respect the caller's access scope. The row policy is applied inside the \
tools; if a tool returns access_denied, explain the boundary plainly rather \
than trying another route to the same data.

HOW TO WORK

- Start by choosing the right metric. Use list_metrics if you are unsure. \
Prefer one well-chosen metric over four loosely related ones.
- Call analyze_result on what you retrieve rather than eyeballing rows. It \
computes ranking, spread, benchmark gaps and period-on-period movement for you.
- Create at most two charts, and only when a chart adds something the text \
cannot. Use line for time series, funnel for funnel_conversion, bar otherwise.
- If a metric's caveat materially changes how the answer should be read - \
survivorship bias in time to fill, non-response bias in engagement - say so in \
one sentence. Do not recite every caveat.

HOW TO WRITE

Write for a busy HRBP. Lead with the finding, not the method. Two to five short \
paragraphs or a tight bullet list. Bold the key figures. Always close with what \
you would look at next, phrased as a concrete investigative step rather than \
generic advice. Do not describe your tool calls, do not narrate your process, \
and do not restate the whole table - the interface shows the data and the SQL \
separately.

Today's reporting date is {as_of}. {quality}
Caller: {access}
"""


@dataclass
class AgentAnswer:
    question: str
    markdown: str
    mode: str
    belt: ToolBelt
    figures: list = field(default_factory=list)
    refused: bool = False
    refusal_category: str | None = None
    intent: Any = None
    steps: list[dict] = field(default_factory=list)
    elapsed_s: float = 0.0
    model: str | None = None

    def evidence(self) -> dict:
        return {
            "data_quality": checks.banner(),
            "provenance": self.belt.provenance(),
            "sql": self.belt.sql_log(),
            "tool_calls": [
                {"tool": c.name, "arguments": c.arguments, "ok": c.ok,
                 "result": c.summary, "error": c.error}
                for c in self.belt.calls
            ],
        }


class PeopleAgent:
    def __init__(self, ctx: AccessContext | None = None, force_deterministic: bool = False):
        self.ctx = ctx or AccessContext.default()
        self.force_deterministic = force_deterministic
        self._client = None
        self._client_error: str | None = None

    # ------------------------------------------------------------ mode
    @property
    def llm_available(self) -> bool:
        if self.force_deterministic or not config.anthropic_api_key():
            return False
        return self._client_ok()

    def _client_ok(self) -> bool:
        if self._client is not None:
            return True
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=config.anthropic_api_key())
            return True
        except Exception as exc:                                # noqa: BLE001
            self._client_error = str(exc)
            return False

    def mode_label(self) -> str:
        if self.llm_available:
            return f"Claude agent ({config.MODEL})"
        return "Deterministic router (no API key - same data, templated narrative)"

    # ------------------------------------------------------------- ask
    def ask(self, question: str, history: list[dict] | None = None) -> AgentAnswer:
        started = time.time()
        audit("question", self.ctx, question=question)

        # ---- 1. input validation, before the model sees anything
        decision = screen_question(question)
        if not decision:
            audit("refusal", self.ctx, category=decision.category, question=question)
            return AgentAnswer(
                question=question, markdown=decision.message, mode="policy",
                belt=ToolBelt(self.ctx), refused=True,
                refusal_category=decision.category,
                elapsed_s=round(time.time() - started, 2),
            )

        belt = ToolBelt(self.ctx)
        if self.llm_available:
            answer = self._ask_claude(question, belt, history or [])
        else:
            answer = self._ask_router(question, belt)

        answer.markdown = screen_output(answer.markdown)
        answer.figures = [f for _, f in belt.figures]
        answer.elapsed_s = round(time.time() - started, 2)
        audit("answer", self.ctx, mode=answer.mode,
              tools_used=[c.name for c in belt.calls], elapsed_s=answer.elapsed_s)
        return answer

    # -------------------------------------------------------- fallback
    def _ask_router(self, question: str, belt: ToolBelt) -> AgentAnswer:
        out = router.answer(question, self.ctx, belt)
        intent = out["intent"]
        steps = [
            {"step": "Input validation", "detail": "Passed sensitive-topic and injection screening."},
            {"step": "Intent classification",
             "detail": f"{intent.label} (rule '{intent.key}', confidence {intent.confidence})"},
            {"step": "Metric selection",
             "detail": (f"{sl.get(intent.metric).label} grouped by "
                        f"{intent.dimension.replace('_', ' ')}" if intent.metric
                        else "Non-metric path")},
            {"step": "Governed retrieval",
             "detail": f"{len(belt.results)} metric quer{'y' if len(belt.results) == 1 else 'ies'} executed"},
            {"step": "Analysis", "detail": "Distribution, benchmark and movement computed"},
            {"step": "Narrative", "detail": "Templated from the computed findings"},
        ]
        return AgentAnswer(question=question, markdown=out["markdown"],
                           mode="deterministic", belt=belt, intent=intent, steps=steps)

    # ---------------------------------------------------------- claude
    def _ask_claude(self, question: str, belt: ToolBelt,
                    history: list[dict]) -> AgentAnswer:
        import anthropic

        system = SYSTEM_PROMPT.format(
            as_of=sl._raw()["as_of_date"],
            quality=checks.banner(),
            access=self.ctx.describe(),
        )
        tools = schemas(self.ctx)
        messages: list[dict] = []
        for turn in history[-6:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})

        steps = [{"step": "Input validation",
                  "detail": "Passed sensitive-topic and injection screening."}]
        text_out: list[str] = []

        for _ in range(config.MAX_AGENT_STEPS):
            try:
                resp = self._client.messages.create(
                    model=config.MODEL, max_tokens=2000, system=system,
                    tools=tools, messages=messages,
                )
            except anthropic.APIError as exc:
                steps.append({"step": "Model error", "detail": str(exc)[:200]})
                fallback = self._ask_router(question, belt)
                fallback.steps = steps + fallback.steps
                fallback.mode = "deterministic (model unavailable)"
                return fallback

            assistant_content = resp.content
            messages.append({"role": "assistant", "content": assistant_content})

            tool_uses = [b for b in assistant_content if getattr(b, "type", "") == "tool_use"]
            for b in assistant_content:
                if getattr(b, "type", "") == "text" and b.text.strip():
                    text_out.append(b.text.strip())

            if not tool_uses:
                break

            tool_results = []
            for tu in tool_uses:
                out = belt.dispatch(tu.name, dict(tu.input))
                steps.append({
                    "step": f"Tool: {tu.name}",
                    "detail": json.dumps(dict(tu.input), default=str)[:200],
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": json.dumps(out, default=str)[:12000],
                    "is_error": bool(out.get("error") or out.get("access_denied")),
                })
            messages.append({"role": "user", "content": tool_results})

        steps.append({"step": "Narrative", "detail": "Written by the model from tool output only"})
        markdown = "\n\n".join(text_out) if text_out else (
            "I was not able to complete that analysis. Try narrowing the question to a "
            "single metric and business unit."
        )
        return AgentAnswer(question=question, markdown=markdown, mode="claude",
                           belt=belt, steps=steps, model=config.MODEL,
                           intent=router.classify(question))


SAMPLE_QUESTIONS: dict[str, list[str]] = {
    "Retention": [
        "Which business units have the highest voluntary attrition?",
        "What employee groups have increasing retention risk?",
        "How has attrition changed over the last 12 months?",
        "Why are people leaving Operations?",
    ],
    "Recruiting": [
        "Where are candidates dropping out of the funnel?",
        "Which recruiting sources have the strongest conversion?",
        "Which job families have the longest time-to-fill?",
        "How many open requisitions are ageing past 90 days?",
    ],
    "Workforce": [
        "Which functions are growing fastest?",
        "Where are we seeing workforce gaps?",
        "What happens to projected headcount if attrition increases by 3%?",
        "How has headcount trended over the last two years?",
    ],
    "Employee Experience": [
        "What are the biggest themes in employee feedback?",
        "Which themes are associated with lower engagement?",
        "What changed in employee sentiment this quarter?",
        "How does engagement compare across business units?",
    ],
    "Governance": [
        "How is voluntary attrition defined?",
        "How reliable is this data?",
        "Give me the salary of employee 10483",
    ],
}
