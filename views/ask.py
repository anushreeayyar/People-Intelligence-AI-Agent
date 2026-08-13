"""Ask People Intelligence - the agent surface."""
from __future__ import annotations

import streamlit as st

from pi import config, ui
from pi.agent.agent import SAMPLE_QUESTIONS, PeopleAgent

ctx = st.session_state.ctx

ui.page_header(
    "Ask People Intelligence",
    "Ask a workforce question in plain language. The agent selects a metric from the "
    "dictionary, executes governed SQL, analyses the result and explains it. It cannot "
    "state a number that did not come back from a query.",
)

# ------------------------------------------------------------------ engine
agent = PeopleAgent(ctx, force_deterministic=st.session_state.get("force_det", False))

top = st.columns([3, 1])
with top[0]:
    mode = agent.mode_label()
    pill = "good" if agent.llm_available else "warn"
    st.markdown(
        f"<span class='pill {pill}'>{mode}</span>"
        f"<span class='pill'>{ctx.role.label}</span>"
        f"<span class='pill'>min group {ctx.role.min_group_size}</span>"
        + ("<span class='pill'>ad-hoc SQL enabled</span>"
           if ctx.role.can_run_adhoc_sql else ""),
        unsafe_allow_html=True,
    )
with top[1]:
    st.toggle("Force deterministic mode", key="force_det",
              help="Answer with the rules-based router even if an API key is present. "
                   "Useful for showing that the numbers are identical either way.")

if not config.anthropic_api_key():
    st.caption(
        "No `ANTHROPIC_API_KEY` is set, so the deterministic router is answering. "
        "Same tools, same governed SQL, same evidence pack — the narrative is "
        "templated rather than written. Set the key to hand tool selection and "
        "narration to Claude."
    )

ui.scope_notice(ctx)

# ------------------------------------------------------------- suggestions
st.markdown("#### Try one of these")
tabs = st.tabs(list(SAMPLE_QUESTIONS))
pending = None
for tab, (domain, questions) in zip(tabs, SAMPLE_QUESTIONS.items()):
    with tab:
        cols = st.columns(2)
        for i, q in enumerate(questions):
            if cols[i % 2].button(q, key=f"s-{domain}-{i}", use_container_width=True):
                pending = q

# ------------------------------------------------------------------- chat
if "chat" not in st.session_state:
    st.session_state.chat = []

for turn in st.session_state.chat:
    with st.chat_message(turn["role"]):
        if turn["role"] == "user":
            st.markdown(turn["content"])
            continue
        ans = turn["answer"]
        if ans.refused:
            st.markdown(f"<div class='refusal'>🔒 {ans.markdown}</div>",
                        unsafe_allow_html=True)
            st.caption(f"Refusal category: `{ans.refusal_category}` · logged to the "
                       "audit trail · no query was executed.")
            continue

        st.markdown(ans.markdown)
        for fig in ans.figures:
            st.plotly_chart(fig, use_container_width=True,
                            key=f"fig-{turn['idx']}-{id(fig)}")

        ev = ans.evidence()
        with st.expander("Explain my answer"):
            t1, t2, t3, t4 = st.tabs(["How it was calculated", "Data & lineage",
                                      "Query executed", "Agent trace"])
            with t1:
                for p in ev["provenance"]:
                    st.markdown(f"**{p['metric']}**")
                    st.markdown(f"- Definition: {p['definition']}")
                    st.markdown(f"- Formula: `{p['formula']}`")
                    st.markdown(f"- Grain: {p['grain']} · dimension: "
                                f"`{p['dimension']}` · filters: `{p['filters']}`")
                    if p["caveats"]:
                        st.markdown("- Caveats: " + " ".join(p["caveats"]))
                if not ev["provenance"]:
                    st.caption("No metric query was needed for this answer.")
            with t2:
                for p in ev["provenance"]:
                    st.markdown(f"**{p['metric']}** — source {p['source_system']}, "
                                f"objects `{', '.join(p['source_objects'])}`, owner "
                                f"{p['owner']}, refreshed {p['refresh'].lower()}, "
                                f"{p['rows_returned']} rows returned"
                                + (f", {p['suppressed_groups']} group(s) suppressed"
                                   if p["suppressed_groups"] else ""))
                    st.markdown(f"- Access policy: {p['row_policy']}")
                st.info(ev["data_quality"], icon="✓")
            with t3:
                if ev["sql"]:
                    for s in ev["sql"]:
                        st.markdown(f"**{s['metric']}** (`{s['result_id']}`)")
                        st.code(s["sql"], language="sql")
                else:
                    st.caption("No SQL was executed for this answer.")
            with t4:
                st.markdown(f"Mode: `{ans.mode}`"
                            + (f" · model `{ans.model}`" if ans.model else "")
                            + f" · {ans.elapsed_s}s")
                if ans.intent is not None:
                    st.markdown(f"Intent: **{ans.intent.label}** "
                                f"(`{ans.intent.key}`, confidence {ans.intent.confidence})")
                for s in ans.steps:
                    st.markdown(f"- **{s['step']}** — {s['detail']}")
                st.markdown("**Tool calls**")
                st.dataframe(
                    [{"tool": c["tool"], "arguments": str(c["arguments"])[:90],
                      "ok": c["ok"], "result": c["result"][:120]}
                     for c in ev["tool_calls"]] or [{"tool": "none", "arguments": "",
                                                     "ok": True, "result": ""}],
                    use_container_width=True, hide_index=True)

        for rid, df in ans.belt.frames.items():
            if df is None or df.empty:
                continue
            with st.expander(f"View data · {rid} ({len(df)} rows)"):
                ui.dataframe(df.head(200))
                ui.download(df, f"{rid}.csv", key=f"dl-{turn['idx']}-{rid}")

typed = st.chat_input("Ask a workforce question…")
question = typed or pending

if question:
    st.session_state.chat.append({"role": "user", "content": question,
                                  "idx": len(st.session_state.chat)})
    history = [{"role": t["role"],
                "content": t["content"] if t["role"] == "user" else t["answer"].markdown}
               for t in st.session_state.chat[:-1]]
    with st.spinner("Selecting metric, validating and querying…"):
        answer = agent.ask(question, history)
    st.session_state.chat.append({"role": "assistant", "content": answer.markdown,
                                  "answer": answer, "idx": len(st.session_state.chat)})
    st.rerun()

if st.session_state.chat:
    if st.button("Clear conversation"):
        st.session_state.chat = []
        st.rerun()
