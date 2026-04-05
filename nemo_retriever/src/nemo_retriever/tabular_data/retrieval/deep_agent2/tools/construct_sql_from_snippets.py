"""Tool: construct_sql_from_multiple_snippets

Maps to the ``construct_sql_from_multiple_snippets`` node in the
omni_lite LangGraph (backed by ``SQLFromMultipleSnippetsAgent``).

Responsibility:
- Use the LLM together with the prepared SQL snippet candidates to
  construct a SQL query that answers the user question.
- Populate ``path_state["current_sql"]`` and set ``state["decision"]``
  to either ``"constructable"`` or ``"unconstructable"``.

Routing (mirrors ``route_decision``):
  "constructable"   → validate_sql_query
  "unconstructable" → unconstructable_sql_response
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

from nemo_retriever.tabular_data.retrieval.deep_agent2.state import (
    load_state,
    log_node_visit,
    save_state,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a SQL construction assistant. You are given a user question and a list
of relevant SQL snippet candidates retrieved from a knowledge base.

Your task:
1. Choose the most relevant snippet(s) and adapt them to answer the user question.
2. If you can construct a valid SQL query, return a JSON object:
   {"decision": "constructable", "sql_code": "<your SQL>",
    "explanation": "<brief explanation>"}
3. If it is impossible to construct SQL from the given snippets, return:
   {"decision": "unconstructable", "sql_code": "",
    "explanation": "<reason why not possible>"}

Rules:
- Return ONLY a JSON object, no extra text.
- "sql_code" must be a single valid SQL SELECT statement.
- Use column/table names exactly as they appear in the snippets.
- Do not invent table or column names not present in the snippets.
"""


def _format_candidates(candidates: list[dict]) -> str:
    lines = []
    for i, c in enumerate(candidates, 1):
        text = (c.get("text") or "").strip()
        if text:
            entity_tag = f" [entity: {c['entity']}]" if c.get("entity") else ""
            lines.append(f"{i}.{entity_tag}\n{text}")
    return "\n\n".join(lines) if lines else "(no snippets available)"


@tool
def construct_sql_from_multiple_snippets(state_path: str) -> str:
    """Build a SQL query from retrieved SQL snippet candidates using the LLM.

    Uses prepared_candidates from the state to construct SQL. Sets
    path_state["current_sql"] and returns a routing decision:
    "constructable" if SQL was built, "unconstructable" if not possible.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "decision", "sql_code", "explanation".
    """
    state = load_state(state_path)
    log_node_visit(state, "construct_sql_from_multiple_snippets")

    path_state = state.get("path_state", {})
    question = (path_state.get("normalized_question") or state.get("initial_question", "")).strip()
    candidates = path_state.get("prepared_candidates", [])

    decision = "unconstructable"
    sql_code = ""
    explanation = "Could not construct SQL from the available snippets."

    try:
        from nemo_retriever.tabular_data.retrieval.generate_sql import _make_llm

        llm = _make_llm()
        snippets_text = _format_candidates(candidates)
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(f"User question: {question}\n\n" f"SQL snippet candidates:\n{snippets_text}")),
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response)).strip()

        brace = content.find("{")
        if brace != -1:
            try:
                obj = json.loads(content[brace:])
                decision = obj.get("decision", "unconstructable")
                sql_code = (obj.get("sql_code") or "").strip()
                explanation = obj.get("explanation", "")
                if decision == "constructable" and not sql_code:
                    decision = "unconstructable"
                    explanation = "LLM returned constructable but no SQL."
            except json.JSONDecodeError:
                # Try to extract SQL from markdown block as fallback.
                import re

                sql_match = re.search(r"```(?:sql)?\s*\n([\s\S]*?)\n```", content, re.IGNORECASE)
                if sql_match:
                    sql_code = sql_match.group(1).strip()
                    if sql_code.upper().startswith("SELECT"):
                        decision = "constructable"
                        explanation = "Extracted from markdown block."

        logger.info(
            "construct_sql_from_multiple_snippets: decision=%r sql_len=%d",
            decision,
            len(sql_code),
        )
    except Exception as exc:
        logger.error("construct_sql_from_multiple_snippets: LLM failed: %s", exc)

    path_state["current_sql"] = sql_code
    path_state["sql_construction_explanation"] = explanation
    state["decision"] = decision
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "construct_sql_from_multiple_snippets",
            "decision": decision,
            "sql_code": sql_code,
            "explanation": explanation,
        }
    )
