"""Tool: reconstruct_sql

Maps to the ``reconstruct_sql`` node in the omni_lite LangGraph
(backed by ``SQLReconstructionAgent``).

Responsibility:
- Fix the current SQL based on validation error or intent feedback.
- Increment ``path_state["reconstruction_count"]``.
- Update ``path_state["current_sql"]`` with the repaired SQL.

Routing: always → validate_sql_query
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
You are a SQL repair specialist. Given a user question, a broken SQL query,
and error/feedback information, produce a corrected SQL SELECT statement.

Return ONLY a JSON object:
  {"sql_code": "<corrected SQL SELECT>",
   "explanation": "<what you changed and why>"}

Rules:
- The corrected SQL must be syntactically valid.
- Address the specific error or intent feedback provided.
- Keep SELECT as the root clause (no mutations).
- If you cannot fix it, set sql_code to an empty string.
"""


@tool
def reconstruct_sql(state_path: str) -> str:
    """Repair the current SQL based on validation error or intent feedback.

    Uses the LLM to fix the broken SQL. Increments reconstruction_count.
    Always routes to validate_sql_query after completion.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "decision", "sql_code", "explanation".
    """
    state = load_state(state_path)
    log_node_visit(state, "reconstruct_sql")

    path_state = state.get("path_state", {})
    question = (path_state.get("normalized_question") or state.get("initial_question", "")).strip()
    broken_sql = (path_state.get("current_sql") or "").strip()
    validation_error = (path_state.get("validation_error") or "").strip()
    intent_feedback = (path_state.get("intent_feedback") or "").strip()

    # Increment reconstruction counter.
    reconstruction_count = path_state.get("reconstruction_count", 0) + 1
    path_state["reconstruction_count"] = reconstruction_count

    sql_code = ""
    explanation = ""

    try:
        from nemo_retriever.tabular_data.retrieval.generate_sql import _make_llm

        llm = _make_llm()
        context_parts = [f"User question: {question}"]
        if broken_sql:
            context_parts.append(f"Current SQL:\n{broken_sql}")
        if validation_error:
            context_parts.append(f"Validation error: {validation_error}")
        if intent_feedback:
            context_parts.append(f"Intent feedback: {intent_feedback}")
        # Include candidates for reference.
        candidates = path_state.get("prepared_candidates", [])
        if candidates:
            snippets = "\n\n".join(
                f"{i}. {c.get('text', '').strip()}" for i, c in enumerate(candidates[:5], 1) if c.get("text")
            )
            if snippets:
                context_parts.append(f"Reference snippets:\n{snippets}")

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content="\n\n".join(context_parts)),
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response)).strip()

        brace = content.find("{")
        if brace != -1:
            try:
                obj = json.loads(content[brace:])
                sql_code = (obj.get("sql_code") or "").strip()
                explanation = obj.get("explanation", "")
            except json.JSONDecodeError:
                import re

                sql_match = re.search(r"```(?:sql)?\s*\n([\s\S]*?)\n```", content, re.IGNORECASE)
                if sql_match:
                    sql_code = sql_match.group(1).strip()
                    explanation = "Extracted from markdown block."

        logger.info(
            "reconstruct_sql: reconstruction_count=%d sql_len=%d",
            reconstruction_count,
            len(sql_code),
        )
    except Exception as exc:
        logger.error("reconstruct_sql: LLM failed: %s", exc)

    path_state["current_sql"] = sql_code
    path_state["sql_construction_explanation"] = explanation
    # Reset validation/intent feedback for next round.
    path_state["validation_error"] = ""
    path_state["intent_feedback"] = ""
    state["path_state"] = path_state
    state["decision"] = "validate_sql_query"

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "reconstruct_sql",
            "decision": "validate_sql_query",
            "sql_code": sql_code,
            "explanation": explanation,
            "reconstruction_count": reconstruction_count,
        }
    )
