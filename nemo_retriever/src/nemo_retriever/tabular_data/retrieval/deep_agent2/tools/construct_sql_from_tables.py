"""Tool: construct_sql_not_from_snippets

Maps to the ``construct_sql_not_from_snippets`` node in the omni_lite
LangGraph (backed by ``SQLFromTablesAgent``).

This is the **fallback** path reached after 4 failed SQL attempts from
snippets.  Instead of using pre-retrieved snippets it asks the LLM to
write SQL purely from its knowledge of the question and any schema
information available.

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
You are a SQL construction assistant operating in fallback mode.
Previous attempts using SQL snippet candidates have failed.

Your task:
- Write a SQL SELECT query that best answers the user question using your
  general knowledge of the expected database structure and common SQL patterns.
- Return ONLY a JSON object:
  {"sql_code": "<your SQL SELECT>", "explanation": "<brief explanation>"}
- sql_code must be a syntactically valid SQL SELECT statement.
- If you truly cannot produce anything, set sql_code to an empty string.
"""


@tool
def construct_sql_not_from_snippets(state_path: str) -> str:
    """Fallback: construct SQL from general knowledge when snippet-based attempts failed.

    This tool is called after 4 failed attempts using construct_sql_from_multiple_snippets.
    It asks the LLM to construct SQL without relying on retrieved snippets.
    Always routes to validate_sql_query after completion.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "decision", "sql_code", "explanation".
    """
    state = load_state(state_path)
    log_node_visit(state, "construct_sql_not_from_snippets")

    path_state = state.get("path_state", {})
    question = (path_state.get("normalized_question") or state.get("initial_question", "")).strip()
    validation_error = path_state.get("validation_error", "")
    prev_sql = path_state.get("current_sql", "")

    sql_code = ""
    explanation = ""

    try:
        from nemo_retriever.tabular_data.retrieval.generate_sql import _make_llm

        llm = _make_llm()
        context_parts = [f"User question: {question}"]
        if prev_sql:
            context_parts.append(f"Previous failed SQL:\n{prev_sql}")
        if validation_error:
            context_parts.append(f"Validation error: {validation_error}")

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
                    explanation = "Extracted from markdown fallback."

        logger.info("construct_sql_not_from_snippets: sql_len=%d", len(sql_code))
    except Exception as exc:
        logger.error("construct_sql_not_from_snippets: LLM failed: %s", exc)

    path_state["current_sql"] = sql_code
    path_state["sql_construction_explanation"] = explanation
    # routing decision: always go to validate_sql_query
    decision = "validate_sql_query"
    state["decision"] = decision
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "construct_sql_not_from_snippets",
            "decision": decision,
            "sql_code": sql_code,
            "explanation": explanation,
        }
    )
