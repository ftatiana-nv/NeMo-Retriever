"""Tool: format_sql_response

Maps to the ``format_sql_response`` node in the omni_lite LangGraph
(backed by ``SQLResponseFormattingAgent``).

Responsibility:
- Produce a polished final SQL and a human-readable explanation.
- Populate ``path_state["final_sql"]`` and ``path_state["final_answer"]``.

Routing: always → execute_sql_query
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
You are a SQL response formatter. Given a validated SQL query and the
original user question, produce a clean, well-formatted final response.

Return ONLY a JSON object:
  {"final_sql": "<cleaned, formatted SQL>",
   "answer": "<1-3 sentence human-readable explanation of what the SQL does>"}

Rules:
- Format the SQL with proper indentation.
- The answer must explain, in plain language, what the query retrieves.
- Do not add any extra keys or prose outside the JSON object.
"""


@tool
def format_sql_response(state_path: str) -> str:
    """Format the validated SQL into a clean final response with explanation.

    Reads current_sql from state, calls the LLM to produce a formatted SQL
    and human-readable answer, and stores both in path_state.
    Always routes to execute_sql_query.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "decision", "final_sql", "answer".
    """
    state = load_state(state_path)
    log_node_visit(state, "format_sql_response")

    path_state = state.get("path_state", {})
    question = (path_state.get("normalized_question") or state.get("initial_question", "")).strip()
    sql = (path_state.get("current_sql") or "").strip()

    final_sql = sql
    answer = ""

    try:
        from nemo_retriever.tabular_data.retrieval.generate_sql import _make_llm

        llm = _make_llm()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(f"User question: {question}\n\n" f"Validated SQL:\n{sql}")),
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response)).strip()

        brace = content.find("{")
        if brace != -1:
            try:
                obj = json.loads(content[brace:])
                final_sql = (obj.get("final_sql") or sql).strip()
                answer = obj.get("answer", "")
            except json.JSONDecodeError:
                pass

        logger.info("format_sql_response: formatted sql_len=%d", len(final_sql))
    except Exception as exc:
        logger.warning("format_sql_response: LLM call failed: %s", exc)

    path_state["final_sql"] = final_sql
    path_state["final_answer"] = answer
    state["decision"] = "execute_sql_query"
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "format_sql_response",
            "decision": "execute_sql_query",
            "final_sql": final_sql,
            "answer": answer,
        }
    )
