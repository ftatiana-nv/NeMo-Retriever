"""Tool: unconstructable_sql_response

Maps to the ``unconstructable_sql_response`` node in the omni_lite
LangGraph (backed by ``CalculationUnconstructableAgent``).

Responsibility:
- Return a clear failure response when SQL cannot be constructed after
  exhausting all retries.

Routing: → END (terminal node)
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from nemo_retriever.tabular_data.retrieval.deep_agent2.state import (
    load_state,
    log_node_visit,
    save_state,
)

logger = logging.getLogger(__name__)


@tool
def unconstructable_sql_response(state_path: str) -> str:
    """Return a failure response when SQL could not be constructed.

    Called when the pipeline exhausts all SQL construction attempts.
    Records the failure in path_state["final_result"] and terminates.
    This is a terminal node — no further routing is needed.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "decision", "sql_code", "answer", "result".
    """
    state = load_state(state_path)
    log_node_visit(state, "unconstructable_sql_response")

    path_state = state.get("path_state", {})
    question = (path_state.get("normalized_question") or state.get("initial_question", "")).strip()
    attempts = path_state.get("sql_attempts", 0)
    last_sql = (path_state.get("current_sql") or "").strip()
    last_error = (path_state.get("validation_error") or "").strip()

    answer = f"Unable to construct a valid SQL query for the question: '{question}'. " f"Tried {attempts} time(s). " + (
        f"Last error: {last_error}" if last_error else "No valid SQL could be generated."
    )

    final_result = {
        "sql_code": last_sql,
        "answer": answer,
        "result": None,
    }
    path_state["final_result"] = final_result
    state["decision"] = "END"
    state["path_state"] = path_state

    save_state(state_path, state)

    logger.error("unconstructable_sql_response: pipeline terminated after %d attempts", attempts)

    return json.dumps(
        {
            "node": "unconstructable_sql_response",
            "decision": "END",
            "sql_code": last_sql,
            "answer": answer,
            "result": None,
        }
    )
