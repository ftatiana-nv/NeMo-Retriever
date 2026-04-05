"""Tool: calc_respond

Maps to the ``calc_respond`` node in the omni_lite LangGraph
(backed by ``CalculationResponseAgent``).

Responsibility:
- Compile the final structured answer from the pipeline results.
- Combines final_sql, final_answer, and execution_result into the
  output dict stored in ``path_state["final_result"]``.

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
def calc_respond(state_path: str) -> str:
    """Compile and return the final SQL answer.

    Reads final_sql, final_answer, and execution_result from the state to
    assemble the terminal output. Stores the result in path_state["final_result"].
    This is a terminal node — no further routing is needed.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "decision", "sql_code", "answer", "result".
    """
    state = load_state(state_path)
    log_node_visit(state, "calc_respond")

    path_state = state.get("path_state", {})
    sql_code = (path_state.get("final_sql") or path_state.get("current_sql") or "").strip()
    answer = (path_state.get("final_answer") or "").strip()
    result = path_state.get("execution_result")

    # If no answer text yet, generate a minimal one.
    if not answer and sql_code:
        answer = f"The SQL query to answer the question has been constructed: {sql_code[:120]}..."
    elif not answer:
        answer = "Could not construct a valid SQL query for the given question."

    final_result = {
        "sql_code": sql_code,
        "answer": answer,
        "result": result,
    }
    path_state["final_result"] = final_result
    state["decision"] = "END"
    state["path_state"] = path_state

    save_state(state_path, state)

    logger.info("calc_respond: pipeline complete, sql_len=%d", len(sql_code))

    return json.dumps(
        {
            "node": "calc_respond",
            "decision": "END",
            "sql_code": sql_code,
            "answer": answer,
            "result": result,
        }
    )
