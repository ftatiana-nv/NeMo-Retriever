"""Tool: execute_sql_query

Maps to the ``execute_sql_query`` node in the omni_lite LangGraph
(backed by ``SQLExecutionAgent``).

Responsibility:
- Execute ``path_state["final_sql"]`` against the configured database.
- Store query results in ``path_state["execution_result"]``.
- Gracefully handle missing DB connection (returns None result).

Routing: always → calc_respond
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

_MAX_ROWS = 100


@tool
def execute_sql_query(state_path: str) -> str:
    """Execute the final SQL query against the configured database.

    Reads final_sql from state and pg_connection_string from state root.
    Stores the query result (up to 100 rows) in path_state["execution_result"].
    If no connection string is configured, stores None and continues gracefully.
    Always routes to calc_respond.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "decision", "row_count", "error".
    """
    state = load_state(state_path)
    log_node_visit(state, "execute_sql_query")

    path_state = state.get("path_state", {})
    sql = (path_state.get("final_sql") or path_state.get("current_sql") or "").strip()
    pg_conn_str = (state.get("pg_connection_string") or "").strip()

    execution_result = None
    error = ""
    row_count = 0

    if not sql:
        error = "No SQL to execute."
        logger.warning("execute_sql_query: no SQL available")
    elif not pg_conn_str:
        logger.info("execute_sql_query: no pg_connection_string configured, skipping execution")
        error = "No database connection configured."
    else:
        try:
            import psycopg2

            conn = psycopg2.connect(pg_conn_str)
            try:
                with conn.cursor() as cur:
                    cur.execute(sql)
                    cols = [desc[0] for desc in (cur.description or [])]
                    rows = cur.fetchmany(_MAX_ROWS)
                    row_count = len(rows)
                    if cols:
                        execution_result = [dict(zip(cols, row)) for row in rows]
                    else:
                        execution_result = rows
                    logger.info("execute_sql_query: executed successfully, %d rows", row_count)
            finally:
                conn.close()
        except ImportError:
            error = "psycopg2 not installed; skipping SQL execution."
            logger.warning("execute_sql_query: %s", error)
        except Exception as exc:
            error = str(exc)
            logger.warning("execute_sql_query: execution failed: %s", exc)

    path_state["execution_result"] = execution_result
    if error:
        path_state["execution_error"] = error
    state["decision"] = "calc_respond"
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "execute_sql_query",
            "decision": "calc_respond",
            "row_count": row_count,
            "error": error,
        }
    )
