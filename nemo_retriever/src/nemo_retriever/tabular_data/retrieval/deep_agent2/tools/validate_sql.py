"""Tool: validate_sql_query

Maps to the ``validate_sql_query`` node in the omni_lite LangGraph
(backed by ``SQLValidationAgent``).

Responsibility:
- Validate the SQL in ``path_state["current_sql"]`` for syntactic
  correctness.
- Track attempt count in ``path_state["sql_attempts"]``.
- Apply the ``route_sql_validation`` routing logic and return the
  appropriate routing decision.

Routing (mirrors ``route_sql_validation``):
  "valid_sql"              → validate_intent
  "skip_intent_validation" → format_sql_response   (if reconstruction_count > 5)
  "invalid_sql"            → reconstruct_sql       (attempts < 8)
  "fallback"               → construct_sql_not_from_snippets  (attempts == 4)
  "unconstructable"        → unconstructable_sql_response     (attempts == 8)
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.tools import tool

from nemo_retriever.tabular_data.retrieval.deep_agent2.state import (
    load_state,
    log_node_visit,
    save_state,
)

logger = logging.getLogger(__name__)

# Keywords that must appear in a valid SELECT statement.
_SELECT_RE = re.compile(r"\bSELECT\b", re.IGNORECASE)
_FROM_RE = re.compile(r"\bFROM\b", re.IGNORECASE)
# Dangerous mutation keywords.
_DANGEROUS_RE = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE)\b", re.IGNORECASE)

_VALIDATION_SYSTEM = """\
You are a strict SQL validator. Given a SQL query, determine if it is
syntactically valid and safe (SELECT only, no mutations).

Return ONLY a JSON object:
  {"valid": true/false, "error": "<error message or empty string>"}
"""


def _quick_validate(sql: str) -> tuple[bool, str]:
    """Fast heuristic validation before calling the LLM."""
    sql = (sql or "").strip()
    if not sql:
        return False, "SQL is empty"
    if _DANGEROUS_RE.search(sql):
        return False, "SQL contains forbidden mutation keyword"
    if not _SELECT_RE.search(sql):
        return False, "SQL does not contain SELECT"
    if not _FROM_RE.search(sql):
        return False, "SQL does not contain FROM"
    return True, ""


@tool
def validate_sql_query(state_path: str) -> str:
    """Validate the current SQL query and return a routing decision.

    Reads current_sql from the state, validates it heuristically and via LLM,
    then applies the route_sql_validation logic to determine the next step.

    Routing decisions:
    - "valid_sql": SQL is valid, proceed to validate_intent
    - "skip_intent_validation": valid SQL but reconstruction_count > 5, skip to format_sql_response
    - "invalid_sql": invalid SQL, retry with reconstruct_sql (attempts < 8)
    - "fallback": 4th failed attempt, try construct_sql_not_from_snippets
    - "unconstructable": 8th failed attempt, give up

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "decision", "valid", "error".
    """
    state = load_state(state_path)
    log_node_visit(state, "validate_sql_query")

    path_state = state.get("path_state", {})
    sql = (path_state.get("current_sql") or "").strip()

    # --- Heuristic pass ---
    valid, error = _quick_validate(sql)

    # --- LLM pass (only if heuristic passes, to avoid wasting calls) ---
    if valid:
        try:
            from nemo_retriever.tabular_data.retrieval.generate_sql import _make_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            llm = _make_llm()
            messages = [
                SystemMessage(content=_VALIDATION_SYSTEM),
                HumanMessage(content=f"SQL to validate:\n{sql}"),
            ]
            response = llm.invoke(messages)
            content = getattr(response, "content", str(response)).strip()
            brace = content.find("{")
            if brace != -1:
                try:
                    obj = json.loads(content[brace:])
                    valid = bool(obj.get("valid", True))
                    error = obj.get("error", "")
                except json.JSONDecodeError:
                    pass
        except Exception as exc:
            logger.warning("validate_sql_query: LLM validation failed: %s", exc)
            # Keep heuristic result on LLM failure.

    # --- Apply route_sql_validation logic ---
    if not valid:
        attempts = path_state.get("sql_attempts", 0)
        path_state["sql_attempts"] = attempts + 1
        path_state["validation_error"] = error
        logger.info("validate_sql_query: invalid_sql (attempt %d)", attempts + 1)

        if attempts == 4:
            decision = "fallback"
            logger.info("⚠️ Fallback: try constructing from tables")
        elif attempts < 8:
            decision = "invalid_sql"
        else:
            decision = "unconstructable"
            logger.error("❌ SQL construction failed after 8 attempts")
    else:
        path_state["validation_error"] = ""
        reconstruction_count = path_state.get("reconstruction_count", 0)
        if reconstruction_count > 5:
            decision = "skip_intent_validation"
            logger.info("⚠️ Skipping intent validation after %d reconstructions", reconstruction_count)
        else:
            decision = "valid_sql"
        logger.info("validate_sql_query: %s", decision)

    state["decision"] = decision
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "validate_sql_query",
            "decision": decision,
            "valid": valid,
            "error": error,
            "sql_attempts": path_state.get("sql_attempts", 0),
        }
    )
