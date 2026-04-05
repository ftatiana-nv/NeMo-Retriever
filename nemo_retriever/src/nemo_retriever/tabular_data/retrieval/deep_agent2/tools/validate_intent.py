"""Tool: validate_intent

Maps to the ``validate_intent`` node in the omni_lite LangGraph
(backed by ``IntentValidationAgent``).

Responsibility:
- Ask the LLM whether the current SQL actually addresses the user's
  intent (not just syntactic validity).
- Apply the ``route_intent_validation`` routing logic.

Routing (mirrors ``route_intent_validation``):
  "valid_sql"    → format_sql_response
  "invalid_sql"  → reconstruct_sql
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
You are a SQL intent validator. Given a user question and a SQL query,
determine whether the SQL correctly addresses the user's intent.

Return ONLY a JSON object:
  {"intent_valid": true/false,
   "feedback": "<brief explanation of what's wrong or 'Looks good'>"}

Be strict: the SQL must retrieve the correct data (right tables, columns,
filters, aggregations) to actually answer the question. If it does, return
intent_valid=true.
"""


@tool
def validate_intent(state_path: str) -> str:
    """Validate whether the current SQL addresses the user question's intent.

    Uses the LLM to check semantic alignment between the question and SQL.
    Applies route_intent_validation logic to produce a routing decision.

    Routing decisions:
    - "valid_sql": SQL correctly addresses the intent → format_sql_response
    - "invalid_sql": SQL does not address the intent → reconstruct_sql

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "decision", "intent_valid", "feedback".
    """
    state = load_state(state_path)
    log_node_visit(state, "validate_intent")

    path_state = state.get("path_state", {})
    question = (path_state.get("normalized_question") or state.get("initial_question", "")).strip()
    sql = (path_state.get("current_sql") or "").strip()

    intent_valid = True
    feedback = "No validation performed (LLM unavailable)."

    try:
        from nemo_retriever.tabular_data.retrieval.generate_sql import _make_llm

        llm = _make_llm()
        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=(f"User question: {question}\n\n" f"SQL query:\n{sql}")),
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response)).strip()

        brace = content.find("{")
        if brace != -1:
            try:
                obj = json.loads(content[brace:])
                intent_valid = bool(obj.get("intent_valid", True))
                feedback = obj.get("feedback", "")
            except json.JSONDecodeError:
                pass

        logger.info("validate_intent: intent_valid=%s feedback=%r", intent_valid, feedback)
    except Exception as exc:
        logger.warning("validate_intent: LLM call failed: %s", exc)

    # --- Apply route_intent_validation logic ---
    if not intent_valid:
        decision = "invalid_sql"
        path_state["intent_feedback"] = feedback
        # Count this as an additional sql_attempt for the retry counter.
        attempts = path_state.get("sql_attempts", 0)
        logger.info("validate_intent: intent_invalid (attempt %d)", attempts)
    else:
        decision = "valid_sql"
        path_state["intent_feedback"] = feedback

    state["decision"] = decision
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "validate_intent",
            "decision": decision,
            "intent_valid": intent_valid,
            "feedback": feedback,
        }
    )
