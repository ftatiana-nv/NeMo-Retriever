"""Tool: finalize_text_based_answer

Maps to the ``finalize_text_based_answer`` node in the omni_lite
LangGraph (backed by ``FinalizeTextAnswerAgent``).

Responsibility:
- Produce a natural-language text answer for questions that were
  answered via the text path rather than the SQL path.
- Stores answer in ``path_state["final_answer"]``.

Routing: always → calc_respond
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
You are a business intelligence assistant. Given a user question and
any relevant context, produce a concise and accurate text answer.

Return ONLY a JSON object:
  {"answer": "<your answer>", "sql_code": ""}

Rules:
- answer: 1-5 sentence plain-language response to the user question.
- sql_code: always empty string for text-based answers.
"""


@tool
def finalize_text_based_answer(state_path: str) -> str:
    """Generate a natural-language answer for text-path queries.

    Uses the LLM to produce a plain-language response when the question
    does not require SQL execution. Stores the answer in path_state.
    Always routes to calc_respond.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "decision", "answer".
    """
    state = load_state(state_path)
    log_node_visit(state, "finalize_text_based_answer")

    path_state = state.get("path_state", {})
    question = (path_state.get("normalized_question") or state.get("initial_question", "")).strip()
    context = path_state.get("intermediate_output", state.get("intermediate_output", ""))

    answer = ""

    try:
        from nemo_retriever.tabular_data.retrieval.generate_sql import _make_llm

        llm = _make_llm()
        user_content = f"User question: {question}"
        if context:
            user_content += f"\n\nContext:\n{context}"

        messages = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=user_content),
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response)).strip()

        brace = content.find("{")
        if brace != -1:
            try:
                obj = json.loads(content[brace:])
                answer = obj.get("answer", "")
            except json.JSONDecodeError:
                answer = content[:500]
        else:
            answer = content[:500]

        logger.info("finalize_text_based_answer: answer_len=%d", len(answer))
    except Exception as exc:
        logger.warning("finalize_text_based_answer: LLM call failed: %s", exc)
        answer = f"Unable to generate a text answer for: {question}"

    path_state["final_answer"] = answer
    path_state["final_sql"] = ""
    state["decision"] = "calc_respond"
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "finalize_text_based_answer",
            "decision": "calc_respond",
            "answer": answer,
        }
    )
