"""Tool: extract_action_input

Maps to the ``extract_action_input`` node in the omni_lite LangGraph
(backed by ``ActionInputExtractionAgent``).

Responsibility:
- Use the LLM to extract entities/concepts and a value-stripped query
  from the user question.
- Populate:
    path_state["normalized_question"]
    path_state["query_no_values"]
    path_state["entities_and_concepts"]
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

_EXTRACTION_SYSTEM = """\
You are a precise entity extractor for SQL pipeline preparation.
Return ONLY a JSON object with these two keys:
  "required_entity_name": list of primary business entities from the question
    (e.g. ["Customer", "Order"]). Ignore dates, numbers, constants.
  "query_no_values": the question with all specific values removed
    (dates, numbers, names), keeping the intent intact.

Example:
  Input: "What is total revenue for Q1 2023 from Customer ACME?"
  Output: {"required_entity_name": ["revenue", "Customer"],
           "query_no_values": "What is total revenue from Customer?"}

Return only the JSON object, nothing else.
"""


@tool
def extract_action_input(state_path: str) -> str:
    """Extract entities and value-stripped query from the user question using the LLM.

    Reads the question from the state file, invokes the LLM to identify key
    entities and produce a query without specific values, then writes results
    to path_state.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "entities", "query_no_values", "decision".
    """
    state = load_state(state_path)
    log_node_visit(state, "extract_action_input")

    question = (state.get("path_state", {}).get("normalized_question") or state.get("initial_question", "")).strip()

    path_state = state.get("path_state", {})
    entities: list[str] = []
    query_no_values = question

    try:
        from nemo_retriever.tabular_data.retrieval.generate_sql import _make_llm

        llm = _make_llm()
        messages = [
            SystemMessage(content=_EXTRACTION_SYSTEM),
            HumanMessage(content=f"User question: {question}"),
        ]
        response = llm.invoke(messages)
        content = getattr(response, "content", str(response)).strip()

        # Parse JSON from the LLM response.
        import re

        brace = content.find("{")
        if brace != -1:
            try:
                obj = json.loads(content[brace:])
                entities = obj.get("required_entity_name") or []
                query_no_values = obj.get("query_no_values") or question
            except json.JSONDecodeError:
                # Fallback: try extracting with regex
                names_match = re.search(r'"required_entity_name"\s*:\s*(\[[^\]]*\])', content)
                if names_match:
                    try:
                        entities = json.loads(names_match.group(1))
                    except Exception:
                        pass
                qnv_match = re.search(r'"query_no_values"\s*:\s*"([^"]*)"', content)
                if qnv_match:
                    query_no_values = qnv_match.group(1)

        logger.info(
            "extract_action_input: entities=%s query_no_values=%r",
            entities,
            query_no_values,
        )
    except Exception as exc:
        logger.warning("extract_action_input: LLM call failed: %s", exc)

    path_state["normalized_question"] = question
    path_state["query_no_values"] = query_no_values
    path_state["entities_and_concepts"] = entities
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "extract_action_input",
            "entities": entities,
            "query_no_values": query_no_values,
            "decision": "done",
        }
    )
