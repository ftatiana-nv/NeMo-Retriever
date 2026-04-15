import logging
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage
from nemo_retriever.tabular_data.retrieval.text_to_sql.text_to_sql_graph import create_graph
from nemo_retriever.tabular_data.retrieval.text_to_sql.state import AgentPayload, AgentState
from nemo_retriever.tabular_data.retrieval.text_to_sql.prompts import main_system_prompt_template, ONTOLOGY, get_ontology_prompt
from nemo_retriever.tabular_data.retrieval.text_to_sql.utils import _make_llm

logger = logging.getLogger(__name__)

try:
    llm_client = _make_llm()
except ValueError as e:
    logger.error("Failed to initialize LLM client: %s", e)
    llm_client = None

graph = create_graph()
app = graph.compile()


def get_agent_response(payload: AgentPayload):
    now = datetime.now()
    main_system_prompt = main_system_prompt_template.format(
        date=now,
        ontology_prompt=get_ontology_prompt(ONTOLOGY),
        dialect=payload.get("dialect"),
    )
    messages = [
        SystemMessage(content=main_system_prompt),
        HumanMessage(content=payload["question"]),
    ]

    initial_path_state = dict(payload.get("path_state") or {})

    state: AgentState = {
        "llm": llm_client,
        "initial_question": payload["question"],
        "dialect": payload.get("dialect"),
        "connector": payload.get("connector"),
        "messages": messages,
        "decision": "",
        "path_state": initial_path_state,
    }

    final_state = state.copy()
    for step in app.stream(state, config={"recursion_limit": 45}):
        logger.info("--- AGENT STEP ---")
        for node_name, node_output in step.items():
            logger.info("Node: %s", node_name)
            if node_output:
                if "path_state" in node_output:
                    if "path_state" not in final_state:
                        final_state["path_state"] = {}
                    final_state["path_state"].update(node_output["path_state"])
                for key, value in node_output.items():
                    if key != "path_state":
                        final_state[key] = value

    path_state = final_state.get("path_state", {})
    final_response = path_state.get("final_response")
    if final_response is None:
        messages_out = final_state.get("messages", [])
        if messages_out:
            if isinstance(messages_out, dict):
                final_response = messages_out
            elif isinstance(messages_out[-1], dict):
                final_response = messages_out[-1]
            else:
                final_response = str(messages_out[-1])
        else:
            final_response = ""

    if isinstance(final_response, dict):
        answer = final_response
    else:
        answer = {"response": str(final_response)}

    logger.info("Final answer to user:\n%s", answer)
    return answer


__all__ = ["get_agent_response", "app", "graph", "llm_client"]
