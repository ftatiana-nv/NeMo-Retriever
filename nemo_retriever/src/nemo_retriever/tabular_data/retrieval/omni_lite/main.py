import logging

from nemo_retriever.tabular_data.retrieval.omni_lite.omni_lite_runtime import (
    _SQLCaptureCallback,
    create_omni_lite_agent,
    extract_structured_answer,
    format_omni_lite_user_prompt,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentPayload
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import _make_llm

logger = logging.getLogger(__name__)

try:
    llm_client = _make_llm()
except ValueError as e:
    logger.error("Failed to initialize LLM client: %s", e)
    llm_client = None


def get_agent_response(payload: AgentPayload):
    """Convert a natural-language question to SQL using the OmniLite Deep Agent.

    The public API signature is unchanged from the LangGraph version:
    accepts an ``AgentPayload`` and returns a response dict with at minimum
    ``sql_code``, ``answer``, and ``result`` keys.

    Args:
        payload: Caller-supplied payload.  Required: ``question``.
            Optional: ``history``, ``dialects``, ``db_connector``, ``path_state``.

    Returns:
        dict with keys ``sql_code``, ``answer``, ``result``,
        ``semantic_elements`` (and possibly others from the agent).
    """
    agent, sql_store = create_omni_lite_agent(payload, llm=llm_client)
    sql_capture = _SQLCaptureCallback(sql_store)

    prompt = format_omni_lite_user_prompt(
        question=payload["question"],
        history=payload.get("history"),
        dialects=payload.get("dialects"),
    )

    max_retries = 3
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            result = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config={"callbacks": [sql_capture]},
            )
            parsed = extract_structured_answer(result)
            if parsed is not None:
                logger.info("OmniLite Deep Agent answer (attempt %d): %s", attempt, parsed)
                return parsed

            # Fallback: return raw final message content
            messages = result.get("messages") or []
            final_msg = messages[-1] if messages else None
            raw = getattr(final_msg, "content", None) if final_msg is not None else None
            if raw is not None:
                answer = {"sql_code": "", "answer": str(raw), "result": None, "semantic_elements": []}
                logger.info("OmniLite Deep Agent raw fallback (attempt %d)", attempt)
                return answer

        except Exception as exc:
            logger.error(
                "OmniLite Deep Agent failed (attempt %d/%d): %s",
                attempt,
                max_retries,
                exc,
            )
            last_error = exc

    return {
        "sql_code": "",
        "answer": f"OmniLite agent failed after {max_retries} attempts: {last_error}",
        "result": None,
        "semantic_elements": [],
    }


__all__ = ["get_agent_response", "llm_client"]
