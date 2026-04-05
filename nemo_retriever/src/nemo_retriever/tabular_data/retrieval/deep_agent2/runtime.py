"""Deep Agent 2 runtime — constructs and invokes the Omni SQL pipeline agent.

This module mirrors the role of ``deep_agent/deep_agent_runtime.py`` but for
the **omni_lite pipeline**, replacing LangGraph nodes with Deep Agent tools.

Entry point used by ``generate_sql.get_omni_deep_agent_sql_response``.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Agent cache (one agent per process; re-use across calls)
# ---------------------------------------------------------------------------

_omni_agent: Any = None
_omni_agent_key: str = ""


def _base_dir() -> str:
    """Absolute path to this package directory."""
    return os.path.dirname(os.path.abspath(__file__))


def _skill_dirs(base: str) -> list[str] | None:
    v = os.environ.get("DEEP_AGENT2_LOAD_SKILLS", "1").strip().lower()
    if v in ("0", "false", "no", "off"):
        return None
    skill_path = os.path.join(base, "skills", "omni-flow")
    return [skill_path] if os.path.isdir(skill_path) else None


def _get_agent(force_new: bool = False) -> Any:
    """Return (and cache) the Deep Agent 2 instance."""
    global _omni_agent, _omni_agent_key

    base = _base_dir()
    skill_dirs = _skill_dirs(base)
    cache_key = f"skills={bool(skill_dirs)}"

    if not force_new and _omni_agent is not None and _omni_agent_key == cache_key:
        return _omni_agent

    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    from nemo_retriever.tabular_data.retrieval.generate_sql import _make_llm
    from nemo_retriever.tabular_data.retrieval.deep_agent2.tools import ALL_TOOLS

    llm = _make_llm()

    logger.info(
        "deep_agent2: creating agent (skills=%s, tools=%d)",
        bool(skill_dirs),
        len(ALL_TOOLS),
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt=None,
        memory=[os.path.join(base, "AGENTS.md")],
        skills=skill_dirs,
        tools=ALL_TOOLS,
        subagents=[],
        backend=FilesystemBackend(root_dir=base),
    )

    _omni_agent = agent
    _omni_agent_key = cache_key
    return agent


# ---------------------------------------------------------------------------
# State file helpers
# ---------------------------------------------------------------------------


def _make_state_file(
    question: str,
    pg_connection_string: str = "",
    language: str = "english",
) -> str:
    """Write a fresh state file and return its path."""
    from nemo_retriever.tabular_data.retrieval.deep_agent2.state import (
        init_state,
        save_state,
    )

    run_id = uuid.uuid4().hex[:12]
    state_dir = os.path.join(_base_dir(), "runs")
    os.makedirs(state_dir, exist_ok=True)
    state_path = os.path.join(state_dir, f"state_{run_id}.json")

    state = init_state(
        question=question,
        pg_connection_string=pg_connection_string,
        language=language,
    )
    save_state(state_path, state)
    logger.info("deep_agent2: state file created at %s", state_path)
    return state_path


def _read_final_result(state_path: str) -> dict | None:
    """Read path_state.final_result from the state file."""
    try:
        from nemo_retriever.tabular_data.retrieval.deep_agent2.state import load_state

        state = load_state(state_path)
        return state.get("path_state", {}).get("final_result")
    except Exception as exc:
        logger.warning("deep_agent2: could not read final_result: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Answer extraction from agent messages
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = frozenset({"sql_code", "answer", "result"})


def _extract_json_answer(content: str) -> dict | None:
    """Find the first JSON object with sql_code, answer, result in *content*."""
    if not isinstance(content, str) or not content.strip():
        return None
    text = content.strip()

    # Fast path: entire content is JSON.
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and _REQUIRED_KEYS.issubset(obj.keys()):
            return obj
    except Exception:
        pass

    # Scan for embedded JSON objects.
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict) and _REQUIRED_KEYS.issubset(obj.keys()):
                    return obj
            except json.JSONDecodeError:
                pass
        i += 1
    return None


def _extract_structured_answer(agent_result: dict) -> dict | None:
    """Scan agent messages (most recent first) for the final JSON answer."""
    messages = agent_result.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if not isinstance(content, str):
            continue
        obj = _extract_json_answer(content)
        if obj is not None:
            return obj
    return None


# ---------------------------------------------------------------------------
# User prompt
# ---------------------------------------------------------------------------


def _format_user_prompt(question: str, state_path: str) -> str:
    return (
        "Execute the Omni SQL generation pipeline for the following question.\n\n"
        f"User question: {question.strip()}\n\n"
        f"State file path: {state_path}\n\n"
        "Follow the pipeline steps defined in AGENTS.md exactly:\n"
        "1. retrieve_candidates(state_path)\n"
        "2. extract_action_input(state_path)\n"
        "3. calculation_search(state_path)\n"
        "4. prepare_candidates(state_path)\n"
        "5. construct_sql_from_multiple_snippets(state_path)\n"
        "   → if decision='constructable': validate_sql_query(state_path)\n"
        "   → if decision='unconstructable': unconstructable_sql_response(state_path) → END\n"
        "6. Follow route_sql_validation and route_intent_validation rules from AGENTS.md.\n"
        "7. End with calc_respond(state_path) or unconstructable_sql_response(state_path).\n\n"
        "After the terminal tool call, output ONLY this JSON (no extra text):\n"
        '{"sql_code": "<SQL>", "answer": "<explanation>", "result": <value_or_null>}'
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_omni_pipeline(
    question: str,
    pg_connection_string: str = "",
    language: str = "english",
    max_retries: int = 2,
) -> dict:
    """Run the Omni SQL generation pipeline as a Deep Agent.

    Parameters
    ----------
    question:
        The natural-language question to answer with SQL.
    pg_connection_string:
        Optional PostgreSQL DSN for SQL execution (e.g.
        ``"postgresql://user:pass@host:5432/db"``).
    language:
        Target language for the response (default: ``"english"``).
    max_retries:
        Number of top-level retries if the agent fails entirely.

    Returns
    -------
    dict
        ``{"sql_code": str, "answer": str, "result": Any}``
    """
    agent = _get_agent()
    state_path = _make_state_file(question, pg_connection_string, language)
    prompt = _format_user_prompt(question, state_path)

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})

            # 1) Try parsing the agent's final message for structured JSON.
            parsed = _extract_structured_answer(result)
            if parsed is not None:
                logger.info(
                    "deep_agent2: answer extracted from agent messages (attempt %d)",
                    attempt,
                )
                return {
                    "sql_code": parsed.get("sql_code", ""),
                    "answer": parsed.get("answer", ""),
                    "result": parsed.get("result"),
                }

            # 2) Fall back to reading final_result from the state file.
            final_result = _read_final_result(state_path)
            if final_result:
                logger.info("deep_agent2: answer read from state file (attempt %d)", attempt)
                return {
                    "sql_code": final_result.get("sql_code", ""),
                    "answer": final_result.get("answer", ""),
                    "result": final_result.get("result"),
                }

            # 3) Return raw last message content if available.
            messages = result.get("messages") or []
            if messages:
                last_content = getattr(messages[-1], "content", None)
                if last_content:
                    return {
                        "sql_code": "",
                        "answer": str(last_content),
                        "result": None,
                    }

        except Exception as exc:
            logger.error(
                "deep_agent2: pipeline attempt %d/%d failed: %s",
                attempt,
                max_retries,
                exc,
            )
            last_error = exc

    return {
        "sql_code": "",
        "answer": f"Omni Deep Agent 2 failed after {max_retries} attempt(s): {last_error}",
        "result": None,
    }
