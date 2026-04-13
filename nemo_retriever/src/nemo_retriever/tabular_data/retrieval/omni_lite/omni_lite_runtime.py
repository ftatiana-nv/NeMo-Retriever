"""
OmniLite Deep Agent runtime.

Creates a Deep Agent equipped with domain-specific tools (entity extraction,
semantic retrieval, SQL validation, SQL execution) and omni_lite skills/AGENTS.md.

Usage
-----
::

    from nemo_retriever.tabular_data.retrieval.omni_lite.omni_lite_runtime import (
        create_omni_lite_agent,
        extract_structured_answer,
    )

    agent = create_omni_lite_agent(payload, llm)
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    answer = extract_structured_answer(result)
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain_core.callbacks import BaseCallbackHandler

from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentPayload
from nemo_retriever.tabular_data.retrieval.omni_lite.tools import build_omni_lite_tools
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import _make_llm

logger = logging.getLogger(__name__)

_SQL_BLOCK_SENTINEL = "__SQL_FROM_MESSAGE__"
_SQL_BLOCK_START = "###SQL_START###"
_SQL_BLOCK_END = "###SQL_END###"
_SQL_BLOCK_RE = re.compile(
    r"###SQL_START###\s*(.*?)\s*###SQL_END###",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# SQL capture callback
# ---------------------------------------------------------------------------


class _SQLCaptureCallback(BaseCallbackHandler):
    """LangChain callback that captures AI messages containing SQL delimiters.

    After the LLM generates a response that includes a ``###SQL_START###``
    block, this callback appends the raw content to *store* before the tool
    node runs — ensuring ``execute_sql`` can read the full SQL via
    ``extract_sql_from_store``.
    """

    def __init__(self, store: list[str]) -> None:
        super().__init__()
        self._store = store

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: ANN401
        """Capture any AI response that contains the SQL delimiter block."""
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    # ChatGeneration has .message.content; plain Generation has .text
                    content = getattr(getattr(gen, "message", None), "content", None)
                    if content is None:
                        content = getattr(gen, "text", "") or ""
                    if _SQL_BLOCK_START in content:
                        self._store.append(content)
        except Exception:  # pragma: no cover — defensive
            pass


def extract_sql_from_store(store: list[str]) -> str | None:
    """Return the SQL from the most recent ``###SQL_START###...###SQL_END###`` block.

    Searches *store* newest-first so the latest generated SQL is always used.
    """
    for content in reversed(store):
        m = _SQL_BLOCK_RE.search(content)
        if m:
            return m.group(1).strip()
    return None


_BASE_DIR = Path(__file__).resolve().parent

# Required keys in the final JSON answer
_REQUIRED_ANSWER_KEYS = frozenset({"sql_code", "answer", "result"})


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


def create_omni_lite_agent(
    payload: AgentPayload,
    llm: Any | None = None,
) -> tuple[Any, list[str]]:
    """Create a Deep Agent for OmniLite Text-to-SQL.

    The agent is equipped with:
    - Four domain tools (``extract_entities``, ``retrieve_semantic_candidates``,
      ``validate_sql``, ``execute_sql``) from ``tools.py``
    - ``AGENTS.md`` as persistent system memory
    - Three SKILL.md files: ``candidate-retrieval``, ``sql-generation``,
      ``answer-formatting``
    - A ``FilesystemBackend`` rooted at the ``omni_lite/`` directory

    Args:
        payload: The ``AgentPayload`` received from the caller.  Used to bind
            session-scoped values (``db_connector``, ``dialects``) to tools.
        llm: Optional pre-built LLM client.  When ``None``, ``_make_llm()`` is
            called automatically.

    Returns:
        Tuple of ``(agent, sql_store)`` where *sql_store* is the mutable list
        that ``_SQLCaptureCallback`` will populate with AI messages containing
        ``###SQL_START###`` blocks.  Pass the callback to ``agent.invoke``
        config so it fires during execution.
    """
    if llm is None:
        llm = _make_llm()

    sql_store: list[str] = []
    tools = build_omni_lite_tools(payload, llm, sql_store=sql_store)

    skill_dirs = _load_skill_dirs()
    agents_md = str(_BASE_DIR / "AGENTS.md")
    memory = [agents_md] if Path(agents_md).exists() else []

    catalog_prompt = _build_catalog_prompt(payload)

    logger.info(
        "Creating OmniLite Deep Agent | tools=%s | skills=%s | memory=%s",
        [getattr(t, "name", "?") for t in tools],
        skill_dirs,
        memory,
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt=catalog_prompt or None,
        memory=memory,
        skills=skill_dirs or None,
        tools=tools,
        subagents=[],
        backend=FilesystemBackend(root_dir=str(_BASE_DIR)),
    )
    return agent, sql_store


def _load_skill_dirs() -> list[str]:
    """Return skill directory paths that exist on disk."""
    skill_names = ["candidate-retrieval", "sql-generation", "answer-formatting"]
    dirs = []
    for name in skill_names:
        path = _BASE_DIR / "skills" / name
        if path.is_dir():
            dirs.append(str(path))
        else:
            logger.debug("Skill directory not found, skipping: %s", path)
    return dirs


def _build_catalog_prompt(payload: AgentPayload) -> str | None:
    """Build an optional system-prompt fragment injecting dialects + date context."""
    now = datetime.now()
    dialects = payload.get("dialects") or []
    lines = [
        f"Today's date: {now.year}-{now.month:02d}-{now.day:02d} " f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}.",
    ]
    if dialects:
        lines.append(f"Allowed SQL dialects: {', '.join(str(d) for d in dialects)}.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# User-prompt formatting
# ---------------------------------------------------------------------------


def format_omni_lite_user_prompt(
    question: str,
    history: list[dict[str, str]] | None = None,
    dialects: list[str] | None = None,
) -> str:
    """Format the user-turn message sent to the Deep Agent.

    Conversation history is inlined as plain text so Deep Agent's flat
    messages list receives a single user turn.

    Args:
        question: The current user question.
        history: Optional list of ``{"question": ..., "response": ...}`` dicts.
        dialects: Optional list of allowed SQL dialect names.

    Returns:
        A formatted string ready to pass as the user message content.
    """
    parts: list[str] = []

    if history:
        parts.append("### Conversation History (most recent last)")
        for turn in history:
            parts.append(f"User: {turn.get('question', '')}")
            parts.append(f"Assistant: {turn.get('response', '')}")
        parts.append("")

    if dialects:
        parts.append(f"Allowed SQL dialects: {', '.join(dialects)}.")

    parts.append(f"User question: {question.strip()}")
    parts.append("")
    parts.append(
        "Step 1 — call extract_entities with the question.\n"
        "Step 2 — call retrieve_semantic_candidates with the outputs of Step 1.\n"
        "Step 3 — generate SQL; wrap it in your message like this (mandatory):\n"
        "  ###SQL_START###\n"
        "  <your SQL here>\n"
        "  ###SQL_END###\n"
        "Step 4 — call validate_sql on the SQL; if invalid, emit a new "
        "###SQL_START###...###SQL_END### block with the fix and re-validate.\n"
        "Step 5 — call execute_sql(sql='__SQL_FROM_MESSAGE__').\n"
        "Step 6 — emit your final answer as a single JSON object (no other text):\n"
        '  {"sql_code": "<exact SQL>", "answer": "<short explanation>", '
        '"result": <DB value or null>, "semantic_elements": []}\n'
        "Nothing before { or after }. No markdown fences. No apologies."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer extraction  (mirrors deep_agent_runtime._extract_structured_answer)
# ---------------------------------------------------------------------------


def extract_structured_answer(result: dict) -> dict | None:
    """Scan Deep Agent messages (newest-first) for the structured JSON answer.

    Tries in order:
    1. Full JSON object with all required keys.
    2. Markdown-style answer with a ```sql block.
    3. Plain-prose SQL — a SQL statement starting at a line boundary and
       delimited by blank lines or end-of-text (handles cases where the agent
       embeds the query inline without fences or JSON).

    Args:
        result: The dict returned by ``agent.invoke()``.

    Returns:
        ``{"sql_code", "answer", "result"}`` dict or ``None`` if not found.
    """
    messages = result.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            obj = _extract_json_answer_object(content)
            if obj is not None:
                return obj
            md = _parse_markdown_answer(content)
            if md is not None:
                return md
            prose = _extract_sql_from_prose(content)
            if prose is not None:
                return prose
        elif isinstance(content, dict):
            if _REQUIRED_ANSWER_KEYS.issubset(content.keys()):
                return content
    return None


def _extract_json_answer_object(content: str) -> dict | None:
    if not content.strip():
        return None
    text = content.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and _REQUIRED_ANSWER_KEYS.issubset(obj.keys()):
            return obj
    except Exception:
        pass
    decoder = json.JSONDecoder()
    i = 0
    while i < len(text):
        if text[i] == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict) and _REQUIRED_ANSWER_KEYS.issubset(obj.keys()):
                    return obj
            except json.JSONDecodeError:
                pass
        i += 1
    return None


def _parse_markdown_answer(text: str) -> dict | None:
    sql_code = answer = None
    result_value: Any = None

    start = text.find("```sql")
    if start != -1:
        start = text.find("\n", start)
        if start != -1:
            end = text.find("```", start)
            if end != -1:
                sql_code = text[start:end].strip()

    answer_marker = "**Answer:**"
    idx = text.find(answer_marker)
    if idx != -1:
        answer = text[idx + len(answer_marker) :].strip()

    result_marker = "**Result:**"
    r_idx = text.find(result_marker)
    if r_idx != -1:
        r_start = r_idx + len(result_marker)
        r_end = text.find("**Answer:**", r_start)
        r_end = r_end if r_end != -1 else len(text)
        section = text[r_start:r_end]
        m = re.search(r"-?\d+(\.\d+)?", section)
        if m:
            try:
                result_value = float(m.group(0))
            except ValueError:
                result_value = m.group(0)
        elif section.strip():
            result_value = section.strip()

    if sql_code and answer:
        return {"sql_code": sql_code, "answer": answer, "result": result_value}
    return None


# Matches a SQL statement that starts at the beginning of a line with a SQL
# keyword and continues until the next blank line (two consecutive newlines)
# or end-of-string.  Works regardless of trailing prose or embedded values.
_SQL_PROSE_RE = re.compile(
    r"(?m)^" r"((?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|EXPLAIN)\b" r".*?)" r"(?=\n[ \t]*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_sql_from_prose(text: str) -> dict | None:
    """Extract SQL embedded in plain prose (no fences, no JSON).

    Locates the first SQL statement that starts at a line boundary (identified
    by a leading SQL keyword) and is delimited by a blank line or end-of-text.
    The surrounding prose becomes the ``answer``.

    Example input::

        Based on the results, the query is:

        SELECT a, b FROM t WHERE x = 'foo'

        This selects rows where x equals 'foo'.

    Args:
        text: Raw message content string.

    Returns:
        ``{"sql_code", "answer", "result"}`` dict or ``None`` if no SQL found.
    """
    m = _SQL_PROSE_RE.search(text)
    if not m:
        return None
    sql_code = m.group(1).strip()
    if not sql_code:
        return None
    # Use the text outside the SQL block as the answer
    answer = text[: m.start()].strip() + " " + text[m.end() :].strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    logger.debug("_extract_sql_from_prose: extracted SQL from prose (%d chars)", len(sql_code))
    return {"sql_code": sql_code, "answer": answer or text.strip(), "result": None, "semantic_elements": []}


__all__ = [
    "create_omni_lite_agent",
    "extract_sql_from_store",
    "extract_structured_answer",
    "format_omni_lite_user_prompt",
    "_SQLCaptureCallback",
]
