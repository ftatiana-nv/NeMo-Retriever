"""Phase 2 — SQL Deep Agent runtime.

Creates a Deep Agent whose only job is to generate and validate SQL from the
``RetrievalContext`` produced by Phase 1.  The context is injected into the
agent's system prompt so the SQL generation starts with a clean context
window that contains no tool-call history from the retrieval phase.

Usage
-----
::

    from nemo_retriever.tabular_data.retrieval.omni_lite.omni_lite_runtime import (
        create_omni_lite_agent,
        extract_structured_answer,
        format_omni_lite_user_prompt,
    )

    agent, store = create_omni_lite_agent(payload, retrieval_ctx, llm=llm_client)
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

from nemo_retriever.tabular_data.retrieval.omni_lite.context import RetrievalContext
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentPayload
from nemo_retriever.tabular_data.retrieval.omni_lite.tools import ExecutionStore, build_omni_lite_tools
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import _make_llm

logger = logging.getLogger(__name__)

_BASE_DIR = Path(__file__).resolve().parent

# Required keys in the final JSON answer
_REQUIRED_ANSWER_KEYS = frozenset({"sql_code", "answer", "result"})


# ---------------------------------------------------------------------------
# Agent creation
# ---------------------------------------------------------------------------


def create_omni_lite_agent(
    payload: AgentPayload,
    retrieval_ctx: RetrievalContext,
    llm: Any | None = None,
) -> tuple[Any, ExecutionStore]:
    """Create a Phase 2 SQL Deep Agent.

    The agent is equipped with a single tool (``validate_sql``) and the
    ``AGENTS.md`` persistent memory.  The ``RetrievalContext`` from Phase 1 is
    injected into the system prompt so the agent has full schema context without
    any retrieval tool-call history in its message thread.

    Args:
        payload: The ``AgentPayload`` from the caller.  Used to bind
            session-scoped values (``dialects``) to ``validate_sql``.
        retrieval_ctx: The ``RetrievalContext`` produced by Phase 1.
        llm: Optional pre-built LLM client.  When ``None``, ``_make_llm()`` is
            called automatically.

    Returns:
        Tuple of (agent, store):
        - agent: Deep Agent instance ready for ``agent.invoke()``.
        - store: ``ExecutionStore`` populated in-place as the agent runs.
          ``store.sql`` holds the last validated SQL.
    """
    if llm is None:
        llm = _make_llm()

    tools, store = build_omni_lite_tools(payload, llm)

    skill_dirs = _load_skill_dirs()
    agents_md = str(_BASE_DIR / "AGENTS.md")
    memory = [agents_md] if Path(agents_md).exists() else []

    system_prompt = _build_system_prompt(payload, retrieval_ctx)

    logger.info(
        "Creating SQL Deep Agent (Phase 2) | tools=%s | skills=%s | memory=%s",
        [getattr(t, "name", "?") for t in tools],
        skill_dirs,
        memory,
    )

    agent = create_deep_agent(
        model=llm,
        system_prompt=system_prompt,
        memory=memory,
        skills=skill_dirs or None,
        tools=tools,
        subagents=[],
        backend=FilesystemBackend(root_dir=str(_BASE_DIR)),
    )
    return agent, store


def _load_skill_dirs() -> list[str]:
    """Return skill directory paths that exist on disk."""
    skill_names = ["sql-generation", "answer-formatting"]
    dirs = []
    for name in skill_names:
        path = _BASE_DIR / "skills" / name
        if path.is_dir():
            dirs.append(str(path))
        else:
            logger.debug("Skill directory not found, skipping: %s", path)
    return dirs


def _build_system_prompt(payload: AgentPayload, retrieval_ctx: RetrievalContext) -> str:
    """Build the system prompt that injects the RetrievalContext for Phase 2.

    The RetrievalContext is serialised as JSON and embedded in the system
    prompt so the agent has full schema context from the start, without
    accumulating retrieval tool-call history in the message thread.
    """
    now = datetime.now()
    dialects = payload.get("dialects") or []

    lines: list[str] = [
        f"Today's date: {now.year}-{now.month:02d}-{now.day:02d} " f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}.",
    ]

    if dialects:
        lines.append(f"Allowed SQL dialects: {', '.join(str(d) for d in dialects)}.")

    lines.append("")
    lines.append("## RetrievalContext (produced by Phase 1 — use this as your schema source)")
    lines.append("")

    # Embed the full context as JSON for the agent to reference
    try:
        ctx_json = json.dumps(retrieval_ctx, default=str, indent=2)
    except Exception:
        ctx_json = "{}"
    lines.append(ctx_json)

    lines.append("")
    lines.append(
        "The RetrievalContext above contains all the tables, columns, foreign keys, "
        "and SQL snippets you need.  Do NOT call any retrieval tools — retrieval is "
        "complete.  Your only tool is `validate_sql`.  Generate SQL from the context "
        "above, then call `validate_sql` immediately.  Fix and retry up to 4 times."
    )

    if not retrieval_ctx.get("coverage_complete"):
        lines.append(
            "\nNote: `coverage_complete` is false — one or more entities could not be "
            "fully resolved.  Construct the best SQL possible from the available context "
            "and note any limitations in your `answer` field."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# User-prompt formatting
# ---------------------------------------------------------------------------


def format_omni_lite_user_prompt(
    question: str,
    history: list[dict[str, str]] | None = None,
    dialects: list[str] | None = None,
) -> str:
    """Format the user-turn message sent to the Phase 2 SQL Deep Agent.

    Conversation history is inlined as plain text so the agent's flat
    messages list receives a single user turn.

    Args:
        question: The current user question.
        history: Optional list of ``{"question": ..., "response": ...}`` dicts.
        dialects: Optional list of allowed SQL dialect names (informational;
            also injected via system prompt).

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
        "Your RetrievalContext is in the system prompt above.\n"
        "Step 1 — Generate SQL using the tables, columns, FKs, and snippets from RetrievalContext.\n"
        "Step 2 — Call validate_sql immediately (wrap SQL in ```sql ... ``` fences).\n"
        "Step 3 — If invalid, fix the SQL and call validate_sql again (up to 4 retries).\n"
        "Step 4 — Emit your final answer as a single JSON object (no other text):\n"
        '  {"sql_code": "<exact SQL>", "answer": "<short explanation>", '
        '"result": null, "semantic_elements": []}\n'
        "Nothing before { or after }. No markdown fences. No apologies."
    )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def extract_structured_answer(result: dict) -> dict | None:
    """Scan Deep Agent messages (newest-first) for the structured JSON answer.

    Tries in order:
    1. Full JSON object with all required keys.
    2. Markdown-style answer with a ```sql block.
    3. Plain-prose SQL — a SQL statement starting at a line boundary and
       delimited by blank lines or end-of-text.

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


_SQL_PROSE_RE = re.compile(
    r"(?m)^" r"((?:SELECT|WITH|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|EXPLAIN)\b" r".*?)" r"(?=\n[ \t]*\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _extract_sql_from_prose(text: str) -> dict | None:
    """Extract SQL embedded in plain prose (no fences, no JSON)."""
    m = _SQL_PROSE_RE.search(text)
    if not m:
        return None
    sql_code = m.group(1).strip()
    if not sql_code:
        return None
    answer = text[: m.start()].strip() + " " + text[m.end() :].strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    logger.debug("_extract_sql_from_prose: extracted SQL from prose (%d chars)", len(sql_code))
    return {"sql_code": sql_code, "answer": answer or text.strip(), "result": None, "semantic_elements": []}


__all__ = [
    "create_omni_lite_agent",
    "extract_structured_answer",
    "format_omni_lite_user_prompt",
]
