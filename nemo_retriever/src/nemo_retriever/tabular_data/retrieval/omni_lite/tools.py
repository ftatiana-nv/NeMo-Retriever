"""
LangChain tools for the OmniLite Deep Agent.

Each tool wraps the existing LangGraph agent logic so the Deep Agent can call
it autonomously rather than following a fixed state-machine route.

Factory
-------
Use ``build_omni_lite_tools(payload, llm)`` to get a list of bound tools that
close over session-scoped values (``db_connector``, ``dialects``).  The LLM is
passed explicitly so the tools can re-use the same client as the agent.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.tools import tool

from nemo_retriever.tabular_data.retrieval.omni_lite.agents.candidates_preparation import (
    CandidatePreparationAgent,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.sql_execution import (
    _run_sql_duckdb,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.agents.query_validation import (
    query_validation,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentPayload
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import (
    _apply_foreign_key_hints,
    clean_results,
    dedupe_merge_relevant_tables,
    extract_candidates,
    # get_all_schemas_ids,  # re-enable with validation
    get_relevant_fks_from_candidates_tables,
    get_relevant_queries,
    get_relevant_tables,
    # get_schemas_slim,  # re-enable with validation
    Labels,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool 1 – extract_entities
# ---------------------------------------------------------------------------


def _make_extract_entities_tool():
    """Return an ``extract_entities`` tool.

    Entity matching is delegated entirely to the vector/graph search in
    ``retrieve_semantic_candidates`` (via ``extract_candidates``), which
    already searches by the raw question text.  No LLM call is needed here —
    mirroring LangGraph's ``CandidateRetrievalAgent`` which receives
    ``entities_and_concepts`` only to augment the search strings, not to
    replace the question itself.
    """

    @tool
    def extract_entities(question: str) -> str:
        """Normalise the question for retrieval.

        Call this as the FIRST step before retrieve_semantic_candidates.
        Returns the question as-is for ``query_no_values`` so the semantic
        search in the next step can use it directly.

        Args:
            question: The raw user question.

        Returns:
            JSON object with:
              - ``query_no_values``: the question (values kept; semantic
                search handles matching)
              - ``entities_and_concepts``: empty list (the vector search
                uses the full question text instead)
        """
        return json.dumps({"query_no_values": question, "entities_and_concepts": []})

    return extract_entities


# ---------------------------------------------------------------------------
# Tool 2 – retrieve_semantic_candidates
# ---------------------------------------------------------------------------


def _make_retrieve_candidates_tool():
    """Return a ``retrieve_semantic_candidates`` tool.

    Mirrors LangGraph nodes ``retrieve_candidates`` (``CandidateRetrievalAgent``)
    and ``prepare_candidates`` (``CandidatePreparationAgent``) combined into
    one tool call so the Deep Agent receives all SQL-generation context at once.
    No LLM call is made.
    """
    _prep_agent = CandidatePreparationAgent()

    @tool
    def retrieve_semantic_candidates(
        question: str,
        query_no_values: str = "",
        entities_and_concepts_json: str = "[]",
    ) -> str:
        """Retrieve semantically relevant tables, columns, FKs, and SQL snippets.

        Implements LangGraph's ``CandidateRetrievalAgent`` + ``CandidatePreparationAgent``
        pipeline: vector/graph search over Neo4j/LanceDB followed by FK expansion
        and deduplication.  No LLM is invoked.

        Call AFTER extract_entities and pass its outputs here.

        Args:
            question: Full user question (with values).
            query_no_values: Question with specific values stripped
                (from extract_entities).
            entities_and_concepts_json: JSON array of entity/concept names
                (from extract_entities).

        Returns:
            JSON object with:
              - ``relevant_tables``: flat list of relevant table dicts
              - ``relevant_fks``: flat list of FK relationship dicts
              - ``complex_candidates_str``: formatted candidate strings
                (SQL snippets / custom analyses)
              - ``relevant_queries``: example queries from the knowledge base
              - ``candidates``: raw candidate list (columns + custom analyses)
        """
        try:
            entities_and_concepts: list[str] = json.loads(entities_and_concepts_json)
        except Exception:
            entities_and_concepts = []

        q_no_vals = query_no_values or question

        try:
            # ── CandidateRetrievalAgent logic ──────────────────────────────
            extracted = extract_candidates(entities_and_concepts, q_no_vals, question)

            if isinstance(extracted, tuple) and len(extracted) == 2:
                custom_raw, column_raw = extracted
                custom_candidates = clean_results(list(custom_raw or []))
                column_candidates = clean_results(list(column_raw or []))
            else:
                merged = [item.get("candidate", item) for item in (extracted or [])]
                cleaned = clean_results(merged)
                custom_candidates = [c for c in cleaned if c.get("label") == Labels.CUSTOM_ANALYSIS]
                column_candidates = [c for c in cleaned if c.get("label") == Labels.COLUMN]

            all_candidates = custom_candidates + column_candidates

            # ── CandidatePreparationAgent logic ───────────────────────────
            relevant_tables, relevant_fks = get_relevant_fks_from_candidates_tables(all_candidates)
            add_tables, add_fks = get_relevant_tables(question, k=5)
            relevant_tables.extend(add_tables)
            relevant_fks.extend(add_fks)
            relevant_tables = dedupe_merge_relevant_tables(relevant_tables)
            _apply_foreign_key_hints(relevant_tables, relevant_fks)

            relevant_queries = get_relevant_queries(all_candidates)
            complex_candidates_str = _prep_agent._build_complex_candidates_str(all_candidates)

            return json.dumps(
                {
                    "relevant_tables": relevant_tables,
                    "relevant_fks": relevant_fks,
                    "complex_candidates_str": complex_candidates_str,
                    "relevant_queries": relevant_queries,
                    "candidates": all_candidates,
                },
                default=str,
            )
        except Exception as exc:
            logger.warning("retrieve_semantic_candidates failed: %s", exc)
            return json.dumps(
                {
                    "relevant_tables": [],
                    "relevant_fks": [],
                    "complex_candidates_str": [],
                    "relevant_queries": [],
                    "candidates": [],
                    "error": str(exc),
                }
            )

    return retrieve_semantic_candidates


# ---------------------------------------------------------------------------
# Tool 3 – validate_sql
# ---------------------------------------------------------------------------


def _make_validate_sql_tool(dialects: list[str] | None):
    """Return a ``validate_sql`` tool bound to *dialects*."""

    _dialects = dialects or []

    @tool
    def validate_sql(sql: str) -> str:
        """Validate a SQL query for schema correctness, SELECT-only enforcement, and FK compliance.

        Call this AFTER generating SQL and BEFORE execute_sql.

        Args:
            sql: The SQL string to validate (no markdown fences, plain SQL only).

        Returns:
            JSON object with:
              - ``valid``: bool
              - ``error``: error message string (empty when valid)
              - ``sql_columns``: list of column IDs referenced by the query
        """
        try:
            # Schema loading skipped while query_validation is bypassed.
            # Re-enable alongside the validation body in query_validation.py:
            #   schemas = get_schemas_slim(list(get_all_schemas_ids()))
            #   result = query_validation(schemas, sql, _dialects, user_participants=[])
            result = query_validation(None, sql, _dialects, user_participants=[])
            if result.get("error"):
                return json.dumps(
                    {
                        "valid": False,
                        "error": result["error"],
                        "sql_columns": [],
                    }
                )
            return json.dumps(
                {
                    "valid": True,
                    "error": "",
                    "sql_columns": result.get("sql_columns") or [],
                }
            )
        except Exception as exc:
            logger.warning("validate_sql failed: %s", exc)
            return json.dumps({"valid": False, "error": str(exc), "sql_columns": []})

    return validate_sql


# ---------------------------------------------------------------------------
# Tool 4 – execute_sql
# ---------------------------------------------------------------------------


def _make_execute_sql_tool(db_connector: Any, sql_store: list[str]):
    """Return an ``execute_sql`` tool bound to *db_connector* and *sql_store*.

    The tool always prefers the most recent ``###SQL_START###...###SQL_END###``
    block captured by ``_SQLCaptureCallback`` in *sql_store*.  This ensures
    the full, untruncated SQL is used even when the JSON-serialised *sql*
    argument is cut short by the LLM.  The *sql* argument is used only as a
    fallback when the store is empty.
    """
    from nemo_retriever.tabular_data.retrieval.omni_lite.omni_lite_runtime import (
        extract_sql_from_store,
    )

    @tool
    def execute_sql(sql: str) -> str:
        """Execute a validated SQL query and return the results.

        Call this AFTER validate_sql confirms the query is valid.
        Pass the SQL as plain text — no markdown fences or backticks.

        The tool internally recovers the full SQL from the
        ``###SQL_START###...###SQL_END###`` block you included in your message,
        so even if the argument is truncated by JSON serialisation the correct
        query is executed.

        Args:
            sql: The validated SQL to execute (DuckDB dialect, plain text).

        Returns:
            JSON object with:
              - ``success``: bool
              - ``result``: list of row dicts (JSON-serialisable) or null
              - ``error``: error message string (empty on success)
        """
        # Always prefer the captured store — it holds the full, untruncated SQL
        # from the ###SQL_START###...###SQL_END### block in the agent message.
        # Fall back to the argument only when the store has no entry yet.
        store_sql = extract_sql_from_store(sql_store)
        if store_sql:
            logger.debug(
                "execute_sql: using SQL from store (%d chars)",
                len(store_sql),
            )
            resolved_sql = store_sql
        else:
            logger.debug(
                "execute_sql: store empty — using sql argument (%d chars)",
                len(sql),
            )
            resolved_sql = sql

        if db_connector is None:
            return json.dumps({"success": False, "result": None, "error": "No db_connector provided in payload."})
        try:
            path_state = {"db_connector": db_connector}
            response = _run_sql_duckdb(resolved_sql, path_state)
            if response is None:
                return json.dumps({"success": False, "result": None, "error": "Infra or auth error during execution."})
            if response.error:
                return json.dumps({"success": False, "result": None, "error": response.error})
            raw = response.result
            result_data = None
            if raw:
                try:
                    result_data = json.loads(raw[0]) if isinstance(raw[0], str) else raw
                except Exception:
                    result_data = raw
            return json.dumps({"success": True, "result": result_data, "error": ""}, default=str)
        except Exception as exc:
            logger.warning("execute_sql failed: %s", exc)
            return json.dumps({"success": False, "result": None, "error": str(exc)})

    return execute_sql


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_omni_lite_tools(payload: AgentPayload, llm: Any, sql_store: list[str] | None = None) -> list:
    """Build and return the list of LangChain tools for the OmniLite Deep Agent.

    Session-scoped values (``db_connector``, ``dialects``) are closed over so
    the tools don't require extra arguments at call time.  Neither
    ``extract_entities`` nor ``retrieve_semantic_candidates`` makes an LLM
    call — all entity matching is done via Neo4j / LanceDB vector search,
    mirroring the LangGraph ``CandidateRetrievalAgent`` behaviour.

    Args:
        payload: The ``AgentPayload`` received from the caller.
        llm: The LLM client (``ChatNVIDIA``) — used only by the Deep Agent
            itself, not by the retrieval tools.
        sql_store: Mutable list shared with ``_SQLCaptureCallback``.  The
            ``execute_sql`` tool reads from this list when it receives the
            ``__SQL_FROM_MESSAGE__`` sentinel.  When ``None`` an empty list is
            created (no capture callback will be attached).

    Returns:
        List of four bound LangChain tools in recommended call order:
        ``[extract_entities, retrieve_semantic_candidates, validate_sql, execute_sql]``
    """
    db_connector = payload.get("db_connector")
    dialects = payload.get("dialects") or []
    store = sql_store if sql_store is not None else []

    return [
        _make_extract_entities_tool(),
        _make_retrieve_candidates_tool(),
        _make_validate_sql_tool(dialects),
        _make_execute_sql_tool(db_connector, store),
    ]


__all__ = ["build_omni_lite_tools"]
