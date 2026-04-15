"""LangChain tools for the Phase 1 Retrieval Deep Agent.

Each tool writes its results directly into a shared ``RetrievalStore`` — the
same pattern used by ``ExecutionStore`` in Phase 2.  The agent never needs to
pass tables or entities as JSON arguments between calls; the store accumulates
state internally.

Tools:
- ``decompose_question``   — splits the question into typed, priority-ordered entities
- ``retrieve_for_entity``  — per-entity candidate + table/FK retrieval with intra-table
                             coverage check (no state arguments required)
- ``synthesize_expression`` — derives a SQL expression for zero-coverage entities

Use ``build_retrieval_tools(payload, llm)`` to get ``(tools, store)``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Literal

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from nemo_retriever.tabular_data.retrieval.omni_lite.agents.candidates_preparation import (
    CandidatePreparationAgent,
)
from nemo_retriever.tabular_data.retrieval.omni_lite.ai_services import invoke_with_structured_output
from nemo_retriever.tabular_data.retrieval.omni_lite.context import EntityCoverage, RetrievalContext
from nemo_retriever.tabular_data.retrieval.omni_lite.state import AgentPayload
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import (
    Labels,
    _apply_foreign_key_hints,
    clean_results,
    dedupe_merge_relevant_tables,
    extract_candidates,
    get_relevant_fks_from_candidates_tables,
    get_relevant_tables,
)

logger = logging.getLogger(__name__)

_prep_agent = CandidatePreparationAgent()


# ---------------------------------------------------------------------------
# RetrievalStore
# ---------------------------------------------------------------------------


class RetrievalStore:
    """Mutable per-request store written by the retrieval tools, read by the runtime.

    Mirrors the ``ExecutionStore`` pattern from Phase 2:
    - ``decompose_question`` writes to ``entities``.
    - ``retrieve_for_entity`` appends to ``entity_results``, ``accumulated_tables``,
      ``accumulated_fks``, and ``custom_candidates``.
    - ``synthesize_expression`` patches the matching entry in ``entity_results``.

    The runtime calls ``as_context()`` after the agent finishes to build the
    ``RetrievalContext`` directly from store state — no JSON parsing of agent
    messages required.
    """

    def __init__(self) -> None:
        self.entities: list[dict] = []
        self.entity_results: list[dict] = []
        self.accumulated_tables: list[dict] = []
        self.accumulated_fks: list[dict] = []
        self.custom_candidates: list[dict] = []

    # ------------------------------------------------------------------
    # Helpers used by tools
    # ------------------------------------------------------------------

    def _resolved_as(self, result: dict) -> str:
        entity_type = result.get("entity_type", "")
        if entity_type == "value":
            return "value"
        if entity_type == "time_filter":
            return "time_filter"
        if result.get("covered_by_existing_table"):
            return "column"
        candidates = result.get("candidates", [])
        if any(c.get("label") == Labels.CUSTOM_ANALYSIS for c in candidates):
            return "custom_analysis"
        if candidates or result.get("relevant_tables"):
            return "column"
        if result.get("sql_expression"):
            return "expression"
        return "unresolved"

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def as_context(self) -> RetrievalContext | None:
        """Build a ``RetrievalContext`` from accumulated store state.

        Returns ``None`` when no tools have written any data yet.
        """
        if not self.entity_results and not self.entities:
            return None

        entity_coverage: list[EntityCoverage] = []
        for r in self.entity_results:
            entity_coverage.append(
                {
                    "entity": r.get("entity", ""),
                    "entity_type": r.get("entity_type", "dimension"),
                    "resolved_as": self._resolved_as(r),
                    "candidates": r.get("candidates", []),
                    "sql_expression": r.get("sql_expression"),
                    "filter_field_hint": r.get("filter_field_hint"),
                    "matched_table": r.get("matched_table"),
                    "matched_column": r.get("matched_column"),
                }
            )

        coverage_complete = all(
            ec["resolved_as"] != "unresolved" for ec in entity_coverage if ec["entity_type"] in ("metric", "dimension")
        )

        complex_candidates_str = _prep_agent._build_complex_candidates_str(self.custom_candidates)

        return {
            "entity_coverage": entity_coverage,
            "relevant_tables": self.accumulated_tables,
            "relevant_fks": self.accumulated_fks,
            "complex_candidates_str": complex_candidates_str,
            "relevant_queries": [],
            "coverage_complete": coverage_complete,
        }


# ---------------------------------------------------------------------------
# Pydantic schemas for structured LLM outputs
# ---------------------------------------------------------------------------


class _EntityItem(BaseModel):
    term: str = Field(..., description="The entity/concept term as it appears in the question.")
    entity_type: Literal["metric", "dimension", "time_filter", "value"] = Field(
        ...,
        description=(
            "metric — a measurable value to compute (revenue, profit, count, average …); "
            "dimension — a schema concept that maps to a table/column (product, student, customer …); "
            "time_filter — a time period or date (last month, Q1 2024, yesterday …); "
            "value — a specific named literal that will become a WHERE filter "
            "(e.g. 'Seattle' in 'students from Seattle', 'Enterprise' in 'Enterprise customers'). "
        ),
    )
    filter_field_hint: str | None = Field(
        default=None,
        description=(
            "For value entities only: the field/concept this value filters on. "
            "Example: entity='Seattle', filter_field_hint='city'. "
            "Leave null for non-value entities."
        ),
    )
    priority: int = Field(
        ...,
        description=(
            "Importance order for retrieval. Lower = retrieved first. "
            "1 = primary subject (the main thing being queried, e.g. 'students', 'orders'); "
            "2 = grouping/join dimension (related objects, e.g. 'product', 'department'); "
            "3 = filter attribute concept (a field used to narrow results, e.g. 'city', 'status'); "
            "4 = filter value or time period (specific literal, e.g. 'Seattle', 'last month')."
        ),
    )


class _DecomposeResult(BaseModel):
    entities: list[_EntityItem] = Field(
        ...,
        description=("All entities extracted from the question, ordered by priority ascending " "(priority=1 first)."),
    )


class _ExpressionResult(BaseModel):
    expression: str = Field(
        ...,
        description=(
            "A SQL expression (NOT a full query) that computes the entity from the given columns. "
            "Example: 'income - outcome' or 'SUM(s.sales_amount) / COUNT(DISTINCT s.customer_id)'. "
            "Use ONLY column names from the provided available_columns list."
        ),
    )
    columns_used: list[str] = Field(
        ...,
        description="Fully-qualified column names used in the expression.",
    )
    confidence: Literal["high", "medium", "low"] = Field(
        ...,
        description="Confidence that the expression correctly represents the entity.",
    )


# ---------------------------------------------------------------------------
# Intra-table coverage helper
# ---------------------------------------------------------------------------


def _find_entity_in_tables(
    entity_term: str,
    tables: list[dict],
) -> tuple[bool, str | None, str | None]:
    """Check whether *entity_term* is already satisfied by a column in *tables*.

    Bidirectional substring match covers plurals, abbreviations, and partials
    (e.g. "city" ↔ "studcity", "grade" ↔ "grade_level").

    Returns:
        ``(covered, matched_table_name, matched_column_name)``
    """
    term = entity_term.lower().strip()
    if not term:
        return False, None, None

    for table in tables:
        table_name = table.get("name") or ""
        for col in table.get("columns") or []:
            if isinstance(col, dict):
                col_name = (col.get("name") or "").lower()
                col_desc = (col.get("description") or "").lower()
            elif isinstance(col, str):
                col_name = col.lower()
                col_desc = ""
            else:
                continue

            if term in col_name or col_name in term or (col_desc and term in col_desc):
                return True, table_name, col_name

    return False, None, None


# ---------------------------------------------------------------------------
# Tool 1 — decompose_question
# ---------------------------------------------------------------------------


def _make_decompose_question_tool(llm: Any, store: RetrievalStore):
    """Return a ``decompose_question`` tool that writes entities into *store*."""

    @tool
    def decompose_question(question: str) -> str:
        """Decompose the user question into typed, priority-ordered entities.

        Call this as the FIRST step.  The entities are stored internally —
        you do not need to pass them to subsequent tool calls.

        Returns a confirmation listing the entities found and their priority order.

        Args:
            question: The raw user question.
        """
        prompt = f"""You are extracting entities and concepts from a user question for database retrieval.

User Question:
{question}

Classify EVERY meaningful term as one of:
- metric: a measurable value the user wants to compute (revenue, profit, count, avg, total …)
- dimension: a schema concept that maps to a DB table or column (student, product, order, customer …)
- time_filter: a time period or date (last month, Q1 2024, this year, yesterday …)
- value: a SPECIFIC NAMED LITERAL that will become a WHERE clause filter value.
  Examples: 'Seattle' in "students from Seattle" → value (city='Seattle'),
            'Enterprise' in "enterprise customers" → value (tier='Enterprise').
  Key: if the term is a PROPER NOUN or SPECIFIC NAMED INSTANCE rather than a general
  concept/field name, it is a value, not a dimension.

Assign a priority (integer, lower = retrieved first):
1 = primary subject (the main thing being queried: "students", "orders", "products")
2 = grouping/join dimension (related objects: "department", "category", "region")
3 = filter attribute concept (field concept used to narrow: "city", "status", "tier")
4 = filter value or time period (specific literal: "Seattle", "last month", "Active")

For filter_field_hint: only for value entities. State the field the value filters on.
E.g. entity='Seattle' → filter_field_hint='city'.

Order the entities list by priority ascending in your output."""

        messages = [SystemMessage(content=prompt)]
        result = invoke_with_structured_output(llm, messages, _DecomposeResult)

        if result is None:
            store.entities = []
            return "decompose_question failed — no entities extracted."

        entities = sorted(result.model_dump()["entities"], key=lambda e: e.get("priority", 99))
        store.entities = entities

        lines = [f"Extracted {len(entities)} entities (call retrieve_for_entity for each):"]
        for i, e in enumerate(entities, 1):
            hint = f" → filter on '{e['filter_field_hint']}'" if e.get("filter_field_hint") else ""
            lines.append(f"  {i}. [{e['entity_type']} p={e['priority']}] {e['term']}{hint}")
        return "\n".join(lines)

    return decompose_question


# ---------------------------------------------------------------------------
# Tool 2 — retrieve_for_entity
# ---------------------------------------------------------------------------


def _make_retrieve_for_entity_tool(store: RetrievalStore):
    """Return a ``retrieve_for_entity`` tool that reads/writes *store*.

    The intra-table coverage check uses ``store.accumulated_tables`` so the
    agent never needs to pass accumulated state as an argument.
    """

    @tool
    def retrieve_for_entity(entity_term: str, entity_type: str = "dimension") -> str:
        """Retrieve semantically relevant candidates for a single entity term.

        Call this ONCE for EACH entity returned by decompose_question, in priority order.
        The tool automatically checks whether the entity is already covered by a
        previously-retrieved table — no state arguments needed.

        Args:
            entity_term: The entity term to search for (e.g. "revenue", "city", "students").
            entity_type: The entity type from decompose_question
                         ("metric", "dimension", "time_filter", or "value").

        Returns:
            A short summary: covered/not-covered, tables found, whether search was skipped.
        """
        # ── Intra-table coverage check ─────────────────────────────────────
        covered_existing, matched_table, matched_col = _find_entity_in_tables(entity_term, store.accumulated_tables)
        if covered_existing:
            logger.info(
                "retrieve_for_entity: '%s' already covered by %s.%s — skipping vector search",
                entity_term,
                matched_table,
                matched_col,
            )
            store.entity_results.append(
                {
                    "entity": entity_term,
                    "entity_type": entity_type,
                    "candidates": [],
                    "relevant_tables": [],
                    "relevant_fks": [],
                    "covered_by_existing_table": True,
                    "matched_table": matched_table,
                    "matched_column": matched_col,
                    "sql_expression": None,
                    "filter_field_hint": None,
                }
            )
            return (
                f"'{entity_term}' already covered by column '{matched_col}' "
                f"in table '{matched_table}' — vector search skipped."
            )

        # ── Full vector/graph search ───────────────────────────────────────
        try:
            custom_raw, column_raw = extract_candidates(
                entities_and_concepts=[entity_term],
                query_no_values=entity_term,
                query_with_values=entity_term,
            )
            custom_candidates = clean_results(list(custom_raw or []))
            column_candidates = clean_results(list(column_raw or []))

            # Column candidates are used only to derive their parent tables.
            # Custom analysis candidates go into the candidates list.
            all_for_tables = custom_candidates + column_candidates
            relevant_tables, relevant_fks = get_relevant_fks_from_candidates_tables(all_for_tables)
            add_tables, add_fks = get_relevant_tables(entity_term, k=3)
            relevant_tables.extend(add_tables)
            relevant_fks.extend(add_fks)
            relevant_tables = dedupe_merge_relevant_tables(relevant_tables)
            _apply_foreign_key_hints(relevant_tables, relevant_fks)

            # Accumulate into store
            store.custom_candidates.extend(custom_candidates)
            store.accumulated_tables = dedupe_merge_relevant_tables(store.accumulated_tables + relevant_tables)
            store.accumulated_fks.extend(relevant_fks)

            covered = len(custom_candidates) > 0 or len(relevant_tables) > 0

            store.entity_results.append(
                {
                    "entity": entity_term,
                    "entity_type": entity_type,
                    "candidates": custom_candidates,
                    "relevant_tables": relevant_tables,
                    "relevant_fks": relevant_fks,
                    "covered_by_existing_table": False,
                    "matched_table": None,
                    "matched_column": None,
                    "sql_expression": None,
                    "filter_field_hint": None,
                }
            )

            status = "COVERED" if covered else "NOT COVERED"
            return (
                f"'{entity_term}' — {status}. "
                f"Found {len(custom_candidates)} custom analyses, "
                f"{len(relevant_tables)} tables."
            )

        except Exception as exc:
            logger.warning("retrieve_for_entity failed for %r: %s", entity_term, exc)
            store.entity_results.append(
                {
                    "entity": entity_term,
                    "entity_type": entity_type,
                    "candidates": [],
                    "relevant_tables": [],
                    "relevant_fks": [],
                    "covered_by_existing_table": False,
                    "matched_table": None,
                    "matched_column": None,
                    "sql_expression": None,
                    "filter_field_hint": None,
                }
            )
            return f"'{entity_term}' — retrieval failed: {exc}"

    return retrieve_for_entity


# ---------------------------------------------------------------------------
# Tool 3 — synthesize_expression
# ---------------------------------------------------------------------------


def _make_synthesize_expression_tool(llm: Any, store: RetrievalStore):
    """Return a ``synthesize_expression`` tool that reads columns from *store*
    and patches the matching entity result.
    """

    @tool
    def synthesize_expression(entity_term: str) -> str:
        """Derive a SQL expression for an entity that has no direct candidate match.

        Call this ONLY when retrieve_for_entity returned NOT COVERED for an entity
        and covered_by_existing_table is false.
        Uses all columns accumulated in the store — no column list argument needed.

        Args:
            entity_term: The entity that has no direct candidate (e.g. "revenue").

        Returns:
            A summary: expression derived or failure reason.
        """
        # Collect all column names from accumulated tables
        col_names: list[str] = []
        for table in store.accumulated_tables:
            for col in table.get("columns") or []:
                if isinstance(col, dict):
                    name = col.get("name") or col.get("id") or ""
                elif isinstance(col, str):
                    name = col
                else:
                    continue
                if name:
                    col_names.append(name)

        if not col_names:
            _patch_expression(store, entity_term, "", False)
            return f"'{entity_term}' — no columns available for synthesis."

        prompt = f"""You are a SQL expression composer.

The user question refers to "{entity_term}", but there is no database column with that exact name.
Compose a SQL expression for "{entity_term}" using ONLY the columns listed below.

Available columns (ONLY use these — never invent new column names):
{json.dumps(col_names, indent=2)}

Rules:
1. Use ONLY column names from the list above.
2. Output a SQL expression fragment (NOT a full SELECT statement).
3. Common patterns: subtraction (income - cost = profit), ratio, SUM, COUNT, AVG.
4. If you cannot express "{entity_term}" from these columns, set confidence to "low".

Return a JSON object with keys: expression, columns_used, confidence."""

        messages = [SystemMessage(content=prompt)]
        result = invoke_with_structured_output(llm, messages, _ExpressionResult)

        if result is None:
            _patch_expression(store, entity_term, "", False)
            return f"'{entity_term}' — synthesis LLM call failed."

        valid_col_set = {n.lower() for n in col_names}
        validated_cols = [c for c in result.columns_used if str(c).split(".")[-1].lower() in valid_col_set]
        success = bool(result.expression and result.confidence in ("high", "medium"))
        _patch_expression(store, entity_term, result.expression if success else "", success)

        if success:
            return (
                f"'{entity_term}' — expression synthesized: {result.expression} "
                f"(confidence={result.confidence}, columns={validated_cols})"
            )
        return (
            f"'{entity_term}' — synthesis low confidence ({result.confidence}): "
            f"{result.expression or 'no expression produced'}"
        )

    return synthesize_expression


def _patch_expression(store: RetrievalStore, entity_term: str, expression: str, success: bool) -> None:
    """Write the synthesized expression back into the matching entity_result entry."""
    for r in store.entity_results:
        if r.get("entity") == entity_term:
            r["sql_expression"] = expression if success else None
            return
    # No existing entry — add one marked as unresolved
    store.entity_results.append(
        {
            "entity": entity_term,
            "entity_type": "metric",
            "candidates": [],
            "relevant_tables": [],
            "relevant_fks": [],
            "covered_by_existing_table": False,
            "matched_table": None,
            "matched_column": None,
            "sql_expression": expression if success else None,
            "filter_field_hint": None,
        }
    )


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------


def build_retrieval_tools(payload: AgentPayload, llm: Any) -> tuple[list, RetrievalStore]:
    """Build and return the list of Phase 1 Retrieval Agent tools and a shared ``RetrievalStore``.

    Tools close over the store so all state is accumulated automatically as the
    agent calls them — no JSON passing between tool calls required.

    Args:
        payload: The ``AgentPayload`` from the caller (reserved for future
            session-scoped filtering).
        llm: The LLM client used by ``decompose_question`` and
            ``synthesize_expression``.

    Returns:
        ``(tools, store)`` — tools list and the ``RetrievalStore`` that will be
        populated as the agent runs.
    """
    store = RetrievalStore()
    return [
        _make_decompose_question_tool(llm, store),
        _make_retrieve_for_entity_tool(store),
        _make_synthesize_expression_tool(llm, store),
    ], store


__all__ = ["build_retrieval_tools", "RetrievalStore", "_find_entity_in_tables"]
