"""
Candidate Preparation Agent

This agent prepares and fetches all candidates needed for SQL construction.
It runs before SQL generation agents to gather all necessary context.

Responsibilities:
- Fetch relevant tables and foreign keys from candidates
- Retrieve relevant queries for context
- Find similar questions from conversation history
- Filter and process complex candidates (with SQL snippets, metrics, analyses)
- Store all prepared data in path_state for downstream agents

Design Decisions:
- Runs before SQL generation to separate data fetching from SQL construction logic
- Stores fetched data in path_state for reusability across multiple SQL agents
- Handles embeddings and conversation history lookup
"""

import logging
from typing import Dict, Any

import networkx as nx

from search.api.omni.agent.agents.shared.types import AgentState
from search.api.omni.agent.agents.base import BaseAgent
from search.api.omni.agent.agents.shared.helpers import (
    get_relevant_fks_from_candidates_tables,
    get_relevant_queries,
    get_relevant_tables,
)
from enrichments.vector_search.embeddings import get_embeddings
from search.api.omni.agent.agents.calculation.utils import (
    find_similar_questions,
    get_question_for_processing,
)
from shared.graph.model.reserved_words import Labels

logger = logging.getLogger(__name__)


class CandidatePreparationAgent(BaseAgent):
    """
    Agent that prepares and fetches all candidates for SQL construction.

    This agent gathers all necessary context before SQL generation:
    - Relevant tables and foreign keys
    - Relevant queries for context
    - Similar questions from conversation history
    - Filtered complex candidates with SQL snippets

    Input Requirements:
    - path_state["candidates"]: List of candidates
    - state["initial_question"]: User's question
    - state["account_id"]: Account ID
    - state["user_participants"]: User IDs

    Output:
    - path_state["candidates"]: All candidates with entities
    - path_state["tables_with_entities"]: Tables with entity info
        Each: {"table": table_obj, "entities": [...], "candidate_ids": [...]}
    - path_state["fks_with_entities"]: FKs with entity info
        Each: {"fk": fk_obj, "entities": [...]}
    - path_state["table_groups"]: Tables grouped by FK relationships with entity info
        Each group: {"tables": [...], "entities": [...], "candidate_ids": [...], "fks": [...]}
    - path_state["relevant_queries"]: Relevant queries for context
    - path_state["similar_questions"]: Similar questions from history
    - path_state["complex_candidates"]: Filtered complex candidates
    - path_state["complex_candidates_str"]: String representation for prompts
    """

    def __init__(self):
        super().__init__("candidate_preparation")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that candidates are available."""
        path_state = state.get("path_state", {})
        if not path_state.get("candidates"):
            self.logger.warning("No candidates found for candidate preparation")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Prepare and fetch all candidates for SQL construction.

        Gathers tables, foreign keys, queries, similar questions, and processes complex candidates.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains all prepared candidate data
        """
        path_state = state.get("path_state", {})
        question = get_question_for_processing(state)
        account_id = state["account_id"]
        user_id = state["user_participants"][0] if state["user_participants"] else None
        candidates_with_entities = path_state["candidates"]

        self.logger.info(
            f"Preparing candidates for {len(candidates_with_entities)} candidates"
        )

        # Get tables and FKs with entity information
        tables_with_entities, fks_with_entities = (
            get_relevant_fks_from_candidates_tables(
                account_id, candidates_with_entities
            )
        )

        # Extract flat tables for additional table search (exclude tables we already have)
        exclude_ids = [item["table"]["id"] for item in tables_with_entities]
        additional_tables, additional_fks = get_relevant_tables(
            account_id,
            state["user_participants"],
            question,
            k=5,
            exclude_ids=exclude_ids,
        )

        # Add additional tables with entity=None (they're from question search, not entity-specific)
        for table in additional_tables:
            tables_with_entities.append(
                {
                    "table": table,
                    "entities": [],  # Additional tables don't have specific entities
                    "candidate_ids": [],
                }
            )

        # Add additional FKs with their entities (based on table connections)
        table_name_to_entities = {
            item["table"].get("name", ""): item["entities"]
            for item in tables_with_entities
        }
        for fk in additional_fks:
            table1 = fk.get("table1")
            table2 = fk.get("table2")
            entities = set()
            if table1 in table_name_to_entities:
                entities.update(table_name_to_entities[table1])
            if table2 in table_name_to_entities:
                entities.update(table_name_to_entities[table2])

            fks_with_entities.append({"fk": fk, "entities": list(entities)})

        self.logger.info(
            f"Found {len(tables_with_entities)} relevant tables and {len(fks_with_entities)} foreign keys"
        )

        # Group tables by their foreign key relationships using networkx
        table_groups = self._group_tables_by_fks_with_entities(
            tables_with_entities, fks_with_entities
        )
        self.logger.info(
            f"Grouped {len(tables_with_entities)} tables into {len(table_groups)} connected groups"
        )

        # Get relevant queries for context (extract just candidate objects)
        candidates = [item["candidate"] for item in candidates_with_entities]
        relevant_queries = get_relevant_queries(
            account_id,
            candidates,
            user_participants=state["user_participants"],
            query_tag="illumex-omni",
        )
        self.logger.info(f"Found {len(relevant_queries)} relevant queries")

        # Find similar questions from conversation history
        similar_questions = []
        embeddings_client = get_embeddings(account_id, is_embeddings=True)
        if embeddings_client and user_id:
            similar_questions = find_similar_questions(
                embeddings_client.embed_query(question), user_id
            )
        self.logger.info(
            f"Found {len(similar_questions)} similar questions from conversations"
        )

        # Filter complex candidates (those with SQL snippets, metrics, analyses)
        complex_candidates = [
            {
                "name": x["name"],
                "sql": x.get("sql") or "",
                "label": x["label"],
                "description": x.get("description") or "",
            }
            for x in candidates
            if x.get("complex_attribute", False)
            or x.get("label") in [Labels.ANALYSIS, Labels.METRIC]
            or x.get("certified", "pending") == "certified"
            or len(x.get("documents", [])) > 0
        ]
        self.logger.info(f"Filtered {len(complex_candidates)} complex candidates")

        # Build string representation of complex candidates for prompts
        complex_candidates_str = self._build_complex_candidates_str(candidates)
        self.logger.info(
            f"Built string representation with {len(complex_candidates_str)} entries"
        )

        # Store all prepared data in path_state
        return {
            "path_state": {
                **path_state,
                "candidates": candidates_with_entities,  # Keep original structure with entities
                "tables_with_entities": tables_with_entities,  # Tables with entity info
                "fks_with_entities": fks_with_entities,  # FKs with entity info
                "table_groups": table_groups,  # Grouped tables by FK relationships
                "relevant_queries": relevant_queries,
                "similar_questions": similar_questions,
                "complex_candidates": complex_candidates,
                "complex_candidates_str": complex_candidates_str,
            }
        }

    def _group_tables_by_fks_with_entities(
        self,
        tables_with_entities: list[dict],
        fks_with_entities: list[dict],
    ) -> list[dict]:
        """
        Group tables by their foreign key relationships using networkx.

        Creates a graph where tables are nodes and foreign keys are edges,
        then finds connected components to group related tables together.

        Args:
            tables_with_entities: List of {"table": table_obj, "entities": [...], "candidate_ids": [...]}
            fks_with_entities: List of {"fk": fk_obj, "entities": [...]}

        Returns:
            List of table group dictionaries with structure:
            {
                "tables": [table1, table2, ...],
                "entities": [entity1, entity2, ...],  # Entities associated with this group
                "candidate_ids": [id1, id2, ...],  # Candidate IDs that contributed tables to this group
                "fks": [fk1, fk2, ...]  # Foreign keys connecting tables within this group
            }
        """
        if not tables_with_entities:
            return []

        # Create a mapping of table_name -> table info
        table_by_name = {}
        for item in tables_with_entities:
            table = item["table"]
            table_name = table.get("schema_name", "") + "." + table.get("name", "")
            if table_name:
                if table_name not in table_by_name:
                    table_by_name[table_name] = {
                        "table": table,
                        "entities": set(),
                        "candidate_ids": set(),
                    }
                # Merge entities and candidate_ids
                table_by_name[table_name]["entities"].update(item.get("entities", []))
                table_by_name[table_name]["candidate_ids"].update(
                    item.get("candidate_ids", [])
                )

        # Create graph
        G = nx.Graph()

        # # Add all tables as nodes (using table names)
        # for table_name in table_by_name.keys():
        #     G.add_node(table_name)

        covered_nodes = set()

        # Add edges based on foreign keys
        for fk_item in fks_with_entities:
            fk = fk_item["fk"]
            table1 = fk.get("table1")
            table2 = fk.get("table2")
            if table1 not in covered_nodes:
                G.add_node(table1)
                covered_nodes.add(table1)
            if table2 not in covered_nodes:
                G.add_node(table2)
                covered_nodes.add(table2)

            # if table1 and table2 and table1 in table_by_name and table2 in table_by_name:
            G.add_edge(table1, table2)

        # Find connected components
        connected_components = list(nx.connected_components(G))

        # Group tables from table_by_name by connected_components
        grouped_tables_by_component = []
        for component in connected_components:
            # Get tables that exist in both component and table_by_name
            tables_in_component = [
                {
                    "table_name": table_name,
                    "table": table_by_name[table_name]["table"],
                    "entities": table_by_name[table_name]["entities"],
                    "candidate_ids": table_by_name[table_name]["candidate_ids"],
                }
                for table_name in component
                if table_name in table_by_name
            ]
            if tables_in_component:  # Only add non-empty groups
                grouped_tables_by_component.append(
                    {"component": component, "tables": tables_in_component}
                )

        # Convert component sets to table group dicts with entity information
        table_groups = []
        for component in connected_components:
            # Get all tables in this component
            group_tables = [
                table_by_name[table_name]["table"]
                for table_name in component
                if table_name in table_by_name
            ]

            # Collect all entities and candidate IDs associated with tables in this group
            entities = set()
            candidate_ids = set()
            for table_name in component:
                if table_name in table_by_name:
                    entities.update(table_by_name[table_name]["entities"])
                    candidate_ids.update(table_by_name[table_name]["candidate_ids"])

            # Collect all FKs that connect tables within this group
            group_fks = []
            for fk_item in fks_with_entities:
                fk = fk_item["fk"]
                table1 = fk.get("table1")
                table2 = fk.get("table2")
                # Include FK if both tables are in this component
                if table1 in component and table2 in component:
                    group_fks.append(fk)

            table_groups.append(
                {
                    "tables": group_tables,
                    "entities": list(entities),
                    "candidate_ids": list(candidate_ids),
                    "fks": group_fks,
                }
            )

        # add all tables that are not in any component to the table_groups
        for table_name in table_by_name.keys():
            if table_name not in covered_nodes:
                table_groups.append(
                    {
                        "tables": [table_by_name[table_name]["table"]],
                        "entities": table_by_name[table_name]["entities"],
                        "candidate_ids": table_by_name[table_name]["candidate_ids"],
                    }
                )

        # Sort groups by size (largest first) for consistency
        table_groups.sort(key=lambda g: len(g["tables"]), reverse=True)

        return table_groups

    def _build_complex_candidates_str(self, candidates: list) -> list[str]:
        """
        Build string representation of complex candidates for prompts.

        Prioritizes certified candidates by sorting them first and including
        certification status in the string representation.

        Args:
            candidates: List of all candidates

        Returns:
            List of formatted candidate strings (certified ones first)
        """
        complex_candidates = []
        for x in candidates:
            if (
                x.get("complex_attribute", False)
                or x.get("label") in [Labels.ANALYSIS, Labels.METRIC]
                or x.get("certified", "pending") == "certified"
                or len(x.get("documents", [])) > 0
            ):
                complex_candidates.append(x)

        # Sort to prioritize certified candidates
        # Certified candidates come first, then others sorted by score
        def sort_key(candidate):
            is_certified = candidate.get("certified", "pending") == "certified"
            score = candidate.get("score", 0)
            # Return tuple: (not_certified, -score) so certified=True sorts first
            # Then by score descending
            return (not is_certified, -score)

        complex_candidates.sort(key=sort_key)

        complex_candidates_str = []
        for x in complex_candidates:
            is_certified = x.get("certified", "pending") == "certified"
            certified_marker = " [CERTIFIED]" if is_certified else ""

            preview = self._get_cleaned_sql(x)
            if preview:
                complex_candidates_str.append(
                    f"name: {x['name']}, label: {x['label']}, id: {x['id']}{certified_marker}, sql_snippet: {preview}"
                )
            else:
                # No SQL preview available; still include basic metadata
                complex_candidates_str.append(
                    f"name: {x['name']}, label: {x['label']}, id: {x['id']}{certified_marker}"
                )
        return complex_candidates_str

    def _get_cleaned_sql(self, candidate: dict) -> str:
        """
        Build a short, clean SQL preview for prompts.

        - Uses the first sql snippet's `sql_code` when available.
        - Avoids dumping full Python list/dict repr with heavy escaping.

        Args:
            candidate: Candidate dictionary

        Returns:
            Cleaned SQL string
        """
        sql_entries = candidate.get("sql") or []
        if isinstance(sql_entries, list) and sql_entries:
            raw = (
                sql_entries[0].get("sql_code")
                or sql_entries[0].get("snippet")
                or sql_entries[0].get("sql_snippet")
                or ""
            )
            if not isinstance(raw, str):
                raw = str(raw)
            # Light cleanup: reduce common escaping that confuses the model
            cleaned = raw.replace('\\"', '"')
            # Turn escaped newlines into real newlines for readability
            cleaned = cleaned.replace("\n", " ")
            return cleaned
        return ""
