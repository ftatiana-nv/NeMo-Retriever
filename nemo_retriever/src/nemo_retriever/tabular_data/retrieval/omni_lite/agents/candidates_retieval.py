"""
Candidate Retrieval Agent

This agent performs semantic search to retrieve relevant candidates from the graph.
It's a foundational agent that provides evidence for routing decisions.

Responsibilities:
- Perform semantic search for graph entities (custom analyses)
- Clean and expand candidate properties
- Build summary for routing agent

Design Decisions:
- Generic retrieval 
- Results by CalculationAgent
- Categorization helps routing agent understand what's available
"""

import logging
from typing import Dict, Any

from nemo_retriever.tabular_data.retrieval.omni_lite.graph import AgentState
from nemo_retriever.tabular_data.retrieval.omni_lite.base import BaseAgent
from search.api.omni.agent.agents.shared.helpers import (
    clean_results,
    update_candidate_properties,
)
from search.api.omni.dal import expand_info
from nemo_retriever.tabular_data.retrieval.omni_lite.utils import (
    Labels as OmniLiteLabels,
    extract_candidates,
)

logger = logging.getLogger(__name__)

def get_question_for_processing(state: Dict[str, Any]) -> str:
    """
    Get the appropriate question for SQL processing.

    Uses normalized_question if available in path_state (for calculation_sql flow),
    otherwise falls back to initial_question.

    Args:
        state: Agent state containing path_state and initial_question

    Returns:
        The question to use for processing (normalized or original)
    """
    path_state = state.get("path_state", {})
    normalized_question = path_state.get("normalized_question")

    if normalized_question:
        return normalized_question

    initial_question = state.get("initial_question", "")
    return initial_question



class CandidateRetrievalAgent(BaseAgent):
    """
    Agent that retrieves candidates from semantic search.

    This agent performs the initial retrieval that informs routing decisions.
    It searches for relevant graph entities and categorizes them.

    Retrieval Strategy:
    - Semantic search over all entity types (customanalyses)
    - Top 10 candidates (balance between coverage and latency)
    - Clean and expand candidate properties

    Input Requirements:
    - path_state["normalized_question"]: Normalized English question (from LanguageDetectionAgent)

    Output:
    - path_state["retrieved_candidates"]: All cleaned candidates (custom analyses + columns)
    - path_state["retrieved_custom_analyses"]: Subset with label ``custom_analysis``
    - path_state["retrieved_column_candidates"]: Subset with label ``column``
    """

    def __init__(self):
        super().__init__("candidate_retrieval")

    def validate_input(self, state: AgentState) -> bool:
        """Validate that question is available."""
        question = get_question_for_processing(state)
        if not question:
            self.logger.warning("No question available for retrieval")
            return False
        return True

    def execute(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve candidates from semantic search.

        Performs semantic search, cleans results, expands properties, and
        categorizes candidates for routing decisions.

        Args:
            state: Current agent state

        Returns:
            Dictionary with:
            - path_state: Contains retrieved candidates and categorization
        """
        account_id = state["account_id"]
        user_participants = state.get("user_participants", [])
        path_state = state.get("path_state", {})

        question = get_question_for_processing(state)

        try:
            # Semantic search: custom analyses + columns (see extract_candidates).
            entities = state.get("entities_and_concepts", []) 
            query_no_values = path_state.get(
                "query_no_values", ""
            )

            extracted = extract_candidates(
                entities,
                query_no_values,
                state.get("initial_question", "") or "",
            )

            # Support both formats:
            # - legacy list[{"candidate": ..., "entity": ...}]
            # - tuple(custom_candidates, column_candidates)
            if isinstance(extracted, tuple) and len(extracted) == 2:
                custom_raw, column_raw = extracted
                merged_raw_candidates = (custom_raw or []) + (column_raw or [])
            else:
                merged_raw_candidates = []
                for item in extracted or []:
                    merged_raw_candidates.append(item.get("candidate", item))

            # Clean and expand candidates
            cleaned_candidates = clean_results(account_id, merged_raw_candidates)
            ids_and_labels = [
                {"label": c["label"], "id": c["id"]} for c in cleaned_candidates
            ]
            candidates_properties = expand_info(
                account_id, user_participants, ids_and_labels
            )

            # Update candidate properties
            for candidate in cleaned_candidates:
                update_candidate_properties(
                    account_id, candidate, candidates_properties
                )

            # Store in path_state (combined + split by label)
            path_state["retrieved_candidates"] = cleaned_candidates
            path_state["retrieved_custom_analyses"] = [
                c
                for c in cleaned_candidates
                if c.get("label") == OmniLiteLabels.CUSTOM_ANALYSIS
            ]
            path_state["retrieved_column_candidates"] = [
                c
                for c in cleaned_candidates
                if c.get("label") == OmniLiteLabels.COLUMN
            ]

            self.logger.info(
                f"Retrieved {len(cleaned_candidates)} candidates "
                f"({len(path_state['retrieved_custom_analyses'])} custom_analysis, "
                f"{len(path_state['retrieved_column_candidates'])} column)"
            )

            return {"path_state": path_state}

        except Exception as e:
            # Fallback: empty candidates (routing agent will handle)
            self.logger.warning(
                f"Candidate retrieval failed: {e}, returning empty candidates"
            )
            path_state["retrieved_candidates"] = []
            path_state["retrieved_custom_analyses"] = []
            path_state["retrieved_column_candidates"] = []

            return {"path_state": path_state}
