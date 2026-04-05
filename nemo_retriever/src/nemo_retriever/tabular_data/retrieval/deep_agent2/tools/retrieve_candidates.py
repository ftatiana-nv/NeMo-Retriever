"""Tool: retrieve_candidates

Maps to the ``retrieve_candidates`` node in the omni_lite LangGraph.

Responsibility:
- Perform semantic search against the LanceDB tabular index.
- Populate ``path_state["retrieved_candidates"]`` with the top-k hits.
"""

from __future__ import annotations

import json
import logging
import os

from langchain_core.tools import tool

from nemo_retriever.tabular_data.retrieval.deep_agent2.state import (
    load_state,
    log_node_visit,
    save_state,
)

logger = logging.getLogger(__name__)


@tool
def retrieve_candidates(state_path: str) -> str:
    """Retrieve semantic search candidates from LanceDB for the user question.

    Reads the question from the state file, queries the LanceDB tabular index,
    and stores the top-k hits in path_state["retrieved_candidates"].

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "candidate_count", "decision".
    """
    state = load_state(state_path)
    log_node_visit(state, "retrieve_candidates")

    question = (state.get("path_state", {}).get("normalized_question") or state.get("initial_question", "")).strip()

    top_k = int(os.environ.get("OMNI_RETRIEVER_TOP_K", "15"))
    candidates: list[dict] = []

    try:
        from nemo_retriever.retriever import Retriever

        lancedb_table = os.environ.get("OMNI_LANCEDB_TABLE", "nv-ingest-tabular")
        retriever = Retriever(lancedb_table=lancedb_table, top_k=top_k)
        hits = retriever.query(question)
        for i, hit in enumerate(hits):
            text = (hit.get("text") or "").strip()
            if text:
                candidates.append(
                    {
                        "id": str(i),
                        "label": "custom_analysis",
                        "text": text,
                        "score": hit.get("_distance", 0.0),
                    }
                )
        logger.info("retrieve_candidates: fetched %d hits from LanceDB", len(candidates))
    except Exception as exc:
        logger.warning("retrieve_candidates: LanceDB query failed: %s", exc)

    path_state = state.get("path_state", {})
    path_state["retrieved_candidates"] = candidates
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "retrieve_candidates",
            "candidate_count": len(candidates),
            "decision": "done",
        }
    )
