"""Tool: prepare_candidates

Maps to the ``prepare_candidates`` node in the omni_lite LangGraph
(backed by ``CandidatePreparationAgent``).

Responsibility:
- Rank and filter the merged candidates from calculation_search.
- For each candidate keep only the highest-usage SQL snippet.
- Populate ``path_state["prepared_candidates"]`` – the final list used
  for SQL construction.
"""

from __future__ import annotations

import json
import logging

from langchain_core.tools import tool

from nemo_retriever.tabular_data.retrieval.deep_agent2.state import (
    load_state,
    log_node_visit,
    save_state,
)

logger = logging.getLogger(__name__)

_MAX_PREPARED = 10


@tool
def prepare_candidates(state_path: str) -> str:
    """Rank and filter candidates, retaining the best SQL snippet per candidate.

    Reads calculation_candidates from the state, deduplicates, keeps only the
    most relevant entries, and stores the result as prepared_candidates.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "candidate_count", "decision".
    """
    state = load_state(state_path)
    log_node_visit(state, "prepare_candidates")

    path_state = state.get("path_state", {})
    candidates: list[dict] = path_state.get("calculation_candidates", []) or path_state.get("retrieved_candidates", [])

    # Sort by score ascending (lower distance = more similar).
    try:
        candidates = sorted(candidates, key=lambda c: float(c.get("score", 0.0)))
    except Exception:
        pass

    # Deduplicate on text.
    seen_texts: set[str] = set()
    prepared: list[dict] = []
    for cand in candidates:
        text = (cand.get("text") or "").strip()
        if not text or text in seen_texts:
            continue
        seen_texts.add(text)
        prepared.append(cand)
        if len(prepared) >= _MAX_PREPARED:
            break

    logger.info("prepare_candidates: %d → %d after dedup/rank", len(candidates), len(prepared))

    path_state["prepared_candidates"] = prepared
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "prepare_candidates",
            "candidate_count": len(prepared),
            "decision": "done",
        }
    )
