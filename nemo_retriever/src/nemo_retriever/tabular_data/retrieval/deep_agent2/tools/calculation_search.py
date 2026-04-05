"""Tool: calculation_search

Maps to the ``calculation_search`` node in the omni_lite LangGraph
(backed by ``CalculationSearchAgent``).

Responsibility:
- Perform per-entity searches to ensure every extracted entity is covered
  by at least one candidate.
- Merge entity-specific hits with the already-retrieved candidates.
- Populate ``path_state["calculation_candidates"]`` with the merged list.
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

_MAX_PER_ENTITY = 3
_MAX_TOTAL = 20


@tool
def calculation_search(state_path: str) -> str:
    """Run entity-specific semantic searches to ensure candidate coverage.

    For each entity extracted by extract_action_input, queries LanceDB with
    the entity name. Merges results with the existing retrieved_candidates,
    deduplicating by text content, and stores them as calculation_candidates.

    Args:
        state_path: Absolute path to the pipeline state JSON file.

    Returns:
        JSON string with keys "node", "candidate_count", "decision".
    """
    state = load_state(state_path)
    log_node_visit(state, "calculation_search")

    path_state = state.get("path_state", {})
    entities: list[str] = path_state.get("entities_and_concepts", [])
    query_no_values: str = path_state.get("query_no_values", "") or path_state.get("normalized_question", "")

    # Start with already retrieved candidates.
    existing: list[dict] = list(path_state.get("retrieved_candidates", []))
    seen_texts: set[str] = {c.get("text", "") for c in existing if c.get("text")}
    merged: list[dict] = list(existing)

    try:
        from nemo_retriever.retriever import Retriever

        lancedb_table = os.environ.get("OMNI_LANCEDB_TABLE", "nv-ingest-tabular")

        for entity in entities:
            if len(merged) >= _MAX_TOTAL:
                break
            try:
                retriever = Retriever(lancedb_table=lancedb_table, top_k=_MAX_PER_ENTITY)
                hits = retriever.query(entity)
                for hit in hits:
                    text = (hit.get("text") or "").strip()
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        merged.append(
                            {
                                "id": f"entity_{entity}_{len(merged)}",
                                "label": "custom_analysis",
                                "text": text,
                                "score": hit.get("_distance", 0.0),
                                "entity": entity,
                            }
                        )
            except Exception as exc:
                logger.warning("calculation_search: entity search for %r failed: %s", entity, exc)

        # Also search with the value-stripped query if different from original.
        if query_no_values and len(merged) < _MAX_TOTAL:
            try:
                retriever = Retriever(lancedb_table=lancedb_table, top_k=_MAX_PER_ENTITY)
                hits = retriever.query(query_no_values)
                for hit in hits:
                    text = (hit.get("text") or "").strip()
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        merged.append(
                            {
                                "id": f"qnv_{len(merged)}",
                                "label": "custom_analysis",
                                "text": text,
                                "score": hit.get("_distance", 0.0),
                                "entity": None,
                            }
                        )
            except Exception as exc:
                logger.warning("calculation_search: query_no_values search failed: %s", exc)

    except Exception as exc:
        logger.warning("calculation_search: search failed: %s", exc)

    path_state["calculation_candidates"] = merged[:_MAX_TOTAL]
    state["path_state"] = path_state

    save_state(state_path, state)

    return json.dumps(
        {
            "node": "calculation_search",
            "candidate_count": len(path_state["calculation_candidates"]),
            "decision": "done",
        }
    )
