"""Shared state schema for the Omni Deep Agent 2 pipeline.

The state is persisted as a JSON file so every tool can read and write
it without needing a live Python object reference across tool calls.
"""

from __future__ import annotations

import json
import os


# ---------------------------------------------------------------------------
# State init & I/O helpers
# ---------------------------------------------------------------------------


def init_state(
    question: str,
    pg_connection_string: str = "",
    language: str = "english",
) -> dict:
    """Return a fresh pipeline state for *question*."""
    return {
        "initial_question": question,
        "messages": [],
        "decision": "",
        "intermediate_output": "",
        "thoughts": "",
        "pg_connection_string": pg_connection_string,
        "path_state": {
            # ---- set by extract_action_input ----
            "normalized_question": "",
            "query_no_values": "",
            "entities_and_concepts": [],
            # ---- set by retrieve_candidates ----
            "retrieved_candidates": [],
            # ---- set by calculation_search ----
            "calculation_candidates": [],
            # ---- set by prepare_candidates ----
            "prepared_candidates": [],
            # ---- set by SQL construction / reconstruction ----
            "current_sql": "",
            "sql_construction_explanation": "",
            # ---- set by validation agents ----
            "validation_error": "",
            "intent_feedback": "",
            # ---- counters (routing logic) ----
            "sql_attempts": 0,
            "reconstruction_count": 0,
            # ---- set by format / execute / respond ----
            "final_sql": "",
            "final_answer": "",
            "final_result": None,
            "execution_result": None,
            # ---- diagnostics ----
            "node_visit_counts": {},
        },
        "language": language,
    }


def load_state(state_path: str) -> dict:
    """Read pipeline state from *state_path*."""
    with open(state_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_state(state_path: str, state: dict) -> None:
    """Write pipeline state to *state_path* (creates parent dirs)."""
    os.makedirs(os.path.dirname(os.path.abspath(state_path)), exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2, default=str)


def log_node_visit(state: dict, node_name: str) -> None:
    """Increment the visit counter for *node_name* in *state*."""
    path_state = state.get("path_state", {})
    counts = path_state.get("node_visit_counts", {})
    counts[node_name] = counts.get(node_name, 0) + 1
    path_state["node_visit_counts"] = counts
    state["path_state"] = path_state
    total = sum(counts.values())
    import logging

    logging.getLogger(__name__).info("🔁 Node visits: %s | Total: %d", counts, total)
