"""Omni Deep Agent 2 — LangGraph-free implementation of the omni_lite pipeline.

This package re-implements the omni_lite LangGraph flow using Deep Agents:

- Each LangGraph node becomes a LangChain tool (in ``tools/``).
- Routing logic (``route_decision``, ``route_sql_validation``,
  ``route_intent_validation``) is encoded inside the tools and in
  ``AGENTS.md``, which instructs the deep agent how to orchestrate them.
- State is persisted in a per-run JSON file so all tools share it.

Entry point: ``runtime.run_omni_pipeline``
"""

from __future__ import annotations

from nemo_retriever.tabular_data.retrieval.deep_agent2.runtime import run_omni_pipeline

__all__ = ["run_omni_pipeline"]
