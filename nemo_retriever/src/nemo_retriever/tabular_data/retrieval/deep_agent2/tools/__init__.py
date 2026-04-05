"""Omni Deep Agent 2 – tool registry.

Each tool corresponds to one node in the omni_lite LangGraph.
"""

from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.retrieve_candidates import (
    retrieve_candidates,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.extract_action_input import (
    extract_action_input,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.calculation_search import (
    calculation_search,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.prepare_candidates import (
    prepare_candidates,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.construct_sql_from_snippets import (
    construct_sql_from_multiple_snippets,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.construct_sql_from_tables import (
    construct_sql_not_from_snippets,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.validate_sql import (
    validate_sql_query,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.validate_intent import (
    validate_intent,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.reconstruct_sql import (
    reconstruct_sql,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.format_sql_response import (
    format_sql_response,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.execute_sql import (
    execute_sql_query,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.calc_respond import (
    calc_respond,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.unconstructable_response import (
    unconstructable_sql_response,
)
from nemo_retriever.tabular_data.retrieval.deep_agent2.tools.finalize_text_answer import (
    finalize_text_based_answer,
)

ALL_TOOLS = [
    retrieve_candidates,
    extract_action_input,
    calculation_search,
    prepare_candidates,
    construct_sql_from_multiple_snippets,
    construct_sql_not_from_snippets,
    validate_sql_query,
    validate_intent,
    reconstruct_sql,
    format_sql_response,
    execute_sql_query,
    calc_respond,
    unconstructable_sql_response,
    finalize_text_based_answer,
]

__all__ = [
    "ALL_TOOLS",
    "retrieve_candidates",
    "extract_action_input",
    "calculation_search",
    "prepare_candidates",
    "construct_sql_from_multiple_snippets",
    "construct_sql_not_from_snippets",
    "validate_sql_query",
    "validate_intent",
    "reconstruct_sql",
    "format_sql_response",
    "execute_sql_query",
    "calc_respond",
    "unconstructable_sql_response",
    "finalize_text_based_answer",
]
