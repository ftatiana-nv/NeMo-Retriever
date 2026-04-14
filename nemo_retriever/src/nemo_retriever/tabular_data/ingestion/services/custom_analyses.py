# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
import uuid

from nemo_retriever.tabular_data.ingestion.dal.queries_dal import add_query
from nemo_retriever.tabular_data.ingestion.model.neo4j_node import Neo4jNode
from nemo_retriever.tabular_data.ingestion.model.reserved_words import Labels, Props
from nemo_retriever.tabular_data.ingestion.services.queries import parse_query_single

logger = logging.getLogger(__name__)


def populate_custom_analyses(
    schemas: dict,
    analyses: list[dict],
    dialect: str,
):
    """Parse custom analysis SQL snippets and write them to the graph.

    Creates two nodes per analysis:
      - **CustomAnalysis** node with ``name``, ``description``, and ``tags``.
      - **Sql** node with the parsed SQL and table/column edges (via
        ``parse_query_single``).

    A ``HAS_SQL`` edge connects ``CustomAnalysis`` → ``Sql``.
    """
    before = time.time()
    logger.info("Starting to ingest %d custom analyses.", len(analyses))

    ingested = 0
    for entry in analyses:
        name = entry.get("name", "")
        sql = (entry.get("sql") or "").strip()
        if not sql:
            logger.warning("Skipping custom analysis %r — no SQL provided.", name)
            continue

        query_obj = parse_query_single(sql=sql, dialect=dialect, schemas=schemas)
        if query_obj is None:
            logger.warning(
                "Could not resolve any tables for custom analysis %r — skipping.",
                name,
            )
            continue

        analysis_id = str(uuid.uuid4())
        analysis_node = Neo4jNode(
            name=name,
            label=Labels.CUSTOM_ANALYSIS,
            props={
                "name": name,
                "description": entry.get("description", ""),
            },
            existing_id=analysis_id,
        )

        edge_props = {Props.ANALYSIS_ID: analysis_id}
        query_obj.edges.append((analysis_node, query_obj.sql_node, edge_props))

        add_query(query_obj.get_edges())
        ingested += 1

    logger.info(
        "Ingested %d/%d custom analyses in %.2fs.",
        ingested,
        len(analyses),
        time.time() - before,
    )
