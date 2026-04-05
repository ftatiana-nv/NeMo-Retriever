# SPDX-FileCopyrightText: Copyright (c) 2024-25, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Omni-lite extension of :class:`nemo_retriever.retriever.Retriever` with optional
LanceDB ``label`` column filtering. Keeps upstream ``retriever.py`` unchanged.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from nemo_retriever.retriever import Retriever


class OmniLiteRetriever(Retriever):
    """Same as :class:`~nemo_retriever.retriever.Retriever`, plus optional ``label_in``.

    When the LanceDB table has a top-level ``label`` column, ``query`` / ``queries``
    can pass ``label_in`` to apply ``WHERE label IN (...)``. Labels stored only in
    JSON ``metadata`` are not filtered server-side; filter in application code.
    """

    @staticmethod
    def _sql_in_literals(values: Sequence[str]) -> str:
        """Escape single-quoted strings for LanceDB ``IN (...)`` predicates."""
        return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)

    def _search_lancedb(
        self,
        *,
        lancedb_uri: str,
        lancedb_table: str,
        query_vectors: list[list[float]],
        query_texts: list[str],
        label_in: Optional[Sequence[str]] = None,
    ) -> list[list[dict[str, Any]]]:
        import lancedb  # type: ignore
        import numpy as np

        db = lancedb.connect(lancedb_uri)
        table = db.open_table(lancedb_table)

        field_names = [f.name for f in table.schema]
        label_where: Optional[str] = None
        if label_in:
            labels = [str(x) for x in label_in if x is not None and str(x).strip()]
            if labels and "label" in field_names:
                label_where = f"label IN ({self._sql_in_literals(labels)})"

        effective_nprobes = int(self.nprobes)
        if effective_nprobes <= 0:
            try:
                for idx in table.list_indices():
                    num_parts = getattr(idx, "num_partitions", None)
                    if num_parts and int(num_parts) > 0:
                        effective_nprobes = int(num_parts)
                        break
            except Exception:
                pass
            if effective_nprobes <= 0:
                effective_nprobes = 16

        results: list[list[dict[str, Any]]] = []
        for i, vector in enumerate(query_vectors):
            q = np.asarray(vector, dtype="float32")
            top_k = self.top_k if not self.reranker else self.top_k * self.reranker_refine_factor
            if self.hybrid:
                from lancedb.rerankers import RRFReranker  # type: ignore

                chain = (
                    table.search(query_type="hybrid")
                    .vector(q)
                    .text(query_texts[i])
                    .nprobes(effective_nprobes)
                    .refine_factor(int(self.refine_factor))
                )
                if label_where:
                    chain = chain.where(label_where)
                hybrid_cols = [
                    "text",
                    "metadata",
                    "source",
                    "page_number",
                    "pdf_page",
                    "pdf_basename",
                    "source_id",
                    "path",
                ]
                if "label" in field_names:
                    hybrid_cols = ["label"] + hybrid_cols
                hits = (
                    chain.select(hybrid_cols)
                    .limit(int(top_k))
                    .rerank(RRFReranker())
                    .to_list()
                )
            else:
                dense_select = [
                    "text",
                    "metadata",
                    "source",
                    "page_number",
                    "_distance",
                    "pdf_page",
                    "pdf_basename",
                    "source_id",
                    "path",
                ]
                if "label" in field_names:
                    dense_select = ["label"] + dense_select
                chain = (
                    table.search(q, vector_column_name=self.vector_column_name)
                    .nprobes(effective_nprobes)
                    .refine_factor(int(self.refine_factor))
                )
                if label_where:
                    chain = chain.where(label_where)
                hits = chain.select(dense_select).limit(int(top_k)).to_list()
            results.append(hits)
        return results

    def query(
        self,
        query: str,
        *,
        embedder: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
        lancedb_table: Optional[str] = None,
        label_in: Optional[Sequence[str]] = None,
    ) -> list[dict[str, Any]]:
        return self.queries(
            [query],
            embedder=embedder,
            lancedb_uri=lancedb_uri,
            lancedb_table=lancedb_table,
            label_in=label_in,
        )[0]

    def queries(
        self,
        queries: Sequence[str],
        *,
        embedder: Optional[str] = None,
        lancedb_uri: Optional[str] = None,
        lancedb_table: Optional[str] = None,
        label_in: Optional[Sequence[str]] = None,
    ) -> list[list[dict[str, Any]]]:
        query_texts = [str(q) for q in queries]
        if not query_texts:
            return []

        resolved_embedder = str(embedder or self.embedder)
        resolved_lancedb_uri = str(lancedb_uri or self.lancedb_uri)
        resolved_lancedb_table = str(lancedb_table or self.lancedb_table)

        endpoint = self._resolve_embedding_endpoint()
        if endpoint is not None:
            vectors = self._embed_queries_nim(
                query_texts,
                endpoint=endpoint,
                model=resolved_embedder,
            )
        else:
            vectors = self._embed_queries_local_hf(
                query_texts,
                model_name=resolved_embedder,
            )

        results = self._search_lancedb(
            lancedb_uri=resolved_lancedb_uri,
            lancedb_table=resolved_lancedb_table,
            query_vectors=vectors,
            query_texts=query_texts,
            label_in=label_in,
        )

        if self.reranker:
            results = self._rerank_results(query_texts, results)

        return results
