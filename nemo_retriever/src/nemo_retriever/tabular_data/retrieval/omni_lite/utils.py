import logging
from enum import StrEnum
from itertools import groupby

from langchain_community.vectorstores import PGVector


logger = logging.getLogger(__name__)

# ==================== CONSTANTS ====================

# Hard ceiling on how many candidate snippets we want to reason over for a single question.
# Larger numbers tend to confuse the LLM and increase latency.
MAX_CALCULATION_CANDIDATES = 15

# k for each get_semantic_candidates_information call (qnv, qwv, per-entity).
SEMANTIC_CANDIDATES_K = 10

# Per-attempt batch size when pulling semantic candidates. We keep this modest
# to balance recall with response time and to leave room for deduping.
ATTR_CANDIDATE_BATCH_SIZE = 6

# Maximum number of batch pulls before we give up looking for fresh attributes;
# empirically, three rounds either fill the quota or show the backend is stuck on duplicates.
ATTR_CANDIDATE_MAX_ATTEMPTS = 3

# Budget for entity-specific candidate searches. This limits how many candidates we fetch
# per entity when ensuring entity coverage. Distributed evenly across entities (at least 1 per entity).
# This prevents over-fetching when there are many entities while ensuring each gets coverage.
ENTITY_CANDIDATE_BUDGET = 4


QUERIES_USAGE_PERCENTILE = "queries_usage_percentile"
TABLES_USAGE_PERCENTILE = "tables_usage_percentile"
COLUMNS_USAGE_PERCENTILE = "columns_usage_percentile"




class Labels(StrEnum):
    """Semantic labels used by omni-lite candidate retrieval."""

    CUSTOM_ANALYSIS = "custom_analysis"
    COLUMN = "column"

def store_usage_percentiles(
    percentiles_type_name: str,
    usage_percentile_25: int,
    usage_percentile_75: int,
):
    query = """
            MATCH (n:db)
            WITH n
            CALL apoc.create.setProperties(n, [$percentiles_type_name_25, $percentiles_type_name_75], [$usage_percentile_25, $usage_percentile_75]) 
            YIELD node
            RETURN n
            """
    conn.query_write(
        query=query,
        parameters={
            "percentiles_type_name_25": f"{percentiles_type_name}_25",
            "percentiles_type_name_75": f"{percentiles_type_name}_75",
            "usage_percentile_25": usage_percentile_25,
            "usage_percentile_75": usage_percentile_75,
        },
    )



def get_stored_usage_percentiles(percentiles_type_name: str):
    query = f"""
                MATCH (n:db)
                RETURN n.{f"{percentiles_type_name}_25"} as usage_percentile_25, n.{f"{percentiles_type_name}_75"} as usage_percentile_75
                """
    results = conn.query_read_only(
        query=query,
        parameters={
        },
    )
    return results

def get_count_str_by_month(alias: str):
    current = date.today().replace(day=1)
    count_3_month = []

    for i in range(0, 3):
        count_3_month.append(f"coalesce({alias}.cnt_{current.month}_{current.year}, 0)")
        prev = current - timedelta(days=1)
        current = prev.replace(day=1)

    count_str = "+".join(count_3_month)
    return count_str

def init_queries_usage_percentiles():
    count_string = get_count_str_by_month("n")
    query_all = f"""match(n:sql{{is_sub_select:FALSE}})
                     return collect({count_string}) as usages """
    usages_result = conn.query_read_only(
        query=query_all, parameters={}
    )
    usages = usages_result[0]["usages"]
    if len(usages) == 0:
        return 0, 0
    usage_percentile_25 = np.percentile(usages, 25)
    usage_percentile_75 = np.percentile(usages, 75)
    store_usage_percentiles(
        QUERIES_USAGE_PERCENTILE, usage_percentile_25, usage_percentile_75
    )
    return usage_percentile_25, usage_percentile_75


def get_usage_percentiles():
    stored_percentiles = get_stored_usage_percentiles(
        QUERIES_USAGE_PERCENTILE
    )
    if len(stored_percentiles) == 0 or (
        stored_percentiles[0]["usage_percentile_25"] is None
    ):
        usage_percentile_25, usage_percentile_75 = init_queries_usage_percentiles()
    else:
        usage_percentile_25 = stored_percentiles[0]["usage_percentile_25"]
        usage_percentile_75 = stored_percentiles[0]["usage_percentile_75"]

    return usage_percentile_25, usage_percentile_75


def get_queries_usage_percentiles(node_str="node"):
    count_str = get_count_str_by_month(node_str)
    usage_percentile_25, usage_percentile_75 = get_usage_percentiles()
    return usage_percentile_25, usage_percentile_75, count_str


def clean_results(raw_candidates: list[dict]) -> list[dict]:
    """
    Normalize raw semantic hits: require id, dedupe by (label, id), preserve order.
    """
    out: list[dict] = []
    seen: set[tuple[str | None, str]] = set()
    for c in raw_candidates or []:
        if not isinstance(c, dict):
            continue
        cid = c.get("id")
        if cid is None:
            continue
        key = (c.get("label"), str(cid))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def update_candidate_properties(candidate: dict, props_by_id: dict) -> None:
    """Merge graph-expanded properties from ``expand_info`` into ``candidate``."""
    cid = candidate.get("id")
    if cid is None:
        return
    extra = props_by_id.get(cid) or props_by_id.get(str(cid))
    if isinstance(extra, dict):
        # values in ``extra`` may be list-wrapped from groupBy; take first dict if so
        candidate.update(extra)


def expand_info(ids_and_labels):
    # define a function for key
    def key_func(k):
        return k["label"]

    results = {}

    (
        queries_percentile_25,
        queries_percentile_75,
        _cnt_str,
    ) = get_queries_usage_percentiles("sql_node")

    for label, ids in groupby(ids_and_labels, key_func):
        label_id_pairs_for_current_label = list(ids)
        query = f"""UNWIND $label_id_pairs as label_id
                    MATCH (n:{label} {{id:label_id.id}})
                    CALL apoc.case([
                        n:custom_analysis,
                            'MATCH(n)-[:analysis_of]->(sql:sql)
                            WITH n, collect(distinct {{sql_code: sql.sql_full_query}}) as sql
                            RETURN apoc.map.setKey(properties(n), "sql", sql) as item',
                        n:column,
                            'MATCH(n)<-[:schema]-(parent)
                            RETURN apoc.map.setPairs(properties(n),[["table_name", parent.name],["table_type", parent.type], ["parent_id", parent.id]]) as item'
                        ],
                        'with n RETURN n{{ .*}} as item ',
                        {{n:n, sql_type: $sql_type, usage_percentile_25: $usage_percentile_25,
                        usage_percentile_75: $usage_percentile_75, {queries_for_columns_params_keys} }}
                        )
                    YIELD value as response
                    WITH collect(response.item) as all_items
                    RETURN apoc.map.groupBy(all_items,'id') as ids_to_props
                    """
        params = {
            "label_id_pairs": label_id_pairs_for_current_label,
            "sql_type": SQLType.QUERY,
            "usage_percentile_25": queries_percentile_25,
            "usage_percentile_75": queries_percentile_75,
        }
        params.update(queries_for_columns_params)
        result = conn.query_read_only(
            query=query,
            parameters=params,
        )
        if len(result) > 0:
            results = results | result[0]["ids_to_props"]

    return results


def search_vector_index(
    index_name,
    allowed_ids,
    user_question,
    k=30,
    label_filter: list[str] | None = None,
):
    vectorstore = PGVector.from_existing_index(
        embedding=get_embeddings(account_id),
        collection_name=index_name,
        connection_string=pg_conn.connection_string,
    )

    filter_kwargs = {}
    if label_filter:
        filter_kwargs["label"] = {"in": label_filter}

    # PGVector returns distance here; convert to a similarity-style score where
    # higher is better by applying (1 - distance).
    result = vectorstore.similarity_search_with_score(
        user_question,
        k=k,
        filter=filter_kwargs,
    )
    # double check for account_id
    return result


def get_semantic_candidates_information(
    entity: str,
    k: int = 5,
    threshold: float = 0,
    list_of_semantic: list | None = None,
):
    """
    Vector search over indexed nodes, then merge graph properties from ``expand_info``.

    Matches the call shape used by ``extract_candidates``:
    ``get_semantic_candidates_information(text, k=..., list_of_semantic=[...])``.
    """
    if list_of_semantic is None:
        list_of_semantic = [Labels.CUSTOM_ANALYSIS, Labels.COLUMN]
    labels = list_of_semantic
    results: list[dict] = []

    nodes_results = search_vector_index(
        "nodes_vector_store",
        entity,
        k=k,
        label_filter=labels,
    )
    results.extend([item["metadata"] for item in nodes_results])

    ids_and_labels = [{"label": x["label"], "id": x["id"]} for x in results]
    props_by_id = expand_info(ids_and_labels)
    for c in results:
        cid = c.get("id")
        if cid is None:
            continue
        extra = props_by_id.get(cid) or props_by_id.get(str(cid))
        if isinstance(extra, dict):
            c.update(extra)

    results.sort(key=lambda item: item.get("score", 0), reverse=True)
    return results


def _dedupe_best_score_sort_cap(combined: list[dict]) -> list[dict]:
    """Deduplicate by (label, id), keep best score, sort descending, cap."""
    best_by_key: dict[tuple[str | None, str], dict] = {}
    for c in combined:
        cid = c.get("id")
        if cid is None:
            continue
        key = (c.get("label"), str(cid))
        score = float(c.get("score") or 0)
        prev = best_by_key.get(key)
        if prev is None or score > float(prev.get("score") or 0):
            best_by_key[key] = c

    unique = list(best_by_key.values())
    unique.sort(key=lambda x: float(x.get("score") or 0), reverse=True)
    return unique[:MAX_CALCULATION_CANDIDATES]


def extract_candidates(
    entities: list[str],
    query_no_values: str,
    query_with_values: str = "",
) -> tuple[list[dict], list[dict]]:
    """
    One semantic search per string: ``query_no_values``, ``query_with_values`` (if distinct),
    and each entity name. For each string, pull custom analyses and columns via
    ``get_semantic_candidates_information``. Merge streams, dedupe by (label, id) keeping best score,
    sort by score, cap at ``MAX_CALCULATION_CANDIDATES`` per stream.

    Returns:
        ``(custom_analysis_candidates, column_candidates)``
    """
    qnv_text = (query_no_values or "").strip()
    qwv_text = (query_with_values or "").strip()

    pulls: list[str] = []
    if qnv_text:
        pulls.append(qnv_text)
    if qwv_text and qwv_text != qnv_text:
        pulls.append(qwv_text)
    for ent in entities or []:
        t = (ent or "").strip()
        if t:
            pulls.append(t)

    combined_custom: list[dict] = []
    combined_columns: list[dict] = []
    for text in pulls:
        t = (text or "").strip()
        if not t:
            continue
        combined_custom.extend(
            get_semantic_candidates_information(
                t,
                k=SEMANTIC_CANDIDATES_K,
                list_of_semantic=[Labels.CUSTOM_ANALYSIS],
            )
            or []
        )
        combined_columns.extend(
            get_semantic_candidates_information(
                t,
                k=SEMANTIC_CANDIDATES_K,
                list_of_semantic=[Labels.COLUMN],
            )
            or []
        )

    out_custom = _dedupe_best_score_sort_cap(combined_custom)
    out_columns = _dedupe_best_score_sort_cap(combined_columns)

    logger.info(
        f"extract_candidates: {len(out_custom)} custom_analysis, {len(out_columns)} column "
        f"(max {MAX_CALCULATION_CANDIDATES} each), {len(pulls)} pulls"
    )

    return out_custom, out_columns
