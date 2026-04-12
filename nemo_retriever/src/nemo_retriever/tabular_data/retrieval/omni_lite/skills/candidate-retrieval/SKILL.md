---
name: candidate-retrieval
description: How to call extract_entities and retrieve_semantic_candidates, and how to interpret their outputs for SQL generation.
---

# Candidate Retrieval Skill

## Purpose

This skill guides you through the two retrieval tools — `extract_entities` and
`retrieve_semantic_candidates` — that prepare the context you need before writing SQL.

---

## Step 1 — extract_entities

**When**: Always call this as the very first tool on every new question.

**Input**:
- `question` — the raw user question exactly as received.

**Output** (JSON):
```json
{
  "query_no_values": "What is the average order value?",
  "entities_and_concepts": ["Order", "Customer"]
}
```

- `query_no_values`: the question with specific values (dates, numbers, names) stripped.
  Use this as the retrieval query to improve semantic matching.
- `entities_and_concepts`: primary domain concepts used to seed entity-level search.

**Important**: pass `query_no_values` and `entities_and_concepts` directly to
`retrieve_semantic_candidates` in the next step.

---

## Step 2 — retrieve_semantic_candidates

**When**: Call immediately after `extract_entities`, before generating any SQL.

**Inputs**:
- `question` — full question (with values).
- `query_no_values` — stripped question from Step 1.
- `entities_and_concepts_json` — JSON-encoded array from Step 1.

**Output** (JSON object with these fields):

### `complex_candidates_str` — highest-priority context

A list of formatted strings describing semantic candidates:
```
name: <name>, label: custom_analysis, id: <id> [CERTIFIED], sql_snippet: SELECT ...
```

- `[CERTIFIED]` candidates have been validated — prefer them for SQL structure.
- `sql_snippet` shows how the business metric is typically computed — use as a **pattern reference**, not a copy-paste template.
- Do NOT copy table aliases from snippets.  Define your own aliases in FROM/JOIN.

### `table_groups` — connected table clusters

```json
[
  {
    "tables": [{"name": "SALES", "schema_name": "PUBLIC", ...}, ...],
    "fks": [{"table1": "PUBLIC.ORDERS", "table2": "PUBLIC.ORDERLINES", "columns": [...]}]
  }
]
```

- Each group is a set of tables connected by foreign keys.
- Use the FK list inside a group for JOIN conditions.
- Pick the group whose tables are most relevant to the question.
- Tables across different groups have NO FK path — do not join them.

### `relevant_tables` — flat table list with column details

Used as fallback when snippets are insufficient.  Each entry includes the table's
available columns.  Only reference columns that appear in this list.

### `relevant_fks` — foreign key relationships

Flat list of FK dicts used to validate JOIN conditions.
**Never invent a JOIN** that does not appear in this list.

### `relevant_queries` — example SQL patterns

Historical queries for style reference.  Use for idioms (date functions, aggregation
patterns) but adapt them to the current question and retrieved tables.

---

## Interpreting Results

| Scenario | Action |
|----------|--------|
| `complex_candidates_str` has `[CERTIFIED]` snippets | Use as primary reference for SQL structure |
| `complex_candidates_str` has non-certified snippets | Use with caution; verify columns against `relevant_tables` |
| `complex_candidates_str` is empty | Build SQL from `table_groups` + `relevant_fks` alone |
| `table_groups` is empty | Fall back to `relevant_tables`; skip FK-based joins |
| All fields empty | Return `sql_code: ""` and explain in `answer` |

---

## Common Mistakes to Avoid

- Do NOT call `retrieve_semantic_candidates` without calling `extract_entities` first.
- Do NOT use table or column names that are not in `relevant_tables`.
- Do NOT copy aliases from `sql_snippet` examples — always define your own.
- Do NOT join tables from different `table_groups` unless you find a FK linking them.
