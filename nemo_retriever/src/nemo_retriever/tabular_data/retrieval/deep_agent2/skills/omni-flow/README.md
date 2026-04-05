# Skill: Omni SQL Pipeline Flow

This skill provides reference documentation for the Omni SQL generation flow.
It supplements the routing rules in `AGENTS.md` with background context.

---

## Pipeline Architecture

The Omni pipeline is a directed graph with deterministic routing. It mirrors
the structure of the original `omni_lite` LangGraph but runs as a Deep Agent
with explicit tool calls instead of LangGraph nodes.

### Node → Tool Mapping

| LangGraph Node                          | Deep Agent 2 Tool                        |
|-----------------------------------------|------------------------------------------|
| `retrieve_candidates`                   | `retrieve_candidates`                    |
| `extract_action_input`                  | `extract_action_input`                   |
| `calculation_search`                    | `calculation_search`                     |
| `prepare_candidates`                    | `prepare_candidates`                     |
| `construct_sql_from_multiple_snippets`  | `construct_sql_from_multiple_snippets`   |
| `construct_sql_not_from_snippets`       | `construct_sql_not_from_snippets`        |
| `validate_sql_query`                    | `validate_sql_query`                     |
| `validate_intent`                       | `validate_intent`                        |
| `reconstruct_sql`                       | `reconstruct_sql`                        |
| `format_sql_response`                   | `format_sql_response`                    |
| `execute_sql_query`                     | `execute_sql_query`                      |
| `calc_respond`                          | `calc_respond`                           |
| `unconstructable_sql_response`          | `unconstructable_sql_response`           |
| `finalize_text_based_answer`            | `finalize_text_based_answer`             |

---

## Routing Functions → Tool Return Values

The original LangGraph had three routing functions. These are now encoded
inside the tools themselves, and the routing decision is returned as the
`"decision"` key in each tool's JSON output.

### `route_decision` (was on `construct_sql_from_multiple_snippets`)
```
"constructable"   → validate_sql_query
"unconstructable" → unconstructable_sql_response
```

### `route_sql_validation` (was on `validate_sql_query`)
```
"valid_sql"              → validate_intent
"skip_intent_validation" → format_sql_response  (reconstruction_count > 5)
"invalid_sql"            → reconstruct_sql       (attempts < 8)
"fallback"               → construct_sql_not_from_snippets  (attempts == 4)
"unconstructable"        → unconstructable_sql_response     (attempts == 8)
```

### `route_intent_validation` (was on `validate_intent`)
```
"valid_sql"   → format_sql_response
"invalid_sql" → reconstruct_sql
```

---

## State File Structure

All tools share a single JSON state file. Key fields:

```json
{
  "initial_question": "...",
  "pg_connection_string": "...",
  "language": "english",
  "decision": "<last routing decision>",
  "path_state": {
    "normalized_question": "...",
    "query_no_values": "...",
    "entities_and_concepts": ["..."],
    "retrieved_candidates": [...],
    "calculation_candidates": [...],
    "prepared_candidates": [...],
    "current_sql": "SELECT ...",
    "validation_error": "...",
    "intent_feedback": "...",
    "sql_attempts": 0,
    "reconstruction_count": 0,
    "final_sql": "SELECT ...",
    "final_answer": "...",
    "final_result": {...},
    "execution_result": [...],
    "node_visit_counts": {"retrieve_candidates": 1, ...}
  }
}
```

---

## Debugging Tips

- Check `path_state.node_visit_counts` to see which nodes executed.
- Check `path_state.sql_attempts` to see how many validation rounds occurred.
- Check `path_state.reconstruction_count` to see how many reconstructions ran.
- Check `path_state.validation_error` and `path_state.intent_feedback` to understand why SQL failed.
- If `sql_attempts >= 8`, the pipeline terminates via `unconstructable_sql_response`.
- If `reconstruction_count > 5`, `validate_sql_query` returns `"skip_intent_validation"` to break loops.
