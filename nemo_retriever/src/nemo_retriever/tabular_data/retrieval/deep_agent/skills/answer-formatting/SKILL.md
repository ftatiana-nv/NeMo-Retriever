---
name: answer-formatting
description: For formatting the final SQL-based answer into a consistent structured output with sql_code, answer, and result
---

## Final turn: JSON only

Your **entire** final assistant message must be **one** JSON object. **Do not** output:

- Apologies (“I apologize…”), help text, or “To fix this…”
- Markdown code fences (`` ``` ``) or ` ```sql ` blocks in the **message body** (the JSON string values may contain SQL text with quotes)
- “Here is an example” or tutorial-style SQL for the user
- Any line of text **before** `{` or **after** `}`

Required shape (keys exactly), one line of JSON:

    {"sql_code": "...", "answer": "...", "result": ...}

- `sql_code` (string): exact SQL executed against DuckDB
- `answer` (string): short explanation tied to the user question
- `result`: raw DB result (scalar, rows, JSON, or `null`)

If the query failed, still emit JSON with `sql_code` set to the attempted SQL, `answer` explaining the failure briefly, and `result`: null — do not replace JSON with a long prose fix-it guide.
