main_system_prompt_template = (
    "Today's date is: {{ 'Year': {date.year}, 'Month': {date.month}, 'Day': {date.day}, "
    "'Time': '{date.hour:02}:{date.minute:02}:{date.second:02}' }}.\n\n"
    "Ontology: {ontology_prompt}\n\n"
    "dialects: {dialects}"
)

format_create_sql_general_prompt = (
    "Question: {question}\n"
    "Relevant Tables: {relevant_tables}\n"
    "Relevant FKs: {relevant_fks}\n"
)

sql_response_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "Assistant_to_generate_sql_response",
        "description": "Schema for extracting a sql that directly answers the user's question.",
        "schema": {
            "type": "object",
            "properties": {
                "sql_code": {
                    "type": "string",
                    "description": "A valid SQL that directly answers the user's question",
                }
            },
            "required": ["sql_code"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

sql_scoring_schema = {
    "type": "json_schema",
    "json_schema": {
        "name": "Assistant_to_score_sql_code",
        "description": "Schema for scoring SQL code against ground truth and question relevance.",
        "schema": {
            "type": "object",
            "properties": {
                "logic_match": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Score from 0 to 1 indicating how well the SQL logic answers the given question",
                },
                "logic_issues": {
                    "type": "string",
                    "description": "Short text explaining what issues reduced the logic_match score",
                },
                "semantic_match": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Score from 0 to 1 indicating how well the SQL response types match the expected types for the question",
                },
                "final_weighted_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Combined score (50% logic_match + 50% semantic_match) indicating overall question answering quality",
                },
                "sql_compared_to_ground_truth_score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "Score from 0 to 1 indicating how similar the SQL is to the ground truth SQL",
                },
                "is_valid_sql": {
                    "type": "boolean",
                    "description": "Whether the SQL code is syntactically valid and executable",
                },
                "is_sql_returns_data": {
                    "type": "boolean",
                    "description": "Whether the SQL execution returned any data/results",
                },
            },
            "required": [
                "logic_match",
                "logic_issues",
                "semantic_match",
                "final_weighted_score",
                "sql_compared_to_ground_truth_score",
                "is_valid_sql",
                "is_sql_returns_data",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

sql_scoring_prompt = """
You are an expert SQL evaluator. Your task is to score SQL code based on three separate criteria:

1. **Logic Match**: How well does the SQL logic answer the given question?
2. **Semantic Match**: How well do the SQL response types match the expected types for the question?
3. **Ground Truth Similarity**: How similar is the SQL to the provided ground truth SQL?

**Logic Match Scoring Guidelines:**
- Scores should be between 0.0 and 1.0
- Evaluate how well the SQL logic addresses the question's intent:
  - Does the SQL use appropriate tables, columns, and filters?
  - Does the query logic match the question's requirements?
  - Are the joins, aggregations, and conditions correct?
  - Does it capture the business logic behind the question?
- Provide short text in logic_issues explaining what reduced the score (e.g., "Missing WHERE clause for date filter", "Wrong aggregation function", "Incorrect table join")

**Semantic Match Scoring Guidelines:**
- Scores should be between 0.0 and 1.0
- Evaluate if the SQL response types match what the question expects:
  - **Format Appropriateness**: Does the response format match what the question is asking for?
    * Questions asking for "the earliest date" or "the maximum value" should return a single value, not a table of multiple values
    * Questions asking for "top 5" should return exactly 5 rows (or fewer if data doesn't exist)
    * Questions asking for specific single values should not return multiple rows
  - **Data Completeness**: Does the response include all relevant information requested?
    * Questions asking for "sales with and without discount" should include BOTH categories in results
    * Questions asking for comparisons should include all relevant comparison groups
    * Questions asking for detailed breakdowns should include all requested dimensions
  - **Important**: If the SQL expected types include the user's question expected types, it is acceptable (e.g., returning more columns than strictly needed is fine if the required ones are present)
  
  **Scoring Examples:**
  - If asking "what is the earliest order date?" and getting a single date value: HIGH score
  - If asking "what is the earliest order date?" and getting a table of all dates: LOWER score (wrong format)
  - If asking "sales with and without discount" and getting both categories: HIGH score
  - If asking "sales with and without discount" and getting only one category: LOWER score (incomplete)

**Final Weighted Score:**
- Calculate as: (logic_match * 0.5) + (semantic_match * 0.5)
- This represents the overall quality of how well the SQL answers the question

**Ground Truth Similarity Guidelines:**
- Compare the SQL structure, logic, tables used, and expected results
- Be lenient with minor syntax differences or equivalent approaches
- Focus on semantic similarity rather than exact text matching

**Question:** {question}

**SQL Code to Evaluate:** 
{sql_code}

**Ground Truth SQL:**
{ground_truth_sql}

**SQL Result Preview (if available):**
{sql_result_preview}

**Evaluation Instructions:**
1. Analyze the SQL logic and score how well it addresses the question's requirements (logic_match)
2. Identify and document any logic issues that reduced the score (logic_issues)
3. Examine the response format and completeness against question expectations (semantic_match)
4. Calculate the final_weighted_score as 50% logic_match + 50% semantic_match
5. Provide the ground truth similarity score as usual
6. Determine if the SQL is valid and returns data

Please provide all scores, logic issues text, and boolean flags based on your comprehensive evaluation.
"""

dual_sql_scoring_prompt = """
You are an expert SQL evaluator. Your task is to score TWO SQL codes (from Omni tool and SQL tool) for the SAME question.
This ensures consistent penalty application across both tools.

**IMPORTANT**: You must apply the SAME penalty criteria to both SQL codes. If you find an issue in one SQL that reduces its score, 
check if the same issue exists in the other SQL and apply the same penalty reduction.

**Scoring Criteria:**

1. **Logic Match**: How well does the SQL logic answer the given question?
2. **Semantic Match**: How well do the SQL response types match the expected types for the question?
3. **Ground Truth Similarity**: How similar is the SQL to the provided ground truth SQL?

**Logic Match Scoring Guidelines:**
- Scores should be between 0.0 and 1.0
- Evaluate how well the SQL logic addresses the question's intent:
  - Does the SQL use appropriate tables, columns, and filters?
  - Does the query logic match the question's requirements?
  - Are the joins, aggregations, and conditions correct?
  - Does it capture the business logic behind the question?
- Provide short text in logic_issues explaining what reduced the score

**Semantic Match Scoring Guidelines:**
- Scores should be between 0.0 and 1.0
- Evaluate if the SQL response types match what the question expects:
  - **Format Appropriateness**: Does the response format match what the question is asking for?
  - **Data Completeness**: Does the response include all relevant information requested?
  - If the SQL expected types include the user's question expected types, it is acceptable

**Final Weighted Score:**
- Calculate as: (logic_match * 0.5) + (semantic_match * 0.5)

**Ground Truth Similarity Guidelines:**
- Compare the SQL structure, logic, tables used, and expected results
- Be lenient with minor syntax differences or equivalent approaches

**Question:** {question}

**Omni Tool SQL Code:** 
{omni_sql_code}

**Omni Tool SQL Result Preview:**
{omni_sql_result_preview}

**SQL Tool SQL Code:**
{sql_tool_sql_code}

**SQL Tool SQL Result Preview:**
{sql_tool_sql_result_preview}

**Ground Truth SQL:**
{ground_truth_sql}

**Evaluation Instructions:**
1. Analyze BOTH SQL codes simultaneously
2. Apply consistent penalty criteria to both (e.g., if missing date filter is -0.2 for one, it should be -0.2 for the other)
3. Score each SQL's logic_match, semantic_match, and provide logic_issues
4. Calculate final_weighted_score for each as 50% logic_match + 50% semantic_match
5. Provide ground truth similarity score for each
6. Determine if each SQL is valid and returns data

Please provide all scores for both tools with consistent penalty application.
"""


def format_deep_agent_user_prompt(question: str) -> str:
    """User message for the Deep Agent benchmark.

    Identity, safety, and SQL behavior are defined in ``AGENTS.md`` and ``skills/``;
    this string is only the per-question task (symmetrical to how ``sql_tool``
    injects the question into its LLM prompt).
    """
    if not (question and str(question).strip()):
        raise ValueError("question must be non-empty")
    return (
        "You are a SQL benchmark assistant.\n\n"
        f"User question: {question.strip()}\n\n"
        "When calling sql_db_query, pass ONLY the raw SQL text (DuckDB dialect). "
        "Do not wrap it in markdown code fences (no ```sql), no backticks, no commentary.\n\n"
        "### Final message (mandatory output contract)\n"
        "Your **last** assistant message must be **only** a single JSON object — no other characters "
        "before or after it. Do not write apologies, preambles, headings, or markdown.\n"
        "Schema:\n"
        '  {"sql_code": "<exact SQL you ran>", "answer": "<short explanation>", '
        '"result": <raw DB value or null>}\n'
        "Rules:\n"
        "- `sql_code`: one string; no ``` fences inside the message (the value may contain quotes).\n"
        "- `answer`: plain text summary for the user question.\n"
        "- `result`: whatever the database returned (number, list of rows, string, or null).\n"
        "- Forbidden in the final message: ``` blocks, 'Here is an example', 'I apologize', or any "
        "text outside the JSON object.\n\n"
    )
