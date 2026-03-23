import argparse
import csv
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from threading import Lock

import pandas as pd
from tqdm import tqdm
from langchain_core.messages import SystemMessage
from langchain_nvidia_ai_endpoints import ChatNVIDIA


def _make_llm() -> ChatNVIDIA:
    return ChatNVIDIA(
        base_url=os.environ.get("LLM_INVOKE_URL"),
        api_key=os.environ.get("LLM_API_KEY"),
        model=os.environ.get("LLM_MODEL", "meta/llama-3.1-70b-instruct"),
    )

from typing import Annotated
from pydantic import BaseModel, Field, ConfigDict

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

def score_sql_code(question: str, sql_response: dict, ground_truth_sql: str):
    sql_code = sql_response["sql_code"]
    sql_result_preview = sql_response.get("result", "Not available")

    # Use LLM to score the SQL code
    messages = [
        SystemMessage(
            content=sql_scoring_prompt.format(
                question=question,
                sql_code=sql_code,
                ground_truth_sql=ground_truth_sql,
                sql_result_preview=sql_result_preview,
            )
        )
    ]

    try:
        from langchain_core.messages import HumanMessage
        # Append explicit JSON instruction so the model returns parseable output
        json_messages = messages + [
            HumanMessage(content=(
                "Respond ONLY with a valid JSON object matching this schema — no markdown, no explanation:\n"
                '{"logic_match": <float 0-1>, "logic_issues": "<string>", '
                '"semantic_match": <float 0-1>, "final_weighted_score": <float 0-1>, '
                '"sql_compared_to_ground_truth_score": <float 0-1>, "is_valid_sql": <true|false>}'
            ))
        ]
        raw_response = _make_llm().invoke(json_messages)
        raw_text = raw_response.content
        print(f"[LLM scoring] raw response: {raw_text[:500]}")

        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.DOTALL).strip()
        data = json.loads(cleaned)
        response = SqlScoringResponse(**data)

        return {
            "logic_match": response.logic_match,
            "logic_issues": response.logic_issues,
            "semantic_match": response.semantic_match,
            "final_weighted_score": response.final_weighted_score,
            "sql_compared_to_ground_truth_score": response.sql_compared_to_ground_truth_score,
            "is_valid_sql": response.is_valid_sql,
        }

    except Exception as e:
        # Fallback scoring in case of LLM failure
        print(f"Error in LLM scoring: {e}")
        return {
            "logic_match": 0.5,  # Neutral score when unable to evaluate
            "logic_issues": "LLM scoring failed",
            "semantic_match": 0.5,  # Neutral score when unable to evaluate
            "final_weighted_score": 0.5,  # Neutral score when unable to evaluate
            "sql_compared_to_ground_truth_score": 0.5,  # Neutral score when unable to evaluate
            "is_valid_sql": True,
        }


def get_results_csv_headers() -> list:
    """Define and return CSV headers for benchmark results."""
    return [
        "test_index",
        "test_question",
        "ground_truth_sql",
        "sql_tool_sql_code",
        "sql_tool_logic_match",
        "sql_tool_logic_issues",
        "sql_tool_semantic_match",
        "sql_tool_final_weighted_score",
        "sql_tool_sql_compared_to_ground_truth_score",
        "sql_tool_is_valid_sql",
    ]

def initialize_results_file(results_file_path: str) -> None:
    """Initialize the results CSV file with headers."""
    headers = get_results_csv_headers()
    with open(results_file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)


def prepare_result_row_data(
    test_index: int,
    question: str,
    ground_truth_sql: str,
    sql_tool_response: dict,
    sql_tool_scoring_result: dict,
) -> list:
    """Prepare row data for CSV output from benchmark results."""
    return [
        test_index,
        question,
        ground_truth_sql,
        sql_tool_response.get("sql_code", ""),
        sql_tool_scoring_result["logic_match"],
        sql_tool_scoring_result["logic_issues"],
        sql_tool_scoring_result["semantic_match"],
        sql_tool_scoring_result["final_weighted_score"],
        sql_tool_scoring_result["sql_compared_to_ground_truth_score"],
        sql_tool_scoring_result["is_valid_sql"],
    ]

def append_result_to_file(results_file_path: str, row_data: list) -> None:
    """Append a single result row to the CSV file."""
    with open(results_file_path, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_data)


def process_summaries_and_create_csv(
    rows_summaries: list, input_file_path: str
) -> tuple:
    """Process the collected row summaries and create summary CSV file."""

    # Generate summary file path
    if input_file_path:
        input_dir = os.path.dirname(input_file_path)
        input_filename = os.path.basename(input_file_path)
        summary_filename = input_filename.replace(".csv", "_summary.csv")
        summary_file_path = os.path.join(input_dir, summary_filename)
    else:
        summary_file_path = "benchmark_summary.csv"

    if not rows_summaries:
        return summary_file_path, {}

    # Extract data from summaries
    sql_tool_scores = [row["sql_tool_score"] for row in rows_summaries]

    # Calculate statistics
    total_processed = len(rows_summaries)
    avg_sql_tool_score = sum(sql_tool_scores) / total_processed

    # Create summary data for CSV
    summary_data = [
        ["Metric", "SQL Tool"],
        [
            "Average Score",
            f"{avg_sql_tool_score:.4f}",
        ],
        ["Total Processed", str(total_processed), str(total_processed), ""],
    ]

    # Write summary CSV
    with open(summary_file_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(summary_data)

    # Create summary stats for return value
    summary_stats = {
        "total_processed": total_processed,
        "sql_tool_avg_score": round(avg_sql_tool_score, 4),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total Questions Processed: {total_processed}")
    print(f"SQL Tool Average Score: {avg_sql_tool_score:.4f}")
    print("=" * 60)

    return summary_file_path, summary_stats



# ==== start of script ====
class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",  # forbid extra fields
        validate_assignment=True,  # re-check on assignment
        str_min_length=1,  # all strings must be non-empty by default
    )


NonEmptyStr = Annotated[str, Field(min_length=1, description="Non-empty string")]


class SqlResponse(StrictModel):
    sql_code: NonEmptyStr = Field(
        ...,
        description="A valid SQL that directly answers the user's question",
    )


class SqlScoringResponse(StrictModel):
    logic_match: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score from 0 to 1 indicating how well the SQL logic answers the given question",
    )
    logic_issues: NonEmptyStr = Field(
        ...,
        description="Short text explaining what issues reduced the logic_match score",
    )
    semantic_match: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score from 0 to 1 indicating how well the SQL response types match the expected types for the question",
    )
    final_weighted_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Combined score (50% logic_match + 50% semantic_match) indicating overall question answering quality",
    )
    sql_compared_to_ground_truth_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Score from 0 to 1 indicating how similar the SQL is to the ground truth SQL",
    )
    is_valid_sql: bool = Field(
        ...,
        description="Whether the SQL code is syntactically valid and executable",
    )


# ==== end of script ====

class TeeOutput:
    """Mirror stdout/stderr to both console and a logfile with thread safety."""

    def __init__(self, filename: str):
        self.console = sys.stdout
        self.file = open(filename, "w")
        self.lock = Lock()

    def write(self, message: str) -> None:
        with self.lock:
            self.console.write(message)
            self.file.write(message)

    def flush(self) -> None:
        with self.lock:
            self.console.flush()
            self.file.flush()

    def close(self) -> None:
        self.file.close()


sys.stdout = TeeOutput("log.txt")
sys.stderr = sys.stdout


TOTAL_GB_PROCESSED = 0.0
GB_LOCK = Lock()


@lru_cache(maxsize=None)
def load_gold_csv(file_path: str) -> pd.DataFrame:
    """Cache gold CSV loads to avoid repeated disk reads during evaluation."""
    return pd.read_csv(file_path)


def load_jsonl_to_dict(jsonl_file: str) -> dict:
    data_dict = {}
    with open(jsonl_file, "r") as file:
        for line in file:
            item = json.loads(line.strip())
            instance_id = item["instance_id"]
            data_dict[instance_id] = item
    return data_dict


def compare_multi_pandas_table(pred: pd.DataFrame, multi_gold, multi_condition_cols=None, multi_ignore_order=False) -> int:
    if not multi_gold:
        return 0

    if multi_condition_cols in (None, [], [[]], [None]):
        multi_condition_cols = [[] for _ in range(len(multi_gold))]
    elif len(multi_gold) > 1 and not all(isinstance(sublist, list) for sublist in multi_condition_cols):
        multi_condition_cols = [multi_condition_cols for _ in range(len(multi_gold))]

    multi_ignore_order = [multi_ignore_order for _ in range(len(multi_gold))]

    for i, gold in enumerate(multi_gold):
        if compare_pandas_table(pred, gold, multi_condition_cols[i], multi_ignore_order[i]):
            return 1
    return 0


def compare_pandas_table(pred: pd.DataFrame, gold: pd.DataFrame, condition_cols=None, ignore_order: bool = False) -> int:
    tolerance = 1e-2

    def normalize(value):
        if pd.isna(value):
            return 0
        return value

    def vectors_match(v1, v2, tol=tolerance, ignore_order_=False):
        v1 = [normalize(x) for x in v1]
        v2 = [normalize(x) for x in v2]

        if ignore_order_:
            v1 = sorted(v1, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))
            v2 = sorted(v2, key=lambda x: (x is None, str(x), isinstance(x, (int, float))))

        if len(v1) != len(v2):
            return False

        for a, b in zip(v1, v2):
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if not math.isclose(float(a), float(b), abs_tol=tol):
                    return False
            elif a != b:
                return False
        return True

    if condition_cols:
        if not isinstance(condition_cols, (list, tuple)):
            condition_cols = [condition_cols]
        gold_cols = gold.iloc[:, condition_cols]
    else:
        gold_cols = gold

    pred_cols = pred
    t_gold_list = gold_cols.transpose().values.tolist()
    t_pred_list = pred_cols.transpose().values.tolist()
    score = 1

    for gold_vector in t_gold_list:
        if not any(vectors_match(gold_vector, pred_vector, ignore_order_=ignore_order) for pred_vector in t_pred_list):
            score = 0
            break

    return score


def get_duckdb_result(db_path: str, query: str, save_dir=None, file_name: str = "result.csv", chunksize: int = 500, instance_id: str = None, schema: str = None):
    from nemo_retriever.relational_db.connectors.duckdb_engine import DuckDBEngine

    prefix = f"[{instance_id}] " if instance_id else ""

    try:
        engine = DuckDBEngine({"database": db_path})
        if schema:
            engine.set_schema(schema)
        if save_dir:
            engine.execute_to_csv(query, os.path.join(save_dir, file_name), chunksize=chunksize)
            return True, None
        return True, engine.execute(query)
    except Exception as e:
        print(f"{prefix}An error occurred: {e}")
        return False, str(e)


def extract_sql_query(pred_sql_query: str) -> str:
    pattern = r"```sql\n(.*?)\n```"
    match = re.search(pattern, pred_sql_query, re.DOTALL)

    if match:
        return match.group(1).strip()
    return pred_sql_query


def resolve_gold_paths(instance_id: str, gold_result_dir: str):
    base_path = Path(gold_result_dir) / f"{instance_id}.csv"
    if base_path.exists():
        return [base_path], True

    if "_" in instance_id:
        pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")
    else:
        pattern = re.compile(rf"^{re.escape(instance_id)}(_[a-z])?\.csv$")

    csv_files = sorted(
        file for file in os.listdir(gold_result_dir)
        if pattern.match(file)
    )
    return [Path(gold_result_dir) / file for file in csv_files], False


def evaluate_single_sql_instance(
    instance_id: str,
    eval_standard_dict: dict,
    spider2sql_metadata: dict,
    pred_result_dir: str,
    gold_result_dir: str,
    timeout: int = 60,
    consolidated_db_path: Path = None,
):
    del timeout  # timeout currently unused for lite databases

    print(f"[{instance_id}] starting ...")

    error_info = None
    score = 0
    llm_scoring: dict = {}
    pred_sql_query = ""

    try:
        pred_sql_path = Path(pred_result_dir) / f"{instance_id}.sql"
        pred_sql_query = pred_sql_path.read_text()
        pred_sql_query = extract_sql_query(pred_sql_query)

        if instance_id.startswith("local"):
            metadata = spider2sql_metadata.get(instance_id, {})
            db_name = metadata.get("db")
            question = metadata.get("question", "")
            if not db_name:
                exe_flag = False
                dbms_error_info = f"Missing database mapping for {instance_id}"
            else:
                if not consolidated_db_path:
                    exe_flag = False
                    dbms_error_info = "No database path configured. Pass --db_path or set DUCKDB_PATH."
                else:
                    exe_flag, pred_pd = get_duckdb_result(
                        str(consolidated_db_path),
                        pred_sql_query,
                        instance_id=instance_id,
                        schema=db_name,
                    )
        else:
            exe_flag = False
            pred_pd = f"Unsupported instance id prefix: {instance_id}"

        if not exe_flag:
            score = 0
            error_info = pred_pd  # holds the error string when exe_flag is False
        else:
            gold_sql_dir = Path(gold_result_dir).parent / "sql"
            gold_sql_path = gold_sql_dir / f"{instance_id}.sql"
            if not gold_sql_path.exists():
                score = 0
                error_info = f"Gold SQL not found: {gold_sql_path}"
            else:
                gold_sql = extract_sql_query(gold_sql_path.read_text())
                gold_exe_flag, gold_pd = get_duckdb_result(
                    str(consolidated_db_path),
                    gold_sql,
                    instance_id=instance_id,
                    schema=db_name,
                )

                pred_result_preview = str(pred_pd.head(5).to_dict()) if isinstance(pred_pd, pd.DataFrame) else str(pred_pd)[:500]
                llm_scoring = score_sql_code(
                    question=question,
                    sql_response={"sql_code": pred_sql_query, "result": pred_result_preview},
                    ground_truth_sql=gold_sql,
                )

                if not gold_exe_flag:
                    score = 0
                    error_info = f"Gold SQL execution failed: {gold_pd}"
                else:
                    standard = eval_standard_dict.get(instance_id, {})
                    condition_cols = standard.get("condition_cols")
                    ignore_order = standard.get("ignore_order", False)
                    try:
                        score = compare_pandas_table(pred_pd, gold_pd, condition_cols, ignore_order)
                    except Exception as e:
                        print(f"{instance_id}: compare failed: {e}")
                        score = 0
                        error_info = f"Python Script Error:{str(e)}"
                    if score == 0 and error_info is None:
                        error_info = "Result Error"

    except Exception as e:
        print(f"Error evaluating {instance_id}: {e}")
        score = 0
        error_info = f"Evaluation Error: {str(e)}"
        pred_sql_query = ""

    print(f"[{instance_id}] done — score={score}" + (f", error={error_info}" if error_info else ""))
    return {
        "instance_id": instance_id,
        "score": score,
        "pred_sql": pred_sql_query,
        "error_info": error_info,
        "llm_logic_match": llm_scoring.get("logic_match"),
        "llm_logic_issues": llm_scoring.get("logic_issues"),
        "llm_semantic_match": llm_scoring.get("semantic_match"),
        "llm_final_weighted_score": llm_scoring.get("final_weighted_score"),
        "llm_ground_truth_score": llm_scoring.get("sql_compared_to_ground_truth_score"),
        "llm_is_valid_sql": llm_scoring.get("is_valid_sql"),
    }


def evaluate_single_exec_result_instance(
    instance_id: str,
    eval_standard_dict: dict,
    pred_result_dir: str,
    gold_result_dir: str,
):
    error_info = None

    try:
        pred_pd = pd.read_csv(Path(pred_result_dir) / f"{instance_id}.csv")

        gold_paths, is_single = resolve_gold_paths(instance_id, gold_result_dir)
        standard = eval_standard_dict.get(instance_id, {})
        condition_cols = standard.get("condition_cols")
        ignore_order = standard.get("ignore_order", False)

        if not gold_paths:
            score = 0
            error_info = "No matching gold file found"
        elif is_single:
            try:
                gold_pd = load_gold_csv(str(gold_paths[0]))
                score = compare_pandas_table(pred_pd, gold_pd, condition_cols, ignore_order)
            except Exception as e:
                print(f"{instance_id}: compare against {gold_paths[0]} failed: {e}")
                score = 0
                error_info = f"Python Script Error:{str(e)}"
            if score == 0 and error_info is None:
                error_info = "Result Error"
        else:
            try:
                gold_pds = [load_gold_csv(str(path)) for path in gold_paths]
                score = compare_multi_pandas_table(pred_pd, gold_pds, condition_cols, ignore_order)
            except Exception as e:
                print(f"{instance_id}: multi-compare against {gold_paths} failed: {e}")
                score = 0
                error_info = f"Python Script Error:{str(e)}"
            if score == 0 and error_info is None:
                error_info = "Result Error"

    except Exception as e:
        print(f"{instance_id} ERROR!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {e}")
        score = 0
        error_info = f"Evaluation Error: {str(e)}"

    return {
        "instance_id": instance_id,
        "score": score,
        "pred_sql": None,
        "error_info": error_info,
    }


def save_correct_ids_to_csv(output_results, result_dir: str):
    correct_ids = [item["instance_id"] for item in output_results if item["score"] == 1]

    transformed_ids = []
    for item in correct_ids:
        if item.startswith(("bq", "ga", "local")):
            transformed_ids.append(f"sf_{item}")
        else:
            transformed_ids.append(item)

    csv_file = f"{result_dir}-ids.csv"
    pd.DataFrame({"instance_id": transformed_ids}).to_csv(csv_file, index=False)
    print(f"Correct IDs saved to: {csv_file}")
    return csv_file


def evaluate_spider2sql(args):
    mode = args.mode
    gold_result_dir = os.path.join(args.gold_dir, "exec_result")
    pred_result_dir = args.result_dir

    eval_standard_dict = load_jsonl_to_dict(os.path.join(args.gold_dir, "spider2lite_eval.jsonl"))

    root_dir = Path(__file__).resolve().parent
    metadata_file = root_dir / "spider2-lite.jsonl"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Spider2-Lite metadata file not found: {metadata_file}\n"
            "Download it from https://github.com/xlang-ai/Spider2 and place it in the benchmark/ folder."
        )
    spider2sql_metadata = load_jsonl_to_dict(str(metadata_file))

    from nemo_retriever.relational_db.connectors.setup_spider2 import DEFAULT_DB_PATH

    if getattr(args, "db_path", None):
        consolidated_db_path = Path(args.db_path).expanduser().resolve()
    elif os.environ.get("DUCKDB_PATH"):
        consolidated_db_path = Path(os.environ["DUCKDB_PATH"]).expanduser().resolve()
    else:
        consolidated_db_path = DEFAULT_DB_PATH.expanduser().resolve()

    if not consolidated_db_path.exists():
        raise FileNotFoundError(
            f"DuckDB database not found: {consolidated_db_path}\n"
            "Run setup_spider2.py to build it, or pass --db_path / set DUCKDB_PATH."
        )

    pred_ids = []
    if mode == "sql":
        pred_ids = [Path(file).stem for file in os.listdir(pred_result_dir) if file.endswith(".sql")]
    elif mode == "exec_result":
        pred_ids = [Path(file).stem for file in os.listdir(pred_result_dir) if file.endswith(".csv")]

    gold_ids = list(eval_standard_dict.keys())
    eval_ids = sorted(set(gold_ids).intersection(pred_ids))

    if not eval_ids:
        print("No overlapping prediction IDs with gold set. Nothing to evaluate.")
        return []

    max_workers = getattr(args, "max_workers", 8)
    max_workers = min(max_workers, len(eval_ids)) or 1

    output_results = []
    results_csv = Path(__file__).resolve().parent / "evaluation_results.csv"
    _result_fields = [
        "instance_id", "score", "pred_sql", "error_info",
        "llm_logic_match", "llm_logic_issues", "llm_semantic_match",
        "llm_final_weighted_score", "llm_ground_truth_score", "llm_is_valid_sql",
    ]

    def _append_result(result: dict) -> None:
        """Write a single result row to the CSV immediately."""
        write_header = not results_csv.exists()
        with open(results_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_result_fields)
            if write_header:
                writer.writeheader()
            writer.writerow({k: result.get(k) for k in _result_fields})

    # Clear any previous run's file so we start fresh.
    if results_csv.exists():
        results_csv.unlink()

    if mode == "sql":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(
                    evaluate_single_sql_instance,
                    instance_id,
                    eval_standard_dict,
                    spider2sql_metadata,
                    pred_result_dir,
                    gold_result_dir,
                    timeout=args.timeout,
                    consolidated_db_path=consolidated_db_path,
                ): instance_id
                for instance_id in eval_ids
            }

            for future in tqdm(as_completed(future_to_id), total=len(eval_ids), desc="Evaluating SQL"):
                result = future.result()
                output_results.append(result)
                _append_result(result)

    elif mode == "exec_result":
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_id = {
                executor.submit(
                    evaluate_single_exec_result_instance,
                    instance_id,
                    eval_standard_dict,
                    pred_result_dir,
                    gold_result_dir,
                ): instance_id
                for instance_id in eval_ids
            }

            for future in tqdm(as_completed(future_to_id), total=len(eval_ids), desc="Evaluating Exec Results"):
                result = future.result()
                output_results.append(result)
                _append_result(result)

    output_results.sort(key=lambda item: item["instance_id"])
    print(f"Evaluation results saved to: {results_csv}")

    print({item["instance_id"]: item["score"] for item in output_results})
    correct_examples = sum(item["score"] for item in output_results)
    total = len(output_results)

    def _avg(key):
        vals = [item[key] for item in output_results if item.get(key) is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    avg_weighted, n_weighted       = _avg("llm_final_weighted_score")
    avg_gt,       n_gt             = _avg("llm_ground_truth_score")
    avg_logic,    n_logic          = _avg("llm_logic_match")
    avg_semantic, n_semantic       = _avg("llm_semantic_match")

    def _fmt(avg, n): return f"{avg:.4f}  (n={n})" if avg is not None else "n/a"

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total evaluated              : {total}")
    print(f"Exact match (score=1)        : {correct_examples}  ({correct_examples / total:.1%})")
    print(f"Exact match / 547            : {correct_examples / 547:.1%}")
    print(f"LLM avg logic match          : {_fmt(avg_logic, n_logic)}")
    print(f"LLM avg semantic match       : {_fmt(avg_semantic, n_semantic)}")
    print(f"LLM avg final weighted score : {_fmt(avg_weighted, n_weighted)}")
    print(f"LLM avg ground truth sim     : {_fmt(avg_gt, n_gt)}")
    print("=" * 60)

    print(f"\nFinal score: {correct_examples / total}, Correct examples: {correct_examples}, Total examples: {total}")
    print(f"Real score: {correct_examples / 547}, Correct examples: {correct_examples}, Total examples: 547")

    save_correct_ids_to_csv(output_results, pred_result_dir)
    print(f"TOTAL_GB_PROCESSED: {TOTAL_GB_PROCESSED:.5f} GB")

    return output_results


if __name__ == "__main__":
    _script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run evaluations for NLP models.")
    parser.add_argument("--mode", type=str, choices=["sql", "exec_result"], default="sql", help="Mode of submission results")
    parser.add_argument("--result_dir", type=str, default=str(_script_dir / "generated_sql"), help="Path to result directory containing predicted .sql files")
    parser.add_argument("--gold_dir", type=str, default=str(_script_dir / "gold"), help="Path to gold directory")
    parser.add_argument(
        "--db_path",
        type=str,
        default=None,
        help=(
            "Path to a consolidated DuckDB file that contains all databases as schemas "
            "(e.g. spider2.duckdb). Used as a fallback when per-database files are absent. "
            "Defaults to the DUCKDB_PATH environment variable if set."
        ),
    )
    parser.add_argument(
        "--sqlite_base_dir",
        type=str,
        default=None,
        help=(
            "Directory containing per-database .duckdb or .sqlite files "
            "(e.g. Airlines.duckdb, california_schools.sqlite). "
            "Defaults to benchmark/resource/databases/. "
            "Override this to point at the folder where your Spider2-Lite databases live."
        ),
    )
    parser.add_argument("--is_sql_debug", action="store_true", default=False)
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of worker threads")
    parser.add_argument("--timeout", type=int, default=60, help="SQL execution timeout in seconds")

    args = parser.parse_args()

    evaluate_spider2sql(args)
