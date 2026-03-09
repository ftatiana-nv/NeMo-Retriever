import logging
import re

from infra.Neo4jConnection import get_neo4j_conn

from enrichments.ai_services.functions import get_ontology_prompt
from enrichments.description_suggestion.system_prompts import (
    term_system_prompt,
    attribute_system_prompt,
    metric_system_prompt,
    analysis_system_prompt,
    analysis_name_system_prompt,
    sql_system_prompt,
    tableau_dashboard_system_prompt,
    tableau_visual_system_prompt,
    tableau_embed_ds_system_prompt,
    tableau_workbook_system_prompt,
    tableau_published_ds_system_prompt,
    quicksight_dashboard_system_prompt,
    quicksight_system_prompt,
    quicksight_dataset_system_prompt,
    sisense_dashboard_system_prompt,
    sisense_visual_system_prompt,
    sisense_datamodel_system_prompt,
    sisense_table_system_prompt,
    powerbi_dashboard_system_prompt,
    powerbi_table_system_prompt,
    powerbi_report_system_prompt,
    powerbi_visual_system_prompt,
    looker_dashboard_system_prompt,
    looker_explore_visual_system_prompt,
    looker_look_view_system_prompt,
    field_system_prompt,
    system_intro_prompt,
    table_system_prompt,
    schema_system_prompt,
    column_system_prompt,
)

logger = logging.getLogger("description suggestion")
conn = get_neo4j_conn()


def get_final_prompt(prompt, system_prompt, ontology):
    ontology_prompt = get_ontology_prompt(ontology)
    system_prompt = system_intro_prompt + ontology_prompt + system_prompt
    return prompt, system_prompt


def create_table_prompt(data, ontology):
    if data is None or not data.keys() >= {"schema_name", "table_name", "columns"}:
        return None, None

    prompt = f"The table name: '{data['table_name']}', schema name: '{data['schema_name']}', columns: {data['columns']}."
    prompt, system_prompt = get_final_prompt(prompt, table_system_prompt, ontology)
    return prompt, system_prompt


def create_schema_prompt(data, ontology):
    if data is None or not data.keys() >= {"schema_name", "db_name", "tables"}:
        return None, None

    prompt = f"The schema name: '{data['schema_name']}', database name: '{data['db_name']}', tables: {data['tables']}."
    prompt, system_prompt = get_final_prompt(prompt, schema_system_prompt, ontology)
    return prompt, system_prompt


def create_column_prompt(data, ontology):
    if data is None or not data.keys() >= {"table_name", "column_name", "columns"}:
        return None, None

    prompt = f"The specific column name: '{data['column_name']}', table name: '{data['table_name']}', a list of other column names within the same table: {data['columns']}."
    prompt, system_prompt = get_final_prompt(prompt, column_system_prompt, ontology)
    return prompt, system_prompt


def create_term_prompt(data, ontology):
    if data is None or not data.keys() >= {"name"}:
        return None, None

    prompt = f"The term: '{data['name']}'."
    prompt, system_prompt = get_final_prompt(prompt, term_system_prompt, ontology)
    return prompt, system_prompt


def create_attribute_prompt(data, ontology):
    if data is None or not data.keys() >= {"term_name", "attribute_name"}:
        return None, None

    prompt = f"The attribute: '{data['attribute_name']}', the term that contains the attribute: '{data['term_name']}'."
    prompt, system_prompt = get_final_prompt(prompt, attribute_system_prompt, ontology)
    return prompt, system_prompt


def create_metric_prompt(data, ontology):
    if data is None or not data.keys() >= {"formula", "name"}:
        return None, None

    prompt = (
        f"The metric name: '{data['name']}', the metric formula: '{data['formula']}'."
    )
    prompt, system_prompt = get_final_prompt(prompt, metric_system_prompt, ontology)
    return prompt, system_prompt


def create_analysis_prompt(data, ontology):
    if data is None or not data.keys() >= {"query_text", "name"}:
        return None, None

    prompt = (
        f"the analysis name: '{data['name']}', the SQL query: '{data['query_text']}'."
    )
    prompt, system_prompt = get_final_prompt(prompt, analysis_system_prompt, ontology)
    return prompt, system_prompt


def create_analysis_name_prompt(data, ontology):
    if data is None or not data.keys() >= {"query_text"}:
        return None, None

    sql = remove_sql_comments(data["query_text"])
    prompt = f"the SQL query: '{sql}'."
    prompt, system_prompt = get_final_prompt(
        prompt, analysis_name_system_prompt, ontology
    )
    return prompt, system_prompt


# This was used in population, should be used if we will have for single query
def create_query_prompt(data, ontology):
    if data is None or not data.keys() >= {"query_text"}:
        return None, None

    sql = remove_sql_comments(data["query_text"])
    prompt = f"the SQL query: '{sql}'."
    prompt, system_prompt = get_final_prompt(prompt, sql_system_prompt, ontology)

    return prompt, system_prompt


def create_ta_dashboard_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "project",
        "workbook",
        "dashboard",
        "sibling_dshs",
        "visuals",
    }:
        return None, None

    prompt = (
        f"The dashboard name: '{data['dashboard']}', project name: '{data['project']}', "
        f"workbook name: '{data['workbook']}', other dashboards under the same workbook: {data['sibling_dshs']}, "
        f"visuals: {data['visuals']}."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, tableau_dashboard_system_prompt, ontology
    )
    return prompt, system_prompt


def create_ta_visual_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "project",
        "workbook",
        "dashboard",
        "visual",
        "sibling_visuals",
        "fields",
    }:
        return None, None

    prompt = (
        f"The visual name: '{data['visual']}', project name: '{data['project']}', "
        f"workbook name: '{data['workbook']}', dashboard name: '{data['dashboard']}', "
        f"other visuals under the same dashboard: {data['sibling_visuals']}, "
        f"fields: {data['fields']}."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, tableau_visual_system_prompt, ontology
    )
    return prompt, system_prompt


def create_ta_embed_ds_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "project",
        "workbook",
        "embed_ds",
        "sibling_emb_dses",
        "fields",
    }:
        return None, None

    prompt = (
        f"The embedded datasource name: '{data['embed_ds']}', project name: '{data['project']}', "
        f"workbook name: '{data['workbook']}', other embedded data sources names under the same workbook: {data['sibling_emb_dses']}, "
        f"fields names under the embedded data source: {data['fields']}."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, tableau_embed_ds_system_prompt, ontology
    )
    return prompt, system_prompt


def create_ta_workbook_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "project",
        "workbook",
        "dashboards",
        "visuals",
        "embedded_dss",
        "sibling_wbs",
    }:
        return None, None

    prompt = (
        f"The workbook name: '{data['workbook']}', project name: '{data['project']}', "
        f"dashboards' names under the workbook: {data['dashboards']}, "
        f"visuals names under the workbook: {data['visuals']}, "
        f"datasources names under the workbook: {data['embedded_dss']}, "
        f"other workbooks names under the same project: {data['sibling_wbs']}."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, tableau_workbook_system_prompt, ontology
    )
    return prompt, system_prompt


def create_ta_published_ds_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "project",
        "pub_ds",
        "fields",
        "sibling_pub_dses",
    }:
        return None, None

    prompt = (
        f"The published datasource name: '{data['pub_ds']}', project name: '{data['project']}', "
        f"fields names under the published data source: '{data['fields']}', "
        f"other published data sources names under the same project: {data['sibling_pub_dses']}."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, tableau_published_ds_system_prompt, ontology
    )
    return prompt, system_prompt


def create_qs_dashboard_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "dashboard",
        "visuals",
    }:
        return None, None

    prompt = f"The dashboard name: '{data['dashboard']}', visuals names under the dashboard: {data['visuals']}."
    prompt, system_prompt = get_final_prompt(
        prompt, quicksight_dashboard_system_prompt, ontology
    )
    return prompt, system_prompt


def create_qs_visual_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "item_name",
        "item_label",
        "visual",
        "sibling_visuals",
        "fields",
    }:
        return None, None

    quicksight_visual_system_prompt = quicksight_system_prompt + (
        f"You will be provided with the visual name, fields' names under the visual, "
        f"{data['item_label']} name that contains the visual, and other visuals' names under the same {data['item_label']}.\n"
        "Your job is to describe in up to 3 sentences the given visual's BI role to the user who understands BI."
    )
    prompt = (
        f"The visual name: '{data['visual']}', fields names under the visual: {data['fields']}, "
        f"{data['item_label']} name that contains the visual: '{data['item_name']}', "
        f"other visuals names under the same {data['item_label']}: {data['sibling_visuals']}."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, quicksight_visual_system_prompt, ontology
    )
    return prompt, system_prompt


def create_qs_dataset_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "dataset",
        "fields",
    }:
        return None, None

    prompt = f"The dataset name: '{data['dataset']}', fields' names under the dataset: {data['fields']}."
    prompt, system_prompt = get_final_prompt(
        prompt, quicksight_dataset_system_prompt, ontology
    )
    return prompt, system_prompt


def create_sis_dashboard_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "dashboard",
        "visuals",
        "sibling_dshes",
        "folder",
    }:
        return None, None

    prompt = (
        f"The dashboard name: '{data['dashboard']}', visuals names under the dashboard: {data['visuals']}, "
        f"other dashboards names under the same folder: {data['sibling_dshes']}, and folder name: '{data['folder']}'."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, sisense_dashboard_system_prompt, ontology
    )
    return prompt, system_prompt


def create_sis_visual_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "dashboard",
        "visual",
        "fields",
        "sibling_visuals",
        "folder",
    }:
        return None, None

    prompt = (
        f"The visual name: '{data['visual']}', the dashboard name: '{data['dashboard']}', "
        f"fields' names under the visual: {data['fields']}, other visuals' names under the same dashboard: {data['sibling_visuals']}, "
        f"and folder name: '{data['folder']}'."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, sisense_visual_system_prompt, ontology
    )
    return prompt, system_prompt


def create_sis_datamodel_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "datamodel",
        "table_fields",
    }:
        return None, None

    prompt = (
        f"The data model name: '{data['datamodel']}', "
        f"the list of tables with fields for each table: '{data['table_fields']}'."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, sisense_datamodel_system_prompt, ontology
    )
    return prompt, system_prompt


def create_sis_table_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "datamodel",
        "table",
        "table_fields",
        "sibling_tables",
    }:
        return None, None

    prompt = (
        f"The table name: '{data['table']}', fields names under the table: {data['table_fields']}, "
        f"data model name: '{data['datamodel']}', and the list of tables under the same data model: {data['sibling_tables']}."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, sisense_table_system_prompt, ontology
    )
    return prompt, system_prompt


def create_powerbi_datasource_prompt(data, ontology):
    prompt = (
        f"The {data['datasource_type']} name: '{data['datasource_name']}', "
        f"the list of tables with fields for each table: '{data['table_fields']}'."
    )

    print(prompt)
    prompt, system_prompt = get_final_prompt(
        prompt, sisense_datamodel_system_prompt, ontology
    )
    return prompt, system_prompt


def create_powerbi_dashboard_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "dashboard",
        "reports",
        "sibling_dshes",
        "workspace",
    }:
        return None, None

    prompt = (
        f"The dashboard name: '{data['dashboard']}', reports names under the dashboard: {data['reports']}, "
        f"other dashboards names under the same workspace: {data['sibling_dshes']}, and workspace name: '{data['workspace']}'."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, powerbi_dashboard_system_prompt, ontology
    )
    return prompt, system_prompt


def create_powerbi_table_prompt(data, ontology):
    prompt = (
        f"The table name: '{data['table']}', fields names under the table: {data['table_fields']}, "
        f"{data['parent_type']} name: '{data['parent_name']}', and the list of tables under the same {data['parent_type']}: {data['sibling_tables']}."
    )
    print(prompt)
    prompt, system_prompt = get_final_prompt(
        prompt, powerbi_table_system_prompt, ontology
    )
    return prompt, system_prompt


def create_powerbi_report_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "report",
        "visuals",
    }:
        return None, None

    prompt = (
        f"The report name: '{data['report']}', "
        f"visuals' names under the report: {data['visuals']} "
    )
    prompt, system_prompt = get_final_prompt(
        prompt, powerbi_report_system_prompt, ontology
    )
    return prompt, system_prompt


def create_powerbi_visual_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "report",
        "visual",
        "sibling_visuals",
        "fields",
    }:
        return None, None

    prompt = (
        f"The visual name: '{data['visual']}', the report name: '{data['report']}', "
        f"fields' names under the visual: {data['fields']}, other visuals' names under the same report: {data['sibling_visuals']}, "
    )
    prompt, system_prompt = get_final_prompt(
        prompt, powerbi_visual_system_prompt, ontology
    )
    return prompt, system_prompt


def create_field_prompt(data, ontology):
    if data is None or not data.keys() >= {"parent_name", "field_name", "fields"}:
        return None, None

    prompt = (
        f"The specific field name: '{data['field_name']}', parent name: '{data['parent_name']}', "
        f"a list of other field names within the same parent: {data['fields']}."
    )
    prompt, system_prompt = get_final_prompt(prompt, field_system_prompt, ontology)
    return prompt, system_prompt


def fix_response(descriptions: str):
    return descriptions.replace("\n", "").replace("'", "").replace('"', "")


def remove_sql_comments(sql_query: str):
    # Remove single-line comments
    sql_query = re.sub(r"--.*$", "", sql_query, flags=re.MULTILINE)
    sql_query = re.sub(r"//.*$", "", sql_query, flags=re.MULTILINE)

    # Remove multi-line comments
    sql_query = re.sub(r"/\*.*?\*/", "", sql_query, flags=re.DOTALL)

    return sql_query


def create_looker_dashboard_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "dashboard",
        "visuals",
        "sibling_dshes",
        "folder",
    }:
        return None, None

    prompt = (
        f"The dashboard name: '{data['dashboard']}', visuals names under the dashboard: {data['visuals']}, "
        f"other dashboards names under the same folder: {data['sibling_dshes']}, and folder name: '{data['folder']}'."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, looker_dashboard_system_prompt, ontology
    )
    return prompt, system_prompt


def create_looker_explore_visual_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "dashboard",
        "visual",
        "fields",
        "sibling_visuals",
        "folder",
    }:
        return None, None

    prompt = (
        f"The visual name: '{data['visual']}', the dashboard name: '{data['dashboard']}', "
        f"fields' names under the visual: {data['fields']}, other visuals' names under the same dashboard: {data['sibling_visuals']}, "
        f"and folder name: '{data['folder']}'."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, looker_explore_visual_system_prompt, ontology
    )
    return prompt, system_prompt


def create_looker_look_view_prompt(data, ontology):
    if data is None or not data.keys() >= {
        "model",
        "visual",
        "fields",
        "sibling_visuals",
        "folder",
    }:
        return None, None

    prompt = (
        f"The visual name: '{data['visual']}', the logical model name: '{data['model']}', "
        f"fields' names under the visual: {data['fields']}, other visuals' names under the same logical model: {data['sibling_visuals']}, "
        f"and folder name: '{data['folder']}'."
    )
    prompt, system_prompt = get_final_prompt(
        prompt, looker_look_view_system_prompt, ontology
    )
    return prompt, system_prompt
