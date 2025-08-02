"""Prompt templates for query generation tasks."""

QUERY_GENERATION_DATASET_PROMPT = """
Your task is to generate potential user query, that is looking for a specific item.
You'll obeserve other queries, mimic as close as possible their style, tone of voice, level of specificity and length.
Generate query for which the item would be a good match, but don't make it too specific.
Your query should be different from any query presented in the examples.

Example queries:
{queries}

Item information:
{item}

Reply in the following json format:
{{
    "query": string, // query in Portuguese, 4-8 words max
}}
"""

QUERY_GENERATION_DATASET_2_PROMPT = """
Your task is to generate potential user query.
You'll obeserve other queries, mimic as close as possible their style, tone of voice, level of specificity and length.
Your query should be different from any query presented in the examples.

Example queries:
{queries}

Reply in the following json format:
{{
    "query": string, // query in Portuguese, 4-8 words max
}}
"""
