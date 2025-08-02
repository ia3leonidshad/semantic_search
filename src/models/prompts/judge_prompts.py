"""Prompt templates for relevance judging tasks."""

JUDGE_PROMPT = """
Your task is to judge the retrieval system of food/groceries delivery service.
You'll be presented the user query and retrieved item.
You need evaluate how well item matches the query and assign on of 3 scores:
2 = Highly relevant (exact dish match)
1 = Partially relevant (shares key attributes, could satisfy)
0 = Irrelevant

User query:
{query}

Item info:
{item}

Reply in English.

Reply in the following json format:
{{
    "reason": string, // 3-5 sentences reasoning about your judgement, what user asked for, what they got, how well it matches the intent
    "score": int, // relevance score, one of {{0, 1, 2}}
}}
"""
