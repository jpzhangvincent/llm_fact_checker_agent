"""Tools for data enrichment.

This module contains functions that are directly exposed to the LLM as tools.
These tools can be used for tasks such as web searching and scraping.
Users can edit and extend these tools as needed.
"""

import weave
import json
import csv
from typing import Any, Optional, cast, List
from langchain_core.runnables import RunnableConfig
from langgraph.prebuilt import InjectedState
from typing_extensions import Annotated
import aiohttp
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import InjectedToolArg
from langchain_experimental.synthetic_data import (
    DatasetGenerator,
    create_data_generation_chain,
)
import pandas as pd

from llm_eval_agent.configuration import Configuration
from llm_eval_agent.state import State
from llm_eval_agent.utils import init_model

@weave.op()
async def search(
    query: str, *, config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[list[dict[str, Any]]]:
    """Query a search engine.

    This function queries the web to fetch comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events. Provide as much context in the query as needed to ensure high recall.
    """
    configuration = Configuration.from_runnable_config(config)
    wrapped = TavilySearchResults(max_results=configuration.max_search_results)
    result = await wrapped.ainvoke({"query": query})
    return cast(list[dict[str, Any]], result)


_INFO_PROMPT = """You are doing web research on behalf of a user. You are trying to find out this information:

<info>
{info}
</info>



Based on the website content below, jot down some notes about the website.

<Website content>
{content}
</Website content>"""

@weave.op()
async def scrape_website(
    url: str,
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg],
) -> str:
    """Scrape and summarize content from a given URL.

    Returns:
        str: A summary of the scraped content, tailored to the extraction schema.
    """
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.text()

    p = _INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        url=url,
        content=content[:40_000],
    )
    raw_model = init_model(config)
    result = await raw_model.ainvoke(p)
    return str(result.content)


_DATA_GENERATION_PROMPT = """You are an expert at generating fact-based synthetic data for LLM evaluation. You are tasked with creating a dataset based on the following topic:

<topic>
{topic}
</topic>

Based on this topic, generate a diverse set of fact-based synthetic data points that are suitable for evaluating an LLM's performance in understanding and responding to prompts in this domain.

Your output should be a JSON object that follows the structure defined in the extraction schema below:

<extraction_schema>
{schema}
</extraction_schema>

Here are some example data points to guide your generation:

<example_data>
{example_data}
</example_data>

Generate the dataset now, ensuring it follows the schema and is inspired by but not identical to the example data:"""

@weave.op()
async def synthetic_data_generator(
    content: str = "",
    fields: List[str] = [],
    *,
    state: Annotated[State, InjectedState],
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> Optional[dict[str, Any]]:
    """Generate synthetic data based on the topic, research content, and example data provided"""
    raw_model = init_model(config)
    data_gen_chain = create_data_generation_chain(raw_model)

    # Read example data if provided
    example_data = []
    if state.example_data_csv_path:
        try:
            df = pd.read_csv(state.example_data_csv_path)
            example_data = df.to_dict('records')
            if not fields:
                fields = df.columns.tolist()
        except Exception as e:
            print(f"Error reading CSV file: {e}")

    # Generate schema if it's not provided
    if state.extraction_schema is None:
        schema_prompt = f"Generate an extraction schema for the following topic: {state.topic}"
        if fields:
            schema_prompt += f" Include these fields: {', '.join(fields)}"
        if example_data:
            schema_prompt += f" Use this example data as a reference: {json.dumps(example_data[:5])}"
        schema_result = await raw_model.ainvoke(schema_prompt)
        try:
            state.extraction_schema = json.loads(schema_result.content)
        except json.JSONDecodeError:
            # If the model didn't return valid JSON, use a default schema
            state.extraction_schema = {
                "type": "object",
                "properties": {
                    "data_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {field: {"type": "string"} for field in fields} if fields else {
                                "input": {"type": "string"},
                                "output": {"type": "string"}
                            },
                            "required": fields if fields else ["input", "output"]
                        }
                    }
                },
                "required": ["data_points"]
            }

    # Prepare the prompt with example data
    p = _DATA_GENERATION_PROMPT.format(
        topic=state.topic,
        schema=json.dumps(state.extraction_schema, indent=2),
        example_data=json.dumps(example_data[:5]) if example_data else "No example data provided"
    )
    result = await data_gen_chain.ainvoke(p)

    # Add export button to the result
    result['export_button'] = weave.ops.Button(
        "Export JSON",
        lambda: export_json(result)
    )

    return cast(dict[str, Any], result)

@weave.op()
def export_json(data: dict) -> str:
    """
    Export the current state data as a JSON string.
    """
    return json.dumps(data, indent=2)
