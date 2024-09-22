"""Enrichment for a pre-defined schema."""

import weave
import os
from llm_eval_agent.graph import graph

weave.init(os.environ['WEAVE_PROJECT_NAME'])
__all__ = ["graph"]
