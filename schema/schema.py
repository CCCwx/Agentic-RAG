"""
Schema for enforcing structured agent / LLM outputs in the RAG workflow.
"""
from pydantic import BaseModel, Field
from typing import List
from langchain_core.prompts import PromptTemplate


# --- Stage 1: Intent routing ---
class IntentResponse(BaseModel):
    """Intent classification: No Retrieval | Retrieval Needed."""
    response: str = Field(
        ...,
        description="One of: 'No Retrieval' or 'Retrieval Needed'."
    )


# --- Stage 1: Query expansion ---
class ExpandedQueriesResponse(BaseModel):
    """1-2 rewritten/expanded search queries."""
    queries: List[str] = Field(
        ...,
        min_length=1,
        max_length=2,
        description="List of 1 or 2 query strings for retrieval."
    )


# --- Stage 5: Utility check ---
class UtilityResponse(BaseModel):
    """Whether the answer is useful for the user's question."""
    response: str = Field(
        ...,
        description="One of: 'Useful' or 'Not Useful'."
    )

