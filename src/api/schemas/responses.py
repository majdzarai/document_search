"""
API Response Schemas Module
============================

This module defines the structure of API responses using Pydantic models.

WHY DEFINE RESPONSE SCHEMAS?
----------------------------
1. Consistency: All responses follow the same structure
2. Documentation: FastAPI generates accurate response examples in docs
3. Validation: Ensures our code returns properly formatted data
4. Type Hints: Provides autocompletion for developers using the API client
"""

from typing import List
from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    """
    Schema for the POST /query endpoint response.

    This model defines what data the API returns after processing a query.
    FastAPI uses this to validate our responses and generate documentation.

    Example response:
    {
        "answer": "RAG stands for Retrieval-Augmented Generation...",
        "sources": ["document1.pdf", "document2.pdf"]
    }
    """

    answer: str = Field(
        ...,  # Required field
        description="The generated answer to the user's question",
        examples=["RAG is a technique that combines retrieval with generation..."]
    )

    sources: List[str] = Field(
        default_factory=list,  # Default to empty list if not provided
        description="List of source documents used to generate the answer",
        examples=[["rag_overview.pdf", "vector_databases_guide.pdf"]]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "Based on the retrieved documents, RAG (Retrieval-Augmented Generation) is a technique that enhances LLMs by providing relevant context from a knowledge base.",
                    "sources": ["rag_overview.pdf", "vector_databases_guide.pdf"]
                }
            ]
        }
    }


class ErrorResponse(BaseModel):
    """
    Schema for error responses.

    This provides a consistent format for all error messages returned by the API.
    Clients can rely on this structure to handle errors programmatically.

    Example error response:
    {
        "detail": "Question cannot be empty",
        "error_code": "INVALID_INPUT"
    }
    """

    detail: str = Field(
        ...,
        description="Human-readable error message explaining what went wrong"
    )

    error_code: str = Field(
        default="INTERNAL_ERROR",
        description="Machine-readable error code for programmatic handling"
    )
