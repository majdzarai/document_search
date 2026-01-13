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


class CreateCollectionResponse(BaseModel):
    """
    Schema for collection creation response.

    Example:
        {
            "success": true,
            "message": "Collection created successfully",
            "collection_name": "rag_client123",
            "collection_id": "client123"
        }
    """

    success: bool = Field(
        ...,
        description="Whether the collection was created successfully"
    )

    message: str = Field(
        ...,
        description="Human-readable status message"
    )

    collection_name: str = Field(
        ...,
        description="Full name of the created collection in Qdrant"
    )

    collection_id: str = Field(
        ...,
        description="The collection ID provided by the client"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "Collection created successfully",
                    "collection_name": "rag_client123",
                    "collection_id": "client123"
                }
            ]
        }
    }


class DeleteCollectionResponse(BaseModel):
    """
    Schema for collection deletion response.

    Example:
        {
            "success": true,
            "message": "Collection deleted successfully",
            "collection_id": "client123"
        }
    """

    success: bool = Field(
        ...,
        description="Whether the collection was deleted successfully"
    )

    message: str = Field(
        ...,
        description="Human-readable status message"
    )

    collection_id: str = Field(
        ...,
        description="The ID of the deleted collection"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "success": True,
                    "message": "Collection deleted successfully",
                    "collection_id": "client123"
                }
            ]
        }
    }
