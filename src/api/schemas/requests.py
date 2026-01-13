"""
API Request Schemas Module
===========================

This module defines the structure of incoming API requests using Pydantic models.

WHAT IS PYDANTIC?
-----------------
Pydantic is a Python library for data validation and settings management.
It allows us to:
1. Define the expected structure of data using Python classes
2. Automatically validate incoming data against that structure
3. Get helpful error messages when data doesn't match expectations
4. Generate automatic API documentation (OpenAPI/Swagger)

WHY USE REQUEST SCHEMAS?
------------------------
1. Validation: Automatically reject malformed requests with clear error messages
2. Documentation: FastAPI uses these schemas to generate API docs
3. Type Safety: IDE autocompletion and type checking work properly
4. Security: Prevents unexpected data from reaching our business logic
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """
    Schema for the POST /query endpoint request body.

    This model defines what data the client must send when making a query.
    FastAPI will automatically validate incoming requests against this schema.

    Example valid request body:
    {
        "question": "What is RAG and how does it work?"
    }

    Example invalid request (will be rejected):
    {
        "query": "What is RAG?"  # Wrong field name!
    }
    """

    question: str = Field(
        ...,  # The ... means this field is REQUIRED (not optional)
        min_length=1,  # Question must have at least 1 character
        max_length=10000,  # Reasonable limit to prevent abuse
        description="The question to ask the RAG system",
        examples=["What is RAG and how does it work?"]
    )

    # Pydantic v2 uses model_config instead of class Config
    model_config = {
        # This example will appear in the auto-generated API documentation.
        # It helps users understand how to format their requests.
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What are the benefits of using RAG for chatbots?"
                }
            ]
        }
    }


class CreateCollectionRequest(BaseModel):
    """
    Schema for creating a new multi-tenant collection.

    This model defines the request body for creating a new isolated
    collection for a specific tenant.

    Example:
        POST /api/v1/collections/create
        {
            "collection_id": "client123"
        }
    """

    collection_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique identifier for the collection (tenant ID)",
        pattern="^[a-zA-Z0-9_-]+$",  # Alphanumeric, underscore, hyphen only
        examples=["client123", "tenant_a", "my-company"]
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "collection_id": "client123"
                }
            ]
        }
    }
