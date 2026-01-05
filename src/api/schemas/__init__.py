"""
Schemas Package
================

This package contains Pydantic models for API request/response validation.

Schemas are used to:
1. Validate incoming request data
2. Define the structure of API responses
3. Generate OpenAPI documentation automatically
4. Provide type hints for IDE autocompletion

Modules:
- requests.py: Request body schemas (what clients send to us)
- responses.py: Response body schemas (what we send to clients)
"""

# Import schemas for convenient access
from src.api.schemas.requests import QueryRequest
from src.api.schemas.responses import QueryResponse, ErrorResponse

# Export list for explicit imports
__all__ = ["QueryRequest", "QueryResponse", "ErrorResponse"]
