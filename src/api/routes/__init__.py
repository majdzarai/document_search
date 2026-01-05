"""
Routes Package
===============

This package contains all API route handlers (endpoints).

Each module in this package defines routes for a specific feature:
- query.py: Handles RAG query endpoints (POST /query)
- health.py: Handles health check endpoints (future)
- ingest.py: Handles document ingestion endpoints (future)
- collections.py: Handles collection management endpoints (future)

Routes are organized this way to:
1. Keep related endpoints together
2. Make the codebase easy to navigate
3. Allow teams to work on different features independently
"""

# Import routers so they can be accessed as: from src.api.routes import query
from src.api.routes import query

# Export list for explicit imports
__all__ = ["query"]
