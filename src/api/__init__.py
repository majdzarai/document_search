"""
API Package
============

This package contains the FastAPI application and all HTTP-related code.

Structure:
- app.py: FastAPI application factory and configuration
- dependencies.py: Dependency injection setup
- routes/: Endpoint handlers organized by feature
- schemas/: Pydantic models for request/response validation
- middleware/: Custom middleware (auth, logging, etc.)
"""

# Import the app for convenient access
from src.api.app import app, create_app

# Export list
__all__ = ["app", "create_app"]
