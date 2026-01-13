"""
FastAPI Application Module
===========================

This module creates and configures the FastAPI application instance.

WHAT IS FASTAPI?
----------------
FastAPI is a modern, high-performance web framework for building APIs with Python.
Key features:
1. Fast: Very high performance, on par with NodeJS and Go
2. Easy: Designed to be easy to use and learn
3. Automatic Docs: Generates interactive API documentation (Swagger/OpenAPI)
4. Type Hints: Uses Python type hints for validation and documentation
5. Async Support: Built-in support for async/await

WHY A SEPARATE APP MODULE?
--------------------------
Separating the app creation from the entry point (main.py) allows us to:
1. Import the app in different contexts (testing, workers, etc.)
2. Configure the app in one place
3. Keep the entry point clean and simple
4. Avoid circular imports

APPLICATION FACTORY PATTERN:
---------------------------
We use a function (create_app) to create the FastAPI instance.
This pattern provides:
1. Flexibility: Can create apps with different configurations
2. Testing: Easy to create test instances with mock dependencies
3. Clarity: All app setup logic is in one function
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the routers from our routes modules.
# Each router handles a specific set of endpoints.
from src.api.routes import query
from src.api.routes import ingest
from src.api.routes import collections

# Import the shared provider initialization
# This ensures all services use the same vector store instance
from src.core.providers import initialize_providers


# =============================================================================
# APPLICATION LIFESPAN (STARTUP/SHUTDOWN EVENTS)
# =============================================================================
# We use a lifespan context manager to initialize shared providers at startup.
# This ensures that:
# 1. All providers (embedding, vector store, LLM) are initialized ONCE
# 2. Both ingestion and query routes use the SAME provider instances
# 3. The application is fully ready before accepting requests
#
# WHY IS THIS IMPORTANT?
# ----------------------
# Previously, providers were created separately by each service:
# - IngestionService created its own vector store
# - RAGPipeline created its own vector store
# - Documents uploaded via ingestion were stored in one store
# - Queries searched a different store → documents not found!
#
# Now with shared providers:
# - Both use the SAME vector store (initialized here at startup)
# - Uploaded documents are correctly found by queries


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    This runs at startup and shutdown:
    - Startup: Initialize shared providers (embedding, vector store, LLM)
    - Shutdown: Clean up resources (if needed)

    WHY INITIALIZE AT STARTUP?
    --------------------------
    1. Fail fast: If a provider can't be initialized, we know immediately
    2. Warm up: First request doesn't have initialization delay
    3. Consistency: All services use the same initialized providers
    """
    # =========================================================================
    # STARTUP: Initialize shared providers
    # =========================================================================
    print("\n" + "=" * 60)
    print("APPLICATION STARTUP")
    print("=" * 60)

    # Initialize all shared providers
    # This creates the SINGLE instances that both ingestion and query will use
    provider_status = initialize_providers()

    # Log the status
    if provider_status["vector_store"]:
        print("\n[Startup] Vector store initialized - ready for ingestion and queries")
    else:
        print("\n[Startup] WARNING: Vector store not available")

    print("=" * 60)
    print("APPLICATION READY")
    print("=" * 60 + "\n")

    # Yield control to the application
    yield

    # =========================================================================
    # SHUTDOWN: Clean up resources
    # =========================================================================
    print("\n[Shutdown] Application shutting down...")
    # Future: Add cleanup logic here if needed


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    This function:
    1. Creates the FastAPI instance with metadata
    2. Configures middleware (CORS, etc.)
    3. Includes all the route handlers
    4. Returns the configured app

    Returns:
        FastAPI: The configured application instance, ready to serve requests.

    Usage:
        app = create_app()
        # Then run with: uvicorn src.api.app:app --reload
    """
    # =========================================================================
    # STEP 1: Create the FastAPI application instance
    # =========================================================================
    # We configure metadata that appears in the auto-generated documentation.
    # Visit /docs after starting the server to see this in action!

    app = FastAPI(
        # Lifespan context manager for startup/shutdown events
        # This initializes shared providers at startup
        lifespan=lifespan,

        # Basic API Information
        # These appear at the top of the documentation page
        title="RAG Service API",
        description="""
        ## Production-Grade RAG Engine

        This API provides Retrieval-Augmented Generation (RAG) capabilities
        for building intelligent question-answering systems.

        ### Features

        * **Query Endpoint**: Ask questions and get AI-generated answers
        * **Source Citations**: Answers include references to source documents
        * **Scalable Architecture**: Designed for production workloads

        ### Current Status

        **Step 1**: Minimal API skeleton with mocked responses.

        Future steps will add:
        - Real vector database integration
        - Embedding model support
        - LLM integration (OpenAI, Anthropic, etc.)
        - Document ingestion pipeline
        """,
        version="0.1.0",  # Semantic versioning (major.minor.patch)

        # Documentation URLs
        # FastAPI automatically generates interactive documentation
        docs_url="/docs",       # Swagger UI (interactive)
        redoc_url="/redoc",     # ReDoc (alternative documentation style)
        openapi_url="/openapi.json",  # OpenAPI schema (machine-readable)

        # Contact information (appears in docs)
        contact={
            "name": "RAG Service Team",
            "email": "support@example.com",
        },

        # License information
        license_info={
            "name": "MIT",
        },
    )

    # =========================================================================
    # STEP 2: Configure CORS (Cross-Origin Resource Sharing)
    # =========================================================================
    # CORS controls which websites can call our API from a browser.
    # Without proper CORS settings, browsers will block requests from
    # web applications hosted on different domains.
    #
    # For development, we allow all origins (*).
    # In production, you should restrict this to your frontend domains.

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production: ["https://yourdomain.com"]
        allow_credentials=True,  # Allow cookies and auth headers
        allow_methods=["*"],     # Allow all HTTP methods (GET, POST, etc.)
        allow_headers=["*"],     # Allow all headers
    )

    # =========================================================================
    # STEP 3: Include Route Handlers
    # =========================================================================
    # We include each router which registers its endpoints with the app.
    # The prefix from the router is used, so query routes become /query/*

    # Include the query router (POST /query)
    app.include_router(query.router)

    # Include the ingest router (POST /ingest) - STEP 5
    app.include_router(ingest.router)

    # Include the collections router (multi-tenant collection management)
    app.include_router(collections.router)

    # =========================================================================
    # STEP 4: Add Root Endpoint
    # =========================================================================
    # A simple endpoint at the root path (/) for basic health checks
    # and to confirm the API is running.

    @app.get(
        "/",
        tags=["Health"],
        summary="Root endpoint",
        description="Returns basic API information. Useful for checking if the service is running.",
    )
    async def root():
        """
        Root endpoint that returns basic API information.

        This is useful for:
        1. Quick health checks (is the service responding?)
        2. Discovering the API version
        3. Getting links to documentation
        """
        return {
            "service": "RAG Service API",
            "version": "0.1.0",
            "status": "running",
            "docs": "/docs",
            "health": "/health",
        }

    # =========================================================================
    # STEP 5: Add Health Check Endpoint
    # =========================================================================
    # Health checks are used by:
    # - Load balancers (to know if the service can receive traffic)
    # - Kubernetes (readiness and liveness probes)
    # - Monitoring systems (to detect outages)

    @app.get(
        "/health",
        tags=["Health"],
        summary="Health check endpoint",
        description="Returns the health status of the service. Used by load balancers and monitoring.",
    )
    async def health_check():
        """
        Health check endpoint for monitoring and orchestration.

        Returns a simple status indicating the service is healthy.
        In production, this would also check:
        - Database connections
        - External service availability
        - Memory/CPU usage
        """
        return {
            "status": "healthy",
            "service": "rag-api",
            "version": "0.1.0",
        }

    # =========================================================================
    # STEP 6: Return the configured application
    # =========================================================================

    return app


# =============================================================================
# CREATE THE APPLICATION INSTANCE
# =============================================================================
# We create the app instance at module level so it can be imported by uvicorn.
# This is the standard pattern for FastAPI applications.
#
# The app variable is what uvicorn looks for when you run:
#   uvicorn src.api.app:app --reload

app = create_app()
