"""
Query Route Module
===================

This module defines the /query endpoint for the RAG API.

WHAT IS A ROUTE?
----------------
A route (also called an endpoint) is a specific URL path that the API responds to.
When a client sends a request to that path, the associated function is executed.

For example:
- POST /query -> Runs the `query_rag` function defined below

WHY SEPARATE ROUTES INTO MODULES?
---------------------------------
1. Organization: Each route file handles a specific feature area
2. Maintainability: Easy to find and modify specific endpoints
3. Scalability: Teams can work on different routes independently
4. Testing: Routes can be tested in isolation

FASTAPI ROUTER CONCEPT:
-----------------------
A Router is like a "mini application" that groups related endpoints.
We create routes here and then include them in the main app.
This keeps the main app.py file clean and focused.
"""

from fastapi import APIRouter, HTTPException, status

# Import our Pydantic schemas for request/response validation.
# These ensure the API accepts and returns data in the correct format.
from src.api.schemas.requests import QueryRequest
from src.api.schemas.responses import QueryResponse

# Import the RAG pipeline that contains our core business logic.
# The route's job is to handle HTTP concerns; the pipeline does the work.
from src.rag.pipeline import RAGPipeline


# =============================================================================
# CREATE THE ROUTER
# =============================================================================
# APIRouter groups related endpoints together.
# The prefix and tags are used for organization and documentation.

router = APIRouter(
    prefix="/query",  # All routes in this file will start with /query
    tags=["Query"],   # Groups these endpoints in the API documentation
)


# =============================================================================
# SHARED RAG PIPELINE INSTANCE
# =============================================================================
# We use a module-level variable that is lazily initialized on first use.
# This ensures the RAGPipeline uses the SHARED providers from src.core.providers.
#
# WHY LAZY INITIALIZATION?
# ------------------------
# If we create RAGPipeline() at import time, it might initialize before
# the shared providers are ready. By using lazy initialization:
# 1. The pipeline is created on the first request
# 2. It uses the already-initialized shared providers
# 3. Both ingestion and query share the same vector store
#
# WHY IS SHARED VECTOR STORE CRITICAL?
# ------------------------------------
# Without sharing:
#   - User uploads document via /api/v1/ingest → stored in Ingest's store
#   - User queries via /query → searches Pipeline's store (different!)
#   - Result: User's documents are NEVER found!
#
# With sharing (this implementation):
#   - Both use the SAME shared vector store
#   - Ingested documents are correctly found by queries

_rag_pipeline = None


def get_rag_pipeline() -> RAGPipeline:
    """
    Get the shared RAG pipeline instance (lazy initialization).

    This function ensures we have a single RAGPipeline instance that:
    1. Uses the SHARED vector store (same as ingestion)
    2. Uses the SHARED embedding provider
    3. Uses the SHARED LLM provider

    Returns:
        The shared RAGPipeline instance.
    """
    global _rag_pipeline
    if _rag_pipeline is None:
        print("[QueryRoute] Creating shared RAG pipeline...")
        _rag_pipeline = RAGPipeline()
        print("[QueryRoute] RAG pipeline ready")
    return _rag_pipeline


# =============================================================================
# QUERY ENDPOINT
# =============================================================================

@router.post(
    "",  # Empty string means the route is just /query (prefix handles the path)
    response_model=QueryResponse,  # Validates and documents the response format
    status_code=status.HTTP_200_OK,  # Returns 200 OK on success
    summary="Query the RAG system",
    description="""
    Send a question to the RAG system and receive an AI-generated answer
    based on retrieved documents.

    The system will:
    1. Analyze your question
    2. Retrieve relevant documents from the knowledge base
    3. Generate an answer using the retrieved context
    4. Return the answer along with source citations

    **Note:** This is Step 1 with mocked responses for demonstration.
    """,
    responses={
        200: {
            "description": "Successful response with answer and sources",
            "model": QueryResponse,
        },
        400: {
            "description": "Invalid request (e.g., empty question)",
        },
        500: {
            "description": "Internal server error during processing",
        },
    },
)
async def query_rag(request: QueryRequest) -> QueryResponse:
    """
    Process a RAG query and return the generated answer.

    This is an ASYNC function because:
    1. In production, we'll make async calls to LLMs and vector DBs
    2. Async allows handling many concurrent requests efficiently
    3. FastAPI is optimized for async request handling

    Args:
        request: The validated request body containing the question.
                 FastAPI automatically validates this using the QueryRequest schema.
                 If validation fails, FastAPI returns a 422 error automatically.

    Returns:
        QueryResponse: The answer and sources from the RAG pipeline.

    Raises:
        HTTPException: If something goes wrong during processing.
                       This is converted to an appropriate HTTP error response.

    FLOW:
    1. FastAPI receives HTTP POST request
    2. Request body is validated against QueryRequest schema
    3. This function is called with the validated data
    4. We call the RAG pipeline with the question
    5. The response is validated against QueryResponse schema
    6. FastAPI returns the JSON response
    """
    try:
        # =================================================================
        # STEP 1: Extract the question from the validated request
        # =================================================================
        # The request object is already validated by FastAPI/Pydantic.
        # We know the question exists and meets our constraints.

        question = request.question

        # =================================================================
        # STEP 2: Log the incoming request (helpful for debugging)
        # =================================================================
        # In production, we would use proper structured logging here.
        # For Step 1, we use a simple print statement.

        print(f"[QUERY] Received question: {question[:100]}...")  # Truncate for logging

        # =================================================================
        # STEP 3: Execute the RAG pipeline
        # =================================================================
        # The pipeline handles all the RAG logic:
        # - Document retrieval
        # - Answer generation
        # - Source extraction
        #
        # We pass just the question; the pipeline does the rest.

        # Get the shared pipeline instance (uses shared vector store)
        pipeline = get_rag_pipeline()
        result = pipeline.run(question=question)

        # =================================================================
        # STEP 4: Log the response (helpful for debugging)
        # =================================================================

        print(f"[QUERY] Generated answer with {len(result['sources'])} sources")

        # =================================================================
        # STEP 5: Return the structured response
        # =================================================================
        # FastAPI will automatically:
        # 1. Validate this against QueryResponse schema
        # 2. Convert it to JSON
        # 3. Set the correct Content-Type header
        # 4. Return it to the client

        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:
        # =================================================================
        # ERROR HANDLING
        # =================================================================
        # If anything goes wrong, we catch the exception and return
        # a proper HTTP error response.
        #
        # In production, we would:
        # 1. Log the full exception with stack trace
        # 2. Report to error monitoring (Sentry, etc.)
        # 3. Return a user-friendly error message (not internal details)

        print(f"[ERROR] Query failed: {str(e)}")

        # Raise an HTTPException which FastAPI converts to an error response.
        # We use 500 Internal Server Error for unexpected failures.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your query. Please try again."
        )
