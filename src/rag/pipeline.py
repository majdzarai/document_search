"""
RAG Pipeline Module
====================

This module contains the core RAG (Retrieval-Augmented Generation) pipeline.

WHAT IS A RAG PIPELINE?
-----------------------
A RAG pipeline is a system that:
1. Takes a user's question as input
2. Retrieves relevant documents from a knowledge base
3. Uses those documents as context to generate an answer
4. Returns the answer along with source references

STEP 4 IMPLEMENTATION - LLM INTEGRATION:
----------------------------------------
In this step, we add REAL LLM ANSWER GENERATION:
- Questions are still embedded and searched in the vector store
- Retrieved documents are passed to an LLM as context
- The LLM generates a real, coherent answer based on the documents
- No more mocked responses!

THE RAG FLOW (Step 4):
----------------------
1. USER ASKS QUESTION: "What is RAG?"
2. EMBED QUESTION: Convert question to embedding vector [0.1, 0.2, ...]
3. SEARCH VECTOR DB: Find documents with similar embeddings
4. GET TOP RESULTS: Return most similar documents with scores
5. GENERATE ANSWER: Send documents + question to LLM (REAL GENERATION!)
6. RETURN RESPONSE: Answer + source citations

WHAT'S NEW IN STEP 4:
---------------------
- LLM provider initialization (OpenRouter)
- Real answer generation using the LLM
- Documents are passed as context to the LLM
- The LLM synthesizes information from documents
- Answers are grounded in retrieved content

WHAT'S STILL TO BE IMPLEMENTED:
-------------------------------
- Document ingestion (we use hardcoded example docs)
- Chunking (documents are used as-is, no splitting)
- Reranking (direct vector search, no reranking)

WHY SEPARATE THE PIPELINE INTO ITS OWN MODULE?
----------------------------------------------
1. Separation of Concerns: The API layer handles HTTP requests,
   while the pipeline handles the business logic.
2. Testability: We can test the pipeline independently of the API.
3. Reusability: The same pipeline can be used by different interfaces
   (API, CLI, background jobs, etc.)
"""

from typing import List, Dict, Any, Optional

# =============================================================================
# STEP 2: Import the embedding layer
# =============================================================================
# We import the base class for type hints.
# IMPORTANT: We use SHARED providers from src.core.providers, NOT the factory!
# This ensures ingestion and query use the SAME vector store instance.

from src.embeddings.base import EmbeddingProvider

# =============================================================================
# STEP 3: Import the vector store layer
# =============================================================================
# The vector store is where we store document embeddings for similarity search.
# When a user asks a question:
# 1. We convert the question to an embedding
# 2. We search the vector store for similar document embeddings
# 3. We return the most similar documents as context
#
# CRITICAL: We use SHARED providers to ensure ingestion and query see the same data!

from src.vectorstore.base import VectorStoreProvider

# =============================================================================
# STEP 4: Import the LLM layer
# =============================================================================
# The LLM (Large Language Model) is responsible for generating answers.
# After we retrieve relevant documents from the vector store, we pass them
# to the LLM along with the user's question. The LLM then generates a
# coherent, helpful answer based on the provided context.
#
# This is the "Generation" part of "Retrieval-Augmented Generation"!

from src.llm.base import LLMProvider

# =============================================================================
# STEP 6: Import the Reranker layer
# =============================================================================
# The reranker is an OPTIONAL second-stage retrieval component.
# It takes documents retrieved by vector search and re-scores them
# to find the most relevant ones before sending to the LLM.
#
# WHY RERANKING?
# --------------
# Vector search is fast but not always precise. Reranking:
# 1. Gets a larger candidate set from vector search (e.g., 20 docs)
# 2. Carefully scores each candidate's relevance to the query
# 3. Returns only the best documents (e.g., top 5) to the LLM
#
# This two-stage approach gives us:
# - SPEED: Vector search quickly narrows down candidates
# - PRECISION: Reranking ensures only the best documents reach the LLM
#
# BACKWARD COMPATIBILITY:
# -----------------------
# Reranking is DISABLED by default. Set ENABLE_RERANKING=true to enable.
# When disabled, the pipeline works exactly as before (pure vector search).

from src.reranker.base import RerankerProvider

# =============================================================================
# STEP 7: Import the Evaluation layer (OPTIONAL)
# =============================================================================
# The evaluator measures RAG quality without affecting the response.
# It computes metrics like:
# - Retrieval: Precision@K, Recall@K, MRR
# - Generation: Faithfulness, Context Coverage, Hallucination Risk
#
# IMPORTANT - BACKWARD COMPATIBILITY:
# -----------------------------------
# Evaluation is DISABLED by default (ENABLE_EVALUATION=false).
# When disabled:
# - No evaluation overhead
# - Response format unchanged
# - Behavior identical to before Step 7
#
# When enabled (ENABLE_EVALUATION=true):
# - Metrics computed after response generation
# - Results optionally included in response
# - Does NOT affect the answer or sources

from src.evaluation.evaluator import get_evaluator, RAGEvaluator

# =============================================================================
# SHARED PROVIDERS - CRITICAL FOR CORRECT RAG OPERATION
# =============================================================================
# We use shared provider instances to ensure that:
# 1. Documents ingested via /api/v1/ingest are stored in the SAME vector store
#    that the query pipeline searches
# 2. We don't waste memory creating multiple provider instances
# 3. The system works correctly as a production RAG engine
#
# WITHOUT shared providers:
#   - Ingestion creates vector store A, stores documents there
#   - Query creates vector store B, searches there (finds nothing!)
#
# WITH shared providers:
#   - Both use the SAME vector store
#   - Ingested documents are found by queries

from src.core.providers import (
    get_embedding_provider,
    get_vector_store,
    get_llm_provider,
    get_reranker
)

# Import settings to check if demo seeding is enabled
from src.core.config import settings


class RAGPipeline:
    """
    The main RAG Pipeline class that orchestrates the entire RAG process.

    This class is responsible for:
    1. Receiving a user's question
    2. Converting the question to an embedding vector (STEP 2)
    3. Searching vector store for similar documents (STEP 3)
    4. Generating an answer using the LLM (STEP 4)
    5. OPTIONALLY reranking documents before generation (STEP 6 - NEW!)
    6. Returning the answer with source citations

    DESIGN PATTERN: This class follows the "Facade" pattern.
    It provides a simple interface (the `run` method) that hides
    the complexity of the underlying RAG process.

    STEP 6 ADDITIONS (RERANKING):
    -----------------------------
    The pipeline now supports an OPTIONAL reranking stage:
    - When ENABLE_RERANKING=false (default): Works exactly as before
    - When ENABLE_RERANKING=true: Adds a reranking step

    With reranking enabled:
    1. Vector search retrieves RETRIEVAL_TOP_K candidates (e.g., 20)
    2. Reranker scores each candidate's relevance to the query
    3. Top FINAL_TOP_K documents (e.g., 5) are sent to the LLM

    WHY RERANKING IMPROVES ACCURACY:
    --------------------------------
    Vector search (embedding similarity) is fast but imprecise.
    A document can have similar embeddings without being truly relevant.
    Reranking uses a more sophisticated model to assess relevance,
    ensuring only the best documents reach the LLM.

    THE RAG PROCESS (Step 6):
    -------------------------
    1. Initialize: Embed provider, vector store, LLM, (optional) reranker
    2. Question comes in: "What is RAG?"
    3. Embed question: Convert to vector [0.1, 0.2, ...]
    4. Search: Find similar vectors in Qdrant
    5. Retrieve: Get top RETRIEVAL_TOP_K documents with scores
    6. RERANK (optional): Re-score and filter to FINAL_TOP_K
    7. Generate: Send documents + question to LLM
    8. Return: Answer + source citations

    BACKWARD COMPATIBILITY:
    -----------------------
    When ENABLE_RERANKING=false (the default):
    - Pipeline fetches FINAL_TOP_K documents directly
    - No reranking overhead
    - Behavior identical to Step 4/5

    Example Usage:
        pipeline = RAGPipeline()
        result = pipeline.run("What is Python?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(
        self,
        embedding_provider: Optional[EmbeddingProvider] = None,
        vector_store_provider: Optional[VectorStoreProvider] = None,
        llm_provider: Optional[LLMProvider] = None,
        reranker_provider: Optional[RerankerProvider] = None
    ) -> None:
        """
        Initialize the RAG Pipeline.

        This constructor sets up all the components needed for the RAG process.

        STEP 6 CHANGES:
        ---------------
        In Step 6, we add optional reranker initialization:
        - Reranker is only initialized if ENABLE_RERANKING=true
        - When enabled, improves retrieval precision
        - When disabled (default), no overhead is added

        WHY ALLOW PASSING PROVIDERS?
        ----------------------------
        Dependency injection allows:
        1. Testing: Pass mock providers in tests
        2. Customization: Use specific provider configurations
        3. Sharing: Multiple pipelines can share providers

        Args:
            embedding_provider: Optional. An EmbeddingProvider instance to use.
                               If None, a provider is created from settings.

            vector_store_provider: Optional. A VectorStoreProvider instance.
                                  If None, a provider is created from settings.

            llm_provider: Optional. An LLMProvider instance to use.
                         If None, a provider is created from settings.

            reranker_provider: Optional. A RerankerProvider instance to use.
                              If None, uses shared provider (if ENABLE_RERANKING=true).
                              Default behavior (ENABLE_RERANKING=false): no reranker.

        Example:
            # Default: creates providers from environment variables
            pipeline = RAGPipeline()

            # Custom: pass your own providers
            from src.embeddings.providers.openrouter import OpenRouterEmbeddingProvider
            from src.vectorstore.providers.qdrant import QdrantVectorStore
            from src.llm.providers.openrouter import OpenRouterLLMProvider

            embed_provider = OpenRouterEmbeddingProvider()
            vector_store = QdrantVectorStore(collection_name="my_docs")
            llm = OpenRouterLLMProvider(model="openai/gpt-4")

            pipeline = RAGPipeline(
                embedding_provider=embed_provider,
                vector_store_provider=vector_store,
                llm_provider=llm
            )
        """
        print("\n" + "=" * 60)
        print("INITIALIZING RAG PIPELINE (Step 6)")
        print("=" * 60)

        # =====================================================================
        # STEP 2: Initialize the embedding provider (USING SHARED INSTANCE)
        # =====================================================================
        # The embedding provider converts text to numerical vectors.
        # We need this to:
        # 1. Embed documents when storing them
        # 2. Embed questions when searching
        #
        # IMPORTANT: We use the SHARED provider from src.core.providers
        # This ensures that both ingestion and query use the same provider.

        self._embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider is not None:
            # Allow override for testing
            self._embedding_provider = embedding_provider
            print("[RAGPipeline] Using provided embedding provider (override)")
        else:
            # USE SHARED PROVIDER - critical for correct operation
            self._embedding_provider = get_embedding_provider()
            if self._embedding_provider:
                print("[RAGPipeline] Using SHARED embedding provider")
            else:
                print("[RAGPipeline] WARNING: Embedding provider not available")
                print("[RAGPipeline] Running without embeddings (retrieval will be mocked)")

        # =====================================================================
        # STEP 3: Initialize the vector store (USING SHARED INSTANCE)
        # =====================================================================
        # The vector store (Qdrant) is where we store document embeddings.
        #
        # CRITICAL: We use the SHARED vector store from src.core.providers
        # This is the most important change for correct RAG operation!
        #
        # WHY SHARED?
        # -----------
        # Without sharing, ingestion and query would use different stores:
        #   - User uploads document → stored in Ingestion's vector store
        #   - User queries → searches Pipeline's vector store (empty!)
        #   - Result: User's documents are never found
        #
        # With sharing:
        #   - User uploads document → stored in SHARED vector store
        #   - User queries → searches SAME SHARED vector store
        #   - Result: User's documents are found correctly!
        #
        # PERSISTENCE MODES:
        # ------------------
        # - In-memory (default): Data lost on restart - good for development
        # - Local server: Set QDRANT_HOST/PORT - data persists
        # - Cloud: Set QDRANT_URL/API_KEY - production ready

        self._vector_store: Optional[VectorStoreProvider] = None
        self._documents_loaded: bool = False  # Track if we've loaded docs

        if vector_store_provider is not None:
            # Allow override for testing
            self._vector_store = vector_store_provider
            print("[RAGPipeline] Using provided vector store provider (override)")
        else:
            # USE SHARED PROVIDER - critical for correct operation
            self._vector_store = get_vector_store()
            if self._vector_store:
                print("[RAGPipeline] Using SHARED vector store")
            else:
                print("[RAGPipeline] WARNING: Vector store not available")
                print("[RAGPipeline] Running with mocked retrieval")

        # =====================================================================
        # STEP 4: Initialize the LLM provider (USING SHARED INSTANCE)
        # =====================================================================
        # The LLM (Large Language Model) generates answers based on the
        # retrieved documents. This is the "Generation" part of RAG.
        #
        # The flow is:
        # 1. User asks a question
        # 2. We retrieve relevant documents
        # 3. We send the documents + question to the LLM
        # 4. The LLM generates a coherent answer
        #
        # Without the LLM, we would just return raw documents.
        # The LLM reads the documents and synthesizes a helpful answer.

        self._llm_provider: Optional[LLMProvider] = None

        if llm_provider is not None:
            # Allow override for testing
            self._llm_provider = llm_provider
            print("[RAGPipeline] Using provided LLM provider (override)")
        else:
            # USE SHARED PROVIDER
            self._llm_provider = get_llm_provider()
            if self._llm_provider:
                print("[RAGPipeline] Using SHARED LLM provider")
            else:
                print("[RAGPipeline] WARNING: LLM provider not available")
                print("[RAGPipeline] Running with mocked generation")

        # =====================================================================
        # STEP 6: Initialize the Reranker (OPTIONAL, CONFIG-DRIVEN)
        # =====================================================================
        # The reranker is an OPTIONAL second-stage retrieval component.
        #
        # IMPORTANT - BACKWARD COMPATIBILITY:
        # -----------------------------------
        # - When ENABLE_RERANKING=false (default): No reranker is initialized
        # - Pipeline works exactly as before (pure vector search)
        # - No additional API calls or overhead
        #
        # When ENABLE_RERANKING=true:
        # - Reranker is initialized from shared providers
        # - Vector search retrieves RETRIEVAL_TOP_K candidates
        # - Reranker filters to FINAL_TOP_K before LLM generation
        # - Better precision but additional processing cost
        #
        # WHY CONFIG-DRIVEN?
        # ------------------
        # Not everyone needs reranking. It adds latency and cost.
        # Making it opt-in ensures existing deployments aren't affected.

        self._reranker: Optional[RerankerProvider] = None

        if reranker_provider is not None:
            # Allow override for testing
            self._reranker = reranker_provider
            print("[RAGPipeline] Using provided reranker provider (override)")
        else:
            # USE SHARED PROVIDER (only if reranking is enabled)
            self._reranker = get_reranker()
            if self._reranker:
                print(f"[RAGPipeline] Using SHARED reranker: {self._reranker.get_provider_name()}")
                print(f"[RAGPipeline] Retrieval: {settings.retrieval_top_k} -> Rerank -> {settings.final_top_k} docs")
            else:
                print("[RAGPipeline] Reranking is DISABLED (default behavior)")
                print("[RAGPipeline] To enable: set ENABLE_RERANKING=true")

        # =====================================================================
        # STEP 7: Initialize the Evaluator (OPTIONAL, CONFIG-DRIVEN)
        # =====================================================================
        # The evaluator measures RAG quality without affecting responses.
        #
        # IMPORTANT - BACKWARD COMPATIBILITY:
        # -----------------------------------
        # - When ENABLE_EVALUATION=false (default): No evaluator overhead
        # - When ENABLE_EVALUATION=true: Metrics computed after each query
        #
        # The evaluator NEVER modifies the answer or sources.
        # It only observes and measures quality.

        self._evaluator: Optional[RAGEvaluator] = None

        if settings.enable_evaluation:
            self._evaluator = get_evaluator()
            print(f"[RAGPipeline] Evaluation ENABLED")
            print(f"[RAGPipeline] RAGAS: {'enabled' if settings.enable_ragas else 'disabled'}")
        else:
            print("[RAGPipeline] Evaluation is DISABLED (default behavior)")
            print("[RAGPipeline] To enable: set ENABLE_EVALUATION=true")

        # =====================================================================
        # Demo document seeding (DISABLED BY DEFAULT FOR PRODUCTION)
        # =====================================================================
        # Demo documents are example documents for testing/learning.
        #
        # IMPORTANT - WHY DEMO SEEDING IS GATED:
        # --------------------------------------
        # In a production RAG system, demo documents should NEVER be seeded:
        # 1. They pollute search results with irrelevant content
        # 2. Users see "example" answers instead of their own data
        # 3. It wastes resources embedding/storing useless documents
        #
        # WHEN DEMO SEEDING HAPPENS:
        # --------------------------
        # Demo documents are ONLY seeded if ALL of these are true:
        # 1. SEED_DEMO_DOCUMENTS=true in environment
        # 2. The vector store is available
        # 3. The embedding provider is available
        # 4. The collection is currently EMPTY
        #
        # This ensures:
        # - Production systems never get demo data (setting is false by default)
        # - If user has already uploaded docs, demos don't overwrite them
        # - Development can still work with demo data when needed

        if self._embedding_provider and self._vector_store:
            self._maybe_seed_demo_documents()
        else:
            print("[RAGPipeline] Skipping document seeding (missing provider)")

        print("=" * 60)
        print("RAG PIPELINE READY!")
        print("=" * 60 + "\n")

    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute the RAG pipeline for a given question.

        This is the main entry point for the RAG process.
        It orchestrates all the steps needed to answer a question.

        STEP 6 CHANGES (RERANKING):
        ---------------------------
        The pipeline now supports optional reranking:

        WITHOUT reranking (ENABLE_RERANKING=false, default):
        1. Question is converted to embedding
        2. Vector search retrieves FINAL_TOP_K documents
        3. Documents + question are sent to the LLM
        4. LLM generates answer

        WITH reranking (ENABLE_RERANKING=true):
        1. Question is converted to embedding
        2. Vector search retrieves RETRIEVAL_TOP_K candidates
        3. Reranker scores and filters to FINAL_TOP_K
        4. Best documents + question are sent to the LLM
        5. LLM generates answer

        Args:
            question: The user's question as a string.
                      Example: "What are the benefits of using Python?"

        Returns:
            A dictionary containing:
            - "answer": The generated answer string
            - "sources": A list of source document identifiers

            Example return value:
            {
                "answer": "Python is a versatile programming language...",
                "sources": ["doc1.pdf", "doc2.txt"]
            }

        STEP-BY-STEP PROCESS:
        1. Validate the input question
        2. Generate embedding for the question
        3. Search vector store for similar documents
        3.5 RERANK documents (if enabled)
        4. Generate answer using documents
        5. Extract and return source citations
        """
        print("\n" + "=" * 60)
        print("RUNNING RAG PIPELINE")
        print("=" * 60)
        print(f"Question: {question}")
        reranking_status = "ENABLED" if self._reranker else "DISABLED"
        print(f"Reranking: {reranking_status}")
        print("=" * 60)

        # =====================================================================
        # STEP 1: Validate the input question
        # =====================================================================
        # We check if the question is valid before processing.
        # This prevents errors and provides better error messages.

        if not question or not question.strip():
            return {
                "answer": "Please provide a valid question.",
                "sources": []
            }

        # =====================================================================
        # STEP 2: Generate embedding for the question
        # =====================================================================
        # Convert the question text into a numerical vector.
        # This vector will be used to find similar documents.

        print("\n[STEP 2] Generating question embedding...")
        question_embedding = self._embed_question(question)

        if question_embedding is not None:
            print(f"[STEP 2] Embedding generated: {len(question_embedding)} dimensions")
            print(f"[STEP 2] First 3 values: {question_embedding[:3]}")
        else:
            print("[STEP 2] Embedding skipped (provider not configured)")

        # =====================================================================
        # STEP 3: Retrieve relevant documents using vector search
        # =====================================================================
        # STEP 6 ENHANCEMENT:
        # -------------------
        # We now support two retrieval modes based on ENABLE_RERANKING:
        #
        # WITHOUT reranking (default):
        #   - Retrieve FINAL_TOP_K documents directly
        #   - Same behavior as before Step 6
        #
        # WITH reranking:
        #   - Retrieve RETRIEVAL_TOP_K candidates (larger set)
        #   - Reranker will filter to FINAL_TOP_K
        #   - Better precision through two-stage retrieval

        # Determine how many documents to retrieve
        # - With reranking: get more candidates for the reranker to score
        # - Without reranking: get only the final documents needed
        if self._reranker is not None:
            retrieval_count = settings.retrieval_top_k
            print(f"\n[STEP 3] Searching for relevant documents (reranking mode)...")
            print(f"[STEP 3] Retrieving {retrieval_count} candidates for reranking")
        else:
            retrieval_count = settings.final_top_k
            print(f"\n[STEP 3] Searching for relevant documents...")
            print(f"[STEP 3] Retrieving top {retrieval_count} documents")

        if question_embedding is not None and self._vector_store is not None:
            # USE REAL VECTOR SEARCH!
            retrieved_documents = self._retrieve_documents(
                question_embedding,
                top_k=retrieval_count
            )
        else:
            # Fall back to mocked retrieval if no embedding/store
            print("[STEP 3] Using mock retrieval (no embedding or vector store)")
            retrieved_documents = self._mock_retrieve_documents(question)

        print(f"[STEP 3] Retrieved {len(retrieved_documents)} documents")

        # =====================================================================
        # STEP 3.5: RERANK documents (OPTIONAL, STEP 6 FEATURE)
        # =====================================================================
        # If reranking is enabled, we score each candidate document
        # and keep only the most relevant ones.
        #
        # WHY RERANKING HELPS:
        # --------------------
        # Vector search finds documents with similar embeddings, but
        # similarity doesn't always mean relevance. Reranking:
        # 1. Scores each document's actual relevance to the query
        # 2. Filters out false positives (high similarity but low relevance)
        # 3. Ensures only the best documents reach the LLM
        #
        # OBSERVABILITY:
        # --------------
        # We log detailed information about the reranking process:
        # - How many documents were retrieved
        # - How many survived reranking
        # - The rerank scores of surviving documents

        if self._reranker is not None and retrieved_documents:
            print(f"\n[STEP 3.5] Reranking {len(retrieved_documents)} documents...")

            # Call the reranker
            reranked_documents = self._reranker.rerank(
                query=question,
                documents=retrieved_documents,
                top_k=settings.final_top_k,
                min_score=settings.reranking_min_score
            )

            # Log reranking results
            print(f"[STEP 3.5] Reranking complete:")
            print(f"  - Input:  {len(retrieved_documents)} candidates")
            print(f"  - Output: {len(reranked_documents)} documents (top {settings.final_top_k})")

            if reranked_documents:
                print(f"  - Score range: {reranked_documents[-1].get('rerank_score', 0):.3f} - {reranked_documents[0].get('rerank_score', 0):.3f}")

            # Use reranked documents for generation
            retrieved_documents = reranked_documents

        elif self._reranker is None and settings.enable_reranking:
            # Reranking was requested but reranker not available
            print("[STEP 3.5] Reranking requested but reranker not available")

        # =====================================================================
        # STEP 4: Generate the answer using the LLM
        # =====================================================================
        # THIS IS THE KEY STEP 4 CHANGE!
        # We now send the question and retrieved documents to an LLM.
        # The LLM reads the documents and generates a coherent answer.
        #
        # If the LLM is not configured, we fall back to mocked generation.

        print("\n[STEP 4] Generating answer...")

        if self._llm_provider is not None and retrieved_documents:
            # USE REAL LLM GENERATION!
            print("[STEP 4] Using LLM for answer generation...")
            answer = self._generate_answer(question, retrieved_documents)
        else:
            # Fall back to mocked generation if no LLM provider
            print("[STEP 4] Using mock generation (no LLM provider or no documents)")
            answer = self._mock_generate_answer(question, retrieved_documents)

        # =====================================================================
        # STEP 5: Extract source citations
        # =====================================================================
        # Extract document sources to attribute where information came from.

        print("\n[STEP 5] Extracting source citations...")
        sources = self._extract_sources(retrieved_documents)

        # =====================================================================
        # STEP 6: Build the response (UNCHANGED FROM PREVIOUS STEPS)
        # =====================================================================
        # Build the response object first. This ensures the response format
        # is IDENTICAL to before Step 7, maintaining backward compatibility.

        response = {
            "answer": answer,
            "sources": sources
        }

        # =====================================================================
        # STEP 7: Evaluate the response (OPTIONAL, NON-BLOCKING)
        # =====================================================================
        # If evaluation is enabled, compute quality metrics.
        #
        # IMPORTANT - NON-BLOCKING:
        # -------------------------
        # Evaluation happens AFTER the response is built.
        # It NEVER modifies the answer or sources.
        # If evaluation fails, the response is still returned.
        #
        # WHAT WE EVALUATE:
        # -----------------
        # - Retrieval: How good were the retrieved documents?
        # - Generation: Is the answer grounded in context?
        #
        # The evaluation result can optionally be included in the response
        # or just logged for monitoring purposes.

        if self._evaluator and self._evaluator.is_enabled:
            try:
                print("\n[STEP 7] Running evaluation (optional, non-blocking)...")

                eval_result = self._evaluator.evaluate(
                    question=question,
                    answer=answer,
                    retrieved_documents=retrieved_documents,
                    # Note: ground_truth_ids would come from a test dataset
                    # In production, this is typically None
                    ground_truth_ids=None
                )

                # Get summary for logging
                summary = eval_result.get_summary()
                if summary:
                    print(f"[STEP 7] Evaluation summary:")
                    for metric, value in summary.items():
                        if value is not None:
                            print(f"  - {metric}: {value:.3f}")

                # Optionally include evaluation in response
                # This is useful for debugging but can be disabled
                if settings.enable_evaluation:
                    response["evaluation"] = eval_result.to_dict()

            except Exception as e:
                # Evaluation failure should NEVER affect the response
                print(f"[STEP 7] WARNING: Evaluation failed (non-blocking): {e}")

        # =====================================================================
        # FINAL: Return the response
        # =====================================================================
        print("\n" + "=" * 60)
        print("RAG PIPELINE COMPLETE")
        print("=" * 60)

        return response

    def _embed_question(self, question: str) -> Optional[List[float]]:
        """
        Convert the question text into an embedding vector.

        WHAT THIS METHOD DOES:
        ----------------------
        1. Takes the user's question as text
        2. Sends it to the embedding provider
        3. Returns a vector (list of numbers) representing the question

        WHY DO WE EMBED THE QUESTION?
        -----------------------------
        To find relevant documents, we need to compare the question to
        documents. But computers can't compare "meaning" directly.
        By converting both the question and documents to vectors (embeddings),
        we can use math to find similar vectors.

        Similar meanings → Similar vectors → High similarity score

        Example:
        - Question embedding: [0.1, 0.2, 0.3, ...]
        - Document A embedding: [0.11, 0.21, 0.29, ...]  (similar - relevant!)
        - Document B embedding: [0.9, -0.5, 0.1, ...]   (different - not relevant)

        Args:
            question: The user's question as a string.

        Returns:
            A list of floating-point numbers (the embedding vector),
            or None if the embedding provider is not configured.

        WHY RETURN OPTIONAL[LIST[FLOAT]]?
        ----------------------------------
        We return None if:
        - The embedding provider wasn't initialized (e.g., no API key)
        - This allows the pipeline to run without embeddings for testing
        """
        # Check if we have an embedding provider
        if self._embedding_provider is None:
            # No provider configured - skip embedding
            # This happens when OPENROUTER_API_KEY is not set
            return None

        try:
            # Call the embedding provider to convert text to vector
            # This makes an API call to the embedding service
            print(f"[RAGPipeline] Generating embedding for question...")
            embedding = self._embedding_provider.embed_text(question)
            print(f"[RAGPipeline] Embedding generated successfully!")
            return embedding

        except Exception as e:
            # If embedding fails, log the error and continue without embeddings
            # This makes the pipeline more resilient - it can still return mocked
            # results even if the embedding service is down
            print(f"[RAGPipeline] WARNING: Failed to generate embedding: {e}")
            print("[RAGPipeline] Continuing with mocked retrieval...")
            return None

    def _mock_retrieve_documents(self, question: str) -> List[Dict[str, str]]:
        """
        MOCK: Simulate document retrieval from a vector database.

        In production, this method would:
        1. Use the question embedding to query a vector database
        2. Find documents with similar embeddings (high cosine similarity)
        3. Return the top-k matching document chunks

        For Step 2, we still return hardcoded fake documents.
        In Step 3+, this will be replaced with real vector database queries.

        Args:
            question: The user's question (used for realistic mock)

        Returns:
            A list of document dictionaries, each containing:
            - "id": Unique document identifier
            - "content": The text content of the document chunk
            - "metadata": Additional information about the document

        WHY RETURN A LIST OF DICTS?
        This structure mirrors what real vector databases return.
        Each document has content (the text) and metadata (source info).
        """
        # These are fake documents for demonstration purposes.
        # In production, these would come from your actual knowledge base.

        mock_documents = [
            {
                "id": "doc_001",
                "content": "RAG (Retrieval-Augmented Generation) is a technique "
                          "that combines information retrieval with text generation. "
                          "It helps LLMs provide more accurate and up-to-date answers "
                          "by grounding them in retrieved documents.",
                "metadata": {
                    "source": "rag_overview.pdf",
                    "page": 1,
                    "chunk_index": 0
                }
            },
            {
                "id": "doc_002",
                "content": "Vector databases store embeddings (numerical representations "
                          "of text) and enable fast similarity search. Popular options "
                          "include Qdrant, Pinecone, Weaviate, and pgvector.",
                "metadata": {
                    "source": "vector_databases_guide.pdf",
                    "page": 5,
                    "chunk_index": 2
                }
            },
            {
                "id": "doc_003",
                "content": "Embeddings are dense vector representations of text that "
                          "capture semantic meaning. Similar texts have similar embeddings, "
                          "which enables semantic search beyond simple keyword matching.",
                "metadata": {
                    "source": "embeddings_explained.pdf",
                    "page": 2,
                    "chunk_index": 1
                }
            }
        ]

        return mock_documents

    def _mock_generate_answer(
        self,
        question: str,
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        MOCK: Simulate answer generation using an LLM.

        In production, this method would:
        1. Build a prompt combining the question and document contents
        2. Send the prompt to an LLM API (OpenAI, Anthropic, etc.)
        3. Return the generated response

        For Step 3, we return a realistic-looking answer that includes
        information about the retrieved documents and their scores.
        In later steps, this will be replaced with real LLM calls.

        Args:
            question: The user's original question
            documents: The retrieved documents to use as context

        Returns:
            A string containing the generated answer

        WHY SEPARATE RETRIEVAL AND GENERATION?
        This separation allows us to:
        1. Test each component independently
        2. Swap out LLM providers without changing retrieval
        3. Log and debug each step separately
        """
        num_docs = len(documents)

        # Build information about retrieved documents (for learning purposes)
        doc_info = []
        for i, doc in enumerate(documents, 1):
            score = doc.get("score", "N/A")
            doc_id = doc.get("id", "unknown")
            source = doc.get("metadata", {}).get("source", "unknown")

            if isinstance(score, float):
                score_str = f"{score:.4f}"
            else:
                score_str = str(score)

            doc_info.append(f"  {i}. {doc_id} (score: {score_str}) from {source}")

        doc_list = "\n".join(doc_info) if doc_info else "  No documents retrieved"

        # Build the mock answer
        mock_answer = (
            f"Based on {num_docs} retrieved documents, here is the answer "
            f"to your question.\n\n"
            f"RETRIEVED DOCUMENTS (ranked by similarity):\n"
            f"{doc_list}\n\n"
            f"ANSWER:\n"
            f"RAG (Retrieval-Augmented Generation) is a powerful technique "
            f"that enhances Large Language Models by providing them with "
            f"relevant context from a knowledge base. This approach combines "
            f"the strengths of information retrieval systems with the natural "
            f"language generation capabilities of LLMs, resulting in more "
            f"accurate, factual, and up-to-date responses.\n\n"
            f"[This is a MOCKED response - Step 3 demonstrates vector storage "
            f"and retrieval with real similarity scores]"
        )

        return mock_answer

    # =========================================================================
    # STEP 4: Real LLM answer generation
    # =========================================================================

    def _generate_answer(
        self,
        question: str,
        documents: List[Dict[str, Any]]
    ) -> str:
        """
        Generate an answer using the LLM with retrieved documents as context.

        THIS IS THE KEY STEP 4 METHOD!
        ------------------------------
        This method sends the question and retrieved documents to the LLM
        and gets back a real, coherent answer.

        HOW IT WORKS:
        -------------
        1. We take the user's question
        2. We take the retrieved documents (from vector search)
        3. We send both to the LLM provider
        4. The LLM generates an answer based on the context
        5. We return the generated answer

        WHY IS THIS IMPORTANT?
        ----------------------
        The LLM is what makes RAG powerful:
        - It reads and understands multiple documents
        - It synthesizes information from different sources
        - It generates a coherent, natural language response
        - It can cite specific information from the documents

        Args:
            question: The user's original question.
                     Example: "What is RAG and how does it work?"

            documents: The retrieved documents to use as context.
                      Each document has:
                      - "content": The document text
                      - "metadata": Additional info (source, page, etc.)
                      - "score": Similarity score (optional)

        Returns:
            The generated answer from the LLM.

        WHAT HAPPENS IF LLM CALL FAILS?
        -------------------------------
        If the LLM call fails (network error, API error, etc.),
        we catch the exception and return an error message.
        This makes the pipeline resilient to failures.
        """
        try:
            # Call the LLM provider's generate_with_context method
            # This method:
            # 1. Builds a prompt with the question and documents
            # 2. Sends it to the LLM API
            # 3. Returns the generated text
            answer = self._llm_provider.generate_with_context(
                question=question,
                context_documents=documents
            )

            return answer

        except Exception as e:
            # If LLM generation fails, log the error and return a fallback
            print(f"[RAGPipeline] ERROR: LLM generation failed: {e}")
            print("[RAGPipeline] Falling back to error message")

            return (
                f"I apologize, but I encountered an error while generating "
                f"the answer. The error was: {str(e)}\n\n"
                f"However, I found {len(documents)} relevant documents that "
                f"might help answer your question about: {question}"
            )

    def _extract_sources(self, documents: List[Dict[str, str]]) -> List[str]:
        """
        Extract source identifiers from retrieved documents.

        This method creates a list of human-readable source references
        that can be included in the API response for transparency.

        Args:
            documents: The list of retrieved document dictionaries

        Returns:
            A list of source strings (e.g., ["doc1.pdf", "doc2.pdf"])

        WHY PROVIDE SOURCES?
        Source citations are crucial for:
        1. Transparency: Users can verify the information
        2. Trust: Shows the answer is based on real documents
        3. Debugging: Helps identify retrieval quality issues
        """
        sources = []

        for doc in documents:
            # Extract the source filename from metadata.
            # We use .get() with a default to handle missing keys gracefully.
            metadata = doc.get("metadata", {})
            source_name = metadata.get("source", "unknown_source")

            # Only add unique sources (avoid duplicates).
            if source_name not in sources:
                sources.append(source_name)

        return sources

    def get_embedding_info(self) -> Dict[str, Any]:
        """
        Get information about the current embedding provider.

        This method is useful for debugging and monitoring.
        It returns information about the configured embedding provider.

        Returns:
            A dictionary containing:
            - "provider_configured": Whether an embedding provider is set up
            - "model_name": The name of the embedding model (if configured)
            - "dimension": The embedding dimension (if configured)

        Example:
            pipeline = RAGPipeline()
            info = pipeline.get_embedding_info()
            print(f"Model: {info['model_name']}")
            print(f"Dimension: {info['dimension']}")
        """
        if self._embedding_provider is None:
            return {
                "provider_configured": False,
                "model_name": None,
                "dimension": None
            }

        return {
            "provider_configured": True,
            "model_name": self._embedding_provider.get_model_name(),
            "dimension": self._embedding_provider.get_dimension()
        }

    # =========================================================================
    # STEP 3: New methods for vector store integration
    # =========================================================================

    def _maybe_seed_demo_documents(self) -> None:
        """
        Conditionally seed demo documents based on configuration.

        IMPORTANT - PRODUCTION SAFETY:
        ------------------------------
        This method checks multiple conditions before seeding demo documents.
        Demo documents should NEVER appear in a production system because
        they would pollute search results with irrelevant content.

        CONDITIONS FOR SEEDING:
        -----------------------
        All of these must be true:
        1. SEED_DEMO_DOCUMENTS=true in environment (default is FALSE)
        2. The vector store collection is currently EMPTY

        If ANY condition is false, no demo documents are seeded.

        WHY CHECK IF COLLECTION IS EMPTY?
        ---------------------------------
        Even in development mode, we don't want to seed demo documents
        if the user has already uploaded their own documents. This prevents:
        - Mixing user data with demo data
        - Confusion about where results come from
        - Duplicate seeding on restart

        HOW TO ENABLE DEMO SEEDING:
        ---------------------------
        Development (enable demo docs):
            export SEED_DEMO_DOCUMENTS=true

        Production (never seed - this is the default):
            Don't set the variable, or set to "false"
        """
        # Check condition 1: Is demo seeding enabled in config?
        if not settings.seed_demo_documents:
            print("[RAGPipeline] Demo document seeding is DISABLED (production mode)")
            print("[RAGPipeline] To enable: set SEED_DEMO_DOCUMENTS=true")
            return

        # Check condition 2: Is the collection empty?
        current_count = self._vector_store.count()
        if current_count > 0:
            print(f"[RAGPipeline] Collection already has {current_count} documents")
            print("[RAGPipeline] Skipping demo seeding to preserve existing data")
            return

        # All conditions met - proceed with seeding
        print("[RAGPipeline] Demo seeding ENABLED and collection is empty")
        print("[RAGPipeline] Seeding demo documents for development/testing...")
        self._seed_example_documents()

    def _seed_example_documents(self) -> None:
        """
        Seed the vector store with example documents.

        WHAT DOES THIS METHOD DO?
        -------------------------
        This method loads hardcoded example documents into the vector store.
        For each document:
        1. Generate an embedding using the embedding provider
        2. Store the embedding + text + metadata in Qdrant

        WHY HARDCODED DOCUMENTS?
        ------------------------
        In a production system, documents would come from:
        - A document ingestion pipeline
        - Uploaded files
        - Web scraping
        - Database exports
        - etc.

        For learning in Step 3, we use simple hardcoded documents to:
        1. Demonstrate how embeddings are stored
        2. Have something to search against
        3. Keep the example self-contained (no external files needed)

        WHY IS THIS CALLED AT STARTUP?
        ------------------------------
        Since we use Qdrant in in-memory mode:
        - The database is empty when the app starts
        - We need to populate it with documents
        - In production with persistent storage, this wouldn't be needed

        WHAT HAPPENS IF CALLED MULTIPLE TIMES?
        --------------------------------------
        The vector store uses "upsert" (update + insert):
        - If a document with the same ID exists, it's updated
        - If it doesn't exist, it's inserted
        - So calling this multiple times is safe!
        """
        print("\n" + "-" * 50)
        print("SEEDING EXAMPLE DOCUMENTS INTO VECTOR STORE")
        print("-" * 50)

        # =====================================================================
        # Define our example documents
        # =====================================================================
        # These documents cover RAG-related topics to demonstrate semantic search.
        # In a real system, these would come from your actual knowledge base.
        #
        # IMPORTANT: These are simple, un-chunked documents.
        # Real production systems would:
        # 1. Split long documents into smaller chunks
        # 2. Add more metadata (author, date, section, etc.)
        # 3. Process many more documents

        example_documents = [
            {
                "id": "doc_001",
                "text": (
                    "RAG (Retrieval-Augmented Generation) is a technique that enhances "
                    "Large Language Models by combining them with external knowledge retrieval. "
                    "Instead of relying solely on the model's training data, RAG systems "
                    "retrieve relevant documents from a knowledge base and use them as context "
                    "for generating responses. This makes answers more accurate, up-to-date, "
                    "and grounded in factual information."
                ),
                "metadata": {
                    "source": "rag_introduction.pdf",
                    "page": 1,
                    "topic": "RAG fundamentals"
                }
            },
            {
                "id": "doc_002",
                "text": (
                    "Vector databases like Qdrant are specialized systems designed to store "
                    "and search through embedding vectors efficiently. They use algorithms "
                    "like HNSW (Hierarchical Navigable Small World) to enable fast similarity "
                    "search even with millions of vectors. Popular vector databases include "
                    "Qdrant, Pinecone, Weaviate, Milvus, and pgvector."
                ),
                "metadata": {
                    "source": "vector_databases_guide.pdf",
                    "page": 3,
                    "topic": "vector databases"
                }
            },
            {
                "id": "doc_003",
                "text": (
                    "Embeddings are numerical representations of text that capture semantic "
                    "meaning. They convert words, sentences, or documents into vectors "
                    "(lists of numbers) where similar meanings result in similar vectors. "
                    "Modern embedding models like OpenAI's text-embedding-3 produce vectors "
                    "with 1536 or 3072 dimensions that can represent subtle semantic relationships."
                ),
                "metadata": {
                    "source": "embeddings_explained.pdf",
                    "page": 2,
                    "topic": "embeddings"
                }
            },
            {
                "id": "doc_004",
                "text": (
                    "Semantic search goes beyond traditional keyword matching by understanding "
                    "the meaning behind queries. For example, searching for 'how to fix bugs' "
                    "might also return results about 'debugging techniques' or 'error resolution' "
                    "because the embeddings capture that these concepts are semantically related."
                ),
                "metadata": {
                    "source": "semantic_search_intro.pdf",
                    "page": 1,
                    "topic": "semantic search"
                }
            },
            {
                "id": "doc_005",
                "text": (
                    "Chunking is the process of splitting large documents into smaller pieces "
                    "for embedding and retrieval. Different strategies include fixed-size chunks, "
                    "sentence-based splitting, paragraph-based splitting, and semantic chunking. "
                    "The optimal chunk size depends on your use case, but typically ranges from "
                    "200 to 1000 tokens."
                ),
                "metadata": {
                    "source": "chunking_strategies.pdf",
                    "page": 5,
                    "topic": "document processing"
                }
            },
            {
                "id": "doc_006",
                "text": (
                    "Python is a versatile programming language widely used for AI and machine "
                    "learning applications. Its extensive ecosystem includes libraries like "
                    "LangChain for building LLM applications, FastAPI for creating web services, "
                    "and numerous embedding providers and vector database clients."
                ),
                "metadata": {
                    "source": "python_for_ai.pdf",
                    "page": 1,
                    "topic": "programming"
                }
            },
        ]

        # =====================================================================
        # Generate embeddings for each document
        # =====================================================================
        # We use the embedding provider to convert each document's text
        # into a numerical vector.

        print(f"\n[Seeding] Processing {len(example_documents)} documents...")

        ids = []
        embeddings = []
        texts = []
        metadata_list = []

        for doc in example_documents:
            print(f"\n[Seeding] Embedding document: {doc['id']}")
            print(f"          Topic: {doc['metadata']['topic']}")
            print(f"          Text preview: {doc['text'][:60]}...")

            try:
                # Generate embedding for this document
                embedding = self._embedding_provider.embed_text(doc["text"])

                # Add to our lists
                ids.append(doc["id"])
                embeddings.append(embedding)
                texts.append(doc["text"])
                metadata_list.append(doc["metadata"])

                print(f"          Embedding dimension: {len(embedding)}")

            except Exception as e:
                print(f"          ERROR: Failed to embed document: {e}")
                continue

        # =====================================================================
        # Store embeddings in the vector store
        # =====================================================================
        # Now we send all the embeddings to Qdrant for storage.

        if embeddings:
            print(f"\n[Seeding] Storing {len(embeddings)} embeddings in vector store...")

            success = self._vector_store.upsert(
                ids=ids,
                embeddings=embeddings,
                texts=texts,
                metadata=metadata_list
            )

            if success:
                print(f"[Seeding] Successfully stored {len(embeddings)} documents!")
                self._documents_loaded = True
            else:
                print("[Seeding] WARNING: Failed to store documents")
        else:
            print("[Seeding] WARNING: No documents were embedded")

        # Display final statistics
        count = self._vector_store.count()
        print(f"\n[Seeding] Vector store now contains {count} documents")
        print("-" * 50 + "\n")

    def _retrieve_documents(
        self,
        query_embedding: List[float],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using real vector similarity search.

        THIS IS THE KEY STEP 3 METHOD!
        ------------------------------
        This method performs actual similarity search in Qdrant:
        1. Takes the question embedding
        2. Searches for similar document embeddings
        3. Returns documents with similarity scores

        HOW SIMILARITY SEARCH WORKS:
        ----------------------------
        1. The query embedding (from the question) is compared to all
           stored document embeddings
        2. Cosine similarity is calculated for each comparison:
           - 1.0 = identical meaning
           - 0.0 = completely unrelated
        3. Results are ranked by similarity score
        4. Top K results are returned

        UNDERSTANDING THE SCORES:
        -------------------------
        - 0.85+ : Very similar (highly relevant)
        - 0.70-0.85: Moderately similar (likely relevant)
        - 0.50-0.70: Somewhat related
        - Below 0.50: Probably not related

        These thresholds vary depending on:
        - The embedding model used
        - The domain/content type
        - The specificity of queries

        Args:
            query_embedding: The embedding vector of the user's question
            top_k: Number of documents to retrieve (default: 3)

        Returns:
            List of document dictionaries with:
            - "id": Document identifier
            - "content": Document text (renamed from 'text' for consistency)
            - "metadata": Additional information
            - "score": Similarity score (NEW in Step 3!)

        WHY RENAME 'text' TO 'content'?
        -------------------------------
        We use 'content' to match the format expected by other pipeline methods.
        The vector store returns 'text', but our pipeline uses 'content'.
        """
        print(f"\n[Retrieval] Performing similarity search (top_k={top_k})...")

        # Call the vector store's search method
        search_results = self._vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k
        )

        # Convert to our standard document format
        documents = []
        for result in search_results:
            doc = {
                "id": result["id"],
                "content": result["text"],  # Rename for pipeline consistency
                "metadata": result.get("metadata", {}),
                "score": result["score"]  # Include similarity score!
            }
            documents.append(doc)

            # Print for learning purposes
            print(f"\n  Retrieved: {doc['id']}")
            print(f"  Score: {doc['score']:.4f} (1.0 = perfect match)")
            print(f"  Preview: {doc['content'][:60]}...")

        return documents

    def get_vector_store_info(self) -> Dict[str, Any]:
        """
        Get information about the current vector store.

        This method is useful for debugging and monitoring.
        It returns information about the configured vector store.

        Returns:
            A dictionary containing:
            - "provider_configured": Whether a vector store is set up
            - "provider": The provider name (e.g., "qdrant")
            - "collection_name": Name of the collection
            - "total_documents": Number of stored documents
            - "documents_loaded": Whether example docs have been loaded

        Example:
            pipeline = RAGPipeline()
            info = pipeline.get_vector_store_info()
            print(f"Provider: {info['provider']}")
            print(f"Documents: {info['total_documents']}")
        """
        if self._vector_store is None:
            return {
                "provider_configured": False,
                "provider": None,
                "collection_name": None,
                "total_documents": 0,
                "documents_loaded": False
            }

        store_info = self._vector_store.get_info()
        return {
            "provider_configured": True,
            "provider": store_info.get("provider", "unknown"),
            "collection_name": store_info.get("collection_name", "unknown"),
            "total_documents": store_info.get("total_vectors", 0),
            "documents_loaded": self._documents_loaded
        }

    def get_llm_info(self) -> Dict[str, Any]:
        """
        Get information about the current LLM provider.

        This method is useful for debugging and monitoring.
        It returns information about the configured LLM provider.

        Returns:
            A dictionary containing:
            - "provider_configured": Whether an LLM provider is set up
            - "model_name": The name of the LLM model (if configured)

        Example:
            pipeline = RAGPipeline()
            info = pipeline.get_llm_info()
            print(f"Model: {info['model_name']}")
        """
        if self._llm_provider is None:
            return {
                "provider_configured": False,
                "model_name": None
            }

        return {
            "provider_configured": True,
            "model_name": self._llm_provider.get_model_name()
        }

    def get_reranker_info(self) -> Dict[str, Any]:
        """
        Get information about the current reranker configuration.

        This method is useful for debugging and monitoring.
        It returns information about whether reranking is enabled
        and what provider is being used.

        STEP 6 ADDITION:
        ----------------
        This method helps understand the retrieval enhancement status.

        Returns:
            A dictionary containing:
            - "enabled": Whether reranking is enabled
            - "provider_name": The reranker provider name (if enabled)
            - "retrieval_top_k": Number of candidates retrieved
            - "final_top_k": Number of documents after reranking
            - "min_score": Minimum rerank score threshold

        Example:
            pipeline = RAGPipeline()
            info = pipeline.get_reranker_info()
            print(f"Reranking enabled: {info['enabled']}")
            print(f"Provider: {info['provider_name']}")
        """
        if self._reranker is None:
            return {
                "enabled": False,
                "provider_name": None,
                "retrieval_top_k": settings.final_top_k,  # Without reranking, we fetch final_top_k directly
                "final_top_k": settings.final_top_k,
                "min_score": None
            }

        return {
            "enabled": True,
            "provider_name": self._reranker.get_provider_name(),
            "retrieval_top_k": settings.retrieval_top_k,
            "final_top_k": settings.final_top_k,
            "min_score": settings.reranking_min_score
        }

    def get_evaluator_info(self) -> Dict[str, Any]:
        """
        Get information about the current evaluator configuration.

        This method is useful for debugging and monitoring.
        It returns information about whether evaluation is enabled
        and what metrics are being computed.

        STEP 7 ADDITION:
        ----------------
        This method helps understand the evaluation status.

        Returns:
            A dictionary containing:
            - "enabled": Whether evaluation is enabled
            - "ragas_enabled": Whether RAGAS metrics are enabled
            - "default_k": Default K value for @K metrics
            - "log_results": Whether results are logged
            - "store_history": Whether history is being stored
            - "history_size": Number of evaluations in history

        Example:
            pipeline = RAGPipeline()
            info = pipeline.get_evaluator_info()
            print(f"Evaluation enabled: {info['enabled']}")
            print(f"RAGAS enabled: {info['ragas_enabled']}")
        """
        if self._evaluator is None:
            return {
                "enabled": False,
                "ragas_enabled": False,
                "default_k": settings.evaluation_default_k,
                "log_results": False,
                "store_history": False,
                "history_size": 0
            }

        return self._evaluator.get_status()
