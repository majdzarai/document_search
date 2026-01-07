"""
Configuration Module for RAG Service
=====================================

WHAT IS THIS MODULE?
--------------------
This module handles all configuration settings for our RAG (Retrieval-Augmented
Generation) service. Configuration includes things like:
- API keys for external services (OpenRouter, OpenAI, etc.)
- Model names and settings
- URLs for API endpoints

WHY DO WE NEED CONFIGURATION MANAGEMENT?
----------------------------------------
1. SECURITY: We never want to hardcode sensitive information like API keys
   directly in our code. If we did, anyone who sees our code would have
   access to our API keys (and potentially our billing accounts!).

2. FLEXIBILITY: By using environment variables, we can easily change settings
   without modifying code. This is especially useful when:
   - Moving from development to production
   - Testing with different models
   - Deploying to different environments (staging, production, etc.)

3. TWELVE-FACTOR APP: This follows the industry best practice of storing
   config in the environment (see https://12factor.net/config).

HOW DOES THIS WORK?
-------------------
1. We define a Settings class that specifies what configuration we need
2. Each setting is read from an environment variable
3. Environment variables are set OUTSIDE the code (in .env files, Docker, etc.)
4. The application reads these values at startup

EXAMPLE USAGE:
--------------
    # Import the settings instance
    from src.core.config import settings

    # Access configuration values
    api_key = settings.openrouter_api_key
    model_name = settings.embedding_model
"""
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Optional


class Settings:
    """
    Application settings loaded from environment variables.

    This class centralizes all configuration for the RAG service.
    Each attribute corresponds to an environment variable that can be set
    externally (in your shell, .env file, Docker, Kubernetes, etc.).

    HOW TO USE:
    -----------
    1. Set environment variables before running the app:

       On Windows (Command Prompt):
           set OPENROUTER_API_KEY=your-api-key-here

       On Windows (PowerShell):
           $env:OPENROUTER_API_KEY="your-api-key-here"

       On Linux/Mac:
           export OPENROUTER_API_KEY=your-api-key-here

    2. Access settings in code:
           from src.core.config import settings
           print(settings.openrouter_api_key)

    IMPORTANT NOTES:
    ----------------
    - API keys should NEVER be committed to version control (git)
    - Use a .env file locally (add .env to .gitignore!)
    - Use secrets management in production (AWS Secrets Manager, Vault, etc.)

    Attributes:
        openrouter_api_key: The API key for OpenRouter service.
        openrouter_base_url: The base URL for OpenRouter API (OpenAI-compatible).
        embedding_model: The name of the embedding model to use.
        embedding_provider: Which provider to use for embeddings (e.g., "openrouter").
    """

    def __init__(self) -> None:
        """
        Initialize settings by reading from environment variables.

        WHAT HAPPENS HERE:
        ------------------
        1. We read each setting from its corresponding environment variable
        2. We use os.getenv() which returns None if the variable isn't set
        3. We provide default values where appropriate (but NOT for secrets!)

        WHY USE os.getenv()?
        --------------------
        - os.getenv("VAR_NAME") returns the value of the environment variable
        - If the variable isn't set, it returns None (or a default if provided)
        - This is the standard Python way to read environment variables
        """

        # =====================================================================
        # OpenRouter API Configuration
        # =====================================================================
        # OpenRouter is a service that provides access to various AI models
        # through a unified API. It's OpenAI-compatible, meaning we can use
        # the same code we'd use for OpenAI's API.

        # The API key is like a password - it authenticates our requests.
        # We NEVER set a default value for API keys because:
        # 1. It would be a security risk if someone copies our code
        # 2. It reminds us to properly configure the environment
        self.openrouter_api_key: Optional[str] = os.getenv("OPENROUTER_API_KEY")

        # The base URL is where we send our API requests.
        # OpenRouter uses a different URL than OpenAI, but the API format is the same.
        # We provide a default here since this isn't sensitive information.
        self.openrouter_base_url: str = os.getenv(
            "OPENROUTER_BASE_URL",
            "https://openrouter.ai/api/v1"  # Default OpenRouter API URL
        )

        # =====================================================================
        # Embedding Model Configuration
        # =====================================================================
        # Embedding models convert text into numerical vectors (lists of numbers).
        # Different models produce different quality embeddings and have different costs.

        # WHAT IS AN EMBEDDING?
        # ---------------------
        # An embedding is a list of numbers (like [0.1, -0.2, 0.8, ...]) that
        # represents the "meaning" of a piece of text. Texts with similar meanings
        # have similar embeddings (their numbers are close to each other).
        #
        # For example:
        # - "cat" and "kitten" would have similar embeddings
        # - "cat" and "economics" would have very different embeddings

        # WHY DO WE NEED EMBEDDINGS IN RAG?
        # ---------------------------------
        # In a RAG system, we:
        # 1. Convert all our documents into embeddings and store them
        # 2. When a user asks a question, we convert the question into an embedding
        # 3. We find documents whose embeddings are similar to the question's embedding
        # 4. We use those similar documents as context for generating an answer
        #
        # This is called "semantic search" - we're searching by meaning, not keywords!

        # The embedding model to use. Different models have different:
        # - Dimensions (how many numbers in the vector): 768, 1024, 1536, 3072, etc.
        # - Quality (how well they capture meaning)
        # - Cost (API pricing)
        # - Speed (how fast they process text)
        #
        # Common choices:
        # - "openai/text-embedding-3-small": Good balance of cost/quality (1536 dimensions)
        # - "openai/text-embedding-3-large": Higher quality but more expensive (3072 dimensions)
        # - "openai/text-embedding-ada-002": Older model, still good (1536 dimensions)
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL",
            "openai/text-embedding-3-small"  # Default: good balance of cost and quality
        )

        # =====================================================================
        # Provider Selection
        # =====================================================================
        # We support multiple providers for flexibility.
        # The provider determines which service we use to generate embeddings.

        # Why support multiple providers?
        # 1. Cost: Different providers have different pricing
        # 2. Performance: Some providers may be faster or more reliable
        # 3. Features: Different providers offer different models
        # 4. Lock-in avoidance: We can switch providers without changing code
        self.embedding_provider: str = os.getenv(
            "EMBEDDING_PROVIDER",
            "openrouter"  # Default: use OpenRouter for embeddings
        )

        # =====================================================================
        # LLM (Language Model) Configuration (STEP 4)
        # =====================================================================
        # The LLM is responsible for generating answers in our RAG system.
        # After we retrieve relevant documents, the LLM reads them and
        # generates a helpful, coherent answer to the user's question.
        #
        # WHAT DOES THE LLM DO IN RAG?
        # ----------------------------
        # 1. Receives the user's question
        # 2. Receives the retrieved documents as context
        # 3. Generates an answer based on both
        # 4. This is the "Generation" part of "Retrieval-Augmented Generation"
        #
        # WHY SEPARATE FROM EMBEDDINGS?
        # -----------------------------
        # Embedding models and generation models are different:
        # - Embedding models: Convert text to vectors (for search)
        # - Generation models: Create new text (for answers)
        # You might use different providers/models for each.

        # Which LLM provider to use for text generation
        # Currently supported: "openrouter"
        # Future: "openai", "anthropic", "ollama"
        self.llm_provider: str = os.getenv(
            "LLM_PROVIDER",
            "openrouter"  # Default: use OpenRouter for LLM
        )

        # The LLM model to use for text generation
        # Different models have different:
        # - Quality (how good the answers are)
        # - Speed (how fast they respond)
        # - Cost (API pricing per token)
        #
        # Popular choices via OpenRouter:
        # - "openai/gpt-3.5-turbo": Fast and affordable
        # - "openai/gpt-4-turbo": Better quality, higher cost
        # - "anthropic/claude-3-sonnet-20240229": Good balance
        # - "anthropic/claude-3-opus-20240229": Highest quality Claude
        # - "meta-llama/llama-3-70b-instruct": Open source option
        self.llm_model: str = os.getenv(
            "LLM_MODEL",
            "openai/gpt-3.5-turbo"  # Default: good balance of cost/quality
        )

        # =====================================================================
        # Vector Store Configuration (STEP 3)
        # =====================================================================
        # A vector store (also called a vector database) is where we store
        # embeddings for fast similarity search. When a user asks a question,
        # we convert it to an embedding and search for similar documents.
        #
        # WHAT IS A VECTOR STORE?
        # -----------------------
        # Unlike regular databases that store text/numbers and use exact matching,
        # vector stores specialize in storing embeddings (lists of numbers) and
        # finding SIMILAR vectors quickly. This enables "semantic search" -
        # finding documents by meaning, not just keywords.
        #
        # Popular vector stores:
        # - Qdrant: Open-source, easy to use, supports in-memory mode
        # - Pinecone: Cloud-native, fully managed
        # - Weaviate: Feature-rich, supports hybrid search
        # - pgvector: PostgreSQL extension
        #
        # For learning, we use Qdrant in IN-MEMORY mode:
        # - No database installation required
        # - Data is stored in RAM (lost when app stops)
        # - Perfect for understanding how vector search works

        # Which vector store provider to use
        # Currently supported: "qdrant"
        # Future: "pinecone", "weaviate", "pgvector"
        self.vector_store_provider: str = os.getenv(
            "VECTOR_STORE_PROVIDER",
            "qdrant"  # Default: use Qdrant (supports in-memory mode)
        )

        # The dimension of embedding vectors
        # IMPORTANT: This MUST match your embedding model!
        # - OpenAI text-embedding-3-small: 1536
        # - OpenAI text-embedding-3-large: 3072
        # - Cohere embed-v3: 1024
        #
        # If this doesn't match, you'll get errors when storing/searching vectors
        self.vector_dimension: int = int(os.getenv(
            "VECTOR_DIMENSION",
            "1536"  # Default: matches text-embedding-3-small
        ))

        # =====================================================================
        # Qdrant-Specific Configuration
        # =====================================================================
        # These settings only apply when using Qdrant as the vector store.
        #
        # QDRANT RUNNING MODES:
        # ---------------------
        # 1. IN-MEMORY (default for learning):
        #    - No configuration needed!
        #    - Leave host, port, url all unset
        #    - Data stored in RAM, lost on restart
        #    - Perfect for testing and learning
        #
        # 2. LOCAL SERVER:
        #    - Run Qdrant locally: docker run -p 6333:6333 qdrant/qdrant
        #    - Set QDRANT_HOST=localhost and QDRANT_PORT=6333
        #    - Data persists in Docker volume
        #
        # 3. QDRANT CLOUD:
        #    - Create account at https://cloud.qdrant.io
        #    - Set QDRANT_URL to your cluster URL
        #    - Set QDRANT_API_KEY to your API key
        #    - Full production features

        # Name of the collection (like a table name in SQL databases)
        # A collection stores vectors with the same dimension
        self.qdrant_collection_name: str = os.getenv(
            "QDRANT_COLLECTION_NAME",
            "rag_documents"  # Default collection name
        )

        # Qdrant server host (for local server mode)
        # Example: "localhost" or "192.168.1.100"
        # Leave empty/None for in-memory mode
        self.qdrant_host: Optional[str] = os.getenv("QDRANT_HOST")

        # Qdrant server port (for local server mode)
        # Default Qdrant port is 6333
        # Leave empty/None for in-memory mode
        qdrant_port_str = os.getenv("QDRANT_PORT")
        self.qdrant_port: Optional[int] = int(qdrant_port_str) if qdrant_port_str else None

        # Qdrant Cloud URL (for cloud mode)
        # Example: "https://xyz-abc.us-east-1-0.aws.cloud.qdrant.io"
        # Leave empty/None for in-memory or local server mode
        self.qdrant_url: Optional[str] = os.getenv("QDRANT_URL")

        # Qdrant Cloud API key (for cloud mode)
        # Get this from your Qdrant Cloud dashboard
        # Leave empty/None for in-memory or local server mode
        self.qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")

        # =====================================================================
        # Demo/Development Mode Configuration
        # =====================================================================
        # These settings control development-only features that should NEVER
        # be enabled in production.

        # SEED_DEMO_DOCUMENTS: Whether to seed demo/example documents at startup
        #
        # IMPORTANT - WHY THIS EXISTS:
        # -----------------------------
        # In development, it's useful to have some example documents to test
        # queries against. However, in production:
        # - Demo documents would pollute real search results
        # - Users would see irrelevant "example" content
        # - It wastes resources embedding/storing useless data
        #
        # DEFAULT BEHAVIOR:
        # -----------------
        # - Default is FALSE (production-safe)
        # - Set to "true" ONLY for development/learning
        # - Demo docs are NEVER seeded if the collection already has data
        #
        # HOW TO USE:
        # -----------
        # Development (seed demo docs):
        #   export SEED_DEMO_DOCUMENTS=true
        #
        # Production (never seed demo docs):
        #   Don't set this variable, or set to "false"
        seed_demo_str = os.getenv("SEED_DEMO_DOCUMENTS", "false").lower()
        self.seed_demo_documents: bool = seed_demo_str in ("true", "1", "yes")

        # =====================================================================
        # Ingestion & Chunking Configuration (STEP 5 ENHANCEMENTS)
        # =====================================================================
        # These settings control how documents are processed during ingestion.
        # They affect text extraction quality and chunking strategy.
        #
        # CHUNKING STRATEGIES:
        # --------------------
        # - "recursive": (DEFAULT) Splits by characters with hierarchy
        #                Best for: General documents, preserves structure
        # - "sentence":  Splits on sentence boundaries
        #                Best for: Articles, news, conversational text
        # - "semantic":  Splits by meaning using embeddings (requires API calls)
        #                Best for: Technical docs, topic-heavy content
        #
        # IMPORTANT: Semantic chunking uses the embedding provider, so it will
        # consume API credits. Use "recursive" or "sentence" to minimize costs.
        self.chunking_strategy: str = os.getenv(
            "CHUNKING_STRATEGY",
            "recursive"  # Default: safe, fast, no extra API calls
        )

        # SEMANTIC_SIMILARITY_THRESHOLD: Controls when to split in semantic chunking
        #
        # HOW IT WORKS:
        # -------------
        # When using semantic chunking, consecutive sentences are compared.
        # If similarity drops BELOW this threshold, a new chunk starts.
        #
        # TUNING GUIDE:
        # - 0.3-0.5: More aggressive splitting, smaller chunks
        # - 0.5-0.7: Balanced (recommended range)
        # - 0.7-0.9: Less splitting, larger chunks
        #
        # Lower threshold = more chunks (finer-grained)
        # Higher threshold = fewer chunks (coarser-grained)
        self.semantic_similarity_threshold: float = float(os.getenv(
            "SEMANTIC_SIMILARITY_THRESHOLD",
            "0.75"  # Default: balanced splitting
        ))

        # MAX_CHUNK_SIZE: Maximum characters per chunk
        #
        # WHY THIS MATTERS:
        # -----------------
        # - Too large: Loses retrieval precision, may exceed embedding limits
        # - Too small: Loses context, creates too many chunks
        # - Sweet spot: 300-1000 characters for most use cases
        #
        # This is a HARD LIMIT - chunks will never exceed this size.
        self.max_chunk_size: int = int(os.getenv(
            "MAX_CHUNK_SIZE",
            "512"  # Default: good balance for most embedding models
        ))

        # MIN_CHUNK_SIZE: Minimum characters per chunk
        #
        # WHY THIS MATTERS:
        # -----------------
        # Prevents creation of tiny, meaningless chunks.
        # Chunks smaller than this will be merged with neighbors.
        #
        # TUNING GUIDE:
        # - 50-100: For dense, technical content
        # - 100-200: For general content (recommended)
        # - 200+: For verbose, narrative content
        self.min_chunk_size: int = int(os.getenv(
            "MIN_CHUNK_SIZE",
            "100"  # Default: prevents tiny chunks
        ))

        # CHUNK_OVERLAP: Characters of overlap between consecutive chunks
        #
        # WHY OVERLAP?
        # ------------
        # Overlap ensures context isn't lost at chunk boundaries.
        # Without overlap, a question spanning two chunks might miss context.
        #
        # TUNING GUIDE:
        # - 0: No overlap (faster, less context preservation)
        # - 50-100: Light overlap (recommended for most cases)
        # - 100-200: Heavy overlap (better context, more redundancy)
        self.chunk_overlap: int = int(os.getenv(
            "CHUNK_OVERLAP",
            "50"  # Default: moderate overlap
        ))

        # ENABLE_PDF_OCR: Whether to use OCR for scanned/image-based PDFs
        #
        # WHAT IS OCR?
        # ------------
        # OCR (Optical Character Recognition) converts images of text into
        # actual text. This is needed for:
        # - Scanned documents
        # - PDFs created from images
        # - PDFs with embedded images containing text
        #
        # REQUIREMENTS:
        # -------------
        # - pytesseract package: pip install pytesseract
        # - pdf2image package: pip install pdf2image
        # - Tesseract OCR installed on system:
        #   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
        #   - Linux: apt-get install tesseract-ocr
        #   - Mac: brew install tesseract
        # - Poppler (for pdf2image):
        #   - Windows: Download from https://github.com/oschwartz10612/poppler-windows
        #   - Linux: apt-get install poppler-utils
        #   - Mac: brew install poppler
        #
        # PERFORMANCE NOTE:
        # -----------------
        # OCR is SLOW and CPU-intensive. Only enable if you need it.
        # Default is FALSE to maintain fast ingestion for regular PDFs.
        enable_ocr_str = os.getenv("ENABLE_PDF_OCR", "false").lower()
        self.enable_pdf_ocr: bool = enable_ocr_str in ("true", "1", "yes")

        # TESSERACT_PATH: Path to Tesseract executable (optional)
        #
        # Only needed on Windows if Tesseract is not in system PATH.
        # Example: C:\Program Files\Tesseract-OCR\tesseract.exe
        self.tesseract_path: Optional[str] = os.getenv("TESSERACT_PATH")

        # =====================================================================
        # Retrieval & Reranking Configuration (STEP 6)
        # =====================================================================
        # These settings control the retrieval enhancement and reranking stage
        # that happens BETWEEN vector search and LLM generation.
        #
        # WHY RERANKING?
        # --------------
        # Vector search (embedding similarity) is fast but not always precise.
        # Reranking is a second-stage process that:
        # 1. Takes a larger candidate set from vector search
        # 2. Scores each candidate more carefully (using cross-encoders or LLMs)
        # 3. Returns only the most relevant documents
        #
        # This two-stage approach gives us:
        # - SPEED: Vector search quickly narrows down candidates
        # - PRECISION: Reranking ensures only the best documents reach the LLM
        #
        # THE RETRIEVAL FLOW WITH RERANKING:
        # -----------------------------------
        # 1. User asks: "What is RAG?"
        # 2. Vector search retrieves RETRIEVAL_TOP_K candidates (e.g., 20)
        # 3. Reranker scores all 20 candidates more carefully
        # 4. Top FINAL_TOP_K documents (e.g., 5) are sent to the LLM
        # 5. LLM generates answer based on the 5 best documents
        #
        # WITHOUT RERANKING (existing behavior preserved by default):
        # 1. User asks: "What is RAG?"
        # 2. Vector search retrieves FINAL_TOP_K candidates (e.g., 5)
        # 3. All 5 documents are sent directly to the LLM

        # RETRIEVAL_TOP_K: Number of candidates to retrieve from vector search
        #
        # WHY A LARGE NUMBER?
        # -------------------
        # When reranking is enabled, we fetch more documents initially
        # because the reranker will filter them down. This increases
        # recall (chance of finding relevant docs) before precision filtering.
        #
        # WHEN RERANKING IS DISABLED:
        # ---------------------------
        # This setting is ignored. We fetch FINAL_TOP_K directly.
        #
        # TUNING GUIDE:
        # - 10-20: Good for small document collections
        # - 20-50: Better recall for larger collections
        # - 50+: Maximum recall but slower reranking
        self.retrieval_top_k: int = int(os.getenv(
            "RETRIEVAL_TOP_K",
            "20"  # Default: fetch 20 candidates for reranking
        ))

        # FINAL_TOP_K: Number of documents to send to the LLM
        #
        # WHY LIMIT DOCUMENTS?
        # --------------------
        # 1. LLM context windows have limits
        # 2. More documents = higher cost (more tokens)
        # 3. Too many documents can confuse the LLM
        # 4. Quality > quantity for answer generation
        #
        # TUNING GUIDE:
        # - 3-5: Good for most use cases
        # - 5-10: For complex questions needing more context
        # - 10+: Rarely needed, may hurt answer quality
        self.final_top_k: int = int(os.getenv(
            "FINAL_TOP_K",
            "5"  # Default: send 5 best documents to LLM
        ))

        # ENABLE_RERANKING: Whether to use the reranking stage
        #
        # IMPORTANT - BACKWARD COMPATIBILITY:
        # -----------------------------------
        # Default is FALSE to preserve existing behavior.
        # When disabled:
        # - Vector search returns FINAL_TOP_K documents directly
        # - No additional API calls for reranking
        # - Fastest retrieval but potentially less precise
        #
        # When enabled:
        # - Vector search returns RETRIEVAL_TOP_K candidates
        # - Reranker scores and filters to FINAL_TOP_K
        # - Better precision but additional processing cost
        enable_reranking_str = os.getenv("ENABLE_RERANKING", "true").lower()
        self.enable_reranking: bool = enable_reranking_str in ("true", "1", "yes")

        # RERANKER_PROVIDER: Which reranking provider to use
        #
        # AVAILABLE PROVIDERS:
        # --------------------
        # - "simple": LLM-based reranker using OpenRouter
        #             Uses the configured LLM to score relevance
        #             No additional dependencies required
        #             Good for getting started
        #
        # FUTURE PROVIDERS (not yet implemented):
        # - "cohere": Cohere Rerank API (fast, accurate)
        # - "cross_encoder": Local cross-encoder model
        # - "bge_reranker": BGE reranker model
        self.reranker_provider: str = os.getenv(
            "RERANKER_PROVIDER",
            "simple"  # Default: LLM-based reranker
        )

        # RERANKING_MIN_SCORE: Minimum relevance score to keep a document
        #
        # WHAT IS THIS?
        # -------------
        # After reranking, each document has a relevance score (0.0 to 1.0).
        # Documents below this threshold are discarded even if they would
        # otherwise make it into the top FINAL_TOP_K.
        #
        # WHY HAVE A THRESHOLD?
        # ---------------------
        # Sometimes there are fewer than FINAL_TOP_K relevant documents.
        # Without a threshold, irrelevant documents would be included.
        # The threshold ensures only actually relevant docs reach the LLM.
        #
        # TUNING GUIDE:
        # - 0.0: No filtering, always return FINAL_TOP_K (if available)
        # - 0.3-0.5: Light filtering, keeps most candidates
        # - 0.5-0.7: Moderate filtering (recommended)
        # - 0.7+: Strict filtering, may return fewer than FINAL_TOP_K
        self.reranking_min_score: float = float(os.getenv(
            "RERANKING_MIN_SCORE",
            "0.0"  # Default: no threshold (backward compatible)
        ))

        # =====================================================================
        # Evaluation Configuration (STEP 7)
        # =====================================================================
        # These settings control the RAG evaluation layer that measures
        # retrieval quality, reranking impact, and answer quality.
        #
        # WHY EVALUATION?
        # ---------------
        # Production RAG systems need measurable quality metrics to:
        # 1. Compare different configurations (chunking, reranking, etc.)
        # 2. Detect regressions when changes are made
        # 3. Report quality to clients/stakeholders
        # 4. Optimize retrieval and generation settings
        #
        # EVALUATION COMPONENTS:
        # ----------------------
        # 1. Retrieval Metrics: Recall@K, Precision@K, MRR, Hit Rate
        # 2. Generation Metrics: Answer relevance, context grounding
        # 3. RAGAS Integration: Optional advanced metrics (if installed)
        #
        # IMPORTANT - NO PIPELINE CHANGES:
        # ---------------------------------
        # Evaluation is OBSERVATIONAL ONLY. It does NOT modify:
        # - The retrieval process
        # - The reranking logic
        # - The LLM generation
        # It simply measures the quality of outputs.

        # ENABLE_EVALUATION: Master switch for evaluation features
        #
        # When enabled:
        # - Evaluation metrics are computed after each query
        # - Results are logged and can be exported
        # - Slight latency increase due to metric computation
        #
        # When disabled (default):
        # - No evaluation overhead
        # - Same behavior as before Step 7
        enable_eval_str = os.getenv("ENABLE_EVALUATION", "false").lower()
        self.enable_evaluation: bool = enable_eval_str in ("true", "1", "yes")

        # EVALUATION_DEFAULT_K: Default K value for @K metrics
        #
        # Used for Recall@K, Precision@K, etc. when not specified.
        # This determines how many top results to consider.
        #
        # TUNING GUIDE:
        # - 3-5: Common for production (matches typical FINAL_TOP_K)
        # - 10: For broader evaluation
        # - Should typically match FINAL_TOP_K for consistency
        self.evaluation_default_k: int = int(os.getenv(
            "EVALUATION_DEFAULT_K",
            "5"  # Default: matches typical FINAL_TOP_K
        ))

        # ENABLE_RAGAS: Whether to use RAGAS library for advanced metrics
        #
        # RAGAS (Retrieval Augmented Generation Assessment) provides:
        # - Faithfulness: Is the answer grounded in context?
        # - Answer Relevancy: Does the answer address the question?
        # - Context Precision: Are retrieved docs relevant?
        # - Context Recall: Are all relevant docs retrieved?
        #
        # REQUIREMENTS:
        # - pip install ragas
        # - May require LLM calls for some metrics
        #
        # When disabled (default):
        # - Only lightweight, deterministic metrics are used
        # - No additional dependencies required
        enable_ragas_str = os.getenv("ENABLE_RAGAS", "false").lower()
        self.enable_ragas: bool = enable_ragas_str in ("true", "1", "yes")

        # EVALUATION_LOG_RESULTS: Whether to log evaluation results
        #
        # When enabled:
        # - Metrics are printed to console/logs after each evaluation
        # - Useful for debugging and monitoring
        #
        # When disabled:
        # - Metrics are computed but only returned, not logged
        eval_log_str = os.getenv("EVALUATION_LOG_RESULTS", "true").lower()
        self.evaluation_log_results: bool = eval_log_str in ("true", "1", "yes")

        # EVALUATION_STORE_HISTORY: Whether to store evaluation history
        #
        # When enabled:
        # - Evaluation results are stored in memory
        # - Can be retrieved for analysis and reporting
        # - Useful for batch evaluation and comparison
        #
        # When disabled:
        # - Each evaluation is independent
        # - Lower memory usage
        eval_history_str = os.getenv("EVALUATION_STORE_HISTORY", "false").lower()
        self.evaluation_store_history: bool = eval_history_str in ("true", "1", "yes")

    def validate(self) -> None:
        """
        Validate that required settings are configured.

        This method checks that essential settings (like API keys) are present.
        It should be called at application startup to fail fast if something
        is missing.

        WHY VALIDATE EARLY?
        -------------------
        It's better to fail immediately at startup with a clear error message
        than to fail later when we try to use an API key that isn't set.
        This is called "fail fast" and makes debugging much easier.

        Raises:
            ValueError: If required settings are missing or invalid.

        Example:
            settings = Settings()
            settings.validate()  # Raises error if OPENROUTER_API_KEY not set
        """
        # List of validation errors we find
        errors = []

        # Check if OpenRouter API key is set
        # This is required for the embedding service to work
        if not self.openrouter_api_key:
            errors.append(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Please set it to your OpenRouter API key. "
                "You can get one at https://openrouter.ai/keys"
            )

        # If we found any errors, raise them all together
        # This way the user sees all missing settings at once, not one at a time
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValueError(error_message)

    def __repr__(self) -> str:
        """
        Return a string representation for debugging.

        IMPORTANT: We hide sensitive values like API keys!
        Never log or print actual API keys - it's a security risk.

        Returns:
            A string showing the settings (with secrets masked).
        """
        # Mask the API keys - show only if they're set, not the actual values
        openrouter_key_status = "SET" if self.openrouter_api_key else "NOT SET"
        qdrant_key_status = "SET" if self.qdrant_api_key else "NOT SET"

        # Determine Qdrant mode for display
        if self.qdrant_url:
            qdrant_mode = "cloud"
        elif self.qdrant_host:
            qdrant_mode = f"server ({self.qdrant_host}:{self.qdrant_port})"
        else:
            qdrant_mode = "in-memory"

        return (
            f"Settings(\n"
            f"  # OpenRouter Configuration\n"
            f"  openrouter_api_key={openrouter_key_status},\n"
            f"  openrouter_base_url={self.openrouter_base_url},\n"
            f"  \n"
            f"  # Embedding Configuration\n"
            f"  embedding_model={self.embedding_model},\n"
            f"  embedding_provider={self.embedding_provider},\n"
            f"  \n"
            f"  # LLM Configuration (STEP 4)\n"
            f"  llm_provider={self.llm_provider},\n"
            f"  llm_model={self.llm_model},\n"
            f"  \n"
            f"  # Vector Store Configuration (STEP 3)\n"
            f"  vector_store_provider={self.vector_store_provider},\n"
            f"  vector_dimension={self.vector_dimension},\n"
            f"  qdrant_collection_name={self.qdrant_collection_name},\n"
            f"  qdrant_mode={qdrant_mode},\n"
            f"  qdrant_api_key={qdrant_key_status},\n"
            f"  \n"
            f"  # Ingestion & Chunking Configuration (STEP 5)\n"
            f"  chunking_strategy={self.chunking_strategy},\n"
            f"  semantic_similarity_threshold={self.semantic_similarity_threshold},\n"
            f"  max_chunk_size={self.max_chunk_size},\n"
            f"  min_chunk_size={self.min_chunk_size},\n"
            f"  chunk_overlap={self.chunk_overlap},\n"
            f"  enable_pdf_ocr={self.enable_pdf_ocr},\n"
            f"  \n"
            f"  # Retrieval & Reranking Configuration (STEP 6)\n"
            f"  enable_reranking={self.enable_reranking},\n"
            f"  reranker_provider={self.reranker_provider},\n"
            f"  retrieval_top_k={self.retrieval_top_k},\n"
            f"  final_top_k={self.final_top_k},\n"
            f"  reranking_min_score={self.reranking_min_score},\n"
            f"  \n"
            f"  # Evaluation Configuration (STEP 7)\n"
            f"  enable_evaluation={self.enable_evaluation},\n"
            f"  evaluation_default_k={self.evaluation_default_k},\n"
            f"  enable_ragas={self.enable_ragas},\n"
            f"  evaluation_log_results={self.evaluation_log_results},\n"
            f"  evaluation_store_history={self.evaluation_store_history},\n"
            f"  \n"
            f"  # Development Settings\n"
            f"  seed_demo_documents={self.seed_demo_documents}\n"
            f")"
        )


# =============================================================================
# Global Settings Instance
# =============================================================================
# We create a single instance of Settings that the entire application uses.
# This is called the "Singleton" pattern - there's only one Settings object.
#
# WHY A GLOBAL INSTANCE?
# ----------------------
# 1. We only need to read environment variables once at startup
# 2. All parts of the application can share the same settings
# 3. It's simple and easy to use: just import and access
#
# USAGE:
# ------
#     from src.core.config import settings
#
#     api_key = settings.openrouter_api_key
#     model = settings.embedding_model

settings = Settings()
