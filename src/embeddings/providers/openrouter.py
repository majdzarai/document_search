"""
OpenRouter Embedding Provider
==============================

WHAT IS OPENROUTER?
-------------------
OpenRouter (https://openrouter.ai) is a unified API that provides access to
many different AI models from various providers. Think of it as a "middleman"
that lets you access OpenAI, Anthropic, and other models through one API.

KEY BENEFITS OF OPENROUTER:
---------------------------
1. ONE API, MANY MODELS: Access GPT-4, Claude, Llama, and more with one API key
2. OPENAI-COMPATIBLE: Uses the same API format as OpenAI, so existing code works
3. COST MANAGEMENT: Set spending limits and track usage across all models
4. FALLBACKS: Automatically switch to backup models if one fails

OPENAI-COMPATIBLE API:
----------------------
OpenRouter uses the same API structure as OpenAI. This means:
- Same request format (JSON with "model" and "input" fields)
- Same response format (JSON with "data" containing embeddings)
- Just different base URL (openrouter.ai instead of api.openai.com)

This is great because:
- Code written for OpenAI works with minimal changes
- You can switch between OpenAI and OpenRouter easily
- Documentation for OpenAI's API applies here too

HOW EMBEDDINGS WORK WITH OPENROUTER:
------------------------------------
1. We send a POST request to https://openrouter.ai/api/v1/embeddings
2. The request includes:
   - The model name (e.g., "openai/text-embedding-3-small")
   - The text(s) to embed
3. OpenRouter forwards the request to the actual provider (e.g., OpenAI)
4. We get back the embedding vectors
"""

import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from src.embeddings.base import EmbeddingProvider
from src.core.config import settings


class OpenRouterEmbeddingProvider(EmbeddingProvider):
    """
    Embedding provider that uses OpenRouter's OpenAI-compatible API.

    This class implements the EmbeddingProvider interface for OpenRouter.
    It sends HTTP requests to OpenRouter's embeddings endpoint and returns
    the resulting embedding vectors.

    HOW TO USE THIS CLASS:
    ----------------------
    1. Set the OPENROUTER_API_KEY environment variable
    2. Optionally set EMBEDDING_MODEL (defaults to text-embedding-3-small)
    3. Create an instance and call embed_text() or embed_texts()

    Example:
        # First, set environment variable:
        # export OPENROUTER_API_KEY=your-key-here

        from src.embeddings.providers.openrouter import OpenRouterEmbeddingProvider

        provider = OpenRouterEmbeddingProvider()
        embedding = provider.embed_text("Hello, world!")
        print(f"Got embedding with {len(embedding)} dimensions")

    WHAT HAPPENS UNDER THE HOOD:
    ----------------------------
    1. The provider reads API key and settings from environment variables
    2. When you call embed_text(), it makes an HTTP POST request to OpenRouter
    3. OpenRouter forwards the request to the actual embedding model
    4. The embedding vector is returned as a list of floats

    SUPPORTED MODELS:
    -----------------
    Through OpenRouter, you can use various embedding models:
    - "openai/text-embedding-3-small" (1536 dimensions, recommended)
    - "openai/text-embedding-3-large" (3072 dimensions, higher quality)
    - "openai/text-embedding-ada-002" (1536 dimensions, older model)

    ERROR HANDLING:
    ---------------
    - If the API key is missing, raises ValueError
    - If the API call fails, raises RuntimeError with details
    - If the response format is unexpected, raises RuntimeError

    Attributes:
        api_key: The OpenRouter API key for authentication.
        base_url: The base URL for the OpenRouter API.
        model: The embedding model to use (e.g., "openai/text-embedding-3-small").
        _dimension: Cached dimension value (set after first API call).
    """

    # =========================================================================
    # Model Dimension Mapping
    # =========================================================================
    # Different embedding models produce vectors of different sizes.
    # We need to know this to:
    # 1. Configure vector databases correctly
    # 2. Verify we're getting the expected output
    #
    # This dictionary maps model names to their output dimensions.
    # Add new models here as you use them.

    MODEL_DIMENSIONS: Dict[str, int] = {
        # OpenAI models (accessed through OpenRouter)
        "openai/text-embedding-3-small": 1536,
        "openai/text-embedding-3-large": 3072,
        "openai/text-embedding-ada-002": 1536,

        # Add more models as needed
        # "cohere/embed-english-v3.0": 1024,
        # "cohere/embed-multilingual-v3.0": 1024,
    }

    # Default dimension if the model isn't in our mapping
    # Most common embedding models use 1536 dimensions
    DEFAULT_DIMENSION: int = 1536

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        """
        Initialize the OpenRouter embedding provider.

        This constructor sets up everything needed to make API calls:
        - API key for authentication
        - Model name to use
        - Base URL for the API

        HOW CONFIGURATION WORKS:
        ------------------------
        1. You can pass values directly to this constructor
        2. If not passed, values are read from environment variables
        3. Environment variables are read through the settings module

        This two-level approach is useful because:
        - Default case: Just use environment variables (no code changes needed)
        - Testing: Pass mock values directly
        - Multiple instances: Use different models in the same app

        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY
                     environment variable.
            model: The embedding model to use. If None, reads from EMBEDDING_MODEL
                   environment variable. Default: "openai/text-embedding-3-small"
            base_url: The base URL for the API. If None, reads from
                      OPENROUTER_BASE_URL environment variable.
                      Default: "https://openrouter.ai/api/v1"

        Raises:
            ValueError: If no API key is provided and OPENROUTER_API_KEY is not set.

        Example:
            # Using environment variables (recommended)
            provider = OpenRouterEmbeddingProvider()

            # Or with explicit values (useful for testing)
            provider = OpenRouterEmbeddingProvider(
                api_key="sk-...",
                model="openai/text-embedding-3-small"
            )
        """
        # =====================================================================
        # Load API Key
        # =====================================================================
        # The API key authenticates our requests to OpenRouter.
        # Without it, the API will reject our requests.

        # Use provided api_key, or fall back to environment variable
        self.api_key: str = api_key if api_key is not None else (
            settings.openrouter_api_key or ""
        )

        # Validate that we have an API key
        # We check this early so we get a clear error message
        if not self.api_key:
            raise ValueError(
                "OpenRouter API key is required. "
                "Either pass it to the constructor or set the "
                "OPENROUTER_API_KEY environment variable. "
                "Get your API key at: https://openrouter.ai/keys"
            )

        # =====================================================================
        # Load Model Name
        # =====================================================================
        # The model determines which embedding model we use.
        # Different models have different quality, cost, and dimensions.

        self.model: str = model if model is not None else settings.embedding_model

        # =====================================================================
        # Load Base URL
        # =====================================================================
        # The base URL is where we send our API requests.
        # OpenRouter's URL is different from OpenAI's.

        self.base_url: str = base_url if base_url is not None else (
            settings.openrouter_base_url
        )

        # =====================================================================
        # Internal State
        # =====================================================================
        # We cache the dimension after the first successful API call.
        # This avoids repeatedly computing it.

        self._dimension: Optional[int] = self.MODEL_DIMENSIONS.get(self.model)

        # Print initialization info (useful for debugging)
        # In production, this would be proper logging
        print(f"[OpenRouterEmbeddingProvider] Initialized with model: {self.model}")
        print(f"[OpenRouterEmbeddingProvider] Expected dimension: {self.get_dimension()}")

    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single piece of text into an embedding vector.

        This is the main method for getting embeddings. It:
        1. Sends the text to OpenRouter's API
        2. Receives the embedding vector
        3. Returns it as a list of floats

        HOW IT WORKS STEP BY STEP:
        --------------------------
        1. We build a JSON request with the text and model name
        2. We send an HTTP POST request to OpenRouter's embeddings endpoint
        3. OpenRouter forwards the request to the actual model (e.g., OpenAI)
        4. We receive a JSON response with the embedding
        5. We extract the embedding list and return it

        Args:
            text: The text to convert into an embedding.
                  Can be a word, sentence, paragraph, or document.
                  Maximum length depends on the model (usually 8192 tokens).

        Returns:
            A list of floating-point numbers representing the text.
            The length equals the model's dimension (e.g., 1536 for small model).

        Raises:
            RuntimeError: If the API call fails or returns an error.
            ValueError: If the response format is unexpected.

        Example:
            provider = OpenRouterEmbeddingProvider()
            embedding = provider.embed_text("What is machine learning?")
            print(f"Embedding has {len(embedding)} dimensions")
            print(f"First 5 values: {embedding[:5]}")
        """
        # Use the batch method with a single text
        # This keeps the code DRY (Don't Repeat Yourself)
        embeddings = self.embed_texts([text])

        # Return the first (and only) embedding
        return embeddings[0]

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple pieces of text into embedding vectors.

        This method is more efficient than calling embed_text() multiple times
        because it sends all texts in a single API request.

        THE API REQUEST:
        ----------------
        We send a POST request to: {base_url}/embeddings

        Request body (JSON):
        {
            "model": "openai/text-embedding-3-small",
            "input": ["text1", "text2", "text3"]
        }

        Response body (JSON):
        {
            "data": [
                {"embedding": [0.1, 0.2, ...], "index": 0},
                {"embedding": [0.3, 0.4, ...], "index": 1},
                {"embedding": [0.5, 0.6, ...], "index": 2}
            ],
            "model": "text-embedding-3-small",
            "usage": {"prompt_tokens": 10, "total_tokens": 10}
        }

        Args:
            texts: A list of text strings to embed.
                   Example: ["Hello", "World", "How are you?"]

        Returns:
            A list of embeddings, one for each input text.
            The order matches the input order.
            Example: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]

        Raises:
            RuntimeError: If the API call fails or returns an error.
            ValueError: If texts is empty or response format is unexpected.

        Example:
            provider = OpenRouterEmbeddingProvider()
            texts = ["cat", "dog", "car"]
            embeddings = provider.embed_texts(texts)

            # Each text gets its own embedding
            for text, embedding in zip(texts, embeddings):
                print(f"'{text}' -> {len(embedding)} dimensions")
        """
        # =====================================================================
        # Input Validation
        # =====================================================================
        # Check that we have at least one text to embed

        if not texts:
            raise ValueError("texts list cannot be empty")

        # Filter out empty strings (the API doesn't like them)
        # Keep track of original indices so we can put results back in order
        non_empty_texts = []
        non_empty_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                non_empty_texts.append(text)
                non_empty_indices.append(i)

        # If all texts were empty, return empty embeddings
        if not non_empty_texts:
            raise ValueError("All texts are empty. Please provide non-empty texts.")

        # =====================================================================
        # Build the API Request
        # =====================================================================
        # The request follows OpenAI's embeddings API format

        # The endpoint URL for embeddings
        url = f"{self.base_url}/embeddings"

        # The request body as a dictionary
        # This will be converted to JSON
        request_body: Dict[str, Any] = {
            "model": self.model,   # Which embedding model to use
            "input": non_empty_texts  # The texts to embed (can be a list)
        }

        # Convert the dictionary to JSON bytes
        # The API expects the body to be JSON
        json_data = json.dumps(request_body).encode("utf-8")

        # =====================================================================
        # Build HTTP Headers
        # =====================================================================
        # Headers tell the API about our request format and authentication

        headers = {
            # Authentication: The API key proves we're allowed to use the API
            # Format: "Bearer <api_key>" (this is standard OAuth2 format)
            "Authorization": f"Bearer {self.api_key}",

            # Content-Type: Tells the API we're sending JSON
            "Content-Type": "application/json",

            # HTTP-Referer: Required by OpenRouter to identify your app
            # In production, use your actual domain
            "HTTP-Referer": "https://github.com/your-app",

            # X-Title: Optional, helps OpenRouter track usage
            "X-Title": "RAG Service"
        }

        # =====================================================================
        # Make the API Request
        # =====================================================================
        # We use urllib.request from Python's standard library
        # This avoids requiring additional dependencies like 'requests' or 'httpx'

        # Create the request object
        request = urllib.request.Request(
            url=url,
            data=json_data,
            headers=headers,
            method="POST"
        )

        try:
            # Send the request and get the response
            # urlopen() opens the URL and returns a response object
            with urllib.request.urlopen(request, timeout=60) as response:
                # Read the response body
                response_body = response.read()

                # Decode bytes to string and parse as JSON
                response_data = json.loads(response_body.decode("utf-8"))

        except urllib.error.HTTPError as e:
            # HTTPError means the server returned an error status code
            # (like 400 Bad Request, 401 Unauthorized, 500 Server Error)

            # Try to read the error message from the response body
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass

            raise RuntimeError(
                f"OpenRouter API request failed with status {e.code}. "
                f"URL: {url}. "
                f"Error: {error_body}"
            )

        except urllib.error.URLError as e:
            # URLError means we couldn't reach the server at all
            # (network issues, DNS problems, etc.)
            raise RuntimeError(
                f"Failed to connect to OpenRouter API. "
                f"URL: {url}. "
                f"Error: {str(e.reason)}"
            )

        except json.JSONDecodeError as e:
            # JSONDecodeError means the response wasn't valid JSON
            raise RuntimeError(
                f"OpenRouter API returned invalid JSON. "
                f"Error: {str(e)}"
            )

        # =====================================================================
        # Parse the Response
        # =====================================================================
        # The response should contain a "data" field with the embeddings

        # Check that we got the expected format
        if "data" not in response_data:
            raise RuntimeError(
                f"Unexpected response format from OpenRouter API. "
                f"Expected 'data' field but got: {list(response_data.keys())}"
            )

        # Extract embeddings from the response
        # The "data" field contains a list of objects with "embedding" and "index"
        data_list = response_data["data"]

        # Sort by index to ensure correct order
        # (the API might return them in a different order)
        data_list.sort(key=lambda x: x.get("index", 0))

        # Extract just the embedding vectors
        embeddings: List[List[float]] = []
        for item in data_list:
            if "embedding" not in item:
                raise RuntimeError(
                    f"Unexpected response format: missing 'embedding' field in data item"
                )
            embeddings.append(item["embedding"])

        # =====================================================================
        # Update Cached Dimension
        # =====================================================================
        # If this is our first successful call, cache the dimension

        if embeddings and self._dimension is None:
            self._dimension = len(embeddings[0])
            print(f"[OpenRouterEmbeddingProvider] Detected dimension: {self._dimension}")

        # =====================================================================
        # Log Success (for learning/debugging)
        # =====================================================================
        # Print some info about what we got

        print(f"[OpenRouterEmbeddingProvider] Successfully embedded {len(texts)} text(s)")
        if embeddings:
            print(f"[OpenRouterEmbeddingProvider] Embedding vector length: {len(embeddings[0])}")

        return embeddings

    def get_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.

        The dimension is the number of values in each embedding vector.
        This is determined by the model being used.

        IMPORTANT FOR VECTOR DATABASES:
        --------------------------------
        When creating a collection in a vector database (like Qdrant, Pinecone),
        you must specify the dimension. All embeddings in that collection must
        have the same dimension. Use this method to get the correct value.

        Returns:
            The dimension (number of values) in embedding vectors.
            For example: 1536 for text-embedding-3-small

        Example:
            provider = OpenRouterEmbeddingProvider()
            dim = provider.get_dimension()
            print(f"Create vector DB collection with dimension={dim}")
        """
        # Return cached dimension if we have it
        if self._dimension is not None:
            return self._dimension

        # Otherwise, look up the dimension for this model
        # If the model isn't in our mapping, use the default
        return self.MODEL_DIMENSIONS.get(self.model, self.DEFAULT_DIMENSION)

    def get_model_name(self) -> str:
        """
        Get the name of the embedding model being used.

        This is useful for logging, debugging, and documentation.
        It helps you verify which model is generating the embeddings.

        Returns:
            The model name string (e.g., "openai/text-embedding-3-small")

        Example:
            provider = OpenRouterEmbeddingProvider()
            print(f"Using embedding model: {provider.get_model_name()}")
        """
        return self.model
