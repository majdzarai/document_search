"""
OpenRouter LLM Provider
========================

WHAT IS OPENROUTER?
-------------------
OpenRouter (https://openrouter.ai) is a unified API that provides access to
many different AI models from various providers. Think of it as a "middleman"
that lets you access GPT-4, Claude, Llama, Mistral, and more through one API.

KEY BENEFITS OF OPENROUTER:
---------------------------
1. ONE API, MANY MODELS: Access GPT-4, Claude, Llama, and more with one API key
2. OPENAI-COMPATIBLE: Uses the same API format as OpenAI, so existing code works
3. COST MANAGEMENT: Set spending limits and track usage across all models
4. FALLBACKS: Automatically switch to backup models if one fails
5. MODEL VARIETY: Try different models easily without changing providers

OPENAI-COMPATIBLE API (Chat Completions):
-----------------------------------------
OpenRouter uses the same API structure as OpenAI for chat completions.
This means:
- Same request format (JSON with "model" and "messages" fields)
- Same response format (JSON with "choices" containing the response)
- Just different base URL (openrouter.ai instead of api.openai.com)

HOW LLM GENERATION WORKS WITH OPENROUTER:
-----------------------------------------
1. We send a POST request to https://openrouter.ai/api/v1/chat/completions
2. The request includes:
   - The model name (e.g., "openai/gpt-4" or "anthropic/claude-3-sonnet")
   - The messages (conversation history including system prompt and user message)
3. OpenRouter forwards the request to the actual provider (e.g., OpenAI)
4. We get back the generated text response

WHY USE OPENROUTER FOR RAG?
---------------------------
In our RAG pipeline, we need to:
1. Take retrieved documents as context
2. Build a prompt that includes the context and question
3. Generate a coherent answer using an LLM

OpenRouter gives us flexibility to experiment with different models
to find the best balance of quality, speed, and cost for our use case.
"""

import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from src.llm.base import LLMProvider
from src.core.config import settings


class OpenRouterLLMProvider(LLMProvider):
    """
    LLM provider that uses OpenRouter's OpenAI-compatible Chat Completions API.

    This class implements the LLMProvider interface for OpenRouter.
    It sends HTTP requests to OpenRouter's chat completions endpoint
    and returns the generated text responses.

    HOW TO USE THIS CLASS:
    ----------------------
    1. Set the OPENROUTER_API_KEY environment variable
    2. Optionally set LLM_MODEL (defaults to openai/gpt-3.5-turbo)
    3. Create an instance and call generate() or generate_with_context()

    Example:
        # First, set environment variable:
        # export OPENROUTER_API_KEY=your-key-here

        from src.llm.providers.openrouter import OpenRouterLLMProvider

        provider = OpenRouterLLMProvider()
        response = provider.generate("Explain what RAG is in simple terms.")
        print(response)

    FOR RAG USAGE:
    --------------
        docs = [
            {"content": "RAG combines retrieval with generation...",
             "metadata": {"source": "rag_intro.pdf"}}
        ]
        answer = provider.generate_with_context(
            question="What is RAG?",
            context_documents=docs
        )
        print(answer)

    SUPPORTED MODELS:
    -----------------
    Through OpenRouter, you can use various LLM models:
    - "openai/gpt-4": Most capable OpenAI model
    - "openai/gpt-4-turbo": Faster and cheaper than GPT-4
    - "openai/gpt-3.5-turbo": Fast and affordable
    - "anthropic/claude-3-opus": Most capable Claude model
    - "anthropic/claude-3-sonnet": Balanced Claude model
    - "meta-llama/llama-3-70b-instruct": Open source option
    - And many more!

    Attributes:
        api_key: The OpenRouter API key for authentication.
        base_url: The base URL for the OpenRouter API.
        model: The LLM model to use.
        default_system_prompt: Default system prompt for RAG generation.
    """

    # =========================================================================
    # Default System Prompt for RAG
    # =========================================================================
    # This prompt instructs the LLM how to behave when answering questions
    # based on retrieved documents. Keep it simple and clear.

    DEFAULT_RAG_SYSTEM_PROMPT: str = (
        "You are a helpful assistant. Answer the user's question based on the "
        "provided context. Be concise and accurate. If the context doesn't "
        "contain enough information to answer the question, say so honestly."
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None
    ) -> None:
        """
        Initialize the OpenRouter LLM provider.

        This constructor sets up everything needed to make API calls:
        - API key for authentication
        - Model name to use
        - Base URL for the API

        Args:
            api_key: OpenRouter API key. If None, reads from OPENROUTER_API_KEY
                    environment variable.
            model: The LLM model to use. If None, reads from LLM_MODEL
                  environment variable. Default: "openai/gpt-3.5-turbo"
            base_url: The base URL for the API. If None, reads from
                     OPENROUTER_BASE_URL environment variable.

        Raises:
            ValueError: If no API key is provided and OPENROUTER_API_KEY is not set.

        Example:
            # Using environment variables (recommended)
            provider = OpenRouterLLMProvider()

            # Or with explicit values (useful for testing)
            provider = OpenRouterLLMProvider(
                api_key="sk-...",
                model="anthropic/claude-3-sonnet"
            )
        """
        # =====================================================================
        # Load API Key
        # =====================================================================
        # The API key authenticates our requests to OpenRouter.

        self.api_key: str = api_key if api_key is not None else (
            settings.openrouter_api_key or ""
        )

        # Validate that we have an API key
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
        # The model determines which LLM we use for generation.

        self.model: str = model if model is not None else settings.llm_model

        # =====================================================================
        # Load Base URL
        # =====================================================================
        # The base URL is where we send our API requests.

        self.base_url: str = base_url if base_url is not None else (
            settings.openrouter_base_url
        )

        # Print initialization info (useful for debugging)
        print(f"[OpenRouterLLMProvider] Initialized with model: {self.model}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM given a prompt.

        This method sends the prompt to OpenRouter and returns the response.
        It's a simple way to get text generation without RAG context.

        THE API REQUEST:
        ----------------
        We send a POST request to: {base_url}/chat/completions

        Request body (JSON):
        {
            "model": "openai/gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Your question here"}
            ]
        }

        Response body (JSON):
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "The generated response..."
                    }
                }
            ]
        }

        Args:
            prompt: The user's input/question.
            system_prompt: Optional. Instructions for the AI's behavior.
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            The generated text response from the LLM.

        Raises:
            RuntimeError: If the API call fails or returns an error.

        Example:
            provider = OpenRouterLLMProvider()
            response = provider.generate(
                prompt="Explain machine learning in simple terms.",
                system_prompt="You are a friendly teacher."
            )
            print(response)
        """
        # =====================================================================
        # Build the messages list
        # =====================================================================
        # The Chat Completions API expects a list of messages, each with:
        # - "role": "system", "user", or "assistant"
        # - "content": The message text

        messages: List[Dict[str, str]] = []

        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })

        # Add the user's prompt
        messages.append({
            "role": "user",
            "content": prompt
        })

        # =====================================================================
        # Make the API request
        # =====================================================================

        return self._call_api(messages, **kwargs)

    def generate_with_context(
        self,
        question: str,
        context_documents: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using retrieved documents as context.

        THIS IS THE KEY METHOD FOR RAG!

        This method:
        1. Takes the user's question
        2. Takes the retrieved documents
        3. Builds a prompt that includes both
        4. Generates an answer grounded in the documents

        HOW THE PROMPT IS STRUCTURED:
        -----------------------------
        We build a simple, clear prompt with three parts:

        [System Prompt]
        You are a helpful assistant. Answer based on the context...

        [User Message with Context and Question]
        Context:
        ---
        Document 1: RAG stands for Retrieval-Augmented Generation...
        Document 2: Vector databases store embeddings...
        ---

        Question: What is RAG?

        WHY THIS STRUCTURE?
        -------------------
        1. System prompt sets the AI's behavior
        2. Context documents provide the knowledge
        3. Clear separation makes it easy for the LLM to understand
        4. The question comes last so it's fresh in the model's "attention"

        Args:
            question: The user's original question.
            context_documents: List of retrieved documents. Each should have:
                              - "content": The document text
                              - "metadata": Optional additional info
            system_prompt: Optional. Custom system prompt. If None, uses default.
            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            The generated answer, grounded in the provided documents.

        Example:
            docs = [
                {"content": "RAG improves accuracy by...",
                 "metadata": {"source": "rag_guide.pdf"}}
            ]
            answer = provider.generate_with_context(
                question="What is RAG?",
                context_documents=docs
            )
        """
        # =====================================================================
        # Build the context string from documents
        # =====================================================================
        # We format each document clearly so the LLM can reference them.

        context_parts: List[str] = []

        for i, doc in enumerate(context_documents, 1):
            # Get the document content
            content = doc.get("content", "")

            # Get optional metadata for reference
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "unknown")

            # Format: "Document 1 (from source.pdf): The content..."
            doc_text = f"Document {i} (from {source}):\n{content}"
            context_parts.append(doc_text)

        # Join all documents with blank lines between them
        context_str = "\n\n".join(context_parts)

        # =====================================================================
        # Build the user message with context and question
        # =====================================================================
        # We clearly separate context from the question.

        user_message = (
            f"Context:\n"
            f"---\n"
            f"{context_str}\n"
            f"---\n\n"
            f"Question: {question}"
        )

        # =====================================================================
        # Build the messages list
        # =====================================================================

        messages: List[Dict[str, str]] = []

        # Add system prompt (use default if not provided)
        effective_system_prompt = system_prompt or self.DEFAULT_RAG_SYSTEM_PROMPT
        messages.append({
            "role": "system",
            "content": effective_system_prompt
        })

        # Add the user message with context
        messages.append({
            "role": "user",
            "content": user_message
        })

        # =====================================================================
        # Log what we're sending (for learning/debugging)
        # =====================================================================

        print(f"\n[OpenRouterLLMProvider] Generating answer with context:")
        print(f"  - Question: {question[:50]}...")
        print(f"  - Context documents: {len(context_documents)}")
        print(f"  - Model: {self.model}")

        # =====================================================================
        # Make the API request
        # =====================================================================

        return self._call_api(messages, **kwargs)

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Make the actual API call to OpenRouter.

        This is an internal method that handles the HTTP request/response.
        Both generate() and generate_with_context() use this method.

        Args:
            messages: The list of messages to send.
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            The generated text from the LLM.

        Raises:
            RuntimeError: If the API call fails.
        """
        # =====================================================================
        # Build the API Request
        # =====================================================================

        url = f"{self.base_url}/chat/completions"

        # Build the request body
        request_body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages
        }

        # Add any additional parameters from kwargs
        # Common ones: temperature, max_tokens, top_p
        for key, value in kwargs.items():
            if value is not None:
                request_body[key] = value

        # Convert to JSON
        json_data = json.dumps(request_body).encode("utf-8")

        # =====================================================================
        # Build HTTP Headers
        # =====================================================================

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-app",
            "X-Title": "RAG Service"
        }

        # =====================================================================
        # Make the API Request
        # =====================================================================

        request = urllib.request.Request(
            url=url,
            data=json_data,
            headers=headers,
            method="POST"
        )

        try:
            # Send the request (use longer timeout for LLM generation)
            with urllib.request.urlopen(request, timeout=120) as response:
                response_body = response.read()
                response_data = json.loads(response_body.decode("utf-8"))

        except urllib.error.HTTPError as e:
            error_body = ""
            try:
                error_body = e.read().decode("utf-8")
            except Exception:
                pass

            raise RuntimeError(
                f"OpenRouter API request failed with status {e.code}. "
                f"Error: {error_body}"
            )

        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Failed to connect to OpenRouter API. "
                f"Error: {str(e.reason)}"
            )

        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"OpenRouter API returned invalid JSON. "
                f"Error: {str(e)}"
            )

        # =====================================================================
        # Parse the Response
        # =====================================================================

        # Check for expected format
        if "choices" not in response_data:
            raise RuntimeError(
                f"Unexpected response format from OpenRouter API. "
                f"Expected 'choices' field but got: {list(response_data.keys())}"
            )

        choices = response_data["choices"]
        if not choices:
            raise RuntimeError("OpenRouter API returned no choices")

        # Extract the message content
        message = choices[0].get("message", {})
        content = message.get("content", "")

        if not content:
            raise RuntimeError("OpenRouter API returned empty content")

        print(f"[OpenRouterLLMProvider] Generated response ({len(content)} chars)")

        return content

    def get_model_name(self) -> str:
        """
        Get the name of the LLM model being used.

        Returns:
            The model name string (e.g., "openai/gpt-3.5-turbo")
        """
        return self.model
