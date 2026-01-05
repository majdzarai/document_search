"""
LLM Provider Base Class
========================

WHAT IS AN LLM?
---------------
LLM stands for Large Language Model. These are AI models trained on massive
amounts of text data that can understand and generate human-like text.

Examples of LLMs include:
- GPT-4, GPT-3.5 (by OpenAI)
- Claude (by Anthropic)
- Llama (by Meta)
- Mistral, Mixtral (by Mistral AI)

LLMs can perform many tasks like:
- Answering questions
- Writing code
- Summarizing documents
- Translating languages
- And much more!

WHY DO WE NEED AN LLM IN RAG?
-----------------------------
In a RAG (Retrieval-Augmented Generation) system, the LLM is responsible for
the "Generation" part. Here's the complete RAG flow:

1. USER ASKS: "What is machine learning?"
2. EMBED: Convert question to a vector (list of numbers)
3. RETRIEVE: Find similar documents in the vector database
4. GENERATE: Use the LLM to create an answer based on retrieved documents
5. RETURN: Send the answer back to the user

Without the LLM, we would just return raw documents. The LLM reads the
documents and synthesizes a coherent, helpful answer.

WHAT IS THIS MODULE?
--------------------
This module defines an ABSTRACT BASE CLASS for LLM providers.

Just like with embeddings, we want to support multiple LLM providers:
- OpenRouter (access to many models via one API)
- OpenAI (direct GPT access)
- Anthropic (Claude models)
- Ollama (local models)
- vLLM (self-hosted models)

By defining an abstract interface, we can swap providers without changing
the rest of our code. The RAG pipeline just calls `generate()` and doesn't
care which specific provider is being used.

DESIGN PATTERN: Strategy Pattern
---------------------------------
This follows the Strategy Pattern:
- The interface (LLMProvider) stays the same
- The implementation (OpenRouterLLM, OpenAILLM, etc.) can be swapped
- The client code (RAGPipeline) works with any implementation
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class LLMProvider(ABC):
    """
    Abstract base class defining the interface for LLM providers.

    This is an abstract class - you cannot create an instance of it directly.
    Instead, you create subclasses that implement the abstract methods.

    WHY USE AN ABSTRACT CLASS?
    --------------------------
    1. STANDARDIZATION: All LLM providers have the same interface
    2. SUBSTITUTION: We can swap providers without changing other code
    3. DOCUMENTATION: The methods here document what every provider must do
    4. SAFETY: Python raises an error if you forget to implement a method

    HOW TO CREATE A NEW PROVIDER:
    -----------------------------
    1. Create a new class that inherits from LLMProvider
    2. Implement all methods marked with @abstractmethod
    3. Register it in the factory (see factory.py)

    Example:
        class MyLLMProvider(LLMProvider):
            def generate(self, prompt: str, **kwargs) -> str:
                # Your implementation here
                return "Generated response..."

            def generate_with_context(
                self,
                question: str,
                context_documents: List[Dict[str, Any]],
                **kwargs
            ) -> str:
                # Your implementation here
                return "Generated response with context..."

            def get_model_name(self) -> str:
                return "my-model-name"
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM given a prompt.

        This is the core method of any LLM provider. It takes a prompt
        (the text input) and returns the model's generated response.

        WHAT HAPPENS INSIDE (conceptually):
        -----------------------------------
        1. The prompt is sent to the LLM API
        2. The model processes the text and generates a response
        3. The response is returned as a string

        Args:
            prompt: The input text/prompt to send to the LLM.
                   This is the main content the model will respond to.
                   Example: "Explain what machine learning is."

            system_prompt: Optional. A system-level instruction that sets
                          the behavior or role of the AI assistant.
                          Example: "You are a helpful assistant that explains
                                   technical concepts simply."

            **kwargs: Additional arguments specific to the provider.
                     Common options include:
                     - temperature: Controls randomness (0.0 = deterministic)
                     - max_tokens: Maximum length of the response

        Returns:
            The generated text response from the LLM.

        Example:
            provider = OpenRouterLLMProvider()
            response = provider.generate(
                prompt="What is Python?",
                system_prompt="You are a helpful programming tutor."
            )
            print(response)  # "Python is a versatile programming language..."
        """
        pass

    @abstractmethod
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
        --------------------------------
        This method is specifically designed for RAG pipelines. It:
        1. Takes the user's question
        2. Takes the retrieved documents (from vector search)
        3. Builds a prompt that includes both
        4. Generates an answer grounded in the documents

        WHY A SEPARATE METHOD?
        ----------------------
        We could just use generate() and build the prompt ourselves.
        But having a dedicated method:
        1. Makes the RAG integration cleaner
        2. Allows providers to optimize prompt construction
        3. Documents the expected document format

        Args:
            question: The user's original question.
                     Example: "What are the benefits of RAG?"

            context_documents: List of retrieved documents to use as context.
                              Each document is a dictionary with:
                              - "content": The text content
                              - "metadata": Additional info (source, page, etc.)
                              - "score": Similarity score (optional)

                              Example:
                              [
                                  {
                                      "content": "RAG improves accuracy...",
                                      "metadata": {"source": "rag_guide.pdf"},
                                      "score": 0.89
                                  },
                                  ...
                              ]

            system_prompt: Optional. Instructions for how the LLM should behave.
                          If None, a default RAG-focused prompt is used.

            **kwargs: Additional arguments (temperature, max_tokens, etc.)

        Returns:
            The generated answer, grounded in the provided documents.

        HOW THE PROMPT IS BUILT (typical structure):
        --------------------------------------------
        [System Prompt]
        You are a helpful assistant. Answer questions based on the
        provided context. If the context doesn't contain the answer,
        say so.

        [Context]
        Document 1: RAG stands for Retrieval-Augmented Generation...
        Document 2: Vector databases store embeddings...

        [Question]
        What is RAG and how does it work?

        Example:
            provider = OpenRouterLLMProvider()
            docs = [
                {"content": "RAG combines retrieval with generation...",
                 "metadata": {"source": "rag_intro.pdf"}}
            ]
            answer = provider.generate_with_context(
                question="What is RAG?",
                context_documents=docs
            )
            print(answer)  # "Based on the documents, RAG is..."
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the LLM model being used.

        This is useful for:
        1. LOGGING: Know which model generated which responses
        2. DEBUGGING: Verify the correct model is being used
        3. COST TRACKING: Different models have different costs

        Returns:
            A string with the model name/identifier.
            Example: "openai/gpt-4" or "anthropic/claude-3-opus"

        Example:
            provider = OpenRouterLLMProvider()
            print(f"Using model: {provider.get_model_name()}")
        """
        pass
