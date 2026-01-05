"""
Embedding Provider Base Class
==============================

WHAT ARE EMBEDDINGS?
--------------------
Embeddings are numerical representations of text. They convert human-readable
text (like "Hello, world!") into a list of numbers (like [0.1, -0.3, 0.8, ...]).

Think of embeddings as a way to capture the "meaning" of text in numbers.
Texts with similar meanings will have similar numbers. For example:
- "dog" and "puppy" → similar embeddings (close numbers)
- "dog" and "quantum physics" → different embeddings (far apart numbers)

The list of numbers is called a "vector" (a fancy word for a list of numbers).
Each embedding typically has hundreds or thousands of numbers, called "dimensions".
For example:
- OpenAI's text-embedding-3-small produces 1536 dimensions
- This means each piece of text becomes a list of 1536 numbers

WHY ARE EMBEDDINGS IMPORTANT FOR RAG?
--------------------------------------
In a RAG (Retrieval-Augmented Generation) system, embeddings are essential for
finding relevant documents. Here's how it works:

1. INDEXING PHASE (done once, when documents are ingested):
   - We take each document and split it into chunks
   - We convert each chunk into an embedding (a vector of numbers)
   - We store these embeddings in a vector database

2. QUERY PHASE (done every time a user asks a question):
   - User asks: "How do I train a neural network?"
   - We convert this question into an embedding
   - We search the vector database for similar embeddings
   - Documents with similar embeddings are likely relevant to the question

3. WHY DOES THIS WORK?
   - The embedding model is trained to put similar meanings close together
   - "Training neural networks" and "deep learning tutorial" have similar embeddings
   - This enables "semantic search" - searching by meaning, not just keywords

WHAT IS THIS MODULE?
--------------------
This module defines an ABSTRACT BASE CLASS for embedding providers.

WHAT IS AN ABSTRACT BASE CLASS?
-------------------------------
An abstract base class (ABC) is like a template or contract. It says:
"Any class that wants to be an embedding provider MUST have these methods."

This is useful because:
1. We can swap providers without changing the rest of our code
2. We know exactly what methods every provider will have
3. Python will error if we forget to implement a required method

For example:
- OpenRouterEmbeddingProvider implements this interface for OpenRouter
- OpenAIEmbeddingProvider would implement it for OpenAI directly
- CohereEmbeddingProvider would implement it for Cohere

All of them have the same methods, so we can use them interchangeably!
"""

from abc import ABC, abstractmethod
from typing import List


class EmbeddingProvider(ABC):
    """
    Abstract base class defining the interface for embedding providers.

    WHAT IS THIS CLASS?
    -------------------
    This is an abstract class - you cannot create an instance of it directly.
    Instead, you create subclasses that implement all the abstract methods.

    Think of it as a "contract" that says: "To be an embedding provider,
    you must implement these methods exactly as specified."

    WHY USE AN ABSTRACT CLASS?
    --------------------------
    1. STANDARDIZATION: All embedding providers have the same interface
    2. SUBSTITUTION: We can swap providers without changing other code
    3. DOCUMENTATION: The methods here document what every provider must do
    4. SAFETY: Python raises an error if you forget to implement a method

    DESIGN PATTERN: This follows the "Strategy Pattern"
    - The interface stays the same (EmbeddingProvider)
    - The implementation changes (OpenRouter, OpenAI, Cohere, etc.)
    - The rest of the code doesn't care which provider is used

    HOW TO CREATE A NEW PROVIDER:
    -----------------------------
    1. Create a new class that inherits from EmbeddingProvider
    2. Implement all methods marked with @abstractmethod
    3. Register it in the factory (see factory.py)

    Example:
        class MyEmbeddingProvider(EmbeddingProvider):
            def embed_text(self, text: str) -> List[float]:
                # Your implementation here
                return [0.1, 0.2, 0.3, ...]

            def embed_texts(self, texts: List[str]) -> List[List[float]]:
                # Your implementation here
                return [[0.1, 0.2, ...], [0.3, 0.4, ...]]

            def get_dimension(self) -> int:
                return 1536  # Your model's dimension

            def get_model_name(self) -> str:
                return "my-model-name"
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single piece of text into an embedding vector.

        This is the core method of any embedding provider. It takes text
        and returns a list of numbers (floats) representing that text.

        WHAT HAPPENS INSIDE (conceptually):
        -----------------------------------
        1. The text is tokenized (split into meaningful pieces)
        2. The tokens are processed by a neural network
        3. The network outputs a list of numbers (the embedding)
        4. These numbers capture the "meaning" of the text

        Args:
            text: The input text to embed.
                  Can be a word, sentence, paragraph, or document.
                  Example: "What is machine learning?"

        Returns:
            A list of floating-point numbers representing the text.
            The length of this list depends on the model (its "dimension").
            Example: [0.123, -0.456, 0.789, ...] (could be 1536 numbers)

        IMPORTANT NOTES:
        ----------------
        - The same text always produces the same embedding (deterministic)
        - Different texts can have similar embeddings if meanings are similar
        - Embeddings can be compared using "cosine similarity" or "dot product"

        Example:
            provider = OpenRouterEmbeddingProvider()
            embedding = provider.embed_text("Hello, world!")
            print(len(embedding))  # e.g., 1536
            print(embedding[:3])   # e.g., [0.123, -0.456, 0.789]
        """
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple pieces of text into embedding vectors (batch processing).

        This method processes multiple texts at once, which is more efficient
        than calling embed_text() multiple times.

        WHY BATCH PROCESSING?
        ---------------------
        1. SPEED: One API call with 100 texts is faster than 100 API calls
        2. COST: Many APIs charge per request, not per text
        3. EFFICIENCY: Network overhead is reduced

        WHEN TO USE THIS METHOD:
        ------------------------
        - When indexing documents (you have many chunks to embed)
        - When comparing multiple queries
        - Any time you have more than one text to embed

        Args:
            texts: A list of text strings to embed.
                   Example: ["Hello", "World", "How are you?"]

        Returns:
            A list of embeddings, one for each input text.
            Each embedding is a list of floats.
            The order matches the input order.
            Example: [
                [0.1, 0.2, ...],  # embedding for "Hello"
                [0.3, 0.4, ...],  # embedding for "World"
                [0.5, 0.6, ...],  # embedding for "How are you?"
            ]

        IMPORTANT:
        ----------
        - The output list has the same length as the input list
        - The i-th output corresponds to the i-th input

        Example:
            provider = OpenRouterEmbeddingProvider()
            texts = ["cat", "dog", "car"]
            embeddings = provider.embed_texts(texts)
            print(len(embeddings))     # 3 (one per input)
            print(len(embeddings[0]))  # e.g., 1536 (embedding dimension)
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension (size) of embedding vectors produced by this provider.

        WHAT IS EMBEDDING DIMENSION?
        ----------------------------
        The dimension is how many numbers are in each embedding vector.
        Different models produce different dimensions:

        - OpenAI text-embedding-3-small: 1536 dimensions
        - OpenAI text-embedding-3-large: 3072 dimensions
        - OpenAI text-embedding-ada-002: 1536 dimensions
        - Cohere embed-english-v3.0: 1024 dimensions
        - BGE-large: 1024 dimensions

        WHY DOES DIMENSION MATTER?
        --------------------------
        1. VECTOR DATABASE: You must tell the database the dimension when creating
           a collection. All embeddings in a collection must have the same dimension.

        2. CONSISTENCY: If you embed documents with model A (1536 dim) and queries
           with model B (1024 dim), the search won't work!

        3. STORAGE: Higher dimensions use more storage space.

        4. QUALITY: Generally, higher dimensions can capture more nuance,
           but there are diminishing returns.

        Returns:
            An integer representing the number of dimensions.
            Example: 1536

        Example:
            provider = OpenRouterEmbeddingProvider()
            dimension = provider.get_dimension()
            print(f"This model produces {dimension}-dimensional vectors")
        """
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the embedding model being used.

        This is useful for:
        1. LOGGING: Know which model generated which embeddings
        2. DEBUGGING: Verify the correct model is being used
        3. DOCUMENTATION: Record which model was used for a dataset

        WHY TRACK MODEL NAMES?
        ----------------------
        - Different models produce incompatible embeddings
        - If you change models, you need to re-embed all documents
        - For reproducibility, you need to know which model was used

        Returns:
            A string with the model name/identifier.
            Example: "openai/text-embedding-3-small"

        Example:
            provider = OpenRouterEmbeddingProvider()
            print(f"Using model: {provider.get_model_name()}")
        """
        pass
