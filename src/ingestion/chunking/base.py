"""
Chunking Base Class
====================

WHAT IS CHUNKING?
-----------------
Chunking is the process of splitting large documents into smaller pieces.
This is essential for RAG systems because:

1. EMBEDDING LIMITS: Embedding models have maximum input lengths
   - text-embedding-3-small: ~8192 tokens
   - Most models work best with shorter texts

2. RETRIEVAL PRECISION: Smaller chunks = more precise retrieval
   - A 10-page document as one chunk → vague matches
   - Same document as 50 chunks → precise paragraph-level matches

3. CONTEXT WINDOW: LLMs have limited context windows
   - GPT-3.5: ~16K tokens
   - GPT-4: ~128K tokens
   - Smaller chunks allow more relevant context

WHY OVERLAP?
------------
Chunks often have "overlap" - shared text between consecutive chunks.

Without overlap:
    Chunk 1: "Machine learning is a subset of AI."
    Chunk 2: "It enables computers to learn from data."

    Problem: If someone asks about "AI and data", neither chunk
    alone contains both concepts.

With overlap:
    Chunk 1: "Machine learning is a subset of AI. It enables"
    Chunk 2: "subset of AI. It enables computers to learn from data."

    Now both chunks have context about the transition.

CHUNK SIZE GUIDELINES:
----------------------
- Too small (< 100 chars): Loses context, too many chunks
- Too large (> 2000 chars): Less precise, may exceed limits
- Sweet spot: 300-1000 characters (or 50-200 tokens)

WHAT IS THIS MODULE?
--------------------
This module defines the abstract interface for all chunking strategies.
Different strategies have different approaches:
- RecursiveSplitter: Split by characters with hierarchy
- SentenceSplitter: Split on sentence boundaries
- SemanticSplitter: Split by semantic meaning (advanced)
"""

from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass
class Chunk:
    """
    Represents a single chunk of text.

    Attributes:
        text: The text content of this chunk.
        index: The position of this chunk (0, 1, 2, ...).
        start_char: Character offset where this chunk starts in original text.
        end_char: Character offset where this chunk ends in original text.

    Example:
        chunk = Chunk(
            text="This is a chunk of text.",
            index=0,
            start_char=0,
            end_char=24
        )
    """
    text: str
    index: int
    start_char: int
    end_char: int


class Chunker(ABC):
    """
    Abstract base class for text chunking strategies.

    All chunkers must inherit from this class and implement
    the `split` method.

    HOW TO CREATE A NEW CHUNKER:
    ----------------------------
    1. Create a new class that inherits from Chunker
    2. Implement the `split` method
    3. Optionally implement `split_with_metadata` for detailed info

    Example:
        class MyChunker(Chunker):
            def split(self, text: str) -> List[str]:
                # Your splitting logic
                return ["chunk1", "chunk2", ...]
    """

    @abstractmethod
    def split(self, text: str) -> List[str]:
        """
        Split text into chunks.

        This is the main method that each chunker must implement.
        It takes a piece of text and returns a list of smaller pieces.

        Args:
            text: The text to split into chunks.

        Returns:
            A list of text chunks.
            Each chunk is a string.
            The chunks should cover the entire original text.

        Example:
            chunker = RecursiveCharacterSplitter(chunk_size=100)
            chunks = chunker.split("A very long document...")
            # Returns: ["A very long ", "long docume", "document..."]
        """
        pass

    def split_with_metadata(self, text: str) -> List[Chunk]:
        """
        Split text into chunks with position metadata.

        This method provides more detail than `split()`,
        including the position of each chunk in the original text.

        Args:
            text: The text to split.

        Returns:
            A list of Chunk objects with text and position info.

        Note:
            Default implementation calls split() and estimates positions.
            Override for more accurate position tracking.
        """
        chunks = self.split(text)
        result = []
        current_pos = 0

        for i, chunk_text in enumerate(chunks):
            # Find where this chunk starts in the original text
            # This is an approximation - may not be exact with overlap
            start = text.find(chunk_text, current_pos)
            if start == -1:
                start = current_pos  # Fallback

            result.append(Chunk(
                text=chunk_text,
                index=i,
                start_char=start,
                end_char=start + len(chunk_text)
            ))

            current_pos = start + 1  # Move forward to find next

        return result
