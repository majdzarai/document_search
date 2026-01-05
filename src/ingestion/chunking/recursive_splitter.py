"""
Recursive Character Text Splitter
===================================

WHAT IS A RECURSIVE SPLITTER?
-----------------------------
A recursive splitter tries to split text in a "smart" way by:
1. First trying to split on paragraph breaks (\n\n)
2. If chunks are still too large, split on newlines (\n)
3. If still too large, split on sentences (. ! ?)
4. If still too large, split on spaces
5. Finally, split on characters if needed

This hierarchy preserves meaning better than just splitting every N characters.

WHY IS THIS APPROACH BETTER?
----------------------------
Consider this text:
    "Machine learning is important.

     It helps solve complex problems."

Dumb splitting (every 20 chars):
    ["Machine learning is ", "important.\n\nIt helps", " solve complex probl", "ems."]

    Problem: Splits happen mid-word and mid-sentence!

Recursive splitting:
    ["Machine learning is important.", "It helps solve complex problems."]

    Better: Respects paragraph boundaries!

HOW IT WORKS:
-------------
1. Try to split on the first separator (paragraph break)
2. For each resulting piece:
   - If it's small enough → keep it
   - If it's too large → recursively split with next separator
3. Apply overlap between chunks

PARAMETERS:
-----------
- chunk_size: Target size for each chunk (default: 500 chars)
- chunk_overlap: Characters shared between chunks (default: 50)
- separators: List of separators to try, in order of preference
"""

from typing import List, Optional

from src.ingestion.chunking.base import Chunker


class RecursiveCharacterSplitter(Chunker):
    """
    Splits text recursively using a hierarchy of separators.

    This is the most commonly used splitter in RAG systems.
    It tries to keep chunks semantically coherent by splitting
    on natural boundaries (paragraphs, sentences, words).

    Example:
        splitter = RecursiveCharacterSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split(long_document)

    Attributes:
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        separators: List of separators to try, in order of preference.
    """

    # Default separators to try, from most preferred to least
    # Each separator represents a natural breaking point in text
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraph breaks (most preferred)
        "\n",    # Line breaks
        ". ",    # Sentence endings
        "? ",    # Question endings
        "! ",    # Exclamation endings
        "; ",    # Semicolon breaks
        ", ",    # Comma breaks
        " ",     # Word boundaries
        ""       # Character by character (last resort)
    ]

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the recursive splitter.

        Args:
            chunk_size: Target size for each chunk in characters.
                       Default is 500, which is good for most use cases.

            chunk_overlap: Number of characters to overlap between chunks.
                          Default is 50 (10% of chunk_size).
                          Set to 0 for no overlap.

            separators: Custom list of separators to use.
                       If None, uses DEFAULT_SEPARATORS.

        Example:
            # Default settings
            splitter = RecursiveCharacterSplitter()

            # Custom settings
            splitter = RecursiveCharacterSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

        # Validate settings
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        print(f"[RecursiveSplitter] Initialized with chunk_size={chunk_size}, overlap={chunk_overlap}")

    def split(self, text: str) -> List[str]:
        """
        Split text into chunks using recursive splitting.

        This method implements the recursive splitting algorithm:
        1. Try separators in order of preference
        2. Split text on the first separator that works
        3. Recursively split any chunks that are too large
        4. Apply overlap between final chunks

        Args:
            text: The text to split.

        Returns:
            A list of text chunks, each approximately chunk_size characters.
        """
        # Handle empty or very short text
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        print(f"[RecursiveSplitter] Splitting text of {len(text)} characters")

        # Start the recursive splitting process
        chunks = self._split_recursive(text, self.separators)

        # Apply overlap between chunks
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks)

        print(f"[RecursiveSplitter] Created {len(chunks)} chunks")

        return chunks

    def _split_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Internal recursive splitting method.

        Args:
            text: Text to split.
            separators: Remaining separators to try.

        Returns:
            List of chunks.
        """
        # Base case: text is small enough
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        # No more separators to try - force split by characters
        if not separators:
            return self._force_split(text)

        # Get the current separator to try
        separator = separators[0]
        remaining_separators = separators[1:]

        # Special case: empty separator means split by character
        if separator == "":
            return self._force_split(text)

        # Try to split on this separator
        splits = text.split(separator)

        # If the separator wasn't found, try the next one
        if len(splits) == 1:
            return self._split_recursive(text, remaining_separators)

        # Process each split
        result = []
        current_chunk = ""

        for i, split in enumerate(splits):
            # Add separator back (except for first piece)
            piece = split if i == 0 else separator + split

            # Check if adding this piece would exceed chunk_size
            if len(current_chunk) + len(piece) <= self.chunk_size:
                current_chunk += piece
            else:
                # Save current chunk if not empty
                if current_chunk.strip():
                    # If current chunk is still too large, recursively split it
                    if len(current_chunk) > self.chunk_size:
                        result.extend(self._split_recursive(
                            current_chunk, remaining_separators
                        ))
                    else:
                        result.append(current_chunk.strip())

                # Start new chunk with this piece
                current_chunk = piece

        # Don't forget the last chunk
        if current_chunk.strip():
            if len(current_chunk) > self.chunk_size:
                result.extend(self._split_recursive(
                    current_chunk, remaining_separators
                ))
            else:
                result.append(current_chunk.strip())

        return result

    def _force_split(self, text: str) -> List[str]:
        """
        Force split text by characters when no separator works.

        This is the fallback when text has no natural breaking points.
        We split exactly at chunk_size characters.

        Args:
            text: Text to split.

        Returns:
            List of chunks.
        """
        result = []
        for i in range(0, len(text), self.chunk_size):
            chunk = text[i:i + self.chunk_size].strip()
            if chunk:
                result.append(chunk)
        return result

    def _apply_overlap(self, chunks: List[str]) -> List[str]:
        """
        Apply overlap between consecutive chunks.

        This method adds the end of each chunk to the beginning
        of the next chunk, creating overlap.

        Args:
            chunks: List of chunks without overlap.

        Returns:
            List of chunks with overlap applied.
        """
        if len(chunks) <= 1:
            return chunks

        result = [chunks[0]]  # First chunk stays as is

        for i in range(1, len(chunks)):
            # Get overlap from the end of previous chunk
            prev_chunk = chunks[i - 1]
            overlap_text = prev_chunk[-self.chunk_overlap:] if len(prev_chunk) >= self.chunk_overlap else prev_chunk

            # Prepend overlap to current chunk
            current_chunk = overlap_text + " " + chunks[i]
            result.append(current_chunk.strip())

        return result
