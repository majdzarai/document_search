"""
Sentence-Based Text Splitter
=============================

WHAT IS SENTENCE SPLITTING?
---------------------------
Sentence splitting divides text into chunks at sentence boundaries.
This ensures that each chunk contains complete sentences, which:
- Preserves semantic meaning
- Makes chunks more readable
- Improves retrieval quality

WHY SPLIT ON SENTENCES?
-----------------------
Sentences are natural units of meaning in text.
Splitting mid-sentence can break meaning:

    Bad:  "Machine learning is a subset of artificial"
    Good: "Machine learning is a subset of artificial intelligence."

CHALLENGES:
-----------
Detecting sentence boundaries is tricky because:
- Periods aren't always sentence ends: "Dr. Smith", "U.S.A.", "3.14"
- Some sentences end with ? or !
- Abbreviations: "etc.", "e.g.", "i.e."
- Decimal numbers: "The value was 2.5 million."

This implementation uses a simple rule-based approach.
For more accuracy, consider:
- spaCy (NLP library with sentence detection)
- NLTK (Natural Language Toolkit)

HOW IT WORKS:
-------------
1. Find potential sentence endings (. ? !)
2. Check if followed by whitespace and capital letter
3. Group sentences until chunk_size is reached
4. Apply overlap between chunks
"""

import re
from typing import List

from src.ingestion.chunking.base import Chunker


class SentenceSplitter(Chunker):
    """
    Splits text into chunks at sentence boundaries.

    This splitter ensures that chunks contain complete sentences,
    improving semantic coherence and retrieval quality.

    Example:
        splitter = SentenceSplitter(chunk_size=500)
        chunks = splitter.split(document_text)
        # Each chunk contains complete sentences

    Attributes:
        chunk_size: Target maximum size for each chunk.
        chunk_overlap: Number of overlapping characters between chunks.
        min_chunk_size: Minimum size before starting a new chunk.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_chunk_size: int = 100
    ):
        """
        Initialize the sentence splitter.

        Args:
            chunk_size: Target maximum size for each chunk in characters.
                       Chunks may be slightly larger to avoid splitting sentences.

            chunk_overlap: Number of characters to overlap between chunks.
                          This adds context from the previous chunk.

            min_chunk_size: Minimum chunk size before starting a new chunk.
                           Prevents very small chunks.

        Example:
            splitter = SentenceSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

        print(f"[SentenceSplitter] Initialized with chunk_size={chunk_size}")

    def split(self, text: str) -> List[str]:
        """
        Split text into chunks at sentence boundaries.

        This method:
        1. Splits text into individual sentences
        2. Groups sentences into chunks up to chunk_size
        3. Applies overlap between chunks

        Args:
            text: The text to split.

        Returns:
            List of text chunks, each containing complete sentences.
        """
        # Handle empty or very short text
        if not text or len(text) <= self.chunk_size:
            return [text.strip()] if text and text.strip() else []

        print(f"[SentenceSplitter] Splitting text of {len(text)} characters")

        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)
        print(f"[SentenceSplitter] Found {len(sentences)} sentences")

        if not sentences:
            return [text.strip()] if text.strip() else []

        # Step 2: Group sentences into chunks
        chunks = self._group_sentences(sentences)

        # Step 3: Apply overlap
        if self.chunk_overlap > 0 and len(chunks) > 1:
            chunks = self._apply_overlap(chunks, sentences)

        print(f"[SentenceSplitter] Created {len(chunks)} chunks")

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into individual sentences.

        Uses regex to find sentence boundaries.
        Handles common abbreviations and edge cases.

        Args:
            text: Text to split into sentences.

        Returns:
            List of sentences.
        """
        # Common abbreviations that shouldn't end a sentence
        # These are replaced temporarily to avoid false splits
        abbreviations = [
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.",
            "Inc.", "Ltd.", "Corp.", "Co.",
            "Jr.", "Sr.",
            "vs.", "etc.", "e.g.", "i.e.",
            "U.S.", "U.K.", "U.N.",
            "a.m.", "p.m.",
            "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."
        ]

        # Replace abbreviations with placeholders
        modified_text = text
        for i, abbr in enumerate(abbreviations):
            placeholder = f"<<ABBR{i}>>"
            modified_text = modified_text.replace(abbr, placeholder)

        # Split on sentence endings followed by whitespace
        # Pattern: period/question/exclamation + space + capital letter or end
        pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        raw_sentences = re.split(pattern, modified_text)

        # Restore abbreviations
        sentences = []
        for sentence in raw_sentences:
            for i, abbr in enumerate(abbreviations):
                placeholder = f"<<ABBR{i}>>"
                sentence = sentence.replace(placeholder, abbr)
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)

        return sentences

    def _group_sentences(self, sentences: List[str]) -> List[str]:
        """
        Group sentences into chunks up to chunk_size.

        Adds sentences to a chunk until adding another would
        exceed chunk_size, then starts a new chunk.

        Args:
            sentences: List of sentences.

        Returns:
            List of chunks, each containing one or more sentences.
        """
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If this sentence alone is larger than chunk_size,
            # we have to include it anyway (can't split a sentence)
            if len(sentence) > self.chunk_size:
                # Save current chunk if not empty
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Add the long sentence as its own chunk
                chunks.append(sentence)
                current_chunk = ""
                continue

            # Check if adding this sentence exceeds chunk_size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential_chunk) <= self.chunk_size:
                # Add sentence to current chunk
                current_chunk = potential_chunk
            else:
                # Current chunk is full, start a new one
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence

        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _apply_overlap(self, chunks: List[str], sentences: List[str]) -> List[str]:
        """
        Apply overlap between chunks using complete sentences.

        Instead of character-based overlap, we add the last sentence(s)
        from the previous chunk to provide context.

        Args:
            chunks: List of chunks.
            sentences: Original list of sentences.

        Returns:
            List of chunks with overlap.
        """
        if len(chunks) <= 1:
            return chunks

        result = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            current_chunk = chunks[i]

            # Find overlap text from end of previous chunk
            # Take last N characters where N = chunk_overlap
            if len(prev_chunk) >= self.chunk_overlap:
                # Get the last part of previous chunk
                overlap_text = prev_chunk[-self.chunk_overlap:]
                # Try to start at a sentence/word boundary
                space_idx = overlap_text.find(" ")
                if space_idx > 0:
                    overlap_text = overlap_text[space_idx:].strip()
            else:
                overlap_text = prev_chunk

            # Add overlap to current chunk
            overlapped = f"...{overlap_text} {current_chunk}"
            result.append(overlapped.strip())

        return result
