"""
Semantic Text Splitter (Full Implementation)
=============================================

WHAT IS SEMANTIC CHUNKING?
--------------------------
Semantic chunking uses AI/embeddings to find natural topic boundaries.
Instead of splitting on characters or sentences, it detects where
the meaning of the text changes.

HOW IT WORKS:
-------------
1. Split text into sentences (base units)
2. Embed each sentence using the shared embedding provider
3. Compare embeddings of consecutive sentences
4. When similarity drops significantly → topic change → chunk boundary
5. Group sentences into chunks respecting min/max size limits

Example:
    Sentence 1: "Machine learning uses data." → embedding [0.1, 0.2, ...]
    Sentence 2: "It learns patterns."         → embedding [0.15, 0.22, ...] (similar)
    Sentence 3: "Python is a language."       → embedding [0.8, -0.3, ...]  (different!)
                                                          ↑
                                                    Chunk boundary here!

WHY IS THIS BETTER?
-------------------
- Respects topic boundaries (chunks contain coherent ideas)
- Creates more meaningful chunks for retrieval
- Better embedding quality (each chunk is semantically focused)
- Improved retrieval precision (chunks match queries better)

TRADE-OFFS:
-----------
- Slower than character/sentence splitting (requires embedding API calls)
- Uses embedding API credits
- May create uneven chunk sizes

CONFIGURATION:
--------------
- CHUNKING_STRATEGY=semantic (enable semantic chunking)
- SEMANTIC_SIMILARITY_THRESHOLD=0.75 (when to split)
- MAX_CHUNK_SIZE=512 (hard limit)
- MIN_CHUNK_SIZE=100 (minimum viable chunk)

USAGE:
------
    from src.core.providers import get_embedding_provider
    from src.ingestion.chunking import SemanticSplitter

    embedder = get_embedding_provider()
    splitter = SemanticSplitter(embedding_provider=embedder)
    chunks = splitter.split(text)
"""

import math
import re
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

from src.ingestion.chunking.base import Chunker, Chunk
from src.ingestion.text_utils import split_into_sentences


# =============================================================================
# CONSTANTS
# =============================================================================

# Sentence endings for boundary detection
SENTENCE_ENDINGS = {'.', '!', '?', '。', '！', '？'}

# Minimum sentences before considering a split
MIN_SENTENCES_PER_CHUNK = 2

# Maximum sentences to batch for embedding (reduces API calls)
EMBEDDING_BATCH_SIZE = 50


@dataclass
class SentenceGroup:
    """
    Represents a group of consecutive sentences.

    Attributes:
        sentences: List of sentence strings
        start_idx: Index of first sentence in original list
        end_idx: Index of last sentence (exclusive)
        combined_text: All sentences joined together
        char_count: Total character count
    """
    sentences: List[str]
    start_idx: int
    end_idx: int
    combined_text: str
    char_count: int


class SemanticSplitter(Chunker):
    """
    Semantic text splitter using embeddings to find topic boundaries.

    This chunker uses an embedding provider to detect semantic boundaries
    in text and create chunks that represent coherent ideas/topics.

    Features:
    - Embedding-based topic boundary detection
    - Configurable similarity threshold
    - Respects min/max chunk size limits
    - Handles edge cases (short text, tables, lists)
    - Efficient batched embedding calls
    - Graceful fallback if embedding fails

    Requirements:
    - An embedding provider (from src.core.providers)
    - CHUNKING_STRATEGY=semantic in environment

    Example:
        from src.core.providers import get_embedding_provider

        embedder = get_embedding_provider()
        splitter = SemanticSplitter(
            embedding_provider=embedder,
            similarity_threshold=0.75
        )
        chunks = splitter.split(long_document_text)
    """

    def __init__(
        self,
        embedding_provider=None,
        similarity_threshold: Optional[float] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        chunk_overlap: int = 0
    ):
        """
        Initialize the semantic splitter.

        Args:
            embedding_provider: The embedding provider to use for sentence
                               embeddings. If None, will try to get from
                               shared providers.

            similarity_threshold: Threshold for detecting topic changes.
                                 When cosine similarity between consecutive
                                 sentences drops below this, a new chunk starts.
                                 If None, reads from SEMANTIC_SIMILARITY_THRESHOLD.
                                 Range: 0.0-1.0, recommended: 0.5-0.8

            min_chunk_size: Minimum characters per chunk.
                           Chunks smaller than this will be merged.
                           If None, reads from MIN_CHUNK_SIZE.

            max_chunk_size: Maximum characters per chunk.
                           Hard limit - chunks will never exceed this.
                           If None, reads from MAX_CHUNK_SIZE.

            chunk_overlap: Characters of overlap between chunks.
                          For semantic splitting, this adds sentences
                          from the previous chunk to the next.

        Example:
            # Using defaults from config
            splitter = SemanticSplitter(embedding_provider=embedder)

            # Custom settings
            splitter = SemanticSplitter(
                embedding_provider=embedder,
                similarity_threshold=0.6,  # More aggressive splitting
                max_chunk_size=800,
                min_chunk_size=150
            )
        """
        # Import settings for defaults
        from src.core.config import settings

        # Set embedding provider
        self.embedding_provider = embedding_provider
        if self.embedding_provider is None:
            # Try to get from shared providers
            try:
                from src.core.providers import get_embedding_provider
                self.embedding_provider = get_embedding_provider()
            except Exception:
                pass

        # Set parameters from config or arguments
        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
        else:
            self.similarity_threshold = settings.semantic_similarity_threshold

        if min_chunk_size is not None:
            self.min_chunk_size = min_chunk_size
        else:
            self.min_chunk_size = settings.min_chunk_size

        if max_chunk_size is not None:
            self.max_chunk_size = max_chunk_size
        else:
            self.max_chunk_size = settings.max_chunk_size

        self.chunk_overlap = chunk_overlap

        # Validate parameters
        if self.similarity_threshold < 0 or self.similarity_threshold > 1:
            print(f"[SemanticSplitter] Warning: Invalid threshold {self.similarity_threshold}, using 0.75")
            self.similarity_threshold = 0.75

        if self.min_chunk_size >= self.max_chunk_size:
            print(f"[SemanticSplitter] Warning: min_chunk_size >= max_chunk_size, adjusting")
            self.min_chunk_size = self.max_chunk_size // 4

        # Cache for sentence embeddings (avoid re-embedding)
        self._embedding_cache: Dict[str, List[float]] = {}

        # Log configuration
        if self.embedding_provider:
            print(f"[SemanticSplitter] Initialized with threshold={self.similarity_threshold}")
            print(f"[SemanticSplitter] Chunk size range: {self.min_chunk_size}-{self.max_chunk_size}")
        else:
            print("[SemanticSplitter] WARNING: No embedding provider available")
            print("[SemanticSplitter] Will fall back to sentence-based splitting")

    def split(self, text: str) -> List[str]:
        """
        Split text into semantically coherent chunks.

        This method:
        1. Splits text into sentences
        2. Embeds sentences (batched for efficiency)
        3. Finds semantic boundaries using similarity
        4. Groups sentences into chunks
        5. Enforces min/max size limits

        Args:
            text: The text to split into chunks.

        Returns:
            List of text chunks, each representing a coherent topic.

        Example:
            chunks = splitter.split(document_text)
            for i, chunk in enumerate(chunks):
                print(f"Chunk {i}: {len(chunk)} chars")
        """
        # Handle empty/whitespace text
        if not text or not text.strip():
            return []

        # Handle short text (no splitting needed)
        text = text.strip()
        if len(text) <= self.max_chunk_size:
            return [text] if len(text) >= self.min_chunk_size else [text]

        # Check if embedding provider is available
        if self.embedding_provider is None:
            print("[SemanticSplitter] No embedding provider, using fallback")
            return self._fallback_split(text)

        try:
            return self._semantic_split(text)
        except Exception as e:
            print(f"[SemanticSplitter] Semantic splitting failed: {e}")
            print("[SemanticSplitter] Falling back to sentence splitting")
            return self._fallback_split(text)

    def _semantic_split(self, text: str) -> List[str]:
        """
        Perform semantic splitting using embeddings.

        Args:
            text: Text to split.

        Returns:
            List of semantically coherent chunks.
        """
        # Step 1: Split into sentences
        sentences = self._split_into_sentences(text)

        if len(sentences) <= MIN_SENTENCES_PER_CHUNK:
            # Too few sentences to split semantically
            return [text]

        print(f"[SemanticSplitter] Split into {len(sentences)} sentences")

        # Step 2: Get embeddings for all sentences (batched)
        embeddings = self._get_sentence_embeddings(sentences)

        if embeddings is None or len(embeddings) != len(sentences):
            print("[SemanticSplitter] Embedding failed, using fallback")
            return self._fallback_split(text)

        # Step 3: Find semantic boundaries
        boundaries = self._find_semantic_boundaries(sentences, embeddings)

        print(f"[SemanticSplitter] Found {len(boundaries)} semantic boundaries")

        # Step 4: Create chunks from boundaries
        chunks = self._create_chunks_from_boundaries(sentences, boundaries)

        # Step 5: Enforce size limits
        chunks = self._enforce_size_limits(chunks)

        print(f"[SemanticSplitter] Created {len(chunks)} final chunks")

        return chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Uses the robust sentence splitter from text_utils,
        with additional handling for special cases.

        Args:
            text: Text to split.

        Returns:
            List of sentences.
        """
        # Use the shared sentence splitter
        sentences = split_into_sentences(text)

        # Filter out very short "sentences" (likely noise)
        sentences = [s for s in sentences if len(s.strip()) > 10]

        return sentences

    def _get_sentence_embeddings(
        self,
        sentences: List[str]
    ) -> Optional[List[List[float]]]:
        """
        Get embeddings for all sentences, using batching and caching.

        Args:
            sentences: List of sentences to embed.

        Returns:
            List of embedding vectors, or None if failed.
        """
        # Check cache first
        uncached_sentences = []
        uncached_indices = []

        for i, sentence in enumerate(sentences):
            if sentence not in self._embedding_cache:
                uncached_sentences.append(sentence)
                uncached_indices.append(i)

        # Embed uncached sentences in batches
        if uncached_sentences:
            print(f"[SemanticSplitter] Embedding {len(uncached_sentences)} sentences...")

            try:
                # Batch embedding
                for batch_start in range(0, len(uncached_sentences), EMBEDDING_BATCH_SIZE):
                    batch_end = min(batch_start + EMBEDDING_BATCH_SIZE, len(uncached_sentences))
                    batch = uncached_sentences[batch_start:batch_end]

                    batch_embeddings = self.embedding_provider.embed_texts(batch)

                    # Cache the results
                    for j, embedding in enumerate(batch_embeddings):
                        sentence = batch[j]
                        self._embedding_cache[sentence] = embedding

            except Exception as e:
                print(f"[SemanticSplitter] Embedding error: {e}")
                return None

        # Retrieve all embeddings (from cache)
        embeddings = []
        for sentence in sentences:
            if sentence in self._embedding_cache:
                embeddings.append(self._embedding_cache[sentence])
            else:
                # Should not happen, but handle gracefully
                print(f"[SemanticSplitter] Warning: Missing embedding for sentence")
                return None

        return embeddings

    def _find_semantic_boundaries(
        self,
        sentences: List[str],
        embeddings: List[List[float]]
    ) -> List[int]:
        """
        Find semantic boundary indices where topic changes.

        A boundary is detected when the cosine similarity between
        consecutive sentence embeddings drops below the threshold.

        Args:
            sentences: List of sentences.
            embeddings: List of corresponding embeddings.

        Returns:
            List of boundary indices (sentence index where new chunk starts).
        """
        boundaries = [0]  # First chunk always starts at 0

        for i in range(1, len(embeddings)):
            # Calculate similarity with previous sentence
            similarity = self._cosine_similarity(embeddings[i - 1], embeddings[i])

            # If similarity is below threshold, this is a topic boundary
            if similarity < self.similarity_threshold:
                boundaries.append(i)

        return boundaries

    def _create_chunks_from_boundaries(
        self,
        sentences: List[str],
        boundaries: List[int]
    ) -> List[str]:
        """
        Create text chunks from sentence boundaries.

        Args:
            sentences: List of sentences.
            boundaries: List of boundary indices.

        Returns:
            List of chunk strings.
        """
        chunks = []

        for i, start_idx in enumerate(boundaries):
            # Determine end index
            if i + 1 < len(boundaries):
                end_idx = boundaries[i + 1]
            else:
                end_idx = len(sentences)

            # Get sentences for this chunk
            chunk_sentences = sentences[start_idx:end_idx]

            # Join sentences
            chunk_text = ' '.join(chunk_sentences)

            if chunk_text.strip():
                chunks.append(chunk_text.strip())

        return chunks

    def _enforce_size_limits(self, chunks: List[str]) -> List[str]:
        """
        Enforce min/max chunk size limits.

        - Chunks exceeding max_chunk_size are split
        - Chunks below min_chunk_size are merged with neighbors

        Args:
            chunks: List of initial chunks.

        Returns:
            List of size-compliant chunks.
        """
        if not chunks:
            return chunks

        result = []

        for chunk in chunks:
            # Split oversized chunks
            if len(chunk) > self.max_chunk_size:
                sub_chunks = self._split_oversized_chunk(chunk)
                result.extend(sub_chunks)
            else:
                result.append(chunk)

        # Merge undersized chunks
        result = self._merge_undersized_chunks(result)

        return result

    def _split_oversized_chunk(self, chunk: str) -> List[str]:
        """
        Split a chunk that exceeds max_chunk_size.

        Uses sentence boundaries when possible.

        Args:
            chunk: Oversized chunk text.

        Returns:
            List of smaller chunks.
        """
        if len(chunk) <= self.max_chunk_size:
            return [chunk]

        # Try splitting on sentences first
        sentences = self._split_into_sentences(chunk)

        if len(sentences) <= 1:
            # Can't split on sentences, use character split
            return self._character_split(chunk)

        # Group sentences until max size
        sub_chunks = []
        current_chunk = ""

        for sentence in sentences:
            potential = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential) <= self.max_chunk_size:
                current_chunk = potential
            else:
                if current_chunk:
                    sub_chunks.append(current_chunk.strip())
                # Check if single sentence is too long
                if len(sentence) > self.max_chunk_size:
                    sub_chunks.extend(self._character_split(sentence))
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            sub_chunks.append(current_chunk.strip())

        return sub_chunks

    def _character_split(self, text: str) -> List[str]:
        """
        Split text by character count (last resort).

        Tries to split at word boundaries.

        Args:
            text: Text to split.

        Returns:
            List of chunks.
        """
        chunks = []
        words = text.split()
        current_chunk = ""

        for word in words:
            potential = current_chunk + " " + word if current_chunk else word

            if len(potential) <= self.max_chunk_size:
                current_chunk = potential
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = word

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _merge_undersized_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge chunks that are below min_chunk_size with neighbors.

        Args:
            chunks: List of chunks.

        Returns:
            List with small chunks merged.
        """
        if len(chunks) <= 1:
            return chunks

        result = []
        i = 0

        while i < len(chunks):
            current = chunks[i]

            # If current chunk is too small and not the last chunk
            if len(current) < self.min_chunk_size and i + 1 < len(chunks):
                # Merge with next chunk
                merged = current + " " + chunks[i + 1]

                # Check if merged is within limits
                if len(merged) <= self.max_chunk_size:
                    chunks[i + 1] = merged
                    i += 1
                    continue

            result.append(current)
            i += 1

        # Second pass: merge trailing small chunks with previous
        if len(result) > 1 and len(result[-1]) < self.min_chunk_size:
            merged = result[-2] + " " + result[-1]
            if len(merged) <= self.max_chunk_size:
                result = result[:-2] + [merged]

        return result

    def _fallback_split(self, text: str) -> List[str]:
        """
        Fallback splitting method when semantic splitting fails.

        Uses sentence-based splitting without embeddings.

        Args:
            text: Text to split.

        Returns:
            List of chunks.
        """
        if not text or len(text) <= self.max_chunk_size:
            return [text.strip()] if text and text.strip() else []

        # Simple sentence-based splitting
        sentences = self._split_into_sentences(text)

        if not sentences:
            return self._character_split(text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            potential = current_chunk + " " + sentence if current_chunk else sentence

            if len(potential) <= self.max_chunk_size:
                current_chunk = potential
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if len(sentence) > self.max_chunk_size:
                    chunks.extend(self._character_split(sentence))
                    current_chunk = ""
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())

        return self._merge_undersized_chunks(chunks)

    @staticmethod
    def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Cosine similarity measures the angle between two vectors:
        - 1.0 = identical direction (very similar)
        - 0.0 = perpendicular (unrelated)
        - -1.0 = opposite direction (opposite meaning)

        Args:
            vec1: First embedding vector.
            vec2: Second embedding vector.

        Returns:
            Cosine similarity score between -1 and 1.
        """
        if not vec1 or not vec2:
            return 0.0

        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def clear_cache(self) -> None:
        """
        Clear the embedding cache.

        Call this between documents to free memory.
        The cache is per-splitter instance.

        Example:
            splitter.split(document1)
            splitter.clear_cache()  # Free memory
            splitter.split(document2)
        """
        self._embedding_cache.clear()
        print("[SemanticSplitter] Embedding cache cleared")
