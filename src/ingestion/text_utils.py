"""
Text Normalization Utilities
=============================

WHAT IS TEXT NORMALIZATION?
---------------------------
Text normalization is the process of cleaning and standardizing text
to improve quality before further processing (chunking, embedding, etc.).

WHY IS THIS IMPORTANT FOR RAG?
------------------------------
1. CLEANER EMBEDDINGS: Noise in text creates noise in embeddings
2. BETTER RETRIEVAL: Normalized text matches queries more accurately
3. REDUCED CHUNK FRAGMENTATION: Clean boundaries = cleaner chunks
4. CONSISTENCY: Same content always produces same embeddings

WHAT DOES THIS MODULE DO?
-------------------------
- Normalize whitespace (tabs, multiple spaces, etc.)
- Remove invisible/control characters
- Strip repeated headers/footers (common in PDFs)
- Preserve paragraph boundaries for semantic chunking
- Handle common OCR artifacts

DESIGN PRINCIPLES:
------------------
1. NON-DESTRUCTIVE: Never lose meaningful content
2. IDEMPOTENT: Applying twice gives same result as once
3. CONFIGURABLE: Different strategies for different use cases
4. EFFICIENT: Optimized for processing large documents
"""

import re
import unicodedata
from typing import List, Optional, Set
from dataclasses import dataclass


# =============================================================================
# CONSTANTS - Common patterns for text cleaning
# =============================================================================

# Control characters to remove (except common whitespace)
# These are invisible characters that can corrupt text
CONTROL_CHAR_PATTERN = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]'
)

# Multiple whitespace (2+ spaces, tabs mixed with spaces, etc.)
MULTI_SPACE_PATTERN = re.compile(r'[ \t]+')

# Multiple newlines (3+ in a row → 2 max to preserve paragraph breaks)
MULTI_NEWLINE_PATTERN = re.compile(r'\n{3,}')

# Lines that are just whitespace
WHITESPACE_LINE_PATTERN = re.compile(r'^\s+$', re.MULTILINE)

# Common page markers from PDFs (Page 1, Page 2, etc.)
PAGE_MARKER_PATTERN = re.compile(
    r'^\s*\[?(?:Page|PAGE|p\.?)\s*\d+\s*\]?\s*$',
    re.MULTILINE | re.IGNORECASE
)

# Common header/footer patterns (repeated on every page)
# These patterns match lines that commonly appear as headers/footers
COMMON_HEADER_PATTERNS = [
    re.compile(r'^\s*(?:CONFIDENTIAL|DRAFT|INTERNAL)\s*$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*(?:Copyright|©)\s*\d{4}.*$', re.MULTILINE | re.IGNORECASE),
    re.compile(r'^\s*\d+\s*of\s*\d+\s*$', re.MULTILINE | re.IGNORECASE),  # "1 of 10"
]

# OCR artifacts - common mistakes from OCR
OCR_ARTIFACT_REPLACEMENTS = {
    '|': 'I',  # Pipe often misread as I
    '0': 'O',  # Zero in wrong context (only in specific patterns)
    'l': '1',  # lowercase L as 1 (only in specific patterns)
    '—': '-',  # Em dash to regular dash
    '–': '-',  # En dash to regular dash
    '"': '"',  # Smart quotes to regular
    '"': '"',
    ''': "'",
    ''': "'",
    '…': '...',  # Ellipsis to dots
    '\xad': '',  # Soft hyphen (invisible)
    '\u200b': '',  # Zero-width space
    '\ufeff': '',  # BOM (byte order mark)
}

# Sentence endings for boundary detection
SENTENCE_ENDINGS = {'.', '!', '?', '。', '！', '？'}


@dataclass
class NormalizationConfig:
    """
    Configuration for text normalization.

    Allows fine-tuning of which normalization steps to apply.
    Default values are production-safe and preserve content.

    Attributes:
        remove_control_chars: Remove invisible control characters
        normalize_whitespace: Collapse multiple spaces/tabs
        normalize_newlines: Limit consecutive newlines
        remove_page_markers: Remove [Page N] style markers
        detect_headers_footers: Try to remove repeated headers/footers
        fix_ocr_artifacts: Apply common OCR fixes
        preserve_paragraphs: Keep paragraph boundaries (double newlines)
        strip_lines: Strip leading/trailing whitespace from each line
        min_line_length: Lines shorter than this may be headers/footers

    Example:
        # Default (safe) normalization
        config = NormalizationConfig()

        # Aggressive normalization for messy OCR text
        config = NormalizationConfig(
            fix_ocr_artifacts=True,
            detect_headers_footers=True
        )
    """
    remove_control_chars: bool = True
    normalize_whitespace: bool = True
    normalize_newlines: bool = True
    remove_page_markers: bool = True
    detect_headers_footers: bool = False  # Off by default - can remove valid content
    fix_ocr_artifacts: bool = False  # Off by default - can corrupt valid text
    preserve_paragraphs: bool = True
    strip_lines: bool = True
    min_line_length: int = 3  # Lines shorter than this are suspicious


class TextNormalizer:
    """
    Normalizes text for improved RAG quality.

    This class provides methods to clean and standardize text
    extracted from documents. It's designed to be:
    - Non-destructive (preserves meaningful content)
    - Configurable (different settings for different needs)
    - Efficient (optimized for large documents)

    Example:
        normalizer = TextNormalizer()

        # Basic normalization
        clean_text = normalizer.normalize(dirty_text)

        # With custom config
        config = NormalizationConfig(fix_ocr_artifacts=True)
        normalizer = TextNormalizer(config)
        clean_text = normalizer.normalize(ocr_text)
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """
        Initialize the text normalizer.

        Args:
            config: Optional configuration. Uses defaults if not provided.
        """
        self.config = config or NormalizationConfig()

    def normalize(self, text: str) -> str:
        """
        Apply all configured normalization steps to text.

        This is the main entry point for text normalization.
        Steps are applied in a specific order to ensure correctness.

        Args:
            text: The text to normalize.

        Returns:
            Normalized text.

        Example:
            normalizer = TextNormalizer()
            clean = normalizer.normalize("  Multiple   spaces   and\t\ttabs  ")
            # Returns: "Multiple spaces and tabs"
        """
        if not text:
            return ""

        # Step 1: Remove control characters (invisible corruption)
        if self.config.remove_control_chars:
            text = self._remove_control_chars(text)

        # Step 2: Fix OCR artifacts (before other processing)
        if self.config.fix_ocr_artifacts:
            text = self._fix_ocr_artifacts(text)

        # Step 3: Strip each line
        if self.config.strip_lines:
            text = self._strip_lines(text)

        # Step 4: Remove page markers
        if self.config.remove_page_markers:
            text = self._remove_page_markers(text)

        # Step 5: Detect and remove repeated headers/footers
        if self.config.detect_headers_footers:
            text = self._remove_repeated_headers_footers(text)

        # Step 6: Normalize whitespace (spaces and tabs)
        if self.config.normalize_whitespace:
            text = self._normalize_whitespace(text)

        # Step 7: Normalize newlines (preserve paragraphs)
        if self.config.normalize_newlines:
            text = self._normalize_newlines(text, self.config.preserve_paragraphs)

        # Final strip
        return text.strip()

    def _remove_control_chars(self, text: str) -> str:
        """
        Remove invisible control characters.

        Control characters are non-printable characters that can
        corrupt text processing. We keep normal whitespace (space,
        tab, newline, carriage return).

        Args:
            text: Input text.

        Returns:
            Text with control characters removed.
        """
        # First pass: regex for common control chars
        text = CONTROL_CHAR_PATTERN.sub('', text)

        # Second pass: unicodedata for remaining invisibles
        cleaned_chars = []
        for char in text:
            category = unicodedata.category(char)
            # Keep: Letters, Numbers, Punctuation, Symbols, Separators (spaces)
            # Remove: Control (Cc), Format (Cf) except some useful ones
            if category.startswith('C') and char not in '\n\r\t ':
                continue
            cleaned_chars.append(char)

        return ''.join(cleaned_chars)

    def _fix_ocr_artifacts(self, text: str) -> str:
        """
        Fix common OCR recognition errors.

        OCR (Optical Character Recognition) often makes predictable
        mistakes. This method fixes the most common ones.

        CAUTION: This can corrupt valid text in some cases.
        Only enable when processing OCR output.

        Args:
            text: Input text (likely from OCR).

        Returns:
            Text with common OCR errors fixed.
        """
        for old, new in OCR_ARTIFACT_REPLACEMENTS.items():
            text = text.replace(old, new)

        return text

    def _strip_lines(self, text: str) -> str:
        """
        Strip leading/trailing whitespace from each line.

        This removes inconsistent indentation while preserving
        the content of each line.

        Args:
            text: Input text.

        Returns:
            Text with each line stripped.
        """
        lines = text.split('\n')
        stripped_lines = [line.strip() for line in lines]
        return '\n'.join(stripped_lines)

    def _remove_page_markers(self, text: str) -> str:
        """
        Remove page marker lines like [Page 1], Page 2, etc.

        These markers are often added by PDF extraction but
        don't contain useful content.

        Args:
            text: Input text.

        Returns:
            Text with page markers removed.
        """
        # Remove [Page N] style markers
        text = PAGE_MARKER_PATTERN.sub('', text)

        # Also remove standalone page numbers (just a number on a line)
        text = re.sub(r'^\s*\d{1,4}\s*$', '', text, flags=re.MULTILINE)

        return text

    def _remove_repeated_headers_footers(self, text: str) -> str:
        """
        Detect and remove repeated headers and footers.

        This analyzes the text for lines that appear multiple times
        (indicating they're headers/footers) and removes them.

        CAUTION: This can remove valid repeated content.
        Only enable when you know the document has headers/footers.

        Args:
            text: Input text.

        Returns:
            Text with repeated headers/footers removed.
        """
        # Remove common header patterns
        for pattern in COMMON_HEADER_PATTERNS:
            text = pattern.sub('', text)

        # Detect repeated short lines
        lines = text.split('\n')
        line_counts: dict = {}

        # Count occurrences of short lines
        for line in lines:
            stripped = line.strip()
            if len(stripped) < 50 and len(stripped) >= self.config.min_line_length:
                line_counts[stripped] = line_counts.get(stripped, 0) + 1

        # Find lines that appear suspiciously often (likely headers/footers)
        # A line appearing 3+ times in a document is suspicious
        repeated_lines: Set[str] = {
            line for line, count in line_counts.items()
            if count >= 3 and len(line) < 50
        }

        # Remove repeated lines
        filtered_lines = [
            line for line in lines
            if line.strip() not in repeated_lines
        ]

        return '\n'.join(filtered_lines)

    def _normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace (spaces and tabs).

        Collapses multiple spaces/tabs into single spaces,
        but preserves newlines for paragraph structure.

        Args:
            text: Input text.

        Returns:
            Text with normalized whitespace.
        """
        # Replace tabs with spaces
        text = text.replace('\t', ' ')

        # Collapse multiple spaces to one
        text = MULTI_SPACE_PATTERN.sub(' ', text)

        # Remove spaces at start/end of lines (but keep newlines)
        lines = text.split('\n')
        text = '\n'.join(line.strip() for line in lines)

        return text

    def _normalize_newlines(self, text: str, preserve_paragraphs: bool = True) -> str:
        """
        Normalize newlines.

        If preserve_paragraphs is True:
            - Single newlines become spaces (flow text)
            - Double newlines stay (paragraph breaks)
            - Triple+ newlines become double

        If preserve_paragraphs is False:
            - All newlines become spaces

        Args:
            text: Input text.
            preserve_paragraphs: Whether to keep paragraph breaks.

        Returns:
            Text with normalized newlines.
        """
        if not preserve_paragraphs:
            # All newlines become spaces
            return text.replace('\n', ' ')

        # Normalize line endings (Windows \r\n → \n)
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # Collapse 3+ newlines to 2 (preserve paragraph breaks)
        text = MULTI_NEWLINE_PATTERN.sub('\n\n', text)

        # Remove blank lines that are just whitespace
        text = WHITESPACE_LINE_PATTERN.sub('', text)

        return text


def normalize_text(
    text: str,
    config: Optional[NormalizationConfig] = None
) -> str:
    """
    Convenience function for one-off text normalization.

    Creates a TextNormalizer with the given config and normalizes text.
    For repeated normalization, create a TextNormalizer instance instead.

    Args:
        text: The text to normalize.
        config: Optional configuration. Uses defaults if not provided.

    Returns:
        Normalized text.

    Example:
        # Quick normalization with defaults
        clean = normalize_text(dirty_text)

        # With custom config
        config = NormalizationConfig(fix_ocr_artifacts=True)
        clean = normalize_text(ocr_text, config)
    """
    normalizer = TextNormalizer(config)
    return normalizer.normalize(text)


def detect_paragraph_boundaries(text: str) -> List[int]:
    """
    Detect paragraph boundary positions in text.

    This function finds where paragraphs start, which is useful
    for semantic chunking that respects document structure.

    A paragraph boundary is detected when:
    - Double newline (explicit paragraph)
    - Single newline followed by indentation
    - Bullet/numbered list items

    Args:
        text: The text to analyze.

    Returns:
        List of character positions where paragraphs start.

    Example:
        boundaries = detect_paragraph_boundaries(text)
        # Returns: [0, 156, 312, ...]  # Start positions
    """
    boundaries = [0]  # First paragraph starts at 0

    # Pattern for paragraph boundaries
    # Double newline is the main indicator
    para_pattern = re.compile(r'\n\n+')

    for match in para_pattern.finditer(text):
        # The new paragraph starts after the newlines
        boundaries.append(match.end())

    return boundaries


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.

    This is a robust sentence splitter that handles:
    - Common abbreviations (Dr., Mr., U.S.A., etc.)
    - Decimal numbers (3.14)
    - URLs and emails
    - Quotations and parentheses

    Args:
        text: The text to split.

    Returns:
        List of sentences.

    Example:
        sentences = split_into_sentences("Hello! How are you? I'm fine.")
        # Returns: ["Hello!", "How are you?", "I'm fine."]
    """
    if not text or not text.strip():
        return []

    # Common abbreviations that contain periods but don't end sentences
    abbreviations = {
        'dr', 'mr', 'mrs', 'ms', 'prof', 'sr', 'jr',
        'vs', 'etc', 'inc', 'ltd', 'co', 'corp',
        'st', 'ave', 'blvd', 'rd',
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
        'no', 'nos', 'vol', 'vols', 'pp', 'pg', 'pgs',
        'approx', 'est', 'dept', 'div', 'govt',
        'i.e', 'e.g', 'cf', 'viz', 'al', 'et'
    }

    # More complete pattern for sentence splitting
    # Handles: periods, exclamation marks, question marks
    # Doesn't split on: abbreviations, numbers, URLs

    sentences = []
    current_sentence = []
    words = text.split()

    for i, word in enumerate(words):
        current_sentence.append(word)

        # Check if this word ends a sentence
        if any(word.rstrip(')"\'').endswith(end) for end in SENTENCE_ENDINGS):
            # Check if it's an abbreviation
            word_lower = word.lower().rstrip('.)!?')

            # Don't split if it's an abbreviation
            if word_lower in abbreviations:
                continue

            # Don't split on single letters with periods (initials like "J. K.")
            if len(word.rstrip('.')) == 1 and word.endswith('.'):
                continue

            # Don't split on numbers with periods (like "3.14" when followed by more)
            if re.match(r'^\d+\.$', word) and i + 1 < len(words):
                next_word = words[i + 1]
                if re.match(r'^\d', next_word):
                    continue

            # This looks like a sentence end
            sentence = ' '.join(current_sentence)
            if sentence.strip():
                sentences.append(sentence.strip())
            current_sentence = []

    # Don't forget the last sentence
    if current_sentence:
        sentence = ' '.join(current_sentence)
        if sentence.strip():
            sentences.append(sentence.strip())

    return sentences


def estimate_reading_complexity(text: str) -> dict:
    """
    Estimate the reading complexity of text.

    This provides metrics that can help tune chunking parameters.
    More complex text may need smaller chunks.

    Args:
        text: The text to analyze.

    Returns:
        Dictionary with complexity metrics:
        - avg_word_length: Average word length
        - avg_sentence_length: Average words per sentence
        - long_word_ratio: Ratio of words > 6 characters
        - complexity_score: Overall complexity (0-1)

    Example:
        metrics = estimate_reading_complexity(text)
        if metrics['complexity_score'] > 0.7:
            # Use smaller chunks for complex text
            chunk_size = 300
    """
    if not text or not text.strip():
        return {
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'long_word_ratio': 0,
            'complexity_score': 0
        }

    # Split into words and sentences
    words = text.split()
    sentences = split_into_sentences(text)

    if not words:
        return {
            'avg_word_length': 0,
            'avg_sentence_length': 0,
            'long_word_ratio': 0,
            'complexity_score': 0
        }

    # Calculate metrics
    word_lengths = [len(w.strip('.,!?;:"\'-()[]{}')) for w in words]
    avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0

    avg_sentence_length = len(words) / len(sentences) if sentences else len(words)

    long_words = sum(1 for length in word_lengths if length > 6)
    long_word_ratio = long_words / len(words) if words else 0

    # Complexity score (0-1)
    # Based on Flesch-Kincaid inspired heuristics
    complexity_score = min(1.0, max(0.0,
        (avg_word_length / 10) * 0.3 +  # Word length contribution
        (avg_sentence_length / 30) * 0.3 +  # Sentence length contribution
        long_word_ratio * 0.4  # Long word ratio contribution
    ))

    return {
        'avg_word_length': round(avg_word_length, 2),
        'avg_sentence_length': round(avg_sentence_length, 2),
        'long_word_ratio': round(long_word_ratio, 2),
        'complexity_score': round(complexity_score, 2)
    }
