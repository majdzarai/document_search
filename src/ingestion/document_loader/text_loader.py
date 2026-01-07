"""
Text File Loader (Enhanced)
============================

WHAT IS THIS LOADER?
--------------------
This loader handles plain text files (.txt, .text, .md, .markdown).
Plain text is the simplest format - no parsing needed, just read the file!

ENHANCEMENTS IN THIS VERSION:
-----------------------------
1. TEXT NORMALIZATION: Cleans extracted text for better quality
2. CONTROL CHARACTER REMOVAL: Strips invisible corruption
3. WHITESPACE NORMALIZATION: Consistent spacing
4. PARAGRAPH PRESERVATION: Maintains document structure

TEXT FILES ARE SIMPLE:
----------------------
- No binary encoding
- No special structure
- Just raw text content
- May have different character encodings (UTF-8, ASCII, etc.)

WHY HANDLE ENCODING?
--------------------
Text files can be saved in different character encodings:
- UTF-8: Most common, supports all languages
- ASCII: English only, 7-bit
- Latin-1: Western European languages
- UTF-16: Alternative Unicode encoding

We try UTF-8 first (most common), then fall back to other encodings.
"""

import os
from typing import List, Optional

from src.ingestion.document_loader.base import DocumentLoader, LoadedDocument
from src.ingestion.text_utils import (
    TextNormalizer,
    NormalizationConfig,
    normalize_text
)


class TextLoader(DocumentLoader):
    """
    Document loader for plain text files with text normalization.

    This loader reads text files and returns their content,
    with optional normalization for better RAG quality.

    Supported formats:
    - .txt (plain text)
    - .text (alternative extension)
    - .md (markdown)
    - .markdown (markdown)

    Features:
    - Multi-encoding support (UTF-8, Latin-1, etc.)
    - Text normalization (whitespace, control chars)
    - Paragraph boundary preservation

    Example:
        loader = TextLoader()
        doc = loader.load("readme.txt")
        print(doc.text)
    """

    # List of encodings to try, in order of preference
    # UTF-8 is most common, so we try it first
    ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "ascii"]

    def __init__(self, normalize: bool = True):
        """
        Initialize the text loader.

        Args:
            normalize: Whether to normalize extracted text (default: True).
                      Normalization removes control characters, normalizes
                      whitespace, and preserves paragraph boundaries.

        Example:
            # With normalization (default)
            loader = TextLoader()

            # Without normalization
            loader = TextLoader(normalize=False)
        """
        self.normalize = normalize
        self.normalizer = TextNormalizer(NormalizationConfig(
            remove_control_chars=True,
            normalize_whitespace=True,
            normalize_newlines=True,
            remove_page_markers=False,  # Not applicable to text files
            detect_headers_footers=False,  # Risky for text files
            fix_ocr_artifacts=False,  # Not applicable
            preserve_paragraphs=True,
            strip_lines=True
        ))

    @property
    def supported_extensions(self) -> List[str]:
        """
        Return supported file extensions.

        We support common text file extensions including markdown.

        Returns:
            List of supported extensions.
        """
        return [".txt", ".text", ".md", ".markdown"]

    def load(self, file_path: str) -> LoadedDocument:
        """
        Load a text file and return its content.

        This method:
        1. Checks if the file exists
        2. Tries different encodings to read the file
        3. Normalizes the text (if enabled)
        4. Returns the text with metadata

        Args:
            file_path: Path to the text file.

        Returns:
            LoadedDocument with text content and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file cannot be decoded.
        """
        # =====================================================================
        # Step 1: Validate the file exists
        # =====================================================================
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # =====================================================================
        # Step 2: Get file information
        # =====================================================================
        file_name = os.path.basename(file_path)
        _, file_ext = os.path.splitext(file_path)
        file_size = os.path.getsize(file_path)

        print(f"[TextLoader] Loading file: {file_name}")
        print(f"[TextLoader] File size: {file_size} bytes")

        # =====================================================================
        # Step 3: Try to read the file with different encodings
        # =====================================================================
        # We try multiple encodings because text files don't always specify
        # their encoding. UTF-8 is most common, so we try it first.

        text = None
        used_encoding = None

        for encoding in self.ENCODINGS_TO_TRY:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    text = f.read()
                    used_encoding = encoding
                    print(f"[TextLoader] Successfully read with encoding: {encoding}")
                    break  # Success! Stop trying other encodings
            except UnicodeDecodeError:
                # This encoding didn't work, try the next one
                continue
            except Exception as e:
                # Some other error occurred
                raise ValueError(f"Error reading file: {e}")

        # If no encoding worked, raise an error
        if text is None:
            raise ValueError(
                f"Could not decode file '{file_path}' with any supported encoding. "
                f"Tried: {', '.join(self.ENCODINGS_TO_TRY)}"
            )

        # =====================================================================
        # Step 4: Normalize the text
        # =====================================================================
        original_char_count = len(text)
        original_line_count = text.count("\n") + 1

        if self.normalize:
            text = self.normalizer.normalize(text)
            print(f"[TextLoader] Normalized text ({original_char_count} → {len(text)} chars)")

        # =====================================================================
        # Step 5: Build metadata
        # =====================================================================
        metadata = {
            "source": file_name,
            "file_type": file_ext.lstrip(".").lower(),
            "file_path": file_path,
            "file_size_bytes": file_size,
            "encoding": used_encoding,
            "char_count": len(text),
            "original_char_count": original_char_count,
            "line_count": text.count("\n") + 1,
            "original_line_count": original_line_count,
            "normalized": self.normalize
        }

        print(f"[TextLoader] Extracted {len(text)} characters, {metadata['line_count']} lines")

        # =====================================================================
        # Step 6: Return the loaded document
        # =====================================================================
        return LoadedDocument(text=text, metadata=metadata)
