"""
Text File Loader
=================

WHAT IS THIS LOADER?
--------------------
This loader handles plain text files (.txt, .text, .md, .markdown).
Plain text is the simplest format - no parsing needed, just read the file!

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
from typing import List

from src.ingestion.document_loader.base import DocumentLoader, LoadedDocument


class TextLoader(DocumentLoader):
    """
    Document loader for plain text files.

    This loader reads text files and returns their content.
    It handles multiple encodings gracefully.

    Supported formats:
    - .txt (plain text)
    - .text (alternative extension)
    - .md (markdown)
    - .markdown (markdown)

    Example:
        loader = TextLoader()
        doc = loader.load("readme.txt")
        print(doc.text)
    """

    # List of encodings to try, in order of preference
    # UTF-8 is most common, so we try it first
    ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "ascii"]

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
        3. Returns the text with metadata

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
        # Step 4: Build metadata
        # =====================================================================
        metadata = {
            "source": file_name,
            "file_type": file_ext.lstrip(".").lower(),
            "file_path": file_path,
            "file_size_bytes": file_size,
            "encoding": used_encoding,
            "char_count": len(text),
            "line_count": text.count("\n") + 1
        }

        print(f"[TextLoader] Extracted {len(text)} characters, {metadata['line_count']} lines")

        # =====================================================================
        # Step 5: Return the loaded document
        # =====================================================================
        return LoadedDocument(text=text, metadata=metadata)
