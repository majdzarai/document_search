"""
HTML File Loader (Enhanced)
============================

WHAT IS THIS LOADER?
--------------------
This loader handles HTML (HyperText Markup Language) files.
HTML is the standard markup language for web pages.

ENHANCEMENTS IN THIS VERSION:
-----------------------------
1. TEXT NORMALIZATION: Cleans extracted text for better quality
2. IMPROVED PARSING: Better handling of nested tags
3. SCRIPT/STYLE REMOVAL: Removes code that's not content
4. PARAGRAPH PRESERVATION: Maintains document structure

HTML FILE STRUCTURE:
--------------------
HTML files contain:
- Tags: <p>, <h1>, <div>, <a>, etc.
- Text content between tags
- Attributes: class, id, href, etc.
- Scripts and styles (which we want to ignore)

WHAT DO WE NEED TO DO?
----------------------
1. Read the HTML file
2. Remove script and style tags (they contain code, not content)
3. Extract just the text content
4. Clean up whitespace
5. Normalize for RAG quality

WHAT LIBRARY DO WE USE?
-----------------------
We use Python's built-in html.parser module for basic parsing,
and implement a simple tag stripper. For production use, consider:
- BeautifulSoup (most popular)
- lxml (fastest)

We keep it simple here to avoid extra dependencies.
"""

import os
import re
from html.parser import HTMLParser
from typing import List, Optional

from src.ingestion.document_loader.base import DocumentLoader, LoadedDocument
from src.ingestion.text_utils import (
    TextNormalizer,
    NormalizationConfig,
    normalize_text
)


class HTMLTextExtractor(HTMLParser):
    """
    Custom HTML parser that extracts text content.

    This parser:
    - Ignores script and style tags
    - Collects text from all other tags
    - Handles HTML entities (like &amp;)
    - Preserves paragraph boundaries

    WHAT IS HTMLParser?
    -------------------
    HTMLParser is a built-in Python class that reads HTML
    and calls methods when it finds different elements:
    - handle_starttag(): Called for <tag>
    - handle_endtag(): Called for </tag>
    - handle_data(): Called for text content

    We override handle_data() to collect text.
    """

    def __init__(self):
        super().__init__()
        # List to collect text pieces
        self.text_parts: List[str] = []
        # Tags to ignore (their content is not human-readable)
        self.ignore_tags = {"script", "style", "head", "meta", "link", "noscript"}
        # Block-level tags that indicate paragraph boundaries
        self.block_tags = {
            "p", "div", "h1", "h2", "h3", "h4", "h5", "h6",
            "li", "tr", "td", "th", "article", "section",
            "header", "footer", "nav", "aside", "blockquote",
            "pre", "br", "hr"
        }
        # Stack to track current tags
        self.tag_stack: List[str] = []
        # Track if we just ended a block tag (for paragraph breaks)
        self.just_ended_block = False

    def handle_starttag(self, tag: str, attrs):
        """Called when we encounter an opening tag like <p>."""
        tag_lower = tag.lower()
        self.tag_stack.append(tag_lower)

        # Add paragraph break before block-level elements
        if tag_lower in self.block_tags:
            self.text_parts.append("\n\n")

    def handle_endtag(self, tag: str):
        """Called when we encounter a closing tag like </p>."""
        tag_lower = tag.lower()

        if self.tag_stack and self.tag_stack[-1] == tag_lower:
            self.tag_stack.pop()

        # Add paragraph break after block-level elements
        if tag_lower in self.block_tags:
            self.text_parts.append("\n\n")
            self.just_ended_block = True
        else:
            self.just_ended_block = False

    def handle_data(self, data: str):
        """
        Called when we encounter text content.

        We only collect text if we're not inside an ignored tag.
        """
        # Check if we're inside any ignored tag
        if any(tag in self.ignore_tags for tag in self.tag_stack):
            return

        # Clean up the text (but preserve some structure)
        text = data.strip()
        if text:
            # Add space before if not just after a block tag
            if self.text_parts and not self.just_ended_block:
                # Check if last part ends with whitespace
                if self.text_parts[-1] and not self.text_parts[-1].endswith((' ', '\n')):
                    self.text_parts.append(" ")
            self.text_parts.append(text)
            self.just_ended_block = False

    def get_text(self) -> str:
        """Return all collected text, with paragraph structure."""
        return "".join(self.text_parts)


class HTMLLoader(DocumentLoader):
    """
    Document loader for HTML files with text normalization.

    This loader extracts readable text content from HTML files,
    removing scripts, styles, and other non-content elements.

    Supported formats:
    - .html
    - .htm

    Features:
    - Script/style removal
    - Text normalization (whitespace, control chars)
    - Paragraph boundary preservation
    - Title extraction

    Example:
        loader = HTMLLoader()
        doc = loader.load("page.html")
        print(doc.text)  # Clean text without HTML tags
    """

    # Encodings to try when reading HTML files
    ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    def __init__(self, normalize: bool = True):
        """
        Initialize the HTML loader.

        Args:
            normalize: Whether to normalize extracted text (default: True).
                      Normalization removes control characters, normalizes
                      whitespace, and preserves paragraph boundaries.

        Example:
            # With normalization (default)
            loader = HTMLLoader()

            # Without normalization
            loader = HTMLLoader(normalize=False)
        """
        self.normalize = normalize
        self.normalizer = TextNormalizer(NormalizationConfig(
            remove_control_chars=True,
            normalize_whitespace=True,
            normalize_newlines=True,
            remove_page_markers=False,
            detect_headers_footers=False,  # HTML pages rarely have repeated headers
            fix_ocr_artifacts=False,
            preserve_paragraphs=True,
            strip_lines=True
        ))

    @property
    def supported_extensions(self) -> List[str]:
        """Return supported file extensions."""
        return [".html", ".htm", ".HTML", ".HTM"]

    def load(self, file_path: str) -> LoadedDocument:
        """
        Load an HTML file and extract its text content.

        This method:
        1. Reads the HTML file
        2. Parses it to extract text
        3. Normalizes the text (if enabled)
        4. Returns text with metadata

        Args:
            file_path: Path to the HTML file.

        Returns:
            LoadedDocument with extracted text and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ValueError: If file cannot be read or parsed.
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
        file_size = os.path.getsize(file_path)

        print(f"[HTMLLoader] Loading HTML: {file_name}")
        print(f"[HTMLLoader] File size: {file_size} bytes")

        # =====================================================================
        # Step 3: Read the file content
        # =====================================================================
        html_content = None
        used_encoding = None

        for encoding in self.ENCODINGS_TO_TRY:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    html_content = f.read()
                    used_encoding = encoding
                    break
            except UnicodeDecodeError:
                continue

        if html_content is None:
            raise ValueError("Could not decode HTML file with any supported encoding")

        print(f"[HTMLLoader] Read with encoding: {used_encoding}")

        # =====================================================================
        # Step 4: Extract title (if present)
        # =====================================================================
        title = ""
        title_match = re.search(
            r"<title[^>]*>(.*?)</title>",
            html_content,
            re.IGNORECASE | re.DOTALL
        )
        if title_match:
            title = title_match.group(1).strip()
            # Clean the title of any HTML entities
            title = self._decode_html_entities(title)
            print(f"[HTMLLoader] Found title: {title}")

        # =====================================================================
        # Step 5: Parse HTML and extract text
        # =====================================================================
        try:
            parser = HTMLTextExtractor()
            parser.feed(html_content)
            text = parser.get_text()
        except Exception as e:
            raise ValueError(f"Error parsing HTML: {e}")

        # =====================================================================
        # Step 6: Normalize the text
        # =====================================================================
        original_char_count = len(text)

        if self.normalize:
            text = self.normalizer.normalize(text)
            print(f"[HTMLLoader] Normalized text ({original_char_count} → {len(text)} chars)")

        print(f"[HTMLLoader] Extracted {len(text)} characters")

        # =====================================================================
        # Step 7: Build metadata
        # =====================================================================
        metadata = {
            "source": file_name,
            "file_type": "html",
            "file_path": file_path,
            "file_size_bytes": file_size,
            "encoding": used_encoding,
            "title": title,
            "char_count": len(text),
            "original_char_count": original_char_count,
            "normalized": self.normalize
        }

        # =====================================================================
        # Step 8: Return the loaded document
        # =====================================================================
        return LoadedDocument(text=text, metadata=metadata)

    def _decode_html_entities(self, text: str) -> str:
        """
        Decode common HTML entities in text.

        Args:
            text: Text with potential HTML entities.

        Returns:
            Text with entities decoded.
        """
        import html
        return html.unescape(text)
