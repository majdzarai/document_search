"""
HTML File Loader
=================

WHAT IS THIS LOADER?
--------------------
This loader handles HTML (HyperText Markup Language) files.
HTML is the standard markup language for web pages.

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

WHAT LIBRARY DO WE USE?
-----------------------
We use Python's built-in html.parser module for basic parsing,
and implement a simple tag stripper. For production use, consider:
- BeautifulSoup (most popular)
- lxml (fastest)

We keep it simple here to avoid extra dependencies.

EXAMPLE:
--------
Input HTML:
    <html>
        <head><title>My Page</title></head>
        <body>
            <h1>Hello World</h1>
            <p>This is content.</p>
        </body>
    </html>

Output Text:
    My Page
    Hello World
    This is content.
"""

import os
import re
from html.parser import HTMLParser
from typing import List

from src.ingestion.document_loader.base import DocumentLoader, LoadedDocument


class HTMLTextExtractor(HTMLParser):
    """
    Custom HTML parser that extracts text content.

    This parser:
    - Ignores script and style tags
    - Collects text from all other tags
    - Handles HTML entities (like &amp;)

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
        self.ignore_tags = {"script", "style", "head", "meta", "link"}
        # Stack to track current tags
        self.tag_stack: List[str] = []

    def handle_starttag(self, tag: str, attrs):
        """Called when we encounter an opening tag like <p>."""
        self.tag_stack.append(tag.lower())

    def handle_endtag(self, tag: str):
        """Called when we encounter a closing tag like </p>."""
        if self.tag_stack and self.tag_stack[-1] == tag.lower():
            self.tag_stack.pop()

    def handle_data(self, data: str):
        """
        Called when we encounter text content.

        We only collect text if we're not inside an ignored tag.
        """
        # Check if we're inside any ignored tag
        if any(tag in self.ignore_tags for tag in self.tag_stack):
            return

        # Clean up the text (remove excessive whitespace)
        text = data.strip()
        if text:
            self.text_parts.append(text)

    def get_text(self) -> str:
        """Return all collected text, joined with spaces."""
        return " ".join(self.text_parts)


class HTMLLoader(DocumentLoader):
    """
    Document loader for HTML files.

    This loader extracts readable text content from HTML files,
    removing scripts, styles, and other non-content elements.

    Supported formats:
    - .html
    - .htm

    Example:
        loader = HTMLLoader()
        doc = loader.load("page.html")
        print(doc.text)  # Clean text without HTML tags
    """

    # Encodings to try when reading HTML files
    ENCODINGS_TO_TRY = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

    @property
    def supported_extensions(self) -> List[str]:
        """
        Return supported file extensions.

        Returns:
            List containing HTML extensions.
        """
        return [".html", ".htm", ".HTML", ".HTM"]

    def load(self, file_path: str) -> LoadedDocument:
        """
        Load an HTML file and extract its text content.

        This method:
        1. Reads the HTML file
        2. Parses it to extract text
        3. Cleans up the text
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
            raise ValueError(f"Could not decode HTML file with any supported encoding")

        print(f"[HTMLLoader] Read with encoding: {used_encoding}")

        # =====================================================================
        # Step 4: Extract title (if present)
        # =====================================================================
        title = ""
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
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
        # Step 6: Clean up the text
        # =====================================================================
        # Remove excessive whitespace
        text = re.sub(r"\s+", " ", text)
        text = text.strip()

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
            "char_count": len(text)
        }

        # =====================================================================
        # Step 8: Return the loaded document
        # =====================================================================
        return LoadedDocument(text=text, metadata=metadata)
