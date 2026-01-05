"""
PDF File Loader
================

WHAT IS THIS LOADER?
--------------------
This loader handles PDF (Portable Document Format) files.
PDFs are complex binary files that can contain text, images, fonts, and more.

PDF FILE STRUCTURE:
-------------------
- PDFs are NOT plain text - they're binary files
- They contain pages, each with its own content
- Text may be in different fonts, sizes, positions
- Some PDFs have images of text (scanned documents) - these require OCR

WHAT LIBRARY DO WE USE?
-----------------------
We use PyPDF (formerly PyPDF2) to extract text from PDFs.
PyPDF is a pure-Python library that:
- Reads PDF files
- Extracts text from each page
- Handles most common PDF formats

LIMITATIONS:
------------
- Scanned PDFs (images) won't have extractable text
- Complex layouts may not extract perfectly
- Some encrypted PDFs cannot be read
- Tables and columns may merge incorrectly

For better extraction, consider:
- pdfplumber (better table extraction)
- pdf2image + OCR (for scanned documents)
"""

import os
from typing import List

from src.ingestion.document_loader.base import DocumentLoader, LoadedDocument


class PDFLoader(DocumentLoader):
    """
    Document loader for PDF files.

    This loader uses PyPDF to extract text from PDF documents.
    It processes each page and combines the text.

    Supported formats:
    - .pdf

    Requirements:
    - pypdf package must be installed: pip install pypdf

    Example:
        loader = PDFLoader()
        doc = loader.load("document.pdf")
        print(doc.text)
        print(doc.metadata["page_count"])
    """

    @property
    def supported_extensions(self) -> List[str]:
        """
        Return supported file extensions.

        Returns:
            List containing ".pdf"
        """
        return [".pdf", ".PDF"]

    def load(self, file_path: str) -> LoadedDocument:
        """
        Load a PDF file and extract its text content.

        This method:
        1. Opens the PDF file
        2. Iterates through each page
        3. Extracts text from each page
        4. Combines all text with page separators
        5. Returns text with metadata

        Args:
            file_path: Path to the PDF file.

        Returns:
            LoadedDocument with extracted text and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ImportError: If pypdf is not installed.
            ValueError: If PDF cannot be read.
        """
        # =====================================================================
        # Step 1: Check if pypdf is installed
        # =====================================================================
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf is required to read PDF files. "
                "Install it with: pip install pypdf"
            )

        # =====================================================================
        # Step 2: Validate the file exists
        # =====================================================================
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # =====================================================================
        # Step 3: Get file information
        # =====================================================================
        file_name = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)

        print(f"[PDFLoader] Loading PDF: {file_name}")
        print(f"[PDFLoader] File size: {file_size} bytes")

        # =====================================================================
        # Step 4: Open and read the PDF
        # =====================================================================
        try:
            reader = PdfReader(file_path)
            page_count = len(reader.pages)
            print(f"[PDFLoader] Found {page_count} pages")
        except Exception as e:
            raise ValueError(f"Could not read PDF file: {e}")

        # =====================================================================
        # Step 5: Extract text from each page
        # =====================================================================
        # We process each page separately and combine them.
        # This allows us to track which page text came from.

        all_text_parts = []
        pages_with_text = 0

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                # Extract text from this page
                page_text = page.extract_text()

                if page_text and page_text.strip():
                    # Add page marker for reference
                    all_text_parts.append(f"[Page {page_num}]\n{page_text}")
                    pages_with_text += 1
                else:
                    # Page has no extractable text (might be an image)
                    all_text_parts.append(f"[Page {page_num}]\n[No text content]")

            except Exception as e:
                # If a page fails, note it but continue
                print(f"[PDFLoader] Warning: Could not extract page {page_num}: {e}")
                all_text_parts.append(f"[Page {page_num}]\n[Extraction failed]")

        # =====================================================================
        # Step 6: Combine all pages
        # =====================================================================
        full_text = "\n\n".join(all_text_parts)

        print(f"[PDFLoader] Extracted text from {pages_with_text}/{page_count} pages")
        print(f"[PDFLoader] Total characters: {len(full_text)}")

        # =====================================================================
        # Step 7: Build metadata
        # =====================================================================
        # Try to get PDF metadata (title, author, etc.)
        pdf_metadata = {}
        if reader.metadata:
            pdf_metadata = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
            }
            # Remove empty values
            pdf_metadata = {k: v for k, v in pdf_metadata.items() if v}

        metadata = {
            "source": file_name,
            "file_type": "pdf",
            "file_path": file_path,
            "file_size_bytes": file_size,
            "page_count": page_count,
            "pages_with_text": pages_with_text,
            "char_count": len(full_text),
            **pdf_metadata  # Include any PDF metadata we found
        }

        # =====================================================================
        # Step 8: Return the loaded document
        # =====================================================================
        return LoadedDocument(text=full_text, metadata=metadata)
