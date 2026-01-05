"""
DOCX File Loader
=================

WHAT IS THIS LOADER?
--------------------
This loader handles DOCX (Microsoft Word) files.
DOCX is the default format for Microsoft Word documents since 2007.

DOCX FILE STRUCTURE:
--------------------
A DOCX file is actually a ZIP archive containing:
- XML files with document content
- Styles and formatting information
- Embedded images and media
- Document properties

The main content is in word/document.xml inside the ZIP.

WHAT LIBRARY DO WE USE?
-----------------------
We use python-docx to extract text from DOCX files.
python-docx is a popular library that:
- Opens DOCX files
- Provides access to paragraphs, tables, etc.
- Handles the complex XML structure internally

LIMITATIONS:
------------
- Only works with .docx files (not old .doc format)
- Images are not extracted (only text)
- Complex formatting may not be preserved
- Headers/footers require extra handling

For .doc files (old Word format), you would need:
- antiword (command-line tool)
- textract
- LibreOffice conversion
"""

import os
from typing import List

from src.ingestion.document_loader.base import DocumentLoader, LoadedDocument


class DOCXLoader(DocumentLoader):
    """
    Document loader for DOCX (Microsoft Word) files.

    This loader uses python-docx to extract text from Word documents.
    It extracts text from paragraphs and tables.

    Supported formats:
    - .docx (Word 2007 and later)

    NOT supported:
    - .doc (older Word format)

    Requirements:
    - python-docx package: pip install python-docx

    Example:
        loader = DOCXLoader()
        doc = loader.load("report.docx")
        print(doc.text)
    """

    @property
    def supported_extensions(self) -> List[str]:
        """
        Return supported file extensions.

        Note: We only support .docx, not .doc (old format)

        Returns:
            List containing ".docx"
        """
        return [".docx", ".DOCX"]

    def load(self, file_path: str) -> LoadedDocument:
        """
        Load a DOCX file and extract its text content.

        This method:
        1. Opens the DOCX file
        2. Extracts text from all paragraphs
        3. Extracts text from tables (if any)
        4. Returns combined text with metadata

        Args:
            file_path: Path to the DOCX file.

        Returns:
            LoadedDocument with extracted text and metadata.

        Raises:
            FileNotFoundError: If file doesn't exist.
            ImportError: If python-docx is not installed.
            ValueError: If file cannot be read.
        """
        # =====================================================================
        # Step 1: Check if python-docx is installed
        # =====================================================================
        try:
            from docx import Document
        except ImportError:
            raise ImportError(
                "python-docx is required to read DOCX files. "
                "Install it with: pip install python-docx"
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

        print(f"[DOCXLoader] Loading DOCX: {file_name}")
        print(f"[DOCXLoader] File size: {file_size} bytes")

        # =====================================================================
        # Step 4: Open and read the DOCX file
        # =====================================================================
        try:
            doc = Document(file_path)
        except Exception as e:
            raise ValueError(f"Could not read DOCX file: {e}")

        # =====================================================================
        # Step 5: Extract text from paragraphs
        # =====================================================================
        # Paragraphs are the main content containers in Word documents.
        # Each paragraph is a separate block of text.

        paragraph_texts = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:  # Only add non-empty paragraphs
                paragraph_texts.append(text)

        print(f"[DOCXLoader] Found {len(paragraph_texts)} paragraphs")

        # =====================================================================
        # Step 6: Extract text from tables
        # =====================================================================
        # Tables contain cells organized in rows and columns.
        # We extract text from each cell.

        table_texts = []
        for table_idx, table in enumerate(doc.tables, start=1):
            for row in table.rows:
                row_texts = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_texts.append(cell_text)
                if row_texts:
                    # Join cells with tabs (table-like format)
                    table_texts.append("\t".join(row_texts))

        if table_texts:
            print(f"[DOCXLoader] Found {len(doc.tables)} tables")

        # =====================================================================
        # Step 7: Combine all text
        # =====================================================================
        # Combine paragraphs first, then tables
        all_text_parts = []

        if paragraph_texts:
            all_text_parts.extend(paragraph_texts)

        if table_texts:
            all_text_parts.append("\n--- Tables ---\n")
            all_text_parts.extend(table_texts)

        full_text = "\n".join(all_text_parts)

        print(f"[DOCXLoader] Total characters: {len(full_text)}")

        # =====================================================================
        # Step 8: Extract document properties (metadata)
        # =====================================================================
        doc_properties = {}
        try:
            core_props = doc.core_properties
            doc_properties = {
                "title": core_props.title or "",
                "author": core_props.author or "",
                "subject": core_props.subject or "",
                "created": str(core_props.created) if core_props.created else "",
                "modified": str(core_props.modified) if core_props.modified else "",
            }
            # Remove empty values
            doc_properties = {k: v for k, v in doc_properties.items() if v}
        except Exception:
            # If we can't get properties, that's okay
            pass

        # =====================================================================
        # Step 9: Build metadata
        # =====================================================================
        metadata = {
            "source": file_name,
            "file_type": "docx",
            "file_path": file_path,
            "file_size_bytes": file_size,
            "paragraph_count": len(paragraph_texts),
            "table_count": len(doc.tables),
            "char_count": len(full_text),
            **doc_properties  # Include document properties
        }

        # =====================================================================
        # Step 10: Return the loaded document
        # =====================================================================
        return LoadedDocument(text=full_text, metadata=metadata)
