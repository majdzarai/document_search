"""
Document Loader Base Class
===========================

WHAT IS A DOCUMENT LOADER?
--------------------------
A document loader is responsible for reading files and extracting their text content.
Different file formats (PDF, TXT, HTML, DOCX) require different parsing methods,
but they all ultimately produce plain text that can be processed by the RAG system.

WHY DO WE NEED DOCUMENT LOADERS?
--------------------------------
In a RAG system, we need to:
1. Accept documents in various formats from users
2. Extract the text content from each format
3. Pass the text to chunking and embedding stages

Each file format has its own structure:
- PDF: Binary format with pages, fonts, images
- TXT: Plain text, simplest format
- HTML: Markup language with tags
- DOCX: ZIP archive containing XML files

WHAT IS THIS MODULE?
--------------------
This module defines an ABSTRACT BASE CLASS for document loaders.
All document loaders must inherit from this class and implement
the `load` method.

DESIGN PATTERN: Strategy Pattern
---------------------------------
By using an abstract base class, we can:
1. Swap loaders easily based on file type
2. Add new loaders without changing existing code
3. Test loaders independently
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class LoadedDocument:
    """
    Represents a document that has been loaded from a file.

    This dataclass holds the extracted text content along with
    basic metadata about the source file.

    Attributes:
        text: The extracted text content from the document.
        metadata: Dictionary containing information about the source:
                 - source: The filename
                 - file_type: The file extension (pdf, txt, etc.)
                 - page_count: Number of pages (if applicable)
                 - Any other format-specific metadata

    Example:
        loaded = LoadedDocument(
            text="This is the document content...",
            metadata={
                "source": "example.pdf",
                "file_type": "pdf",
                "page_count": 5
            }
        )
    """
    text: str
    metadata: Dict[str, Any]


class DocumentLoader(ABC):
    """
    Abstract base class for document loaders.

    All document loaders must inherit from this class and implement
    the `load` method to extract text from their specific file format.

    HOW TO CREATE A NEW LOADER:
    ---------------------------
    1. Create a new class that inherits from DocumentLoader
    2. Implement the `load` method
    3. Implement the `supported_extensions` property

    Example:
        class MyLoader(DocumentLoader):
            @property
            def supported_extensions(self) -> list:
                return [".xyz"]

            def load(self, file_path: str) -> LoadedDocument:
                # Read and parse the file
                text = "extracted text"
                return LoadedDocument(
                    text=text,
                    metadata={"source": file_path, "file_type": "xyz"}
                )
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> list:
        """
        Return a list of file extensions this loader can handle.

        Extensions should include the dot (e.g., [".pdf", ".PDF"]).

        Returns:
            List of supported file extensions.

        Example:
            @property
            def supported_extensions(self) -> list:
                return [".pdf", ".PDF"]
        """
        pass

    @abstractmethod
    def load(self, file_path: str) -> LoadedDocument:
        """
        Load a document from the given file path and extract its text.

        This is the main method that each loader must implement.
        It should:
        1. Open the file at the given path
        2. Parse the file according to its format
        3. Extract all text content
        4. Return a LoadedDocument with text and metadata

        Args:
            file_path: The path to the file to load.
                      Can be absolute or relative.

        Returns:
            A LoadedDocument containing:
            - text: The extracted text content
            - metadata: Information about the source file

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file format is not supported.
            Exception: For any parsing errors.

        Example:
            loader = PDFLoader()
            doc = loader.load("/path/to/document.pdf")
            print(doc.text)  # The extracted text
            print(doc.metadata["page_count"])  # 5
        """
        pass

    def can_load(self, file_path: str) -> bool:
        """
        Check if this loader can handle the given file.

        This method checks if the file extension matches one of
        the supported extensions for this loader.

        Args:
            file_path: The path to the file to check.

        Returns:
            True if this loader can handle the file, False otherwise.

        Example:
            loader = PDFLoader()
            loader.can_load("doc.pdf")  # True
            loader.can_load("doc.txt")  # False
        """
        import os
        _, ext = os.path.splitext(file_path)
        return ext.lower() in [e.lower() for e in self.supported_extensions]
