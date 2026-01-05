"""
Document Loader Module
=======================

This module provides document loaders for various file formats.
Each loader extracts text content from files for ingestion into the RAG system.

Available Loaders:
- TextLoader: Plain text files (.txt, .md)
- PDFLoader: PDF documents (.pdf)
- HTMLLoader: HTML files (.html, .htm)
- DOCXLoader: Word documents (.docx)

Usage:
    from src.ingestion.document_loader import TextLoader, PDFLoader

    loader = TextLoader()
    doc = loader.load("file.txt")
    print(doc.text)
"""

from src.ingestion.document_loader.base import DocumentLoader, LoadedDocument
from src.ingestion.document_loader.text_loader import TextLoader
from src.ingestion.document_loader.pdf_loader import PDFLoader
from src.ingestion.document_loader.html_loader import HTMLLoader
from src.ingestion.document_loader.docx_loader import DOCXLoader

__all__ = [
    "DocumentLoader",
    "LoadedDocument",
    "TextLoader",
    "PDFLoader",
    "HTMLLoader",
    "DOCXLoader",
]
