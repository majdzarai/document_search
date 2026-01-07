"""
PDF File Loader (Enhanced)
===========================

WHAT IS THIS LOADER?
--------------------
This loader handles PDF (Portable Document Format) files.
PDFs are complex binary files that can contain text, images, fonts, and more.

ENHANCEMENTS IN THIS VERSION:
-----------------------------
1. OCR FALLBACK: Automatically uses OCR for scanned/image-based PDFs
2. TEXT NORMALIZATION: Cleans extracted text for better quality
3. SCANNED PDF DETECTION: Intelligently detects if a PDF needs OCR
4. CONFIGURABLE: OCR is opt-in via environment variable

PDF FILE STRUCTURE:
-------------------
- PDFs are NOT plain text - they're binary files
- They contain pages, each with its own content
- Text may be in different fonts, sizes, positions
- Some PDFs have images of text (scanned documents) - these require OCR

WHAT LIBRARY DO WE USE?
-----------------------
- PyPDF: For text extraction from regular PDFs
- pytesseract + pdf2image: For OCR on scanned PDFs (optional)

OCR REQUIREMENTS:
-----------------
To enable OCR fallback, you need:
1. pip install pytesseract pdf2image
2. Install Tesseract OCR on your system:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - Linux: apt-get install tesseract-ocr
   - Mac: brew install tesseract
3. Install Poppler (for pdf2image):
   - Windows: https://github.com/oschwartz10612/poppler-windows
   - Linux: apt-get install poppler-utils
   - Mac: brew install poppler
4. Set ENABLE_PDF_OCR=true in environment

LIMITATIONS:
------------
- OCR is slow and CPU-intensive
- OCR accuracy depends on image quality
- Some encrypted PDFs cannot be read
- Complex layouts may not extract perfectly
"""

import os
from typing import List, Optional, Tuple

from src.ingestion.document_loader.base import DocumentLoader, LoadedDocument
from src.ingestion.text_utils import (
    TextNormalizer,
    NormalizationConfig,
    normalize_text
)


# =============================================================================
# CONSTANTS
# =============================================================================

# Minimum characters per page to consider it "has text"
# Pages with fewer chars than this are considered potentially scanned
MIN_CHARS_PER_PAGE = 50

# Minimum ratio of pages with text to skip OCR
# If less than this ratio has text, consider OCR
MIN_TEXT_PAGE_RATIO = 0.3

# OCR DPI - higher = better quality but slower
OCR_DPI = 200


class PDFLoader(DocumentLoader):
    """
    Document loader for PDF files with OCR fallback.

    This loader uses PyPDF to extract text from PDF documents.
    If the PDF appears to be scanned (no extractable text),
    it can optionally use OCR to extract text from images.

    Supported formats:
    - .pdf

    Features:
    - Automatic scanned PDF detection
    - Optional OCR fallback for image-based PDFs
    - Text normalization for cleaner output
    - Page-by-page extraction with markers

    Requirements:
    - pypdf package: pip install pypdf
    - For OCR (optional):
      - pytesseract: pip install pytesseract
      - pdf2image: pip install pdf2image
      - Tesseract OCR installed on system
      - Poppler installed on system

    Example:
        loader = PDFLoader()
        doc = loader.load("document.pdf")
        print(doc.text)
        print(doc.metadata["page_count"])
        print(doc.metadata["ocr_used"])  # True if OCR was needed
    """

    def __init__(
        self,
        enable_ocr: Optional[bool] = None,
        tesseract_path: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize the PDF loader.

        Args:
            enable_ocr: Whether to use OCR for scanned PDFs.
                       If None, reads from ENABLE_PDF_OCR env var.
            tesseract_path: Path to Tesseract executable.
                           If None, reads from TESSERACT_PATH env var.
            normalize: Whether to normalize extracted text (default: True).

        Example:
            # Use environment settings
            loader = PDFLoader()

            # Force enable OCR
            loader = PDFLoader(enable_ocr=True)

            # Disable text normalization
            loader = PDFLoader(normalize=False)
        """
        # Import settings
        from src.core.config import settings

        # OCR settings
        if enable_ocr is not None:
            self.enable_ocr = enable_ocr
        else:
            self.enable_ocr = settings.enable_pdf_ocr

        if tesseract_path is not None:
            self.tesseract_path = tesseract_path
        else:
            self.tesseract_path = settings.tesseract_path

        # Text normalization settings
        self.normalize = normalize
        self.normalizer = TextNormalizer(NormalizationConfig(
            remove_control_chars=True,
            normalize_whitespace=True,
            normalize_newlines=True,
            remove_page_markers=False,  # We add our own page markers
            detect_headers_footers=False,  # Risky - disabled by default
            fix_ocr_artifacts=False,  # Will be enabled during OCR
            preserve_paragraphs=True,
            strip_lines=True
        ))

        # OCR normalizer with artifact fixing enabled
        self.ocr_normalizer = TextNormalizer(NormalizationConfig(
            remove_control_chars=True,
            normalize_whitespace=True,
            normalize_newlines=True,
            remove_page_markers=False,
            detect_headers_footers=True,  # OCR often duplicates headers
            fix_ocr_artifacts=True,  # Fix common OCR mistakes
            preserve_paragraphs=True,
            strip_lines=True
        ))

    @property
    def supported_extensions(self) -> List[str]:
        """Return supported file extensions."""
        return [".pdf", ".PDF"]

    def load(self, file_path: str) -> LoadedDocument:
        """
        Load a PDF file and extract its text content.

        This method:
        1. Opens the PDF file with PyPDF
        2. Attempts to extract text from each page
        3. If text is minimal and OCR is enabled, uses OCR
        4. Normalizes the extracted text
        5. Returns text with comprehensive metadata

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
        all_text_parts = []
        pages_with_text = 0
        page_char_counts = []

        for page_num, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                page_text = page_text.strip()

                if page_text and len(page_text) >= MIN_CHARS_PER_PAGE:
                    pages_with_text += 1
                    page_char_counts.append(len(page_text))

                    # Normalize the page text
                    if self.normalize:
                        page_text = self.normalizer.normalize(page_text)

                    all_text_parts.append(f"[Page {page_num}]\n{page_text}")
                else:
                    # Page has minimal or no text
                    page_char_counts.append(len(page_text) if page_text else 0)
                    all_text_parts.append(f"[Page {page_num}]\n[No text content]")

            except Exception as e:
                print(f"[PDFLoader] Warning: Could not extract page {page_num}: {e}")
                page_char_counts.append(0)
                all_text_parts.append(f"[Page {page_num}]\n[Extraction failed]")

        # =====================================================================
        # Step 6: Check if OCR is needed
        # =====================================================================
        ocr_used = False
        text_ratio = pages_with_text / page_count if page_count > 0 else 0

        if text_ratio < MIN_TEXT_PAGE_RATIO and self.enable_ocr:
            print(f"[PDFLoader] Low text ratio ({text_ratio:.1%}), attempting OCR...")

            try:
                ocr_text, ocr_page_count = self._extract_with_ocr(file_path)
                if ocr_text and len(ocr_text) > sum(page_char_counts):
                    # OCR extracted more text - use it
                    print(f"[PDFLoader] OCR successful, extracted {len(ocr_text)} characters")
                    all_text_parts = [ocr_text]
                    ocr_used = True
                    pages_with_text = ocr_page_count
                else:
                    print("[PDFLoader] OCR did not improve text extraction")
            except Exception as e:
                print(f"[PDFLoader] OCR failed: {e}")

        elif text_ratio < MIN_TEXT_PAGE_RATIO and not self.enable_ocr:
            print(f"[PDFLoader] Low text ratio ({text_ratio:.1%})")
            print("[PDFLoader] OCR disabled. Set ENABLE_PDF_OCR=true to enable.")

        # =====================================================================
        # Step 7: Combine all pages
        # =====================================================================
        full_text = "\n\n".join(all_text_parts)

        print(f"[PDFLoader] Extracted text from {pages_with_text}/{page_count} pages")
        print(f"[PDFLoader] Total characters: {len(full_text)}")
        if ocr_used:
            print("[PDFLoader] OCR was used for this document")

        # =====================================================================
        # Step 8: Build metadata
        # =====================================================================
        pdf_metadata = {}
        if reader.metadata:
            pdf_metadata = {
                "title": reader.metadata.get("/Title", ""),
                "author": reader.metadata.get("/Author", ""),
                "subject": reader.metadata.get("/Subject", ""),
                "creator": reader.metadata.get("/Creator", ""),
            }
            pdf_metadata = {k: v for k, v in pdf_metadata.items() if v}

        metadata = {
            "source": file_name,
            "file_type": "pdf",
            "file_path": file_path,
            "file_size_bytes": file_size,
            "page_count": page_count,
            "pages_with_text": pages_with_text,
            "char_count": len(full_text),
            "ocr_used": ocr_used,
            "text_extraction_ratio": round(text_ratio, 2),
            **pdf_metadata
        }

        # =====================================================================
        # Step 9: Return the loaded document
        # =====================================================================
        return LoadedDocument(text=full_text, metadata=metadata)

    def _extract_with_ocr(self, file_path: str) -> Tuple[str, int]:
        """
        Extract text from PDF using OCR.

        This method converts PDF pages to images and runs OCR on them.
        It's slow but works for scanned documents.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Tuple of (extracted_text, pages_processed)

        Raises:
            ImportError: If OCR dependencies are not installed.
            Exception: If OCR fails.
        """
        # =====================================================================
        # Check for OCR dependencies
        # =====================================================================
        try:
            import pytesseract
            from pdf2image import convert_from_path
        except ImportError as e:
            raise ImportError(
                "OCR requires additional packages. Install with:\n"
                "  pip install pytesseract pdf2image\n"
                "Also install Tesseract OCR and Poppler on your system.\n"
                f"Missing: {e}"
            )

        # Configure Tesseract path if provided
        if self.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_path

        # =====================================================================
        # Convert PDF to images
        # =====================================================================
        print(f"[PDFLoader-OCR] Converting PDF to images (DPI={OCR_DPI})...")
        try:
            images = convert_from_path(file_path, dpi=OCR_DPI)
        except Exception as e:
            raise Exception(f"Failed to convert PDF to images: {e}")

        print(f"[PDFLoader-OCR] Processing {len(images)} pages with OCR...")

        # =====================================================================
        # Run OCR on each page
        # =====================================================================
        all_text_parts = []
        pages_with_text = 0

        for page_num, image in enumerate(images, start=1):
            try:
                # Run OCR on this page
                page_text = pytesseract.image_to_string(image)
                page_text = page_text.strip()

                if page_text:
                    # Normalize OCR text (with artifact fixing)
                    page_text = self.ocr_normalizer.normalize(page_text)

                    if page_text and len(page_text) >= MIN_CHARS_PER_PAGE:
                        all_text_parts.append(f"[Page {page_num}]\n{page_text}")
                        pages_with_text += 1
                    else:
                        all_text_parts.append(f"[Page {page_num}]\n[Minimal text]")
                else:
                    all_text_parts.append(f"[Page {page_num}]\n[No text detected]")

            except Exception as e:
                print(f"[PDFLoader-OCR] Warning: OCR failed for page {page_num}: {e}")
                all_text_parts.append(f"[Page {page_num}]\n[OCR failed]")

        full_text = "\n\n".join(all_text_parts)
        return full_text, pages_with_text

    def is_scanned_pdf(self, file_path: str) -> bool:
        """
        Check if a PDF appears to be scanned (image-based).

        This is a quick check that doesn't load the full document.
        Useful for deciding whether to enable OCR before loading.

        Args:
            file_path: Path to the PDF file.

        Returns:
            True if the PDF appears to be scanned, False otherwise.

        Example:
            loader = PDFLoader()
            if loader.is_scanned_pdf("document.pdf"):
                print("This PDF needs OCR")
        """
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)

            if len(reader.pages) == 0:
                return False

            # Check first few pages
            pages_to_check = min(3, len(reader.pages))
            text_chars = 0

            for i in range(pages_to_check):
                page_text = reader.pages[i].extract_text() or ""
                text_chars += len(page_text.strip())

            # If average chars per page is very low, it's likely scanned
            avg_chars = text_chars / pages_to_check
            return avg_chars < MIN_CHARS_PER_PAGE

        except Exception:
            return False
