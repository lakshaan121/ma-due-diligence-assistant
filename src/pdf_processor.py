"""
pdf_processor.py

Responsibility: Extract text from uploaded PDF files, page by page,
so that every piece of text we later retrieve can be traced back to
an exact (document_name, page_number) location.

Why page-level tracking matters:
    If we just extracted the whole PDF as one big string, we would
    lose the ability to tell the user "this claim comes from page 45."
    Citations are a core requirement of this project, so extraction
    keeps text separated per page from the very first step.
"""

import os

from pypdf import PdfReader
from pypdf.errors import PdfReadError, PyPdfError

# Configurable via MAX_UPLOAD_MB (see .env.example). Default 50 MB.
DEFAULT_MAX_UPLOAD_MB = 50


def _max_upload_bytes() -> int:
    try:
        return int(float(os.getenv("MAX_UPLOAD_MB", DEFAULT_MAX_UPLOAD_MB)) * 1024 * 1024)
    except ValueError:
        return DEFAULT_MAX_UPLOAD_MB * 1024 * 1024


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or contains no usable text.

    We use a custom exception (instead of letting raw pypdf errors
    escape) so the Streamlit UI layer can catch ONE known error type
    and show a clean message, instead of leaking internal library
    tracebacks to the user.
    """
    pass


def extract_text_by_page(pdf_file, document_name: str) -> list[dict]:
    """
    Extract text from a PDF, one entry per page.

    Args:
        pdf_file: a file-like object (e.g. what st.file_uploader gives you,
                   or an open('rb') file handle). pypdf can read either.
        document_name: display name to tag every extracted page with,
                        e.g. "ABC_Annual_Report.pdf".

    Returns:
        A list of dicts, one per page that contains text:
            {"document": "ABC_Annual_Report.pdf", "page": 1, "text": "..."}
        Page numbers are 1-indexed because that's how humans (and PDF
        viewers) refer to pages -- it makes citations readable.

    Raises:
        PDFProcessingError: if the file isn't a valid PDF, or if no
                             page in the whole document has extractable
                             text (e.g. a scanned/image-only PDF with
                             no OCR layer -- OCR is out of scope here).
    """
    if not document_name.lower().endswith(".pdf"):
        raise PDFProcessingError(f"'{document_name}' is not a .pdf file.")

    # Size check without reading the whole file into memory twice.
    try:
        pdf_file.seek(0, os.SEEK_END)
        size = pdf_file.tell()
        pdf_file.seek(0)
    except (AttributeError, OSError):
        size = None
    if size is not None:
        if size == 0:
            raise PDFProcessingError(f"'{document_name}' is empty (0 bytes).")
        if size > _max_upload_bytes():
            raise PDFProcessingError(
                f"'{document_name}' is larger than the "
                f"{_max_upload_bytes() // (1024 * 1024)} MB upload limit."
            )

    try:
        reader = PdfReader(pdf_file)
        if reader.is_encrypted:
            # Empty-password PDFs open fine; real passwords are unsupported.
            if not reader.decrypt(""):
                raise PDFProcessingError(
                    f"'{document_name}' is password-protected and cannot be read."
                )
        page_count = len(reader.pages)
    except PDFProcessingError:
        raise
    except (PdfReadError, PyPdfError, ValueError, KeyError, OSError) as exc:
        raise PDFProcessingError(
            f"'{document_name}' does not appear to be a valid PDF file."
        ) from exc

    if page_count == 0:
        raise PDFProcessingError(f"'{document_name}' has no pages.")

    pages_with_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        # extract_text() can return None or "" for image-only pages --
        # we simply skip those rather than storing empty chunks.
        try:
            raw_text = page.extract_text()
        except Exception as exc:  # corrupt page content stream
            raise PDFProcessingError(
                f"'{document_name}' could not be read (page {page_number} is corrupt)."
            ) from exc
        if raw_text and raw_text.strip():
            pages_with_text.append({
                "document": document_name,
                "page": page_number,
                "text": raw_text.strip(),
            })

    if not pages_with_text:
        raise PDFProcessingError(
            f"'{document_name}' contains no extractable text. "
            "It may be a scanned/image-only PDF, which this prototype "
            "does not OCR."
        )

    return pages_with_text
