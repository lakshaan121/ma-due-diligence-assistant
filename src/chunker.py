"""
chunker.py

Responsibility: Split extracted page text into overlapping chunks of
~1000 characters (150 character overlap) so that:
  1. Each chunk is small enough to embed and fit in an LLM prompt.
  2. Overlap prevents losing meaning at chunk boundaries -- a sentence
     that gets cut in half at the end of one chunk still appears
     whole at the start of the next chunk.

Why chunk at all instead of embedding a whole page or document?
    Embedding models produce ONE vector per input. If we embedded an
    entire page, that single vector would have to represent every
    topic on the page at once, making similarity search vague. Small,
    focused chunks give each vector a narrower, more precise meaning,
    which makes retrieval far more accurate.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE,
                 chunk_overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """
    Split each page's text into overlapping chunks.

    Args:
        pages: output of pdf_processor.extract_text_by_page(), i.e.
               [{"document": str, "page": int, "text": str}, ...]
        chunk_size: max characters per chunk (configurable; default 1000).
        chunk_overlap: characters shared between consecutive chunks
                       from the SAME page (configurable; default 150).

    Returns:
        A list of dicts, one per chunk, each still tagged with its
        original document name and page number so citations survive
        the splitting step:
            {"document": str, "page": int, "chunk_text": str}

    Note:
        We split page-by-page (not the whole document at once) so a
        chunk never silently spans two different page numbers -- that
        would make the page citation for that chunk misleading.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on paragraph/sentence boundaries first, only
        # falling back to a hard character cut if nothing else fits.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks = []
    for page in pages:
        page_chunks = splitter.split_text(page["text"])
        for chunk_text in page_chunks:
            all_chunks.append({
                "document": page["document"],
                "page": page["page"],
                "chunk_text": chunk_text,
            })

    return all_chunks
