"""
Print retrieval similarity scores for questions you supply, so you can pick
MIN_RELEVANCE_SCORE from evidence instead of guessing.

Usage (needs the real embedding model, i.e. internet on first run):
    python scripts/calibrate_threshold.py data/sample_documents/sample_annual_report.pdf \
        --on-topic "What are the liabilities?" "Is there litigation?" \
        --off-topic "What is the capital of France?" "Who won the world cup?"

Read the output: choose a threshold above the highest OFF-topic top score and
below the lowest ON-topic top score. If those ranges overlap, no single
threshold separates them for your documents -- say so rather than forcing one.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.chunker import chunk_pages
from src.embeddings import embed_texts
from src.pdf_processor import extract_text_by_page
from src.vector_store import build_index, retrieve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--on-topic", nargs="*", default=[])
    ap.add_argument("--off-topic", nargs="*", default=[])
    ap.add_argument("--top-k", type=int, default=4)
    args = ap.parse_args()

    chunks = []
    for path in args.pdfs:
        with open(path, "rb") as f:
            chunks += chunk_pages(extract_text_by_page(f, os.path.basename(path)))
    index = build_index(embed_texts([c["chunk_text"] for c in chunks]))

    for label, questions in (("ON-topic", args.on_topic), ("OFF-topic", args.off_topic)):
        for q in questions:
            res = retrieve(index, chunks, embed_texts([q]), top_k=args.top_k)
            scores = ", ".join(f"{r['score']:.3f}" for r in res)
            print(f"[{label}] {q!r}\n    top-{args.top_k} scores: {scores}")


if __name__ == "__main__":
    main()
