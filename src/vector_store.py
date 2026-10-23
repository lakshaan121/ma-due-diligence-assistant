"""
vector_store.py

Responsibility: Store chunk embeddings in a FAISS index and perform
fast similarity search to find the chunks whose meaning is closest
to a user's question.

What is FAISS and why use it?
    FAISS (Facebook AI Similarity Search) is a library built specifically
    for finding, out of thousands/millions of vectors, the ones closest
    to a given query vector -- much faster than comparing the query to
    every single vector one by one in plain Python. For a college
    project with a handful of documents this speed difference is small
    in absolute terms, but FAISS is the standard, well-documented tool
    for this job, so using it demonstrates the right approach even at
    small scale.

How similarity is measured here:
    In embeddings.py we normalize every vector to unit length. For unit
    vectors, the INNER PRODUCT (dot product) of two vectors is
    mathematically equal to their COSINE SIMILARITY. So we use FAISS's
    IndexFlatIP ("Flat" = exact search, "IP" = inner product) to get
    cosine-similarity-based search without any extra math ourselves.

    Cosine similarity measures the ANGLE between two vectors, not their
    length -- so it captures "do these two pieces of text point in the
    same semantic direction," regardless of how long the text is. That's
    exactly what we want when comparing a short question to a longer
    chunk.
"""

import os
import re

import faiss
import numpy as np

DEFAULT_TOP_K = 4


def build_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index from an array of chunk embeddings.

    Args:
        embeddings: numpy array of shape (num_chunks, 384), the output
                    of embeddings.embed_texts() run over all chunk texts.

    Returns:
        A FAISS IndexFlatIP populated with all the given vectors, ready
        to be searched. "Flat" means it does an exact (brute-force)
        search rather than an approximate one -- fine for a college
        project's document scale, and simpler to explain than an
        approximate index.
    """
    if embeddings.size == 0:
        raise ValueError("Cannot build a FAISS index from zero embeddings.")

    dimension = embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index


def search(index: faiss.Index, query_embedding: np.ndarray, top_k: int = 4):
    """
    Find the top_k chunks most similar to a query embedding.

    Args:
        index: a FAISS index built by build_index().
        query_embedding: numpy array of shape (1, 384) -- the embedded
                          user question (embed_texts() returns this
                          shape when given a one-item list).
        top_k: how many of the closest chunks to return (default 4).

    Returns:
        A tuple (scores, indices):
            scores: numpy array of shape (1, top_k) with the cosine
                    similarity of each match (higher = more similar).
            indices: numpy array of shape (1, top_k) with the position
                     of each matching chunk in the original embeddings
                     array -- used to look up the actual chunk text
                     and its document/page tags.
    """
    # top_k can't exceed the number of vectors actually stored in the index.
    effective_k = min(top_k, index.ntotal)
    scores, indices = index.search(query_embedding, effective_k)
    return scores, indices


def get_retrieval_settings() -> tuple[int, float | None]:
    """Read TOP_K and MIN_RELEVANCE_SCORE from the environment.

    MIN_RELEVANCE_SCORE is a cosine-similarity floor in [-1, 1]. It is
    DISABLED (None) unless set, because a sensible value depends on the
    embedding model and your documents and must be calibrated on real
    examples -- see scripts/calibrate_threshold.py. Invalid values fall back
    to the defaults instead of crashing the app.
    """
    try:
        top_k = int(os.getenv("TOP_K", DEFAULT_TOP_K))
        if top_k < 1:
            top_k = DEFAULT_TOP_K
    except ValueError:
        top_k = DEFAULT_TOP_K

    raw = os.getenv("MIN_RELEVANCE_SCORE", "").strip()
    min_score = None
    if raw:
        try:
            min_score = float(raw)
        except ValueError:
            min_score = None
    return top_k, min_score


def retrieve(index: faiss.Index, chunks: list[dict], query_embedding: np.ndarray,
             top_k: int = DEFAULT_TOP_K, min_score: float | None = None) -> list[dict]:
    """Search the index and return validated, de-duplicated chunk dicts.

    Each returned dict is a COPY of the stored chunk plus a "score" key
    (cosine similarity, higher = more similar); stored chunks are never
    mutated. Results are ordered best-first.

    Raises:
        ValueError: if the index and chunk list are out of sync (which would
            otherwise attach the wrong document/page to a passage) or the
            query vector has the wrong dimension.
    """
    if index is None or not chunks:
        return []
    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index/chunk mismatch: index has {index.ntotal} vectors but "
            f"{len(chunks)} chunks are stored. Please re-process the documents."
        )
    query_embedding = np.asarray(query_embedding, dtype="float32")
    if query_embedding.ndim != 2 or query_embedding.shape[1] != index.d:
        raise ValueError("Query embedding has the wrong shape for this index.")

    scores, indices = search(index, query_embedding, top_k=top_k)

    results, seen = [], set()
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        if min_score is not None and float(score) < min_score:
            continue
        chunk = chunks[int(idx)]
        key = (chunk["document"], chunk["page"],
               re.sub(r"\s+", " ", chunk["chunk_text"]).strip().lower())
        if key in seen:
            continue
        seen.add(key)
        results.append({**chunk, "score": float(score)})
    return results
