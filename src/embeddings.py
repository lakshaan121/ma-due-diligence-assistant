"""
embeddings.py

Responsibility: Convert text (chunks or questions) into 384-dimensional
numerical vectors using the pre-trained all-MiniLM-L6-v2 Sentence-
Transformers model.

What is an embedding?
    An embedding is a list of numbers (a vector) that represents the
    MEANING of a piece of text, not just its exact words. Two sentences
    with similar meaning end up with vectors that are close together
    in this 384-dimensional space, even if they don't share any words.

    Example:
        "The company has high debt."
        "The company has significant borrowings."
    These use different words but mean nearly the same thing, so their
    embeddings will be close together (high cosine similarity). This is
    exactly what makes semantic search possible -- keyword search would
    miss this connection because there's no shared vocabulary.

Why all-MiniLM-L6-v2 specifically?
    - It's small (~80MB) and fast enough to run on a CPU, so no GPU
      or paid infrastructure is needed for a college project.
    - It's a well-established, pre-trained model -- we don't train or
      fine-tune anything ourselves, which keeps the project's scope
      realistic for a fresher-level demo.
    - 384 dimensions is enough to capture sentence-level meaning for
      this use case without the extra size/cost of larger models.

Why we don't build an embedding model from scratch:
    Training a good embedding model needs huge amounts of text data
    and compute. Using a pre-trained model is standard practice even
    in production RAG systems -- the skill being demonstrated here is
    knowing WHEN and HOW to use one, not reinventing it.
"""

import numpy as np

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


class EmbeddingError(Exception):
    """Raised when the embedding model cannot be loaded or run, so the UI
    can show one clean message instead of a raw library traceback."""

# Module-level cache so the (relatively slow) model load only happens
# once per app run, not once per question or per chunk.
_model_cache = None


def get_embedding_model():
    """
    Load (or return the already-loaded) Sentence-Transformers model.

    Loading a transformer model from disk/network takes a couple of
    seconds. We cache it in a module-level variable so Streamlit
    doesn't reload it on every single user interaction.
    """
    global _model_cache
    if _model_cache is None:
        # Imported lazily so modules/tests that never embed (risk, financial,
        # chunking...) work without loading torch.
        try:
            from sentence_transformers import SentenceTransformer
            _model_cache = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as exc:
            raise EmbeddingError(
                f"Could not load the embedding model '{EMBEDDING_MODEL_NAME}'. "
                "On first run it must be downloaded from huggingface.co, so "
                "check your internet connection and that "
                "sentence-transformers is installed."
            ) from exc
    return _model_cache


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of strings into their embedding vectors.

    Args:
        texts: list of strings (chunk texts, or a single question
               wrapped in a one-item list).

    Returns:
        A numpy array of shape (len(texts), 384) -- one 384-dimensional
        vector per input string. FAISS (Phase 6) expects exactly this
        shape for both building the index and searching it.
    """
    if not texts:
        return np.array([])

    model = get_embedding_model()
    # normalize_embeddings=True scales each vector to unit length, which
    # lets us use a simple inner-product FAISS index to get cosine
    # similarity behaviour -- see vector_store.py for how this is used.
    try:
        # batch_size is explicit so large documents are embedded in batches.
        embeddings = model.encode(texts, normalize_embeddings=True,
                                  batch_size=32)
    except Exception as exc:
        raise EmbeddingError("Embedding the text failed.") from exc
    return np.asarray(embeddings, dtype="float32")
