"""Lightweight Vector Store and RAG retrieval tool.

Provides semantic search over a local or external knowledge base.
Supports both OpenAI dense embeddings and a zero-dependency TF-IDF/cosine similarity
retriever for fast, offline, or mock execution.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Document:
    """Represents a document in the vector store."""
    id: str
    title: str
    content: str
    category: str = "general"
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None


@dataclass
class SearchResult:
    """Represents a scored search result."""
    document: Document
    score: float


class LightweightVectorStore:
    """In-memory vector store with cosine similarity search.

    Implements dual embedding backends:
    1. OpenAI Embeddings (`text-embedding-3-small`) when API key is available.
    2. Zero-dependency TF-IDF Cosine Similarity engine for offline/fallback operation.
    """

    def __init__(self, use_openai: bool = False, api_key: Optional[str] = None) -> None:
        self.use_openai = use_openai and bool(api_key or os.getenv("OPENAI_API_KEY"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.documents: List[Document] = []
        self._doc_vectors: Optional[np.ndarray] = None
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}

    def add_documents(self, docs: List[Document]) -> None:
        """Add documents to the store and build the vector index."""
        self.documents.extend(docs)
        self._build_index()

    def load_from_json(self, file_path: Path | str) -> int:
        """Load documents from a JSON file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        docs = [
            Document(
                id=item.get("id", f"doc_{i}"),
                title=item.get("title", "Untitled Document"),
                content=item.get("content", ""),
                category=item.get("category", "general"),
                metadata=item.get("metadata", {}),
            )
            for i, item in enumerate(raw_docs)
        ]
        self.add_documents(docs)
        return len(docs)

    def _build_index(self) -> None:
        """Builds vector representations for all stored documents."""
        if not self.documents:
            return

        if self.use_openai and self.api_key:
            try:
                self._build_openai_index()
                return
            except Exception as e:
                # Graceful fallback to TF-IDF if OpenAI API fails or is unreachable
                print(f"[RAG Note] OpenAI embedding failed ({e}); falling back to local TF-IDF retriever.")
                self.use_openai = False

        self._build_tfidf_index()

    def _build_openai_index(self) -> None:
        """Generates dense embeddings using OpenAI's API."""
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        texts = [f"{doc.title}\n{doc.content}" for doc in self.documents]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        self._doc_vectors = np.array(embeddings, dtype=np.float32)

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer for TF-IDF."""
        return re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower())

    def _build_tfidf_index(self) -> None:
        """Builds an in-memory TF-IDF index using numpy."""
        all_tokens_per_doc: List[List[str]] = []
        for doc in self.documents:
            tokens = self._tokenize(f"{doc.title} {doc.content}")
            all_tokens_per_doc.append(tokens)

        # Build vocabulary
        vocab: Dict[str, int] = {}
        for tokens in all_tokens_per_doc:
            for token in tokens:
                if token not in vocab:
                    vocab[token] = len(vocab)
        self._vocabulary = vocab

        num_docs = len(self.documents)
        # Compute IDF
        self._idf = {}
        for token, _ in vocab.items():
            doc_freq = sum(1 for tokens in all_tokens_per_doc if token in tokens)
            self._idf[token] = math.log((num_docs + 1) / (doc_freq + 1)) + 1.0

        # Compute TF-IDF matrix
        vectors: List[np.ndarray] = []
        for tokens in all_tokens_per_doc:
            vec = self._vectorize_tokens(tokens)
            vectors.append(vec)

        self._doc_vectors = np.array(vectors, dtype=np.float32)

    def _vectorize_tokens(self, tokens: List[str]) -> np.ndarray:
        """Converts token list to normalized TF-IDF vector."""
        vec = np.zeros(len(self._vocabulary), dtype=np.float32)
        if not tokens:
            return vec

        total_tokens = len(tokens)
        token_counts: Dict[str, int] = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1

        for token, count in token_counts.items():
            if token in self._vocabulary:
                idx = self._vocabulary[token]
                tf = count / total_tokens
                idf = self._idf.get(token, 1.0)
                vec[idx] = tf * idf

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Performs cosine similarity search against indexed documents."""
        if not self.documents or self._doc_vectors is None or len(self._doc_vectors) == 0:
            return []

        if self.use_openai and self.api_key:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=self.api_key)
                resp = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=[query],
                )
                query_vec = np.array(resp.data[0].embedding, dtype=np.float32)
                query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            except Exception:
                # Fallback to TF-IDF query vector
                tokens = self._tokenize(query)
                query_vec = self._vectorize_tokens(tokens)
        else:
            tokens = self._tokenize(query)
            query_vec = self._vectorize_tokens(tokens)

        # Compute cosine similarities
        if np.linalg.norm(query_vec) == 0:
            return []

        norms = np.linalg.norm(self._doc_vectors, axis=1)
        norms = np.where(norms == 0, 1e-10, norms)
        normalized_docs = self._doc_vectors / norms[:, np.newaxis]
        scores = np.dot(normalized_docs, query_vec)

        # Get top-k indices
        top_k = min(top_k, len(self.documents))
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            SearchResult(document=self.documents[idx], score=float(scores[idx]))
            for idx in ranked_indices
            if scores[idx] > 0.05  # minimum relevance filter
        ]
        return results


# Global singleton instance for easy import and tool invocation
_default_kb_path = Path(__file__).parent.parent / "data" / "knowledge_base.json"
vector_store = LightweightVectorStore(use_openai=False)

if _default_kb_path.exists():
    vector_store.load_from_json(_default_kb_path)


def initialize_rag(use_openai_embeddings: bool = False, api_key: Optional[str] = None) -> None:
    """Configures the RAG store with optional OpenAI embeddings."""
    global vector_store
    vector_store = LightweightVectorStore(use_openai=use_openai_embeddings, api_key=api_key)
    if _default_kb_path.exists():
        vector_store.load_from_json(_default_kb_path)


def search_documents(query: str, top_k: int = 3) -> str:
    """Searches the knowledge base for documents matching the query.

    Args:
        query: The search question or keywords.
        top_k: Maximum number of relevant chunks to retrieve (default: 3).

    Returns:
        Formatted string containing relevant excerpts, source titles, and relevance scores.
    """
    if not query or not query.strip():
        return "Search error: query string cannot be empty."

    try:
        results = vector_store.search(query=query.strip(), top_k=top_k)
        if not results:
            return f"No relevant documents found for query: '{query}'."

        formatted_outputs = [f"Found {len(results)} relevant document(s):"]
        for i, res in enumerate(results, start=1):
            doc = res.document
            score_pct = f"{res.score * 100:.1f}%"
            formatted_outputs.append(
                f"\n--- [Document {i}] {doc.title} (Relevance: {score_pct}) ---\n"
                f"Category: {doc.category}\n"
                f"Content: {doc.content}"
            )
        return "\n".join(formatted_outputs)
    except Exception as e:
        return f"Retrieval error: {str(e)}"


# Tool specification for OpenAI / Anthropic function calling
RAG_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "Retrieves relevant technical documentation, policies, and architecture guides "
            "from the vector store. Use this whenever the user asks about system architecture, "
            "RAG evaluation metrics, company policies, remote work, stipends, or AI engineering guidelines."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query or concepts to look up in the knowledge base.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of top matching documents to retrieve (default: 3).",
                    "default": 3,
                },
            },
            "required": ["query"],
        },
    },
}
