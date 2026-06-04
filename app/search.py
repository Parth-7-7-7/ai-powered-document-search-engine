"""
Search layer: keyword (BM25), semantic (FAISS), and hybrid (RRF) search.

Usage:
    from app.search import LegalSearchEngine
    engine = LegalSearchEngine()          # loads indexes once
    results = engine.search("termination clause", mode="hybrid", top_k=10)
"""

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR  = Path("data")
INDEX_DIR = Path("indexes")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K       = 60        # RRF constant — higher = smoother rank fusion


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


class LegalSearchEngine:
    """
    Loads all indexes once and exposes keyword / semantic / hybrid search.
    All three modes return a list of result dicts with identical keys so the
    UI can render them uniformly.
    """

    def __init__(self):
        print("Loading search engine artifacts...")

        # Chunks
        with open(DATA_DIR / "chunks.json", encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        # BM25
        with open(INDEX_DIR / "bm25.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25: BM25Okapi = data["bm25"]

        # FAISS
        self.faiss_index = faiss.read_index(str(INDEX_DIR / "faiss.index"))

        # Embedding model
        self.model = SentenceTransformer(EMBED_MODEL)

        print(f"  Chunks loaded  : {len(self.chunks)}")
        print(f"  FAISS vectors  : {self.faiss_index.ntotal}")
        print("Search engine ready.\n")

    # ── Keyword search (BM25) ─────────────────────────────────────────────────

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict]:
        tokens = _tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices):
            if scores[idx] == 0:
                break
            results.append(self._make_result(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank + 1,
                mode="keyword",
            ))
        return results

    # ── Semantic search (FAISS cosine similarity) ─────────────────────────────

    def semantic_search(self, query: str, top_k: int = 10) -> list[dict]:
        query_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_vec)
        distances, indices = self.faiss_index.search(query_vec, top_k)

        results = []
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx == -1:
                break
            results.append(self._make_result(
                chunk=self.chunks[idx],
                score=float(dist),
                rank=rank + 1,
                mode="semantic",
            ))
        return results

    # ── Hybrid search (Reciprocal Rank Fusion) ────────────────────────────────

    def hybrid_search(self, query: str, top_k: int = 10) -> list[dict]:
        # Run both searches at a larger pool to get stable rankings
        pool = max(top_k * 3, 50)
        kw_results  = self.keyword_search(query,  top_k=pool)
        sem_results = self.semantic_search(query, top_k=pool)

        # RRF score: 1/(k + rank) accumulated per chunk_id
        rrf_scores: dict[str, float] = {}

        for result in kw_results:
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + result["rank"])

        for result in sem_results:
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + result["rank"])

        # Build a lookup: chunk_id → chunk dict
        chunk_lookup = {c["chunk_id"]: c for c in self.chunks}

        # Sort by RRF score descending, take top_k
        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[:top_k]

        results = []
        for rank, cid in enumerate(sorted_ids):
            if cid not in chunk_lookup:
                continue
            results.append(self._make_result(
                chunk=chunk_lookup[cid],
                score=round(rrf_scores[cid], 6),
                rank=rank + 1,
                mode="hybrid",
            ))
        return results

    # ── Unified search dispatcher ─────────────────────────────────────────────

    def search(self, query: str, mode: str = "hybrid", top_k: int = 10) -> list[dict]:
        query = query.strip()
        if not query:
            return []
        if mode == "keyword":
            return self.keyword_search(query, top_k)
        if mode == "semantic":
            return self.semantic_search(query, top_k)
        return self.hybrid_search(query, top_k)

    # ── Result dict builder ───────────────────────────────────────────────────

    @staticmethod
    def _make_result(chunk: dict, score: float, rank: int, mode: str) -> dict:
        return {
            "chunk_id"  : chunk["chunk_id"],
            "text"      : chunk["text"],
            "title"     : chunk["title"],
            "filename"  : chunk["filename"],
            "doc_type"  : chunk["doc_type"],
            "source"    : chunk["source"],
            "score"     : score,
            "rank"      : rank,
            "mode"      : mode,
        }
