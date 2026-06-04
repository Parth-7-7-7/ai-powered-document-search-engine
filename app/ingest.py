"""
Ingestion pipeline: loads all documents from documents/, chunks them,
builds a BM25 index and a FAISS vector index, then saves both to indexes/.

Run once before starting the search app:
    python app/ingest.py
"""

import json
import pickle
import re
import time
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

DOCS_DIR   = Path("documents")
INDEX_DIR  = Path("indexes")
DATA_DIR   = Path("data")

INDEX_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"


# ─── 1. Load documents ────────────────────────────────────────────────────────

def load_documents():
    meta_path = DOCS_DIR / "metadata.json"
    with open(meta_path, encoding="utf-8") as f:
        metadata = json.load(f)

    docs = []
    for entry in metadata:
        fpath = DOCS_DIR / entry["filename"]
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        text = fpath.read_text(encoding="utf-8")
        docs.append({
            "text": text,
            "filename": entry["filename"],
            "title": entry["title"],
            "type": entry["type"],
            "source": entry["source"],
        })
    return docs


# ─── 2. Chunk documents ───────────────────────────────────────────────────────

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for j, split in enumerate(splits):
            split = split.strip()
            if len(split) < 50:
                continue
            chunks.append({
                "chunk_id"    : f"{doc['filename']}__chunk_{j}",
                "text"        : split,
                "filename"    : doc["filename"],
                "title"       : doc["title"],
                "doc_type"    : doc["type"],
                "source"      : doc["source"],
                "chunk_index" : j,
            })
    return chunks


# ─── 3. Build BM25 index ─────────────────────────────────────────────────────

def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())


def build_bm25(chunks):
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25, tokenized


# ─── 4. Build FAISS index ────────────────────────────────────────────────────

def build_faiss(chunks, model):
    texts = [c["text"] for c in chunks]
    print(f"  Encoding {len(texts)} chunks with {EMBED_MODEL}...")
    t0 = time.time()
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    print(f"  Encoding done in {time.time()-t0:.1f}s")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)               # inner product = cosine sim on normalised vecs
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index, embeddings


# ─── 5. Save everything ──────────────────────────────────────────────────────

def save_artifacts(chunks, bm25, tokenized, faiss_index, embeddings):
    # chunks + tokenized corpus
    with open(DATA_DIR / "chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2)

    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized": tokenized}, f)

    faiss.write_index(faiss_index, str(INDEX_DIR / "faiss.index"))

    np.save(str(INDEX_DIR / "embeddings.npy"), embeddings)

    print(f"  chunks.json      -> {DATA_DIR / 'chunks.json'}")
    print(f"  bm25.pkl         -> {INDEX_DIR / 'bm25.pkl'}")
    print(f"  faiss.index      -> {INDEX_DIR / 'faiss.index'}")
    print(f"  embeddings.npy   -> {INDEX_DIR / 'embeddings.npy'}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Loading documents")
    print("=" * 60)
    docs = load_documents()
    print(f"  Loaded {len(docs)} documents")

    print("\n" + "=" * 60)
    print("STEP 2: Chunking documents")
    print("=" * 60)
    chunks = chunk_documents(docs)
    print(f"  Created {len(chunks)} chunks from {len(docs)} documents")
    print(f"  Avg chunk size: {sum(len(c['text']) for c in chunks)//len(chunks)} chars")

    print("\n" + "=" * 60)
    print("STEP 3: Building BM25 index")
    print("=" * 60)
    bm25, tokenized = build_bm25(chunks)
    print(f"  BM25 index built over {len(chunks)} chunks")

    print("\n" + "=" * 60)
    print("STEP 4: Building FAISS vector index")
    print("=" * 60)
    model = SentenceTransformer(EMBED_MODEL)
    faiss_index, embeddings = build_faiss(chunks, model)
    print(f"  FAISS index: {faiss_index.ntotal} vectors, dim={embeddings.shape[1]}")

    print("\n" + "=" * 60)
    print("STEP 5: Saving artifacts")
    print("=" * 60)
    save_artifacts(chunks, bm25, tokenized, faiss_index, embeddings)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print(f"  Documents : {len(docs)}")
    print(f"  Chunks    : {len(chunks)}")
    print(f"  Ready for search.")
    print("=" * 60)
