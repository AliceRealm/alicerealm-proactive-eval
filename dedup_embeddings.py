"""
dedup_embeddings.py — semantic deduplication using embeddings.

Mirrors EigenFlux's two-phase dedup:
  1. Exact content hash (fast, free)
  2. Embedding cosine similarity >= threshold (catches same-story-different-source)

Uses the same embedding provider as EigenFlux for consistency.
"""

import hashlib
import json
import math
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# Config (matches EigenFlux pipeline)
# ---------------------------------------------------------------------------

EMBEDDING_BASE_URL = os.environ.get(
    "EMBEDDING_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
EMBEDDING_API_KEY = os.environ.get(
    "EMBEDDING_API_KEY",
    "sk-8727c8621e9d414ba9082e1773c33878",
)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-v4")
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))
SIM_THRESHOLD = float(os.environ.get("SIM_THRESHOLD", "0.70"))

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "data" / ".embedding_cache"


# ---------------------------------------------------------------------------
# Embedding API
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    """Call the OpenAI-compatible embedding endpoint."""
    url = f"{EMBEDDING_BASE_URL.rstrip('/')}/embeddings"
    body = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": text[:8000],  # truncate to avoid token limits
        "dimensions": EMBEDDING_DIMENSIONS,
    }).encode()
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Authorization": f"Bearer {EMBEDDING_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    resp = urllib.request.urlopen(req, timeout=30)
    data = json.loads(resp.read())
    return data["data"][0]["embedding"]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Cache (avoid re-embedding the same content across runs)
# ---------------------------------------------------------------------------

def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_cached_embedding(text: str) -> list[float] | None:
    path = CACHE_DIR / f"{_cache_key(text)}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_cached_embedding(text: str, embedding: list[float]):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{_cache_key(text)}.json"
    path.write_text(json.dumps(embedding))


def get_embedding_cached(text: str) -> list[float]:
    cached = _load_cached_embedding(text)
    if cached is not None:
        return cached
    emb = get_embedding(text)
    _save_cached_embedding(text, emb)
    return emb


# ---------------------------------------------------------------------------
# Content hash (phase 1: exact dedup)
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    """Normalize whitespace and compute SHA-256 hash."""
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Two-phase dedup
# ---------------------------------------------------------------------------

def semantic_dedup(items: list[dict], content_key: str = "content",
                   threshold: float = SIM_THRESHOLD,
                   verbose: bool = False) -> list[dict]:
    """
    Two-phase dedup matching EigenFlux logic:
      1. Content hash — exact duplicates discarded
      2. Embedding similarity — near-duplicates (>= threshold) discarded,
         keeping the earliest occurrence.

    Returns a filtered list preserving original order.
    """
    # Phase 1: exact hash dedup
    seen_hashes: set[str] = set()
    phase1: list[dict] = []
    hash_dupes = 0
    for item in items:
        h = content_hash(item.get(content_key, ""))
        if h in seen_hashes:
            hash_dupes += 1
            continue
        seen_hashes.add(h)
        phase1.append(item)

    if verbose and hash_dupes:
        print(f"    dedup phase1 (hash): {hash_dupes} exact duplicates removed")

    # Phase 2: embedding similarity dedup
    # Step 2a: pre-fetch all embeddings in parallel (API calls + cache)
    total = len(phase1)
    texts = [item.get(content_key, "") for item in phase1]
    embeddings: list[list[float] | None] = [None] * total

    # Find which ones need API calls (not cached)
    uncached = []
    for i, text in enumerate(texts):
        if not text.strip():
            continue
        cached = _load_cached_embedding(text)
        if cached is not None:
            embeddings[i] = cached
        else:
            uncached.append(i)

    if verbose:
        cached_count = total - len(uncached)
        print(f"    dedup phase2: {total} items, {cached_count} cached, "
              f"{len(uncached)} need embedding API", flush=True)

    # Parallel fetch for uncached
    if uncached:
        done = 0
        with ThreadPoolExecutor(max_workers=8) as pool:
            future_to_idx = {
                pool.submit(get_embedding_cached, texts[i]): i
                for i in uncached
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embeddings[idx] = future.result()
                except Exception:
                    pass  # will be treated as non-dup
                done += 1
                if verbose and done % 20 == 0:
                    print(f"    dedup phase2: embedding {done}/{len(uncached)}",
                          flush=True)

    # Step 2b: serial comparison (order matters — keep earliest)
    kept: list[dict] = []
    kept_embeddings: list[list[float]] = []
    embed_dupes = 0

    for idx, item in enumerate(phase1):
        emb = embeddings[idx]
        if emb is None:
            kept.append(item)
            continue

        is_dup = False
        for existing_emb in kept_embeddings:
            if cosine_similarity(emb, existing_emb) >= threshold:
                is_dup = True
                break

        if is_dup:
            embed_dupes += 1
            continue

        kept.append(item)
        kept_embeddings.append(emb)

    if verbose:
        print(f"    dedup phase2 (embedding): {total} items → {len(kept)} kept, "
              f"{embed_dupes} semantic duplicates removed (threshold={threshold})")

    return kept
