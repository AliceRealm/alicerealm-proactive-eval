"""Semantic deduplication using embeddings.

Mirrors EigenFlux's two-phase dedup:
  1. Exact content hash (fast, free)
  2. Embedding cosine similarity >= threshold (catches same-story-different-source)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from proactive_eval.types import SignalItem
from proactive_eval.config import SCRIPT_DIR

# ---------------------------------------------------------------------------
# Config
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

CACHE_DIR = SCRIPT_DIR / "data" / ".embedding_cache"


# ---------------------------------------------------------------------------
# Embedding API
# ---------------------------------------------------------------------------

def get_embedding(text: str) -> list[float]:
    url = f"{EMBEDDING_BASE_URL.rstrip('/')}/embeddings"
    body = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": text[:8000],
        "dimensions": EMBEDDING_DIMENSIONS,
    }).encode()
    req = urllib.request.Request(
        url, data=body,
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
# Cache
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
# Content hash
# ---------------------------------------------------------------------------

def content_hash(text: str) -> str:
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Two-phase dedup (works on both dicts and SignalItems)
# ---------------------------------------------------------------------------

def semantic_dedup(items: list, content_key: str = "content",
                   threshold: float = SIM_THRESHOLD,
                   verbose: bool = False) -> list:
    """Two-phase dedup. Works with dicts or SignalItem objects."""

    def _get_content(item) -> str:
        if isinstance(item, SignalItem):
            return item.content
        return item.get(content_key, "")

    # Phase 1: exact hash dedup
    seen_hashes: set[str] = set()
    phase1 = []
    hash_dupes = 0
    for item in items:
        h = content_hash(_get_content(item))
        if h in seen_hashes:
            hash_dupes += 1
            continue
        seen_hashes.add(h)
        phase1.append(item)

    if verbose and hash_dupes:
        print(f"    dedup phase1 (hash): {hash_dupes} exact duplicates removed")

    # Phase 2: embedding similarity dedup
    total = len(phase1)
    texts = [_get_content(item) for item in phase1]
    embeddings: list[list[float] | None] = [None] * total

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
                    pass
                done += 1
                if verbose and done % 20 == 0:
                    print(f"    dedup phase2: embedding {done}/{len(uncached)}", flush=True)

    kept = []
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
        print(f"    dedup phase2 (embedding): {total} items -> {len(kept)} kept, "
              f"{embed_dupes} semantic duplicates removed (threshold={threshold})")

    return kept


# ---------------------------------------------------------------------------
# Dedup for SignalItems (convenience)
# ---------------------------------------------------------------------------

def dedup_signal_items(items: list[SignalItem],
                       semantic: bool = True,
                       verbose: bool = False) -> list[SignalItem]:
    """Deduplicate a list of SignalItems.
    Phase 1: exact dedup by item_id (eigenflux) or url (search).
    Phase 2: semantic dedup on search items only (eigenflux has server-side dedup)."""

    # Phase 1: exact dedup by key
    seen: set[str] = set()
    deduped: list[SignalItem] = []
    for item in items:
        if item.source == "search":
            key = f"search:{item.url or item.content}"
        else:
            key = f"{item.source}:{item.item_id or item.content}"
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    if not semantic:
        return deduped

    # Phase 2: semantic dedup on search items only
    search_items = [it for it in deduped if it.source == "search"]
    other_items = [it for it in deduped if it.source != "search"]
    if search_items:
        before = len(search_items)
        search_items = semantic_dedup(search_items, verbose=verbose)
        after = len(search_items)
        if verbose and before != after:
            print(f"    semantic dedup: {before} -> {after} (-{before - after})")

    result = search_items + other_items
    result.sort(key=lambda x: x.received_at)
    return result
