"""Shared utilities for CBDE Lab 1 (impedance mismatch of vector data).

Everything that must stay identical across PostgreSQL, Chroma and pgvector
lives here: the dataset split, the 10 query sentences, the embedding model,
the timing helpers and the statistics (min / max / avg / std) required by the
lab statement.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

SENTENCES_PATH = DATA_DIR / "sentences.parquet"
CHUNKS_PATH = DATA_DIR / "chunks.parquet"
QUERIES_PATH = DATA_DIR / "queries.json"

# --------------------------------------------------------------------------
# Experiment constants (shared by every system)
# --------------------------------------------------------------------------

#: Lightweight transformer recommended by the lab statement.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
#: Output dimensionality of all-MiniLM-L6-v2.
EMBEDDING_DIM = 384

#: Sentences kept from bookCorpus.
TARGET_SENTENCES = 10_000
#: Sentences grouped into a single chunk.
SENTENCES_PER_CHUNK = 20
#: Minimum / maximum sentence length (characters) kept during cleaning.
MIN_SENTENCE_CHARS = 40
MAX_SENTENCE_CHARS = 400
#: Minimum number of words kept during cleaning.
MIN_SENTENCE_WORDS = 6

#: Number of query sentences and neighbours retrieved for each of them.
N_QUERIES = 10
TOP_K = 2

#: Seed used everywhere a deterministic choice is made.
RANDOM_SEED = 42

#: Distance metrics used in *all three* systems.
METRICS = ("l2", "cosine")

#: Batch sizes explored when measuring insertion performance.
BATCH_SIZES = (1, 10, 50, 100, 500, 1000, 2000)
#: Batch size used for the "official" runs once the grid has been explored.
#: Chosen from P0's grid: 28 % faster than 500, while 2000 only saves another
#: 15 % and would leave just 5 batches to compute min/max/avg/std over.
DEFAULT_BATCH_SIZE = 1000

# --------------------------------------------------------------------------
# PostgreSQL connection
# --------------------------------------------------------------------------

#: Database used for the plain-PostgreSQL part (P0-P2). No pgvector here.
PG_DATABASE = "cbde"
#: Separate database for the optional pgvector part (G0-G2).
PGVECTOR_DATABASE = "cbde_pgvector"


def pg_config(dbname: str = PG_DATABASE) -> dict[str, Any]:
    """Connection parameters for psycopg2, overridable through the environment."""
    return {
        # 127.0.0.1, not "localhost": on Windows "localhost" resolves to ::1
        # first, and Docker Desktop's IPv6 port forwarding stalls ~45 ms on
        # every message of roughly 32-70 KB (a 500-row INSERT), which would
        # be measured as if it were PostgreSQL's cost.
        "host": os.getenv("PGHOST", "127.0.0.1"),
        "port": int(os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("PGDATABASE", dbname),
        "user": os.getenv("PGUSER", "cbde"),
        "password": os.getenv("PGPASSWORD", "cbde"),
    }


def pgvector_connect():
    """Connection to the pgvector database, with the extension enabled.

    The extension lives only in ``cbde_pgvector``: the plain-PostgreSQL part
    (database ``cbde``) never enables it. Fails with a clear message if the
    database was not created (see README).
    """
    import psycopg2

    try:
        conn = psycopg2.connect(**pg_config(PGVECTOR_DATABASE))
    except psycopg2.OperationalError as exc:
        if "does not exist" in str(exc):
            raise RuntimeError(
                f"database '{PGVECTOR_DATABASE}' does not exist. Create it once with:\n"
                f'  docker exec cbde_postgres psql -U cbde -d cbde -c '
                f'"CREATE DATABASE {PGVECTOR_DATABASE} OWNER cbde"') from exc
        raise
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn.commit()
    return conn


# --------------------------------------------------------------------------
# Chroma
# --------------------------------------------------------------------------

#: On-disk store of the embedded (in-process) Chroma client; git-ignored.
CHROMA_PATH = ROOT / "chroma" / "chroma_db"
#: One collection per metric: Chroma fixes the distance when a collection is
#: created, so the same data is stored twice.
CHROMA_COLLECTIONS = {"l2": "sentences_l2", "cosine": "sentences_cosine"}


def chroma_client():
    """Persistent in-process Chroma client, with telemetry off.

    Telemetry would make network calls from inside the timed sections.
    """
    import chromadb  # heavy import
    from chromadb.config import Settings

    return chromadb.PersistentClient(path=str(CHROMA_PATH),
                                     settings=Settings(anonymized_telemetry=False))


def chroma_collections(client) -> dict:
    """The existing collections, keyed by metric; fails clearly if C0 was not run."""
    collections = {}
    for space, name in CHROMA_COLLECTIONS.items():
        try:
            collections[space] = client.get_collection(name)
        except Exception as exc:  # noqa: BLE001 - chromadb raises several types
            raise RuntimeError(f"collection '{name}' not found. Run chroma/C0.py first.") from exc
    return collections


# --------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------


def load_sentences() -> pd.DataFrame:
    """The per-sentence split: ``sentence_id``, ``chunk_id``, ``pos``, ``text``.

    This is the table every system must ingest, so that the split is identical
    in PostgreSQL, Chroma and pgvector.
    """
    _require(SENTENCES_PATH)
    return pd.read_parquet(SENTENCES_PATH)


def load_chunks() -> pd.DataFrame:
    """The chunked corpus: ``chunk_id``, ``n_sentences``, ``text``."""
    _require(CHUNKS_PATH)
    return pd.read_parquet(CHUNKS_PATH)


def load_queries() -> list[dict[str, Any]]:
    """The 10 fixed query sentences (``sentence_id``, ``chunk_id``, ``text``)."""
    _require(QUERIES_PATH)
    with QUERIES_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)["queries"]


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python data/prepare_corpus.py` first."
        )


def exact_reference() -> dict[str, dict[int, list[int]]] | None:
    """P2's exact top-k neighbour ids per metric and query, or None if P2 has
    not been run. It is the reference for the recall of C2 and G2."""
    path = RESULTS_DIR / "P2.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        p2 = json.load(fh)
    return {m: {a["sentence_id"]: [n["sentence_id"] for n in a[m]] for a in p2["answers"]}
            for m in METRICS}


def recall_at_k(reference: dict, answers: dict, queries: list[dict], k: int) -> dict[str, float]:
    """Share of the exact neighbours (``reference``) also returned in
    ``answers`` ({metric: {sentence_id: [(neighbour_id, distance), ...]}})."""
    return {
        m: sum(len(set(reference[m][q["sentence_id"]])
                   & {n for n, _ in answers[m][q["sentence_id"]]}) for q in queries)
           / (k * len(queries))
        for m in METRICS
    }


# --------------------------------------------------------------------------
# Embedding model
# --------------------------------------------------------------------------

_MODEL = None


def get_model():
    """Lazily load all-MiniLM-L6-v2 (kept as a singleton across a script run).

    The model is loaded *outside* any timed section on purpose: we measure the
    cost of generating and storing embeddings, not the cost of loading weights
    from disk.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # heavy import

        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _MODEL


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------


class Timer:
    """Context manager measuring wall-clock time with a monotonic clock.

    >>> with Timer() as t:
    ...     do_something()
    >>> t.elapsed  # seconds
    """

    __slots__ = ("_start", "elapsed")

    def __enter__(self) -> "Timer":
        self.elapsed = 0.0
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.elapsed = time.perf_counter() - self._start


def stats(values: Sequence[float]) -> dict[str, float]:
    """min / max / avg / std (plus n and total) for a series of measurements."""
    values = list(values)
    if not values:
        return {"n": 0, "min": 0.0, "max": 0.0, "avg": 0.0, "std": 0.0, "total": 0.0}
    return {
        "n": len(values),
        "min": min(values),
        "max": max(values),
        "avg": statistics.fmean(values),
        # Sample standard deviation; undefined for a single measurement.
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "total": sum(values),
    }


def print_stats(label: str, values: Sequence[float], unit: str = "s") -> dict[str, float]:
    """Print a one-line summary of a series and return its statistics."""
    s = stats(values)
    print(
        f"{label:<38} n={s['n']:>6}  "
        f"min={s['min']:.6f}{unit}  max={s['max']:.6f}{unit}  "
        f"avg={s['avg']:.6f}{unit}  std={s['std']:.6f}{unit}  "
        f"total={s['total']:.3f}{unit}"
    )
    return s


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


def batched(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    """Yield consecutive slices of ``items`` of at most ``size`` elements."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def n_batches(total: int, size: int) -> int:
    return (total + size - 1) // size


def generate_embeddings(texts: list[str], batch_size: int):
    """Encode ``texts`` batch by batch with the shared model.

    Returns ``(embeddings, times)``: a float32 array of shape
    ``(len(texts), EMBEDDING_DIM)`` and the time of each batch. A warm-up call
    runs first, outside the timing, so no batch pays one-off initialisation.
    Used by P1, C1 and G1 so the three systems store identical vectors.
    """
    import numpy as np

    model = get_model()
    model.encode(texts[:8])

    parts = []
    times: list[float] = []
    for batch in batched(texts, batch_size):
        with Timer() as t:
            parts.append(model.encode(list(batch), convert_to_numpy=True))
        times.append(t.elapsed)
    embeddings = np.vstack(parts).astype(np.float32, copy=False)
    if embeddings.shape != (len(texts), EMBEDDING_DIM):
        raise RuntimeError(f"unexpected embedding shape {embeddings.shape}")
    return embeddings, times


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


def save_results(name: str, payload: dict[str, Any]) -> Path:
    """Persist a script's measurements to ``results/<name>.json``."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    payload = {
        "script": name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        **payload,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\n-> results written to {path}")
    return path
