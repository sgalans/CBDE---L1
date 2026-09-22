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
DEFAULT_BATCH_SIZE = 500

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
        "host": os.getenv("PGHOST", "localhost"),
        "port": int(os.getenv("PGPORT", "5432")),
        "dbname": os.getenv("PGDATABASE", dbname),
        "user": os.getenv("PGUSER", "cbde"),
        "password": os.getenv("PGPASSWORD", "cbde"),
    }


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
