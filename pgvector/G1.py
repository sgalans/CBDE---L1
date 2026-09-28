"""[G1] Generate the sentence embeddings and store them as pgvector vectors.

As P1: the script reads the sentences from the database, generates their
embeddings with the shared model and writes them back, now as ``vector(384)``:

    sentence_embeddings(sentence_id INTEGER PK -> sentences, embedding vector(384))

Unlike ``REAL[]``, ``vector(384)`` is a real vector type: PostgreSQL checks the
dimension, and pgvector provides distance operators and indexes for it. The
values are sent through pgvector's official psycopg2 adapter, which turns each
numpy array into one text literal ``'[x1,x2,...]'`` - the same approach as P1's
``'{...}'::real[]`` literal, so the two are directly comparable.

Phases timed separately (``results/G1.json``):

1. **read**     - ``SELECT`` of every sentence (one query; P1's own function,
   since reading text is plain SQL in both).
2. **generate** - ``common.generate_embeddings`` (the same as P1 and C1).
3. **store**    - INSERT per batch for every size in ``common.BATCH_SIZES``
   (``--repeats`` full loads each, as P1) plus an official load at
   ``DEFAULT_BATCH_SIZE``.
4. **index**    - build one HNSW index per metric (``vector_l2_ops`` and
   ``vector_cosine_ops``) on the loaded table, ``--repeats`` times. As in
   Chroma, an index serves one distance; unlike Chroma, the data is stored once.

Usage:
    python pgvector/G1.py
    python pgvector/G1.py --skip-grid
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIM,
    PGVECTOR_DATABASE,
    Timer,
    generate_embeddings,
    get_model,
    n_batches,
    pgvector_connect,
    print_stats,
    save_results,
    stats,
)
from postgres import P1  # noqa: E402  - reading the text is the same plain SQL

TABLE = "sentence_embeddings"

DDL = f"""
DROP TABLE IF EXISTS {TABLE};
CREATE TABLE {TABLE} (
    sentence_id INTEGER PRIMARY KEY REFERENCES sentences (sentence_id),
    embedding   vector({EMBEDDING_DIM}) NOT NULL
);
"""

INSERT_SQL = f"INSERT INTO {TABLE} (sentence_id, embedding) VALUES %s"

#: One HNSW index per metric: the operator class fixes the distance it serves.
INDEXES = {
    "l2": f"CREATE INDEX {TABLE}_l2_idx ON {TABLE} USING hnsw (embedding vector_l2_ops)",
    "cosine": f"CREATE INDEX {TABLE}_cosine_idx ON {TABLE} "
              f"USING hnsw (embedding vector_cosine_ops)",
}


def recreate_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def store(conn, ids: list[int], embeddings: np.ndarray, batch_size: int) -> list[float]:
    """Phase 3: recreate the table and insert every embedding; per-batch times."""
    recreate_table(conn)
    times: list[float] = []
    for start in range(0, len(ids), batch_size):
        stop = start + batch_size
        with Timer() as t:
            # The adapter turns each numpy array into a '[...]' literal here.
            rows = list(zip(ids[start:stop], embeddings[start:stop]))
            with conn.cursor() as cur:
                execute_values(cur, INSERT_SQL, rows, page_size=len(rows))
            conn.commit()
        times.append(t.elapsed)
    return times


def run_store_config(conn, ids, embeddings, batch_size: int, repeats: int) -> dict:
    batch_times: list[float] = []
    totals: list[float] = []
    for _ in range(repeats):
        times = store(conn, ids, embeddings, batch_size)
        batch_times.extend(times)
        totals.append(sum(times))
    total_stats = stats(totals)
    calls = n_batches(len(ids), batch_size)
    return {
        "batch_size": batch_size,
        "repeats": repeats,
        "batch_time": print_stats(f"store batch={batch_size} (per batch)", batch_times),
        "total_time": total_stats,
        "rows_per_second": len(ids) / total_stats["avg"],
        "db_calls": {"statements": calls, "commits": calls},
    }


def build_indexes(conn, repeats: int) -> dict:
    """Phase 4: time the build of each HNSW index; leaves both built for G2."""
    out = {}
    for metric, ddl in INDEXES.items():
        name = ddl.split()[2]  # CREATE INDEX <name> ON ...
        times: list[float] = []
        for _ in range(repeats):
            with conn.cursor() as cur:
                cur.execute(f"DROP INDEX IF EXISTS {name}")
            conn.commit()
            with Timer() as t, conn.cursor() as cur:
                cur.execute(ddl)
                conn.commit()
            times.append(t.elapsed)
        with conn.cursor() as cur:
            cur.execute("SELECT pg_relation_size(%s)", (name,))
            (size,) = cur.fetchone()
        out[metric] = {"build_time": print_stats(f"hnsw {metric} build", times),
                       "index_bytes": size}
    with conn.cursor() as cur:
        cur.execute(f"SELECT pg_relation_size('{TABLE}')")
        (table_size,) = cur.fetchone()
    out["table_bytes"] = table_size
    return out


def validate(conn, ids: list[int], embeddings: np.ndarray) -> dict:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*), min(vector_dims(embedding)), max(vector_dims(embedding)) "
                    f"FROM {TABLE}")
        count, min_dim, max_dim = cur.fetchone()
        # Norms computed by pgvector itself, inside the database.
        cur.execute(f"SELECT min(vector_norm(embedding)), max(vector_norm(embedding)) FROM {TABLE}")
        min_norm, max_norm = cur.fetchone()
        cur.execute(f"SELECT sentence_id, embedding FROM {TABLE} ORDER BY sentence_id")
        stored = dict(cur.fetchall())
    back = np.asarray([stored[i].to_numpy() for i in ids], dtype=np.float32)
    checks = {
        "rows": count, "dim_min": min_dim, "dim_max": max_dim,
        "norm_min": min_norm, "norm_max": max_norm,
        "roundtrip_max_abs_error": float(np.abs(back - embeddings).max()),
    }
    if not (count == len(ids) and min_dim == max_dim == EMBEDDING_DIM
            and checks["roundtrip_max_abs_error"] == 0.0):
        raise RuntimeError(f"validation failed: {checks}")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES),
                        help=f"grid of batch sizes for storing (default: {list(BATCH_SIZES)})")
    parser.add_argument("--repeats", type=int, default=3,
                        help="full loads per grid configuration and index builds (default: 3)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"batch size for generating and the official load "
                             f"(default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--skip-grid", action="store_true",
                        help="only run the official load")
    args = parser.parse_args()

    print("[G1] loading the embedding model (not timed)")
    get_model()

    results: dict = {"database": PGVECTOR_DATABASE, "table": TABLE,
                     "column_type": f"vector({EMBEDDING_DIM})", "dim": EMBEDDING_DIM}

    with closing(pgvector_connect()) as conn:
        register_vector(conn)

        ids, texts, read_time = P1.read_sentences(conn, loader="pgvector/G0.py")
        print(f"[G1] read {len(ids)} sentences in {read_time:.3f}s\n")
        results["n_sentences"] = len(ids)
        results["read"] = {"time": read_time, "db_calls": {"statements": 1}}

        print("[G1] generating embeddings")
        embeddings, gen_times = generate_embeddings(texts, args.batch_size)
        results["generate"] = {
            "batch_size": args.batch_size,
            "batch_time": print_stats(f"generate batch={args.batch_size} (per batch)",
                                      gen_times),
        }

        if not args.skip_grid:
            print("\n[G1] storing: batch-size grid")
            results["grid"] = [run_store_config(conn, ids, embeddings, size, args.repeats)
                               for size in args.batch_sizes]

        # Last, so the table stays populated (and indexed) for G2.
        print("\n[G1] storing: official load")
        results["official"] = run_store_config(conn, ids, embeddings, args.batch_size,
                                               repeats=1)
        results["validation"] = validate(conn, ids, embeddings)

        print("\n[G1] building HNSW indexes")
        results["index"] = build_indexes(conn, args.repeats)

    v = results["validation"]
    print(f"\n[G1] {v['rows']} vectors of {v['dim_min']} dims; norms in "
          f"[{v['norm_min']:.6f}, {v['norm_max']:.6f}]; round-trip error "
          f"{v['roundtrip_max_abs_error']}")
    save_results("G1", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
