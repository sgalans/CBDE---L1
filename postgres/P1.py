"""[P1] Generate the sentence embeddings and store them in plain PostgreSQL.

As the statement asks, the script *connects to the database*, reads the
sentences loaded by P0, generates their embeddings in Python and writes them
back. Without pgvector there is no vector type, so each embedding is stored as
a native ``REAL[]`` (float4, the precision the model produces) in its own table:

    sentence_embeddings(sentence_id INTEGER PK -> sentences, embedding REAL[])

A separate table (instead of ``ALTER TABLE sentences ADD COLUMN`` + ``UPDATE``)
keeps storing the embeddings a pure INSERT: in PostgreSQL an UPDATE writes a
whole new row version and leaves a dead one behind, which would blur what is
being measured.

Three phases are timed separately (``results/P1.json``):

1. **read**     - ``SELECT`` of every sentence from PostgreSQL into Python.
2. **generate** - ``model.encode`` per batch of ``DEFAULT_BATCH_SIZE`` sentences.
3. **store**    - numpy -> text -> INSERT, per batch, for every size in
   ``common.BATCH_SIZES`` (as in P0) plus a final official load at
   ``DEFAULT_BATCH_SIZE``. The conversion is timed together with the INSERT on
   purpose: it is the cost of the impedance mismatch.

PostgreSQL has no binary vector type to receive a numpy array, so every float
travels as text and is parsed again on the server. Two ways of doing that are
measured:

* ``literal`` (used for the grid and the official load): each vector is sent
  as a single array-literal string ``'{0.01,-0.2,...}'`` that the server casts
  to ``REAL[]`` in one step.
* ``adapt`` (reference, at ``DEFAULT_BATCH_SIZE``): psycopg2's default
  adaptation of a Python list, ``ARRAY[0.01,-0.2,...]``. Every element becomes
  a separate float8 literal in the SQL text that the server must parse and cast
  one by one; it is several times slower.

Usage:
    python postgres/P1.py                  # read + generate + grid + official
    python postgres/P1.py --repeats 1      # quicker grid
    python postgres/P1.py --skip-grid      # official load only
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIM,
    Timer,
    generate_embeddings,
    get_model,
    n_batches,
    pg_config,
    print_stats,
    save_results,
    stats,
)

TABLE = "sentence_embeddings"

DDL = f"""
DROP TABLE IF EXISTS {TABLE};
CREATE TABLE {TABLE} (
    sentence_id INTEGER PRIMARY KEY REFERENCES sentences (sentence_id),
    embedding   REAL[]  NOT NULL
);
"""

INSERT_SQL = f"INSERT INTO {TABLE} (sentence_id, embedding) VALUES %s"
# The second value is either an array-literal string ('literal') or an
# ARRAY[...] expression ('adapt'); the cast turns both into float4.
INSERT_TEMPLATE = "(%s, %s::real[])"


def to_literal(vector: list[float]) -> str:
    """Python floats -> PostgreSQL array literal ``{x1,x2,...}``.

    ``repr`` gives the shortest text that round-trips the float exactly, so the
    value parsed back into float4 is bit-identical to the model's output.
    """
    return "{" + ",".join(map(repr, vector)) + "}"


CONVERTERS = {
    "literal": to_literal,
    "adapt": lambda vector: vector,  # let psycopg2 build ARRAY[...]
}


# --------------------------------------------------------------------------
# Phases
# --------------------------------------------------------------------------


def read_sentences(conn, loader: str = "postgres/P0.py") -> tuple[list[int], list[str], float]:
    """Phase 1: fetch every sentence from PostgreSQL (one query).

    Also used by G1 (pgvector): reading the text is plain SQL in both.
    ``loader`` names the script that fills the table, for the error message.
    """
    with Timer() as t, conn.cursor() as cur:
        cur.execute("SELECT sentence_id, text FROM sentences ORDER BY sentence_id")
        rows = cur.fetchall()
    if not rows:
        raise RuntimeError(f"Table 'sentences' is empty. Run {loader} first.")
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return ids, texts, t.elapsed


def recreate_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def store(conn, ids: list[int], embeddings: np.ndarray, batch_size: int,
          method: str) -> list[float]:
    """Phase 3: recreate the table and insert every embedding; per-batch times."""
    convert = CONVERTERS[method]
    recreate_table(conn)
    times: list[float] = []
    for start in range(0, len(ids), batch_size):
        stop = start + batch_size
        with Timer() as t:
            # numpy float32 -> Python floats (psycopg2 cannot adapt ndarrays)
            # -> the text PostgreSQL will parse.
            vectors = [convert(v) for v in embeddings[start:stop].tolist()]
            rows = list(zip(ids[start:stop], vectors))
            with conn.cursor() as cur:
                execute_values(cur, INSERT_SQL, rows,
                               template=INSERT_TEMPLATE, page_size=len(rows))
            conn.commit()
        times.append(t.elapsed)
    return times


def run_store_config(conn, ids, embeddings, batch_size: int, repeats: int,
                     method: str = "literal") -> dict:
    batch_times: list[float] = []
    totals: list[float] = []
    for _ in range(repeats):
        times = store(conn, ids, embeddings, batch_size, method)
        batch_times.extend(times)
        totals.append(sum(times))

    batch_stats = print_stats(f"store {method} batch={batch_size} (per batch)", batch_times)
    total_stats = stats(totals)
    calls = n_batches(len(ids), batch_size)
    return {
        "method": method,
        "batch_size": batch_size,
        "repeats": repeats,
        "batch_time": batch_stats,
        "total_time": total_stats,
        "rows_per_second": len(ids) / total_stats["avg"],
        "db_calls": {"statements": calls, "commits": calls},
    }


def validate(conn, ids: list[int], embeddings: np.ndarray) -> dict:
    """Check, inside PostgreSQL, that what was stored is what was generated."""
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT count(*),
                   min(array_length(embedding, 1)),
                   max(array_length(embedding, 1))
            FROM {TABLE}
        """)
        count, min_dim, max_dim = cur.fetchone()
        # L2 norm of every stored vector, computed in SQL. Squaring in float8:
        # PostgreSQL raises "value out of range: underflow" when the square of
        # a tiny float4 component falls below the float4 range.
        cur.execute(f"""
            SELECT min(norm), max(norm) FROM (
                SELECT sqrt(sum(x::float8 * x::float8)) AS norm
                FROM {TABLE}, unnest(embedding) AS x
                GROUP BY sentence_id
            ) n
        """)
        min_norm, max_norm = cur.fetchone()
        cur.execute(f"SELECT embedding FROM {TABLE} WHERE sentence_id = %s", (ids[0],))
        (first,) = cur.fetchone()

    roundtrip_error = float(np.abs(np.asarray(first, dtype=np.float32) - embeddings[0]).max())
    checks = {
        "rows": count,
        "dim_min": min_dim,
        "dim_max": max_dim,
        "norm_min": min_norm,
        "norm_max": max_norm,
        "roundtrip_max_abs_error": roundtrip_error,
    }
    ok = (count == len(ids) and min_dim == max_dim == EMBEDDING_DIM
          and abs(min_norm - 1) < 1e-3 and abs(max_norm - 1) < 1e-3
          and roundtrip_error == 0.0)
    if not ok:
        raise RuntimeError(f"validation failed: {checks}")
    return checks


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES),
                        help=f"grid of batch sizes for storing (default: {list(BATCH_SIZES)})")
    parser.add_argument("--repeats", type=int, default=3,
                        help="full loads per grid configuration (default: 3)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"batch size for generating and the official load "
                             f"(default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--skip-grid", action="store_true",
                        help="only run the official load")
    args = parser.parse_args()

    print("[P1] loading the embedding model (not timed)")
    get_model()

    results: dict = {"table": TABLE, "column_type": "REAL[]", "dim": EMBEDDING_DIM}

    with closing(psycopg2.connect(**pg_config())) as conn:
        ids, texts, read_time = read_sentences(conn)
        print(f"[P1] read {len(ids)} sentences from PostgreSQL in {read_time:.3f}s\n")
        results["n_sentences"] = len(ids)
        results["read"] = {"time": read_time, "db_calls": {"statements": 1}}

        print("[P1] generating embeddings")
        embeddings, gen_times = generate_embeddings(texts, args.batch_size)
        results["generate"] = {
            "batch_size": args.batch_size,
            "batch_time": print_stats(f"generate batch={args.batch_size} (per batch)",
                                      gen_times),
        }

        if not args.skip_grid:
            print("\n[P1] storing: batch-size grid")
            results["grid"] = [
                run_store_config(conn, ids, embeddings, size, args.repeats)
                for size in args.batch_sizes
            ]
            print("\n[P1] storing: psycopg2 default adaptation (reference)")
            results["adapt"] = run_store_config(conn, ids, embeddings, args.batch_size,
                                                args.repeats, method="adapt")

        # Last, so the table stays populated for P2.
        print("\n[P1] storing: official load")
        results["official"] = run_store_config(conn, ids, embeddings,
                                               args.batch_size, repeats=1)

        results["validation"] = validate(conn, ids, embeddings)

    v = results["validation"]
    print(f"\n[P1] {v['rows']} embeddings of {v['dim_min']} dims stored as REAL[]; "
          f"norms in [{v['norm_min']:.6f}, {v['norm_max']:.6f}]; "
          f"round-trip error {v['roundtrip_max_abs_error']}")
    save_results("P1", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
