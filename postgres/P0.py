"""[P0] Load the sentence split of the corpus into plain PostgreSQL.

The chunks built in phase 1 are loaded already split per sentence
(``data/sentences.parquet``), so PostgreSQL, Chroma and pgvector ingest exactly
the same rows. Each sentence keeps the chunk it belongs to (``chunk_id``) and
its position inside it (``pos``).

What is measured (``results/P0.json``):

* **Batch-size grid** - the whole corpus is inserted with a multi-row
  ``INSERT ... VALUES`` (``psycopg2.extras.execute_values``) and one commit per
  batch, for every size in ``common.BATCH_SIZES``. Size 1 is the row-by-row
  baseline. The table is recreated before every run so all runs start equal.
* **COPY** - the same load through ``COPY ... FROM STDIN`` at the default batch
  size, as the reference "bulk" insertion method of PostgreSQL ([PQ1]).
* **Official load** - a final run at ``common.DEFAULT_BATCH_SIZE`` that leaves
  the table populated for P1. These are the "storing the textual data" times
  reported in the document.

Usage:
    python postgres/P0.py                      # full grid + COPY + official load
    python postgres/P0.py --repeats 1          # quicker grid
    python postgres/P0.py --skip-grid          # official load only
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from contextlib import closing
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    Timer,
    batched,
    load_sentences,
    n_batches,
    pg_config,
    print_stats,
    save_results,
    stats,
)

TABLE = "sentences"

# INTEGER ids (10k rows fit comfortably), SMALLINT position inside a chunk
# (0..19) and TEXT for the sentence: PostgreSQL stores short TEXT values inline
# with no length check, so it is never slower than VARCHAR(n).
DDL = f"""
DROP TABLE IF EXISTS {TABLE} CASCADE;
CREATE TABLE {TABLE} (
    sentence_id INTEGER  PRIMARY KEY,
    chunk_id    INTEGER  NOT NULL,
    pos         SMALLINT NOT NULL,
    text        TEXT     NOT NULL
);
"""

INSERT_SQL = f"INSERT INTO {TABLE} (sentence_id, chunk_id, pos, text) VALUES %s"
COPY_SQL = f"COPY {TABLE} (sentence_id, chunk_id, pos, text) FROM STDIN WITH (FORMAT csv)"

Row = tuple[int, int, int, str]


def recreate_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(DDL)
    conn.commit()


def insert_values(conn, batch: list[Row]) -> None:
    """One multi-row INSERT statement plus one commit for the whole batch."""
    with conn.cursor() as cur:
        execute_values(cur, INSERT_SQL, batch, page_size=len(batch))
    conn.commit()


def insert_copy(conn, batch: list[Row]) -> None:
    """One COPY FROM STDIN plus one commit for the whole batch.

    Serialising the batch to CSV happens inside the timed call, just like
    ``execute_values`` builds its SQL string inside it.
    """
    buf = io.StringIO()
    csv.writer(buf).writerows(batch)
    buf.seek(0)
    with conn.cursor() as cur:
        cur.copy_expert(COPY_SQL, buf)
    conn.commit()


METHODS = {"execute_values": insert_values, "copy": insert_copy}


def load(conn, rows: list[Row], batch_size: int, method: str) -> list[float]:
    """Recreate the table and load every row; return the time of each batch."""
    recreate_table(conn)
    insert = METHODS[method]
    times: list[float] = []
    for batch in batched(rows, batch_size):
        with Timer() as t:
            insert(conn, batch)
        times.append(t.elapsed)
    return times


def check_loaded(conn, expected: int) -> None:
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {TABLE}")
        (count,) = cur.fetchone()
    if count != expected:
        raise RuntimeError(f"{TABLE} has {count} rows, expected {expected}")


def run_config(conn, rows: list[Row], batch_size: int, method: str, repeats: int) -> dict:
    """Load the corpus ``repeats`` times with one configuration and summarise it."""
    batch_times: list[float] = []
    totals: list[float] = []
    for _ in range(repeats):
        times = load(conn, rows, batch_size, method)
        check_loaded(conn, len(rows))
        batch_times.extend(times)
        totals.append(sum(times))

    label = f"{method} batch={batch_size}"
    batch_stats = print_stats(f"{label} (per batch)", batch_times)
    total_stats = stats(totals)
    calls = n_batches(len(rows), batch_size)
    return {
        "method": method,
        "batch_size": batch_size,
        "repeats": repeats,
        "batch_time": batch_stats,
        "total_time": total_stats,
        "rows_per_second": len(rows) / total_stats["avg"],
        # Round trips per load: one statement and one commit per batch.
        "db_calls": {"statements": calls, "commits": calls},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES),
                        help=f"grid of batch sizes (default: {list(BATCH_SIZES)})")
    parser.add_argument("--repeats", type=int, default=3,
                        help="full loads per grid configuration (default: 3)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"batch size of the official load (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--skip-grid", action="store_true",
                        help="only run the official load")
    args = parser.parse_args()

    df = load_sentences()
    rows: list[Row] = [
        (int(r.sentence_id), int(r.chunk_id), int(r.pos), r.text)
        for r in df.itertuples(index=False)
    ]
    print(f"[P0] {len(rows)} sentences to load into '{TABLE}'\n")

    results: dict = {"table": TABLE, "n_sentences": len(rows)}

    # psycopg2's own context manager only ends the transaction; closing() also
    # closes the connection.
    with closing(psycopg2.connect(**pg_config())) as conn:
        if not args.skip_grid:
            print("[P0] batch-size grid (execute_values)")
            results["grid"] = [
                run_config(conn, rows, size, "execute_values", args.repeats)
                for size in args.batch_sizes
            ]
            print("\n[P0] COPY at the official batch size")
            results["copy"] = run_config(conn, rows, args.batch_size, "copy", args.repeats)

        # Last, so the table stays populated for P1.
        print("\n[P0] official load")
        official = run_config(conn, rows, args.batch_size, "execute_values", repeats=1)
        results["official"] = official

    print(f"\n[P0] {TABLE} loaded: {len(rows)} rows, "
          f"{official['total_time']['total']:.3f}s at batch size {args.batch_size}")
    save_results("P0", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
