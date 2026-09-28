"""[G0] Load the sentence split into PostgreSQL with pgvector.

pgvector adds a vector type, operators and indexes to PostgreSQL, but nothing
changes for the text: the table, the data types and the batched INSERT are
exactly those of P0. So this script *reuses P0's code* and only points it to
the ``cbde_pgvector`` database, where the extension is enabled. That the text
part needs no new code at all is itself a result for the discussion: pgvector
extends the relational model instead of replacing it.

What is measured (``results/G0.json``), as in P0:

* **Batch-size grid** with ``execute_values`` for every size in
  ``common.BATCH_SIZES`` (``--repeats`` full loads each, default 3).
* **Official load** at ``DEFAULT_BATCH_SIZE``, which leaves the table
  populated for G1.

(P0's COPY reference is not repeated: it would measure the same thing.)

Usage:
    python pgvector/G0.py
    python pgvector/G0.py --skip-grid
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    BATCH_SIZES,
    DEFAULT_BATCH_SIZE,
    PGVECTOR_DATABASE,
    load_sentences,
    pgvector_connect,
    save_results,
)
from postgres import P0  # noqa: E402  - same table, same loading code


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
    rows: list[P0.Row] = [
        (int(r.sentence_id), int(r.chunk_id), int(r.pos), r.text)
        for r in df.itertuples(index=False)
    ]
    print(f"[G0] {len(rows)} sentences to load into '{P0.TABLE}' ({PGVECTOR_DATABASE})\n")

    results: dict = {"database": PGVECTOR_DATABASE, "table": P0.TABLE,
                     "n_sentences": len(rows), "code": "reuses postgres/P0.py"}

    with closing(pgvector_connect()) as conn:
        if not args.skip_grid:
            print("[G0] batch-size grid (execute_values)")
            results["grid"] = [
                P0.run_config(conn, rows, size, "execute_values", args.repeats)
                for size in args.batch_sizes
            ]
        # Last, so the table stays populated for G1.
        print("\n[G0] official load")
        results["official"] = P0.run_config(conn, rows, args.batch_size, "execute_values",
                                            repeats=1)

    save_results("G0", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
