"""[C1] Generate the sentence embeddings and store them in Chroma.

As in P1, the script *connects to the database*, reads the sentences stored by
C0, generates their embeddings with our model (``common.get_model()``, the same
one as PostgreSQL) and writes them back. C0 could only store the text together
with a vector computed by Chroma's default embedding function (ONNX), so here
those vectors are replaced with ``update(ids, embeddings=...)``: afterwards
Chroma holds exactly the same vectors as PostgreSQL.

Unlike PostgreSQL, Chroma receives the numpy array directly: there is no
float -> text -> float round trip.

Three phases are timed separately (``results/C1.json``):

1. **read**     - ``get(include=["documents"])`` of every record (one call).
2. **generate** - ``model.encode`` per batch of ``DEFAULT_BATCH_SIZE`` sentences
   (comparable with P1's generation).
3. **store**    - ``update(ids, embeddings)`` per batch, in each collection,
   ``--repeats`` times (default 3, as P1's storing) so the stability across
   runs can be compared. Generation is measured once: it is expensive and
   deterministic.

C1 is not part of the batch-size grid (``CLAUDE.md``): only the official batch
size is used.

Checks (not timed): the stored vectors are bit-identical to the generated ones,
and the ONNX vectors written by C0 are compared with ours on a sample.

Usage:
    python chroma/C1.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    CHROMA_COLLECTIONS,
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIM,
    Timer,
    chroma_client,
    chroma_collections,
    generate_embeddings,
    get_model,
    n_batches,
    print_stats,
    save_results,
    stats,
)

#: Records whose C0 (ONNX) vectors are compared with ours before replacing them.
ONNX_SAMPLE = 500


def read_documents(collection) -> tuple[list[str], list[str], float]:
    """Phase 1: fetch every document of the collection (one call)."""
    with Timer() as t:
        got = collection.get(include=["documents"])
    if not got["ids"]:
        raise RuntimeError("collection is empty. Run chroma/C0.py first.")
    # Keep the corpus order (ids are the sentence ids as strings).
    order = sorted(range(len(got["ids"])), key=lambda i: int(got["ids"][i]))
    ids = [got["ids"][i] for i in order]
    docs = [got["documents"][i] for i in order]
    return ids, docs, t.elapsed


def store(collection, ids: list[str], embeddings: np.ndarray, batch_size: int) -> list[float]:
    """Phase 3: replace the vectors of every record; per-batch times."""
    times: list[float] = []
    for start in range(0, len(ids), batch_size):
        stop = start + batch_size
        with Timer() as t:
            collection.update(ids=ids[start:stop], embeddings=embeddings[start:stop])
        times.append(t.elapsed)
    return times


def stored_vectors(collection, ids: list[str]) -> np.ndarray:
    got = collection.get(ids=ids, include=["embeddings"])
    by_id = dict(zip(got["ids"], got["embeddings"]))
    return np.asarray([by_id[i] for i in ids], dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"batch size for generating and storing (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--repeats", type=int, default=3,
                        help="update() runs per collection (default: 3)")
    args = parser.parse_args()

    print("[C1] loading the embedding model (not timed)")
    get_model()

    client = chroma_client()
    collections = chroma_collections(client)
    results: dict = {"collections": dict(CHROMA_COLLECTIONS), "dim": EMBEDDING_DIM}

    # Phase 1: the documents are identical in both collections; read them once.
    ids, docs, read_time = read_documents(collections["l2"])
    print(f"[C1] read {len(ids)} documents from Chroma in {read_time:.3f}s\n")
    results["n_sentences"] = len(ids)
    results["read"] = {"time": read_time, "db_calls": {"get": 1}}

    # Vectors written by C0 (ONNX), kept to compare them with ours afterwards.
    sample = ids[:: max(1, len(ids) // ONNX_SAMPLE)][:ONNX_SAMPLE]
    onnx_vectors = stored_vectors(collections["l2"], sample)

    # Phase 2
    print("[C1] generating embeddings")
    embeddings, gen_times = generate_embeddings(docs, args.batch_size)
    results["generate"] = {
        "batch_size": args.batch_size,
        "batch_time": print_stats(f"generate batch={args.batch_size} (per batch)", gen_times),
    }

    # Phase 3
    print("\n[C1] storing: update() per collection")
    results["store"] = {}
    for space, collection in collections.items():
        times: list[float] = []
        totals: list[float] = []
        for _ in range(args.repeats):  # same vectors each time: idempotent
            run = store(collection, ids, embeddings, args.batch_size)
            times.extend(run)
            totals.append(sum(run))
        results["store"][space] = {
            "collection": CHROMA_COLLECTIONS[space],
            "batch_size": args.batch_size,
            "repeats": args.repeats,
            "batch_time": print_stats(f"update {space} batch={args.batch_size} (per batch)",
                                      times),
            "total_time": stats(totals),
            "db_calls": {"update": n_batches(len(ids), args.batch_size)},
        }

    # Checks (not timed)
    index = {sid: i for i, sid in enumerate(ids)}
    ours_sample = embeddings[[index[s] for s in sample]]
    validation = {"rows": {}, "roundtrip_max_abs_error": {}, "vectors_changed": {}}
    for space, collection in collections.items():
        stored = stored_vectors(collection, ids)
        diff = np.abs(stored - embeddings)
        validation["rows"][space] = collection.count()
        validation["roundtrip_max_abs_error"][space] = float(diff.max())
        validation["vectors_changed"][space] = int((diff.max(axis=1) > 0).sum())
    onnx_diff = float(np.abs(onnx_vectors - ours_sample).max())
    # An exact 0 means the collections did not hold C0's ONNX vectors any more
    # (C1 already ran after the last C0): the comparison would be meaningless.
    validation["onnx_vs_model_max_abs_diff"] = onnx_diff if onnx_diff > 0 else None
    validation["onnx_sample_size"] = len(sample)
    if onnx_diff == 0:
        print("[C1] WARNING: the collections no longer hold C0's ONNX vectors; "
              "run chroma/C0.py first to measure ONNX vs. our model.")
    # l2 must store the vectors bit for bit. The cosine space re-normalises
    # every vector on insert, which changes the last bit of some components of
    # vectors whose float32 norm is not exactly 1: allow that rounding only.
    errors = validation["roundtrip_max_abs_error"]
    if (any(n != len(ids) for n in validation["rows"].values())
            or errors["l2"] != 0.0 or errors["cosine"] > 1e-6):
        raise RuntimeError(f"validation failed: {validation}")
    results["validation"] = validation

    print(f"\n[C1] {len(ids)} embeddings stored in each collection; round-trip error "
          f"{validation['roundtrip_max_abs_error']}; ONNX (C0) vs our model: max diff "
          f"{onnx_diff:.1e} on {len(sample)} records")
    save_results("C1", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
