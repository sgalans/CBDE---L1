"""[C0] Load the same sentence split into Chroma.

Chroma cannot store a record without its vector: with no embedding function
``add(documents=...)`` raises ``ValueError``, and with one, ``add()`` computes
the embeddings itself. So this script loads the text the way Chroma is meant to
be used - ``add(ids, documents, metadatas)`` with the default embedding
function (ONNX ``all-MiniLM-L6-v2``) - and the "text insertion" time
*unavoidably includes* computing the embeddings inside ``add()``. That is the
answer to [CQ1]; C1 then replaces those vectors with our own model's.

Two collections hold the same data, one per metric (``hnsw:space`` is fixed
when a collection is created): ``sentences_l2`` and ``sentences_cosine``.
Every record keeps ``sentence_id``, ``chunk_id`` and ``pos`` as metadata, the
same split as PostgreSQL.

What is measured (``results/C0.json``):

* **Batch-size grid** on ``sentences_l2`` for every size in
  ``common.BATCH_SIZES`` (the collection is recreated before every run). Each
  load re-embeds the whole corpus, so it runs ``--repeats`` times (default 1).
* **Embedding share** - at the official batch size, the default embedding
  function alone is timed on the same batches (nothing stored), to show which
  part of ``add()`` is embedding and which is storing.
* **Official load** at ``DEFAULT_BATCH_SIZE``: ``--official-repeats`` full
  loads into ``sentences_l2`` (default 3, as P0, so the stability across
  loads can be compared) and one into ``sentences_cosine``. These are the
  "storing the textual data" times reported in the document.

Usage:
    python chroma/C0.py                  # grid + embedding share + official
    python chroma/C0.py --skip-grid      # official load only; keeps the grid and
                                         # embedding share already in C0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    BATCH_SIZES,
    CHROMA_COLLECTIONS,
    CHROMA_PATH,
    DEFAULT_BATCH_SIZE,
    RESULTS_DIR,
    Timer,
    batched,
    chroma_client,
    load_sentences,
    n_batches,
    print_stats,
    save_results,
    stats,
)


def default_embedding_function():
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

    return DefaultEmbeddingFunction()


def recreate_collection(client, name: str, space: str, ef):
    try:
        client.delete_collection(name)
    except Exception:  # noqa: BLE001 - it did not exist yet
        pass
    return client.create_collection(name, embedding_function=ef,
                                    configuration={"hnsw": {"space": space}})


def load(client, name: str, space: str, ef, records: dict, batch_size: int) -> list[float]:
    """Recreate the collection and add every record; return per-batch times."""
    collection = recreate_collection(client, name, space, ef)
    ids, docs, metas = records["ids"], records["documents"], records["metadatas"]
    times: list[float] = []
    for start in range(0, len(ids), batch_size):
        stop = start + batch_size
        with Timer() as t:
            collection.add(ids=ids[start:stop], documents=docs[start:stop],
                           metadatas=metas[start:stop])
        times.append(t.elapsed)
    if collection.count() != len(ids):
        raise RuntimeError(f"{name} has {collection.count()} records, expected {len(ids)}")
    return times


def summarise(label: str, batch_times: list[float], totals: list[float],
              n_records: int, batch_size: int, **extra) -> dict:
    total_stats = stats(totals)
    return {
        **extra,
        "batch_size": batch_size,
        "batch_time": print_stats(label, batch_times),
        "total_time": total_stats,
        "records_per_second": n_records / total_stats["avg"],
        # One add() per batch: the embedded client makes no network calls.
        "db_calls": {"add": n_batches(n_records, batch_size)},
    }


def run_config(client, name, space, ef, records, batch_size, repeats) -> dict:
    batch_times: list[float] = []
    totals: list[float] = []
    for _ in range(repeats):
        times = load(client, name, space, ef, records, batch_size)
        batch_times.extend(times)
        totals.append(sum(times))
    return summarise(f"add {space} batch={batch_size} (per batch)", batch_times, totals,
                     len(records["ids"]), batch_size, collection=name, space=space,
                     repeats=repeats)


def embedding_share(ef, documents: list[str], batch_size: int) -> dict:
    """Time the default embedding function alone on the official batches."""
    times: list[float] = []
    for batch in batched(documents, batch_size):
        with Timer() as t:
            ef(list(batch))
        times.append(t.elapsed)
    return {"batch_size": batch_size,
            "batch_time": print_stats(f"embed only batch={batch_size} (per batch)", times)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=list(BATCH_SIZES),
                        help=f"grid of batch sizes (default: {list(BATCH_SIZES)})")
    parser.add_argument("--repeats", type=int, default=1,
                        help="full loads per grid configuration (default: 1; each "
                             "load re-embeds the whole corpus)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"batch size of the official load (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--official-repeats", type=int, default=3,
                        help="official loads into the l2 collection (default: 3; the cosine "
                             "collection is loaded once)")
    parser.add_argument("--skip-grid", action="store_true",
                        help="only run the official load, keeping the grid and embedding "
                             "share of an existing results/C0.json")
    args = parser.parse_args()

    df = load_sentences()
    records = {
        "ids": [str(i) for i in df["sentence_id"]],
        "documents": df["text"].tolist(),
        "metadatas": [{"sentence_id": int(r.sentence_id), "chunk_id": int(r.chunk_id),
                       "pos": int(r.pos)} for r in df.itertuples(index=False)],
    }
    print(f"[C0] {len(df)} sentences to load into Chroma ({CHROMA_PATH})\n")

    client = chroma_client()
    # One embedding function shared by every collection, warmed up before any
    # timing: the first call downloads the ONNX model (~80 MB) and builds the
    # inference session.
    ef = default_embedding_function()
    ef(records["documents"][:8])

    results: dict = {"client": "PersistentClient (embedded, in-process)",
                     "embedding_function": "DefaultEmbeddingFunction (ONNX all-MiniLM-L6-v2)",
                     "n_sentences": len(df)}

    previous = RESULTS_DIR / "C0.json"
    if args.skip_grid and previous.exists():
        # The grid takes ~40 min: reuse it instead of throwing it away.
        with previous.open(encoding="utf-8") as fh:
            old = json.load(fh)
        for key in ("grid", "embedding_only"):
            if key in old:
                results[key] = old[key]
        print(f"[C0] --skip-grid: keeping the grid of {old.get('timestamp', '?')}\n")

    if not args.skip_grid:
        print("[C0] batch-size grid on the l2 collection")
        results["grid"] = [
            run_config(client, CHROMA_COLLECTIONS["l2"], "l2", ef, records, size, args.repeats)
            for size in args.batch_sizes
        ]
        print("\n[C0] embedding share: default embedding function alone")
        results["embedding_only"] = embedding_share(ef, records["documents"], args.batch_size)

    # Last, so both collections stay populated for C1.
    print("\n[C0] official load")
    results["official"] = {
        space: run_config(client, name, space, ef, records, args.batch_size,
                          repeats=args.official_repeats if space == "l2" else 1)
        for space, name in CHROMA_COLLECTIONS.items()
    }

    total = sum(o["total_time"]["total"] for o in results["official"].values())
    print(f"\n[C0] {len(df)} records in each of {len(CHROMA_COLLECTIONS)} collections, "
          f"{total:.1f}s in total at batch size {args.batch_size}")
    save_results("C0", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
