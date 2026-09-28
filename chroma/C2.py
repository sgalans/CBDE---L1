"""[C2] Top-2 most similar sentences in Chroma, for the same 10 queries.

Each metric lives in its own collection (``sentences_l2``, ``sentences_cosine``),
so the same search runs once per collection. Chroma searches with its HNSW
index: the answer is **approximate**, and P2's exact brute force is the
reference to measure its recall.

Chroma cannot search "by record id": ``query()`` needs the vector itself. So
every query is two calls, both timed together (P2's SQL function reads the
query vector inside the database in its single call):

1. ``get(ids=[id], include=["embeddings"])`` - fetch the stored query vector
   (the one written by C1, as in P2; no re-embedding of the text).
2. ``query(query_embeddings=[v], n_results=k,
   where={"sentence_id": {"$ne": id}})`` - k nearest neighbours, excluding the
   query sentence itself through a metadata filter.

Chroma's ``l2`` space returns the *squared* Euclidean distance and ``cosine``
returns 1 - cos, so for unit vectors l2 = 2 * cosine_distance exactly.

What is measured (``results/C2.json``): the time of each (get + query) for the
10 query sentences, per collection, over ``--repeats`` runs after one untimed
warm-up run.

``l2_nofilter`` / ``cosine_nofilter`` are reference-only variants ([CQ1]):
instead of the metadata filter, they ask for ``k + 1`` neighbours and drop the
query sentence in Python. They show what the ``where`` filter costs; the
official answer keeps the exclusion inside the database, as P2 does.

Checks (not timed): top-2 identical in both collections and with and without
the filter, l2 = 2 * cosine distance, and recall against P2's exact answer
(``results/P2.json``).

Usage:
    python chroma/C2.py                 # 3 timed runs
    python chroma/C2.py --repeats 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    CHROMA_COLLECTIONS,
    METRICS,
    TOP_K,
    Timer,
    chroma_client,
    chroma_collections,
    exact_reference,
    load_queries,
    print_stats,
    recall_at_k,
    save_results,
    stats,
)


#: Timed variants: name -> (collection metric, exclude with the where filter?).
#: The "_nofilter" ones are reference-only.
VARIANTS = {**{m: (m, True) for m in METRICS},
            **{f"{m}_nofilter": (m, False) for m in METRICS}}


def top_k(collection, sentence_id: int, k: int, use_filter: bool = True) -> list[tuple[int, float]]:
    """get() the stored query vector, then query() its k nearest neighbours.

    With ``use_filter`` the query sentence is excluded by a metadata filter;
    without it, k + 1 neighbours are requested and it is dropped in Python.
    """
    vector = collection.get(ids=[str(sentence_id)], include=["embeddings"])["embeddings"][0]
    if use_filter:
        res = collection.query(query_embeddings=[vector], n_results=k,
                               where={"sentence_id": {"$ne": sentence_id}},
                               include=["distances"])
    else:
        res = collection.query(query_embeddings=[vector], n_results=k + 1,
                               include=["distances"])
    hits = [(int(i), float(d)) for i, d in zip(res["ids"][0], res["distances"][0])
            if int(i) != sentence_id]
    return hits[:k]


def run_queries(collections: dict, queries: list[dict], repeats: int, k: int):
    """Warm-up once, then time every (query, variant) ``repeats`` times."""
    answers = {name: {q["sentence_id"]: top_k(collections[m], q["sentence_id"], k, flt)
                      for q in queries}
               for name, (m, flt) in VARIANTS.items()}
    times: dict[str, list[float]] = {name: [] for name in VARIANTS}
    for _ in range(repeats):
        for name, (metric, use_filter) in VARIANTS.items():
            for q in queries:
                with Timer() as t:
                    top_k(collections[metric], q["sentence_id"], k, use_filter)
                times[name].append(t.elapsed)
    return answers, times


def validate(answers: dict, queries: list[dict], k: int) -> dict:
    same_ranking = 0
    max_identity_error = 0.0  # | l2_squared - 2 * cosine_distance |
    for q in queries:
        sid = q["sentence_id"]
        l2, cos = answers["l2"][sid], answers["cosine"][sid]
        if [n for n, _ in l2] == [n for n, _ in cos]:
            same_ranking += 1
        for (_, d_l2), (_, d_cos) in zip(l2, cos):
            max_identity_error = max(max_identity_error, abs(d_l2 - 2 * d_cos))
    checks = {
        "queries": len(queries),
        "same_top_k_l2_vs_cosine": same_ranking,
        "max_abs_error_l2sq_vs_2cos": max_identity_error,
        "same_top_k_filter_vs_nofilter": sum(
            [n for n, _ in answers[m][q["sentence_id"]]]
            == [n for n, _ in answers[f"{m}_nofilter"][q["sentence_id"]]]
            for m in METRICS for q in queries),
        "excludes_query_itself": all(
            sid not in [n for n, _ in answers[v][sid]] for v in VARIANTS for sid in answers[v]),
    }

    reference = exact_reference()
    if reference is not None:
        # Recall@k: share of P2's exact neighbours that HNSW also returned.
        checks["recall_vs_P2"] = recall_at_k(reference, answers, queries, k)
    if not checks["excludes_query_itself"]:
        raise RuntimeError(f"a query sentence was returned as its own neighbour: {checks}")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3,
                        help="timed runs over the 10 queries (default: 3)")
    parser.add_argument("--k", type=int, default=TOP_K,
                        help=f"neighbours per query (default: {TOP_K})")
    args = parser.parse_args()

    queries = load_queries()
    client = chroma_client()
    collections = chroma_collections(client)
    print(f"[C2] top-{args.k} for {len(queries)} queries x {len(METRICS)} collections, "
          f"{args.repeats} timed run(s)\n")

    answers, times = run_queries(collections, queries, args.repeats, args.k)

    results: dict = {"k": args.k, "metrics": list(METRICS),
                     "reference_metrics": [v for v in VARIANTS if v not in METRICS],
                     "repeats": args.repeats,
                     "collections": dict(CHROMA_COLLECTIONS),
                     "search": "approximate (HNSW index), get() + query() per query",
                     # Index parameters actually in use (Chroma's defaults), so the
                     # report can explain the recall without hand-typed values.
                     "hnsw": {m: dict(c.configuration.get("hnsw") or {})
                              for m, c in collections.items()},
                     "timing": {}}
    for name in VARIANTS:
        results["timing"][name] = {
            "query_time": print_stats(f"get+query {name} (per query)", times[name]),
            "per_query": {q["sentence_id"]: stats(times[name][i::len(queries)])
                          for i, q in enumerate(queries)},
            "raw": times[name],
        }
    results["db_calls"] = {"per_query_and_metric": 2, "calls": ["get", "query"],
                           "per_run": 2 * len(queries) * len(METRICS)}
    results["validation"] = validate(answers, queries, args.k)

    # Readable answer, also stored so the document can show it.
    neighbour_ids = sorted({n for m in METRICS for a in answers[m].values() for n, _ in a})
    got = collections["l2"].get(ids=[str(i) for i in neighbour_ids], include=["documents"])
    texts = {int(i): d for i, d in zip(got["ids"], got["documents"])}
    results["answers"] = []
    print("\n[C2] results")
    for q in queries:
        sid = q["sentence_id"]
        entry = {"query_rank": q["query_rank"], "sentence_id": sid, "text": q["text"]}
        print(f"\n#{q['query_rank']} [{sid}] {q['text']}")
        for metric in METRICS:
            entry[metric] = [{"sentence_id": n, "distance": d, "text": texts[n]}
                             for n, d in answers[metric][sid]]
            for rank, (n, d) in enumerate(answers[metric][sid], 1):
                print(f"   {metric:<6} {rank}. d={d:.4f} [{n}] {texts[n][:90]}")
        results["answers"].append(entry)

    v = results["validation"]
    print(f"\n[C2] same top-{args.k} l2 vs cosine: {v['same_top_k_l2_vs_cosine']}/{v['queries']}; "
          f"max |l2 - 2*cos_dist| = {v['max_abs_error_l2sq_vs_2cos']:.2e}; "
          f"recall vs P2: {v.get('recall_vs_P2', 'P2.json not found')}")
    save_results("C2", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
