"""[G2] Top-2 most similar sentences with pgvector, for the same 10 queries.

pgvector gives PostgreSQL what P2 had to write by hand: distance operators
(``<->`` L2, ``<=>`` cosine distance) and vector indexes. Each query is a
single SQL statement, and the query vector never leaves the database, as in
P2 - this is pgvector's documented "nearest neighbours of a row" pattern:

    SELECT sentence_id, embedding <-> (SELECT embedding FROM sentence_embeddings
                                       WHERE sentence_id = %s) AS distance
    FROM sentence_embeddings
    WHERE sentence_id <> %s            -- exclude the query itself
    ORDER BY distance LIMIT 2

Two configurations are measured on the same table (``results/G2.json``):

* **exact** - index scans disabled (``SET enable_indexscan = off``): a
  sequential scan with pgvector's native C distance, the same brute force as
  P2 but without P2's interpreted SQL functions.
* **hnsw**  - the HNSW indexes built by G1 (one per operator class): an
  approximate search, as in Chroma. ``EXPLAIN`` checks that the index is used.

Each (configuration, metric) runs for the 10 queries over ``--repeats`` timed
runs after one untimed warm-up run.

Checks (not timed): same top-2 for both metrics, L2^2 = 2 * cosine_distance
(``<->`` returns the plain Euclidean distance), and recall against P2's exact
answer (``results/P2.json``).

Usage:
    python pgvector/G2.py                 # 3 timed runs
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    METRICS,
    PGVECTOR_DATABASE,
    TOP_K,
    Timer,
    exact_reference,
    load_queries,
    pgvector_connect,
    print_stats,
    recall_at_k,
    save_results,
    stats,
)

OPERATORS = {"l2": "<->", "cosine": "<=>"}

TOP_K_SQL = """
SELECT sentence_id, embedding {op} (SELECT embedding FROM sentence_embeddings
                                    WHERE sentence_id = %(q)s) AS distance
FROM sentence_embeddings
WHERE sentence_id <> %(q)s
ORDER BY distance
LIMIT %(k)s
"""

#: Configuration name -> planner setting applied before the queries.
CONFIGS = {"exact": "SET enable_indexscan = off", "hnsw": "SET enable_indexscan = on"}


def top_k(conn, sentence_id: int, metric: str, k: int) -> list[tuple[int, float]]:
    with conn.cursor() as cur:
        cur.execute(TOP_K_SQL.format(op=OPERATORS[metric]), {"q": sentence_id, "k": k})
        return cur.fetchall()


def plan_uses_index(conn, sentence_id: int, metric: str, k: int) -> bool:
    with conn.cursor() as cur:
        cur.execute("EXPLAIN " + TOP_K_SQL.format(op=OPERATORS[metric]),
                    {"q": sentence_id, "k": k})
        plan = "\n".join(r[0] for r in cur.fetchall())
    return "Index Scan using" in plan and "_idx" in plan


def run_config(conn, config: str, queries: list[dict], repeats: int, k: int):
    """Apply the planner setting, warm up once, then time every query."""
    with conn.cursor() as cur:
        cur.execute(CONFIGS[config])
    answers = {m: {q["sentence_id"]: top_k(conn, q["sentence_id"], m, k) for q in queries}
               for m in METRICS}
    uses_index = {m: plan_uses_index(conn, queries[0]["sentence_id"], m, k) for m in METRICS}
    times: dict[str, list[float]] = {m: [] for m in METRICS}
    for _ in range(repeats):
        for metric in METRICS:
            for q in queries:
                with Timer() as t:
                    top_k(conn, q["sentence_id"], metric, k)
                times[metric].append(t.elapsed)
    return answers, times, uses_index


def validate(answers: dict, queries: list[dict], k: int, reference: dict | None) -> dict:
    same_ranking = 0
    max_identity_error = 0.0  # | l2^2 - 2 * cosine_distance |
    for q in queries:
        sid = q["sentence_id"]
        l2, cos = answers["l2"][sid], answers["cosine"][sid]
        if [n for n, _ in l2] == [n for n, _ in cos]:
            same_ranking += 1
        for (_, d_l2), (_, d_cos) in zip(l2, cos):
            max_identity_error = max(max_identity_error, abs(d_l2 ** 2 - 2 * d_cos))
    checks = {
        "same_top_k_l2_vs_cosine": same_ranking,
        "max_abs_error_l2sq_vs_2cos": max_identity_error,
        "excludes_query_itself": all(
            sid not in [n for n, _ in answers[m][sid]] for m in METRICS for sid in answers[m]),
    }
    if reference is not None:
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
    reference = exact_reference()
    results: dict = {"database": PGVECTOR_DATABASE, "k": args.k, "metrics": list(METRICS),
                     "operators": OPERATORS, "repeats": args.repeats, "configs": {}}

    with closing(pgvector_connect()) as conn:
        conn.autocommit = True  # planner settings persist for the session
        with conn.cursor() as cur:
            # pgvector's settings only exist once its library is loaded in the
            # session, which happens the first time the vector type is used.
            cur.execute("SELECT NULL::vector")
            cur.execute("SHOW hnsw.ef_search")
            results["hnsw_ef_search"] = int(cur.fetchone()[0])

        all_answers: dict = {}
        for config in CONFIGS:
            print(f"[G2] {config}: top-{args.k} for {len(queries)} queries x "
                  f"{len(METRICS)} metrics, {args.repeats} timed run(s)")
            answers, times, uses_index = run_config(conn, config, queries, args.repeats, args.k)
            if uses_index != {m: config == "hnsw" for m in METRICS}:
                raise RuntimeError(f"{config}: unexpected plan, index used = {uses_index}")
            validation = validate(answers, queries, args.k, reference)
            # Both P2 and the exact configuration are brute force: they must agree.
            if config == "exact" and reference is not None and any(
                    r != 1.0 for r in validation["recall_vs_P2"].values()):
                raise RuntimeError(f"exact search differs from P2: {validation}")
            results["configs"][config] = {
                "uses_index": uses_index,
                "timing": {
                    m: {"query_time": print_stats(f"  {config} {m} (per query)", times[m]),
                        "per_query": {q["sentence_id"]: stats(times[m][i::len(queries)])
                                      for i, q in enumerate(queries)},
                        "raw": times[m]}
                    for m in METRICS
                },
                "validation": validation,
            }
            all_answers[config] = answers
        results["db_calls"] = {"per_query_and_metric": 1,
                               "per_run": len(queries) * len(METRICS)}

        # Readable answer in the same format as P2 and C2 (from the index run).
        final = all_answers["hnsw"]
        neighbour_ids = sorted({n for m in METRICS for a in final[m].values() for n, _ in a})
        with conn.cursor() as cur:
            cur.execute("SELECT sentence_id, text FROM sentences WHERE sentence_id = ANY(%s)",
                        (neighbour_ids,))
            texts = dict(cur.fetchall())
        results["answers"] = [
            {"query_rank": q["query_rank"], "sentence_id": q["sentence_id"], "text": q["text"],
             **{m: [{"sentence_id": n, "distance": d, "text": texts[n]}
                    for n, d in final[m][q["sentence_id"]]] for m in METRICS}}
            for q in queries
        ]

    for config, r in results["configs"].items():
        v = r["validation"]
        print(f"[G2] {config}: same top-{args.k} l2 vs cosine: "
              f"{v['same_top_k_l2_vs_cosine']}/{len(queries)}; max |l2^2 - 2*cos| = "
              f"{v['max_abs_error_l2sq_vs_2cos']:.2e}; recall vs P2: "
              f"{v.get('recall_vs_P2', 'P2.json not found')}")
    save_results("G2", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
