"""[P2] Top-2 most similar sentences in plain PostgreSQL, computed in the DB.

Without pgvector PostgreSQL has neither a vector type nor distance operators,
so both metrics are written as SQL functions over ``REAL[]`` and the search is
a PL/pgSQL function. Python only sends a sentence id and receives two rows: the
vectors never leave the database (lab requirement).

    l2_distance(a, b)      = sqrt(sum((a_i - b_i)^2))
    cosine_distance(a, b)  = 1 - (a . b) / (|a| |b|)
    unit_cosine_distance   = 1 - (a . b)   (only valid for unit vectors)
    top_k_similar(q_id, metric, k)
        -> the k sentences closest to sentence q_id, excluding q_id itself

Every function computes in float8: squaring very small float4 components
raises "value out of range: underflow" in PostgreSQL instead of rounding to 0.

The search is exact (brute force over every row, no index is possible on
``REAL[]``), so it is the reference for Chroma's approximate HNSW search.

What is measured (``results/P2.json``): the time of each ``top_k_similar``
call for the 10 query sentences of ``data/queries.json``, per metric, over
``--repeats`` runs after one untimed warm-up run.

``cosine_unit`` is an extra, reference-only variant ([PQ1] c): the model
returns unit vectors, so the two norms of the cosine are always 1 and the
distance reduces to ``1 - a . b``. It must give the same neighbours as
``cosine``; the time difference is the cost of recomputing the norms per row.

Checks (outside the timed section): the top-2 of both metrics must coincide
and satisfy L2^2 = 2 * cosine_distance (unit vectors), and must match a numpy
brute force over the stored vectors. numpy is only used to *validate* the
database's answer, never to produce it.

Usage:
    python postgres/P2.py                 # 3 timed runs
    python postgres/P2.py --repeats 5
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

import numpy as np
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    METRICS,
    TOP_K,
    Timer,
    load_queries,
    pg_config,
    print_stats,
    save_results,
    stats,
)

FUNCTIONS_SQL = """
CREATE OR REPLACE FUNCTION l2_distance(a REAL[], b REAL[])
RETURNS DOUBLE PRECISION
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT sqrt(sum(d * d))
    FROM (SELECT x::float8 - y::float8 AS d FROM unnest(a, b) AS t(x, y)) s
$$;

CREATE OR REPLACE FUNCTION cosine_distance(a REAL[], b REAL[])
RETURNS DOUBLE PRECISION
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT 1 - sum(x * y) / (sqrt(sum(x * x)) * sqrt(sum(y * y)))
    FROM (SELECT x::float8 AS x, y::float8 AS y FROM unnest(a, b) AS t(x, y)) s
$$;

-- Valid only for unit vectors (all-MiniLM-L6-v2 normalises its output).
CREATE OR REPLACE FUNCTION unit_cosine_distance(a REAL[], b REAL[])
RETURNS DOUBLE PRECISION
LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS $$
    SELECT 1 - sum(x::float8 * y::float8) FROM unnest(a, b) AS t(x, y)
$$;

CREATE OR REPLACE FUNCTION top_k_similar(q_id INTEGER, metric TEXT, k INTEGER DEFAULT 2)
RETURNS TABLE (neighbour_id INTEGER, distance DOUBLE PRECISION)
LANGUAGE plpgsql STABLE AS $$
DECLARE
    q REAL[];
BEGIN
    SELECT e.embedding INTO q FROM sentence_embeddings e WHERE e.sentence_id = q_id;
    IF q IS NULL THEN
        RAISE EXCEPTION 'sentence % has no embedding (run P1 first)', q_id;
    END IF;

    -- One branch per metric instead of a CASE per row: the metric is chosen
    -- once per call, not once per compared vector.
    IF metric = 'l2' THEN
        RETURN QUERY
            SELECT e.sentence_id, l2_distance(e.embedding, q) AS dist
            FROM sentence_embeddings e
            WHERE e.sentence_id <> q_id          -- exclude the query itself
            ORDER BY dist
            LIMIT k;
    ELSIF metric = 'cosine' THEN
        RETURN QUERY
            SELECT e.sentence_id, cosine_distance(e.embedding, q) AS dist
            FROM sentence_embeddings e
            WHERE e.sentence_id <> q_id
            ORDER BY dist
            LIMIT k;
    ELSIF metric = 'cosine_unit' THEN
        RETURN QUERY
            SELECT e.sentence_id, unit_cosine_distance(e.embedding, q) AS dist
            FROM sentence_embeddings e
            WHERE e.sentence_id <> q_id
            ORDER BY dist
            LIMIT k;
    ELSE
        RAISE EXCEPTION 'unknown metric %', metric;
    END IF;
END;
$$;
"""

TOP_K_SQL = "SELECT neighbour_id, distance FROM top_k_similar(%s, %s, %s)"

#: Reference-only variant, timed but not part of the official metrics.
UNIT_COSINE = "cosine_unit"
TIMED_METRICS = (*METRICS, UNIT_COSINE)


def install_functions(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(FUNCTIONS_SQL)
    conn.commit()


def top_k(conn, sentence_id: int, metric: str, k: int) -> list[tuple[int, float]]:
    with conn.cursor() as cur:
        cur.execute(TOP_K_SQL, (sentence_id, metric, k))
        return cur.fetchall()


def run_queries(conn, queries: list[dict], repeats: int, k: int) -> tuple[dict, dict]:
    """Warm-up once, then time every (query, metric) call ``repeats`` times."""
    answers: dict[str, dict[int, list[tuple[int, float]]]] = {m: {} for m in TIMED_METRICS}
    for metric in TIMED_METRICS:  # warm-up: caches, plan and function compilation
        for q in queries:
            answers[metric][q["sentence_id"]] = top_k(conn, q["sentence_id"], metric, k)

    times: dict[str, list[float]] = {m: [] for m in TIMED_METRICS}
    for _ in range(repeats):
        for metric in TIMED_METRICS:
            for q in queries:
                with Timer() as t:
                    top_k(conn, q["sentence_id"], metric, k)
                times[metric].append(t.elapsed)
    return answers, times


# --------------------------------------------------------------------------
# Validation (not timed)
# --------------------------------------------------------------------------


def fetch_texts(conn, ids: set[int]) -> dict[int, str]:
    with conn.cursor() as cur:
        cur.execute("SELECT sentence_id, text FROM sentences WHERE sentence_id = ANY(%s)",
                    (list(ids),))
        return dict(cur.fetchall())


def numpy_reference(conn, queries: list[dict], k: int) -> dict[str, dict[int, list[int]]]:
    """Exact top-k computed with numpy, only to cross-check the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT sentence_id, embedding FROM sentence_embeddings ORDER BY sentence_id")
        rows = cur.fetchall()
    ids = np.array([r[0] for r in rows])
    emb = np.array([r[1] for r in rows], dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1)
    ref: dict[str, dict[int, list[int]]] = {m: {} for m in METRICS}
    for q in queries:
        v = emb[ids == q["sentence_id"]][0]
        dist = {
            "l2": np.linalg.norm(emb - v, axis=1),
            "cosine": 1 - emb @ v / (norms * np.linalg.norm(v)),
        }
        for metric, d in dist.items():
            d = d.copy()
            d[ids == q["sentence_id"]] = np.inf
            ref[metric][q["sentence_id"]] = ids[np.argsort(d)[:k]].tolist()
    return ref


def validate(answers: dict, reference: dict, queries: list[dict]) -> dict:
    same_ranking = 0
    same_unit = 0
    same_as_numpy = 0
    max_identity_error = 0.0  # | l2^2 - 2 * cosine_distance |
    max_unit_error = 0.0      # | cosine_distance - unit_cosine_distance |
    for q in queries:
        sid = q["sentence_id"]
        l2 = answers["l2"][sid]
        cos = answers["cosine"][sid]
        unit = answers[UNIT_COSINE][sid]
        if [n for n, _ in l2] == [n for n, _ in cos]:
            same_ranking += 1
        if [n for n, _ in unit] == [n for n, _ in cos]:
            same_unit += 1
        for (_, d_cos), (_, d_unit) in zip(cos, unit):
            max_unit_error = max(max_unit_error, abs(d_cos - d_unit))
        if all([n for n, _ in answers[m][sid]] == reference[m][sid] for m in METRICS):
            same_as_numpy += 1
        for (_, d_l2), (_, d_cos) in zip(l2, cos):
            max_identity_error = max(max_identity_error, abs(d_l2 ** 2 - 2 * d_cos))
    checks = {
        "queries": len(queries),
        "same_top_k_l2_vs_cosine": same_ranking,
        "same_top_k_as_numpy": same_as_numpy,
        "max_abs_error_l2sq_vs_2cos": max_identity_error,
        "same_top_k_cosine_vs_unit": same_unit,
        "max_abs_error_cosine_vs_unit": max_unit_error,
    }
    if same_as_numpy != len(queries) or same_unit != len(queries):
        raise RuntimeError(f"database result differs from the numpy reference: {checks}")
    return checks


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repeats", type=int, default=3,
                        help="timed runs over the 10 queries (default: 3)")
    parser.add_argument("--k", type=int, default=TOP_K,
                        help=f"neighbours per query (default: {TOP_K})")
    args = parser.parse_args()

    queries = load_queries()
    results: dict = {"k": args.k, "metrics": list(METRICS), "reference_metrics": [UNIT_COSINE],
                     "repeats": args.repeats,
                     "search": "exact (sequential scan, SQL functions over REAL[])"}

    with closing(psycopg2.connect(**pg_config())) as conn:
        install_functions(conn)
        print(f"[P2] top-{args.k} for {len(queries)} queries x {len(TIMED_METRICS)} metrics, "
              f"{args.repeats} timed run(s)\n")
        answers, times = run_queries(conn, queries, args.repeats, args.k)

        results["timing"] = {}
        for metric in TIMED_METRICS:
            per_query = {
                q["sentence_id"]: stats(times[metric][i::len(queries)])
                for i, q in enumerate(queries)
            }
            results["timing"][metric] = {
                "query_time": print_stats(f"top_k_similar {metric} (per query)", times[metric]),
                "per_query": per_query,
                "raw": times[metric],
            }
        results["db_calls"] = {"per_query_and_metric": 1,
                               "per_run": len(queries) * len(METRICS)}

        results["validation"] = validate(answers, numpy_reference(conn, queries, args.k),
                                         queries)
        neighbour_ids = {n for m in METRICS for a in answers[m].values() for n, _ in a}
        texts = fetch_texts(conn, neighbour_ids)

    # Readable answer, also stored so the document can show it.
    results["answers"] = []
    print("\n[P2] results")
    for q in queries:
        sid = q["sentence_id"]
        entry = {"query_rank": q["query_rank"], "sentence_id": sid, "text": q["text"]}
        print(f"\n#{q['query_rank']} [{sid}] {q['text']}")
        for metric in METRICS:
            entry[metric] = [
                {"sentence_id": n, "distance": d, "text": texts[n]} for n, d in answers[metric][sid]
            ]
            for rank, (n, d) in enumerate(answers[metric][sid], 1):
                print(f"   {metric:<6} {rank}. d={d:.4f} [{n}] {texts[n][:90]}")
        results["answers"].append(entry)

    v = results["validation"]
    print(f"\n[P2] same top-{args.k} L2 vs cosine: {v['same_top_k_l2_vs_cosine']}/{v['queries']}; "
          f"matches numpy: {v['same_top_k_as_numpy']}/{v['queries']}; "
          f"max |L2^2 - 2*cos_dist| = {v['max_abs_error_l2sq_vs_2cos']:.2e}; "
          f"unit cosine same top-{args.k}: {v['same_top_k_cosine_vs_unit']}/{v['queries']} "
          f"(max diff {v['max_abs_error_cosine_vs_unit']:.2e})")
    save_results("P2", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
