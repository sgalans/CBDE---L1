"""Phase 1 - build the shared corpus for every system of the lab.

Streams bookCorpus from HuggingFace, keeps ~10k clean sentences, groups them
into chunks and writes the artefacts that PostgreSQL, Chroma and pgvector all
ingest afterwards:

    data/chunks.parquet     the chunked corpus (the "chunk of data" to publish)
    data/sentences.parquet  the per-sentence split, identical for every system
    data/queries.json       the 10 fixed query sentences used by P2 / C2 / G2

CSV mirrors are written next to the Parquet files so the data is readable
directly on GitHub.

Usage:
    python data/prepare_corpus.py                # default: 10000 sentences
    python data/prepare_corpus.py --n 2000 --force
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import (  # noqa: E402
    CHUNKS_PATH,
    DATA_DIR,
    MAX_SENTENCE_CHARS,
    MIN_SENTENCE_CHARS,
    MIN_SENTENCE_WORDS,
    N_QUERIES,
    QUERIES_PATH,
    RANDOM_SEED,
    SENTENCES_PATH,
    SENTENCES_PER_CHUNK,
    TARGET_SENTENCES,
    TOP_K,
    Timer,
)

#: bookCorpus mirrors, tried in order. The canonical repository still ships a
#: loading script, which datasets>=3.0 refuses to execute; the mirrors below are
#: plain Parquet exports of the very same corpus.
DATASET_CANDIDATES = (
    "bookcorpus/bookcorpus",
    "bookcorpus",
    "rojagtap/bookcorpus",
    "SamuelYang/bookcorpus",
)

#: Safety valve: stop reading the stream even if not enough sentences passed
#: the cleaning filters, so a bad filter can never loop forever.
MAX_RAW_RECORDS = 2_000_000


def open_stream(dataset_id: str | None):
    """Open bookCorpus in streaming mode, trying the known mirrors in order."""
    from datasets import load_dataset

    candidates = (dataset_id,) if dataset_id else DATASET_CANDIDATES
    errors: list[str] = []
    for candidate in candidates:
        for trust in (False, True):
            try:
                kwargs = {"split": "train", "streaming": True}
                if trust:
                    kwargs["trust_remote_code"] = True
                stream = load_dataset(candidate, **kwargs)
                suffix = " (trust_remote_code)" if trust else ""
                print(f"[corpus] streaming from '{candidate}'{suffix}")
                return stream, candidate
            except Exception as exc:  # noqa: BLE001 - we want to try the next mirror
                label = f"{candidate}{' +trust' if trust else ''}"
                errors.append(f"  {label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not open bookCorpus. Attempts:\n" + "\n".join(errors))


def clean(text: str) -> str | None:
    """Return a normalised sentence, or ``None`` if it must be discarded.

    bookCorpus is lowercased and loosely tokenised, so the cleaning is
    deliberately minimal: collapse whitespace and drop sentences that are empty,
    too short to carry meaning, or long enough to be truncated by the 256-token
    window of all-MiniLM-L6-v2.
    """
    if not text:
        return None
    sentence = " ".join(text.split())
    if not (MIN_SENTENCE_CHARS <= len(sentence) <= MAX_SENTENCE_CHARS):
        return None
    if len(sentence.split()) < MIN_SENTENCE_WORDS:
        return None
    return sentence


def collect_sentences(stream, target: int) -> tuple[list[str], dict[str, int]]:
    """Consume the stream until ``target`` clean, unique sentences are gathered."""
    sentences: list[str] = []
    seen: set[str] = set()
    counters = {"read": 0, "empty_or_short": 0, "duplicate": 0}

    for record in stream:
        counters["read"] += 1
        sentence = clean(record.get("text", ""))
        if sentence is None:
            counters["empty_or_short"] += 1
        else:
            key = sentence.lower()
            if key in seen:
                counters["duplicate"] += 1
            else:
                seen.add(key)
                sentences.append(sentence)
                if len(sentences) >= target:
                    break
        if counters["read"] >= MAX_RAW_RECORDS:
            break
        if counters["read"] % 20_000 == 0:
            print(f"[corpus] read={counters['read']:>8}  kept={len(sentences):>6}")

    return sentences, counters


def build_frames(
    sentences: list[str], chunk_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the sentences into chunks and return (sentences_df, chunks_df).

    Chunking is done on *consecutive* sentences so that a chunk keeps the
    narrative locality of the original book, and every sentence carries the id
    of the chunk it came from. Systems ingest the sentence-level frame; the
    chunk-level frame is what the statement asks us to publish.
    """
    rows = [
        {
            "sentence_id": i,
            "chunk_id": i // chunk_size,
            "pos": i % chunk_size,
            "text": sentence,
        }
        for i, sentence in enumerate(sentences)
    ]
    sentences_df = pd.DataFrame(rows)

    chunks_df = (
        sentences_df.sort_values(["chunk_id", "pos"])
        .groupby("chunk_id")["text"]
        .agg(n_sentences="count", text=" ".join)
        .reset_index()
    )
    return sentences_df, chunks_df


def pick_queries(sentences_df: pd.DataFrame, n: int, seed: int) -> list[dict]:
    """Pick ``n`` fixed query sentences, deterministically and spread out.

    They are sampled from the corpus itself because the statement asks for the
    top-2 most similar sentences "among all other sentences": the query is a
    stored sentence and must be excluded from its own result set. One sentence
    is drawn per stratum so the queries cover the whole corpus instead of
    clustering inside a couple of books.
    """
    rng = random.Random(seed)
    total = len(sentences_df)
    stride = total // n
    ids = sorted(rng.randrange(i * stride, (i + 1) * stride) for i in range(n))

    picked = sentences_df.set_index("sentence_id").loc[ids]
    return [
        {
            "query_rank": rank,
            "sentence_id": int(sid),
            "chunk_id": int(row.chunk_id),
            "text": row.text,
        }
        for rank, (sid, row) in enumerate(picked.iterrows())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=TARGET_SENTENCES,
                        help=f"sentences to keep (default: {TARGET_SENTENCES})")
    parser.add_argument("--chunk-size", type=int, default=SENTENCES_PER_CHUNK,
                        help=f"sentences per chunk (default: {SENTENCES_PER_CHUNK})")
    parser.add_argument("--queries", type=int, default=N_QUERIES,
                        help=f"query sentences to freeze (default: {N_QUERIES})")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--dataset", default=None,
                        help="force a specific HuggingFace dataset id")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing artefacts")
    parser.add_argument("--no-csv", action="store_true",
                        help="skip the CSV mirrors")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    if SENTENCES_PATH.exists() and not args.force:
        print(f"{SENTENCES_PATH} already exists. Use --force to rebuild.")
        return 1

    stream, dataset_id = open_stream(args.dataset)

    with Timer() as t_download:
        sentences, counters = collect_sentences(stream, args.n)
    print(f"[corpus] {len(sentences)} sentences collected in {t_download.elapsed:.1f}s "
          f"(read={counters['read']}, discarded={counters['empty_or_short']}, "
          f"duplicates={counters['duplicate']})")

    if len(sentences) < args.n:
        print(f"WARNING: only {len(sentences)}/{args.n} sentences could be collected.")

    sentences_df, chunks_df = build_frames(sentences, args.chunk_size)
    queries = pick_queries(sentences_df, args.queries, args.seed)

    sentences_df.to_parquet(SENTENCES_PATH, index=False)
    chunks_df.to_parquet(CHUNKS_PATH, index=False)
    if not args.no_csv:
        sentences_df.to_csv(SENTENCES_PATH.with_suffix(".csv"), index=False)
        chunks_df.to_csv(CHUNKS_PATH.with_suffix(".csv"), index=False)

    with QUERIES_PATH.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset": dataset_id,
                "seed": args.seed,
                "top_k": TOP_K,
                "n_sentences": int(len(sentences_df)),
                "queries": queries,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    lengths = sentences_df["text"].str.len()
    print(
        f"\n[corpus] {len(sentences_df)} sentences in {len(chunks_df)} chunks "
        f"of {args.chunk_size}\n"
        f"[corpus] sentence length: min={lengths.min()} "
        f"avg={lengths.mean():.1f} max={lengths.max()} chars\n"
        f"[corpus] written: {SENTENCES_PATH.name}, {CHUNKS_PATH.name}, "
        f"{QUERIES_PATH.name}"
    )
    print(f"\n[corpus] the {len(queries)} frozen query sentences:")
    for q in queries:
        preview = q["text"][:80] + ("..." if len(q["text"]) > 80 else "")
        print(f"  #{q['query_rank']:>2}  id={q['sentence_id']:<6} {preview}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
