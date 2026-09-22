# CBDE Lab 1 — Impedance mismatch of vector data

Comparison of how text embeddings are stored and searched in a relational
database (PostgreSQL), a native vector database (Chroma) and a relational
database extended for vectors (pgvector).

- **Corpus**: bookCorpus (HuggingFace, streaming), ~10 000 sentences grouped in
  chunks of 20.
- **Embedding model**: `all-MiniLM-L6-v2` (384 dimensions).
- **Distance metrics**: Euclidean (L2) and cosine, in all three systems.
- **Query workload**: top-2 nearest neighbours for 10 fixed sentences
  (`data/queries.json`), identical everywhere.

## Layout

| Path | Contents |
|---|---|
| `data/prepare_corpus.py` | Phase 1: download, clean, chunk and split the corpus |
| `data/` | `chunks.parquet`, `sentences.parquet`, `queries.json` (+ CSV mirrors) |
| `postgres/` | `P0` load text, `P1` embeddings, `P2` similarity search |
| `chroma/` | `C0`, `C1`, `C2` — same three steps in Chroma |
| `pgvector/` | `G0`, `G1`, `G2` — optional part |
| `results/` | timing measurements produced by each script (JSON) |
| `common.py` | shared data loading, timing, statistics and batching helpers |

## Setup

The environment is not committed; recreate it on whichever machine runs the
experiments (all reported timings must come from the same machine to be
comparable):

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

PostgreSQL runs in Docker (the image also carries the pgvector extension, used
only by the optional part):

```bash
docker compose up -d
```

## Running

```bash
python data/prepare_corpus.py          # phase 1, writes data/
```

The remaining scripts are added in the following phases.
