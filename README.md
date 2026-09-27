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
| `data/` | `chunks.parquet`, `sentences.parquet`, `queries.json` (+ CSV mirrors for GitHub, not read by the scripts) |
| `postgres/` | `P0` load text, `P1` embeddings, `P2` similarity search |
| `chroma/` | `C0`, `C1`, `C2` — same three steps in Chroma |
| `pgvector/` | `G0`, `G1`, `G2` — optional part |
| `results/` | timing measurements produced by each script (JSON) |
| `common.py` | shared data loading, timing, statistics and batching helpers |
| `docker/init/` | SQL run on the first container start (creates `cbde_pgvector`) |
| `docs/` | AI usage log (`ai_log.md`) and, later, the report |

## Setup

Requirements: Git, Python 3.13 and Docker Desktop (WSL 2 backend on Windows).
The environment is not committed; recreate it on every machine.

**1. Clone**

```bash
git clone https://github.com/sgalans/CBDE---L1.git
cd CBDE---L1
```

**2. Python environment**

Windows (PowerShell):

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate            # the prompt must now start with (.venv)
pip install -r requirements.txt
```

macOS / Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate         # the prompt must now start with (.venv)
pip install -r requirements.txt
```

Check that the prompt shows `(.venv)` before running `pip install`; otherwise
the packages end up in the global Python. Run one command at a time.

**3. PostgreSQL (Docker)**

```bash
docker compose up -d                # starts cbde_postgres on localhost:5432
docker ps                           # STATUS should read "(healthy)"
```

The `pgvector/pgvector:pg16` image is plain PostgreSQL 16 with the `vector`
extension installed but not enabled. P0-P2 use the database `cbde` (extension
never enabled); G0-G2 use `cbde_pgvector`.

Both databases are created on the **first** start: `cbde` by the image itself
and `cbde_pgvector` by `docker/init/01-create-pgvector-db.sql`. Init scripts
only run on an empty volume, so if your volume predates that script, check and
create the database once by hand:

```bash
docker exec cbde_postgres psql -U cbde -d cbde -Atc "select datname from pg_database"
docker exec cbde_postgres psql -U cbde -d cbde -c "CREATE DATABASE cbde_pgvector OWNER cbde"
```

Connection parameters are the same
on every machine (`127.0.0.1:5432`, user/password `cbde`) and live in
`common.pg_config()`; they can be overridden with the standard `PG*`
environment variables. The host is deliberately `127.0.0.1` and not
`localhost`: on Windows `localhost` resolves to IPv6 first, and Docker
Desktop's IPv6 forwarding adds ~45 ms to every 32-70 KB message, which would
distort the batch-insertion timings. For a SQL console:

```bash
docker exec -it cbde_postgres psql -U cbde -d cbde
```

## Running

The corpus in `data/` is **already committed**: do not regenerate it
(`prepare_corpus.py --force` could pick a different HuggingFace mirror and
produce a different corpus). Run every script from the repo root with the
`.venv` active. Each script is idempotent (it drops and recreates its tables or
collections) and writes its timings to `results/<script>.json`.

| Order | Script | What it measures |
|---|---|---|
| 1 | `python postgres/P0.py` | text insertion, for several batch sizes |
| 2 | `python postgres/P1.py` | embedding generation + insertion as `REAL[]` |
| 3 | `python postgres/P2.py` | top-2 search with PL/pgSQL L2 and cosine functions |
| 4 | `python chroma/C0.py` | `add()` of text + embeddings, for several batch sizes |
| 5 | `python chroma/C1.py` | embedding generation |
| 6 | `python chroma/C2.py` | top-2 search in the `l2` and `cosine` collections |
| 7 | `python pgvector/G0.py` | *(optional)* text insertion |
| 8 | `python pgvector/G1.py` | *(optional)* embeddings as `vector(384)` |
| 9 | `python pgvector/G2.py` | *(optional)* top-2 with `<->` and `<=>` |

Within each system the order matters (P1 needs P0's table, P2 needs P1's
embeddings); the three systems are independent of each other.

**Official timings** (the ones committed under `results/` and reported in the
document) are produced on a single machine only, so they are comparable.
Timings from any other machine are for testing and must not be committed.

Only needed once, to (re)build the corpus from scratch:

```powershell
python data/prepare_corpus.py
```
