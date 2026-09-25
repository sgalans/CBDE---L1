# CLAUDE.md — CBDE Lab 1: Impedance mismatch de dades vectorials

Context permanent del projecte. Llegeix-lo abans de tocar res.

## Objectiu de la pràctica

Comparar l'emmagatzematge i la cerca de similitud d'embeddings de text en tres
sistemes, per exemplificar l'*impedance mismatch* de les dades vectorials:

1. **PostgreSQL "pur"** — sense pgvector. Els vectors es guarden com a arrays
   natius (`REAL[]`). Scripts `P0`, `P1`, `P2`.
2. **Chroma** — base de dades vectorial nativa. Scripts `C0`, `C1`, `C2`.
3. **pgvector** — extensió vectorial de PostgreSQL. Part **opcional**
   (val 2p extra). Scripts `G0`, `G1`, `G2`.

Per a cada sistema cal mesurar amb `time`:
- temps d'inserció del **text**,
- temps d'inserció/generació dels **embeddings**,
- temps de **càlcul de similitud** (top-2 més semblants) per a 10 frases fixes,
  amb **dues mètriques de distància** diferents.

De cada sèrie de temps: **mínim, màxim, mitjana i desviació estàndard**.

## Decisions ja preses (no les tornis a preguntar)

| Tema | Decisió |
|---|---|
| Corpus | `bookCorpus` de HuggingFace, en mode **streaming**, ~10.000 frases |
| Chunking | blocs de **20 frases** per chunk (~500 chunks) |
| Model d'embeddings | `all-MiniLM-L6-v2` (sentence-transformers), **384 dimensions** |
| Mètriques de distància | **Euclidiana (L2)** i **Cosinus** — a tots tres sistemes |
| PostgreSQL | via **Docker** (`docker-compose.yml` a l'arrel), no instal·lació local |
| Entorn Python | `.venv` creat a cada màquina (mai committejat), Python 3.13 |
| Format de dades | **Parquet** a `data/` |
| Repo | https://github.com/sgalans/CBDE---L1 |

## Màquines de treball

Cada membre de l'equip desenvolupa des del seu portàtil. El repo no conté cap
entorn: a cada màquina es clona i es crea el `.venv` (vegeu README).

- **Mesures finals**: tots els temps que surtin al document (`results/*.json`)
  s'han d'executar al **portàtil d'en Sergi Galán**, perquè siguin comparables. Els
  temps generats en altres màquines serveixen per provar, però no es
  committegen.
- **Dades**: `data/*.parquet` i `data/queries.json` estan committejats i són la
  font única. No es regeneren amb `prepare_corpus.py --force` (el mirror de
  HuggingFace podria canviar i sortiria un corpus diferent).
- **Windows**: executar `.venv\Scripts\activate` per el virtual environment.


## Indicacions explícites del professor (crítiques per a la nota)

- **Insercions per LOTS (batch)**. Cal *experimentar amb diferents mides de lot*
  i mesurar-ne el temps per justificar quina és la més adequada. No inserir
  fila a fila excepte com a baseline comparatiu.
- **El càlcul de distàncies a PostgreSQL (sense pgvector) s'ha de fer DINS de la
  base de dades**, amb una funció o procedure en PL/pgSQL o SQL. No es pot
  baixar els vectors a Python i calcular-hi les distàncies amb numpy. Aquest és
  precisament el punt on es veu l'impedance mismatch.
- Les **mateixes dades i el mateix split per frases** a tots els sistemes.
- Les **10 frases de consulta han de ser les mateixes** a tots els sistemes i
  han d'estar clarament identificades (es guarden a `data/queries.json`).

## Estructura del repo

```
CBDE---L1/
├── data/
│   ├── prepare_corpus.py    # Fase 1: descàrrega, neteja, chunking, split
│   ├── chunks.parquet       # chunks de 20 frases (el "chunk of data" a lliurar)
│   ├── sentences.parquet    # split per frases: sentence_id, chunk_id, pos, text
│   └── queries.json         # les 10 frases fixes de consulta
├── postgres/                # P0 (text), P1 (embeddings), P2 (similitud)
├── chroma/                  # C0, C1, C2
├── pgvector/                # G0, G1, G2 (opcional)
├── results/                 # JSON amb els temps mesurats de cada script
├── common.py                # utilitats compartides
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## `common.py` — què hi ha

Càrrega de dades (`load_sentences`, `load_chunks`, `load_queries`), cronòmetre
(`Timer`), estadístiques (`stats` → min/max/avg/std), iteració per lots
(`batched`), persistència de resultats (`save_results`), paràmetres de connexió
a Postgres i constants del model. **Tot script nou ha de fer servir aquestes
utilitats**, no reimplementar-les.

## Entregables finals

1 document (màx. 10 pàgines) + 6 scripts (9 amb la part opcional). El document
ha d'incloure el link al repo, les decisions d'impedance mismatch de cada
sistema, les respostes a [PQ1] i [CQ1], la discussió comparativa, i **un apartat
obligatori explicant com s'ha fet servir la IA** (racional dels prompts, com
s'han refinat i com s'ha validat el resultat).

## Estil de treball

- Codi i comentaris en **anglès**; conversa amb l'usuari en **català**.
- Els scripts han de ser executables sols i idempotents (`--drop` / recreació de
  taules i col·leccions).
- Cada script escriu els seus temps a `results/<nom>.json` perquè el document
  final es pugui construir a partir d'aquests fitxers.
