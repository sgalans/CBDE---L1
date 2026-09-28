# CLAUDE.md — CBDE Lab 1: Impedance mismatch de dades vectorials

Context permanent del projecte. Llegeix-lo abans de tocar res.

**L'enunciat original complet és a [`docs/enunciat.md`](docs/enunciat.md).** És
la font de veritat: si alguna cosa d'aquest fitxer el contradiu, mana
l'enunciat (i cal avisar l'usuari).

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
| Mètriques de distància | **Euclidiana (L2)** i **Cosinus** — a tots tres sistemes. **Sense L1** (vegeu nota) |
| PostgreSQL | via **Docker** (`docker-compose.yml` a l'arrel), no instal·lació local. Imatge **`pgvector/pgvector:pg16`** per a tot (vegeu nota) |
| Connexió a Postgres | `common.pg_config()`: per defecte **`127.0.0.1:5432`** (no `localhost`: a Windows es resol a IPv6 i Docker Desktop hi afegeix ~45 ms per missatge de 32–70 KB, verificat), usuari/contrasenya `cbde`, iguals que el `docker-compose.yml` i a totes les màquines. Sense `.env`; només es poden sobreescriure amb les variables `PG*` |
| Mides de lot | Grid **únic** per a `P0`/`P1`, `C0` i `G0`/`G1`: `common.BATCH_SIZES = (1, 10, 50, 100, 500, 1000, 2000)`; la mida 1 és el baseline fila a fila. Càrrega "oficial" amb **`common.DEFAULT_BATCH_SIZE = 1000`**, triada amb el grid de `P0`/`P1` pel punt on la corba s'aplana: per al text 2000 gairebé no millora, per als embeddings a partir de ~500 no hi ha guany mesurable (dins la desviació), i lots més grans només fan transaccions i missatges més grans. Al document, justificar-ho **pel comportament del sistema**, no per "tenir més lots per a les estadístiques". Cap script defineix el seu propi grid |
| Base de dades pgvector | `cbde_pgvector`, creada per `docker/init/01-create-pgvector-db.sql` en la primera arrencada del volum (o a mà, vegeu README). `G0` ha de fallar amb un missatge clar si no existeix |
| Top-2 | **Sempre s'exclou la pròpia frase** de la consulta (`sentence_id` de `queries.json`) a `P2`, `C2` i `G2`: "els 2 més semblants entre *totes les altres* frases" |
| Entorn Python | `.venv` creat a cada màquina (mai committejat), Python 3.13 |
| Format de dades | **Parquet** a `data/` |
| Repo | https://github.com/sgalans/CBDE---L1 |

### Nota: L2 i cosinus donen el mateix rànquing (i és intencionat)

`all-MiniLM-L6-v2` acaba amb una capa `Normalize`: tots els embeddings tenen
norma 1. Per a vectors unitaris, ‖a−b‖² = 2 − 2·cos(a,b), així que **els top-2
per L2 i per cosinus seran idèntics** a tots tres sistemes (verificat
empíricament a la mostra del corpus). No és un error de disseny:

- Es manté L2 + cosinus perquè **Chroma només suporta `l2`, `ip` i `cosine`**, i
  amb vectors normalitzats les tres són equivalents. Canviar a L1 obligaria a
  trencar "mateixes mètriques a tots els sistemes". Tampoc s'afegeix L1 com a
  tercera mètrica (decisió de l'equip: complica sense aportar prou).
- Els **temps** sí que poden diferir entre mètriques (el cosinus fa més
  operacions), i això és el que es compara.
- Els scripts de similitud (`P2`, `C2`, `G2`) han de **comprovar** que els top-2
  coincideixen i que les distàncies compleixen la relació, perquè el document
  ho pugui afirmar amb dades.
- El document ho ha d'**explicar explícitament** a [PQ1]/[CQ1]; si no, sembla
  que les mètriques s'han triat sense entendre'n la geometria.

### Nota: una sola imatge Docker per a PostgreSQL i pgvector

`pgvector/pgvector:pg16` és PostgreSQL 16 estàndard amb l'extensió `vector`
**instal·lada però no activada**. `P0`–`P2` fan servir la base de dades `cbde`,
on **mai** s'executa `CREATE EXTENSION vector` (seria fer trampa a la part
"pura"). `G0`–`G2` fan servir una base de dades separada, `cbde_pgvector`, on
sí que s'activa. Un sol `docker-compose.yml` i cap interferència entre parts.
No canviar a la imatge oficial `postgres`: no porta l'extensió.

### Nota per al document: cerca exacta vs. aproximada

- **PostgreSQL pur (`P2`)**: la funció PL/pgSQL recorre totes les files →
  cerca **exacta** (força bruta).
- **Chroma (`C2`)**: sempre fa servir un índex **HNSW** → cerca
  **aproximada**; els top-2 podrien no coincidir exactament amb els de `P2`.
- **pgvector (`G2`)**: sense índex és **exacta**; amb índex HNSW/IVFFlat és
  **aproximada**. Cal deixar clar al document quina configuració s'ha fet
  servir.
- Els resultats exactes de `P2` serveixen de referència: `C2` (i `G2` amb
  índex) poden calcular quants top-2 coincideixen amb `P2` (*recall*). És un
  punt fort per a [CQ1] i la comparativa final.

### PostgreSQL: decisions de disseny (P0/P1/P2)

- `P1`, segons l'enunciat, *"connects to your database, generates the
  embeddings for each sentence, and store the embeddings"*: el text es
  **llegeix de la BD** (no del Parquet), es generen els embeddings i es tornen
  a escriure. Aquest viatge d'anada i tornada BD ↔ Python és part de
  l'impedance mismatch i s'ha de mesurar.
- `P1` (implementat): taula separada
  `sentence_embeddings(sentence_id PK/FK, embedding REAL[])`, no `UPDATE`
  sobre `sentences` (l'UPDATE reescriu la fila i deixa tuples mortes). Tres
  temps separats: lectura, generació, emmagatzematge.
- **El cost d'emmagatzemar vectors és serialitzar floats a text** (verificat
  amb perfil): l'adaptació per defecte de psycopg2 (`ARRAY[...]`, cada float
  un literal `float8`; 8,25 MB de SQL per 1000 vectors) és ~3× més lenta que
  enviar cada vector com un sol literal `'{...}'::real[]`. `P1` fa servir el
  literal i mesura l'adaptació per defecte com a referència. `repr` garanteix
  que el valor tornat és bit-idèntic (error de round-trip 0, validat).
- **Aritmètica en `float8`**: elevar al quadrat components `REAL` molt petits
  dona `value out of range: underflow` (PostgreSQL no arrodoneix a 0). Les
  funcions de distància de `P2` han de fer els càlculs amb `float8`.

- `P2`: el top-2 s'ha de filtrar amb `WHERE sentence_id <> <id de la consulta>`
  **dins de la funció SQL/PL/pgSQL**, no a Python després. Si no, la pròpia
  frase surt sempre primera amb distància 0.
- `P2` (implementat): `l2_distance` i `cosine_distance` en `LANGUAGE sql`
  sobre `unnest(a, b)` en `float8`; `top_k_similar(q_id, metric, k)` en
  PL/pgSQL (una branca per mètrica, no `CASE` per fila). 1 crida per consulta
  i mètrica. numpy **només** valida el resultat (top-2 idèntic 10/10), no el
  calcula. Resultat verificat: top-2 L2 = cosinus 10/10, |L2² − 2·d_cos| ≤ 1e-7.

### Chroma: decisions de disseny (C0/C1/C2)

Verificat amb `chromadb` 1.5.9:

- **Client integrat** (`common.chroma_client()` = `PersistentClient` a
  `chroma/chroma_db/`, telemetria desactivada): Chroma corre dins el procés de
  Python, sense xarxa, mentre que PostgreSQL passa per TCP. Cal dir-ho al
  document en comparar temps. Alternativa no adoptada: servidor Chroma a Docker.
- **Durada**: cada càrrega de `C0` recalcula els 10.000 embeddings (~15–20 s
  a CPU); el grid complet, amb la mida 1 inclosa, pot trigar 10–15 min. S'ha
  d'executar **en segon pla** i avisant abans. **No** es retallen mides del
  grid per estalviar temps (trencaria el "grid únic").
- **Comparabilitat de `C0`**: els seus temps inclouen la generació
  d'embeddings (ONNX, dins l'`add()`), així que **no es comparen amb `P0` sol,
  sinó amb `P0` + la generació de `P1`**. Cal dir-ho explícitament a la secció
  de Chroma; és la prova de [CQ1] que text i vector no es poden separar.
- **Càrrega oficial de `C0`**: **3 càrregues a `sentences_l2`** (com `P0`, per
  poder comparar l'estabilitat entre càrregues a [CQ1]) i 1 a
  `sentences_cosine` (`--official-repeats`). `--skip-grid` reaprofita el grid
  i la referència d'embeddings del `results/C0.json` existent.
- **`C1` repeteix 3 vegades l'`update()`** de cada col·lecció (`--repeats`), com
  l'emmagatzematge de `P1`; la generació es mesura un sol cop.
- **Variant de referència a `C2`**: `l2_nofilter` / `cosine_nofilter` demanen
  k + 1 veïns i descarten la pròpia frase a Python, per mesurar el cost del
  filtre `where` ([CQ1] c). La resposta oficial manté l'exclusió dins la BD.
- **`add()` − embedding = cost d'emmagatzemar a `C0`**: amb la referència
  "només embedding ONNX" es pot comparar amb `update()` de `C1` (cost de
  reubicar vectors a l'índex HNSW).
- **Resultats verificats de Chroma** (per no reinterpretar-los): el 97 % d'un
  `add()` és embedding ONNX; l'ONNX és ~9× més lent que el model PyTorch;
  `update()` costa el mateix que l'`INSERT` de `P1` (no és més ràpid);
  consultes ~20× més ràpides que `P2` amb *recall* 100 %; el filtre `where`
  multiplica ~14× el temps de consulta. L'espai `cosine` **re-normalitza** els
  vectors en desar-los (canvis ≤ 1,5e-8 en ~28 % dels vectors), per això la
  validació de `C1` només exigeix igualtat exacta a `l2`.
- **`onnx_vs_model_max_abs_diff` només és vàlid si `C1` s'executa just després
  de `C0`**: si `C1` ja havia substituït els vectors, el JSON guarda `null`.
  A les mesures oficials, executar sempre `C0 → C1 → C2` seguits.
- **Grid de `C0` (decisió de l'equip, per temps)**: el grid complet només a la
  col·lecció `sentences_l2` i amb **1 repetició** (a `P0` en són 3). La
  càrrega oficial sí que es fa a les dues col·leccions. Que la mètrica afecti
  poc la inserció **no es pot afirmar sense dades**: s'ha de citar la
  comparació L2 vs. cosinus de la càrrega oficial (lots de 1000). Tampoc s'ha
  de dir que la doble generació d'embeddings passa al grid: només passa a la
  càrrega oficial.

- **Dues col·leccions** amb les mateixes dades: `sentences_l2` i
  `sentences_cosine`. La mètrica es fixa en crear la col·lecció
  (`configuration={"hnsw": {"space": ...}}`) i no es pot canviar per consulta.
  Punt d'impedance mismatch per a [CQ1]: a PostgreSQL una sola taula serveix
  per a totes dues mètriques; a Chroma cal duplicar dades i índex.
- **Chroma no permet inserir text sense vector**: amb `embedding_function=None`,
  `add(documents=...)` sense `embeddings` llança `ValueError`. Amb la
  `DefaultEmbeddingFunction` (ONNX de `all-MiniLM-L6-v2`), l'`add()` calcula
  els vectors internament. En cap cas es pot inserir el text "sol": separar la
  inserció de text de la creació d'embeddings no és natiu (resposta a [CQ1]).
- La `DefaultEmbeddingFunction` i `common.get_model()` donen vectors
  pràcticament idèntics (diferència màxima 1,3·10⁻⁷, verificat).
- **Repartiment de fases (decidit; substitueix l'antiga "opció A")**, per
  complir l'enunciat al peu de la lletra (`C1` ha de *generar i guardar*):
  - `C0`: crea les dues col·leccions amb la funció d'embeddings per defecte i
    fa `add(ids, documents, metadatas)` **per lots** (mateix grid que `P0`).
    El temps d'inserció de text **inclou per força** el càlcul intern dels
    embeddings: és la demostració de [CQ1]. Els vectors es calculen un cop per
    col·lecció (dues vegades en total): cal comentar-ho al document.
    Abans de cronometrar cal fer un *warm-up* de la funció per defecte (la
    primera crida baixa el model ONNX, ~80 MB, a `~/.cache/chroma`).
  - `C1`: llegeix el text **de la col·lecció** (`get(include=["documents"])`,
    com `P1` llegeix de la BD), genera els embeddings amb `common.get_model()`
    (temps de generació, comparable amb `P1`) i els desa amb
    `update(ids, embeddings=...)` per lots (temps d'emmagatzematge, per
    separat). A partir d'aquí Chroma té exactament els mateixos vectors que
    PostgreSQL.
  - `C2`: top-2 per a les 10 consultes a cada col·lecció amb
    `query(query_embeddings=..., where={"sentence_id": {"$ne": id}})` per
    excloure la pròpia frase. El vector de consulta és l'emmagatzemat (com a
    `P2`), **no** `query_texts` (afegiria el temps d'embedding a la consulta).
  - Metadades de cada registre: `sentence_id`, `chunk_id`, `pos` (el mateix
    split que PostgreSQL).

### pgvector: decisions de disseny (G0/G1/G2)

- **`G0` reutilitza el codi de `P0`** (`from postgres import P0`): mateixa
  taula, mateixos tipus, mateix `INSERT` per lots. Només canvia la BD
  (`cbde_pgvector`, amb `CREATE EXTENSION vector` a `common.pgvector_connect()`).
  Que el text no necessiti codi nou és un argument per a la Discussió:
  pgvector **estén** el model relacional, no el substitueix. No es repeteix
  la referència `COPY` (mesuraria el mateix que a `P0`).
- **`G1`**: llegeix el text amb `P1.read_sentences` (mateix SQL), genera amb
  `common.generate_embeddings` i desa `vector(384)` amb l'adaptador oficial
  `pgvector.psycopg2` (`register_vector`). L'adaptador envia cada vector com
  **un sol literal de text** `'[x1,...]'`, igual que el literal `'{...}'::real[]`
  de `P1`: són directament comparables. Mateix grid que `P1` (3 repeticions).
- **Índexs HNSW a `G1`, construïts DESPRÉS de carregar** (pràctica recomanada
  per pgvector), un per mètrica (`vector_l2_ops`, `vector_cosine_ops`): com a
  Chroma, un índex serveix una sola distància, però **les dades es desen una
  sola vegada**. Chroma, en canvi, manté l'índex durant cada `add()`: cal
  dir-ho en comparar temps d'inserció. Es mesura el temps de construcció (3
  repeticions) i la mida de taula i índexs.
- **`G2`**: una sola crida SQL per consulta (patró documentat de pgvector,
  `ORDER BY embedding <-> (SELECT ...) LIMIT k` amb `WHERE sentence_id <> q`),
  el vector no surt de la BD, com a `P2`. Dues configuracions sobre la mateixa
  taula: **exacta** (`SET enable_indexscan = off`, força bruta amb la distància
  en C) i **hnsw** (aproximada). `EXPLAIN` comprova que l'índex s'usa només a
  `hnsw`; la configuració exacta **ha de** coincidir al 100 % amb `P2` o el
  script falla. `<->` retorna la L2 **sense** elevar al quadrat (a diferència
  de Chroma): la identitat es comprova com L2² = 2·d_cos.
- `common.exact_reference()` i `common.recall_at_k()` són compartides per `C2`
  i `G2` (abans `C2` tenia la seva còpia).
- **Resultats verificats de pgvector** (per no reinterpretar-los): text igual
  que `P0`; desar `vector(384)` ~13 % més ràpid que `REAL[]`; cerca exacta
  ~90× més ràpida que `P2` (mateix algorisme, distància en C); HNSW ~250×
  `P2` i ~13× Chroma amb filtre, *recall* 100 % (`ef_search` = 40); cada índex
  HNSW ocupa més que la taula. Per llegir `hnsw.ef_search` cal haver carregat
  la llibreria a la sessió (`SELECT NULL::vector`).

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
│   ├── chunks.csv, sentences.csv  # miralls per llegir-los a GitHub (els scripts no els fan servir)
│   └── queries.json         # les 10 frases fixes de consulta
├── docker/init/             # SQL executat en la primera arrencada del volum (crea cbde_pgvector)
├── postgres/                # P0 (text), P1 (embeddings), P2 (similitud)
├── chroma/                  # C0, C1, C2
├── pgvector/                # G0, G1, G2 (opcional)
├── results/                 # JSON amb els temps mesurats de cada script
├── docs/
│   ├── enunciat.md          # enunciat original (font de veritat)
│   ├── informe.qmd          # el document a lliurar (Quarto → PDF)
│   └── ai_log.md            # registre d'ús de la IA (per a l'apartat del document)
├── common.py                # utilitats compartides
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## `common.py` — què hi ha

Càrrega de dades (`load_sentences`, `load_chunks`, `load_queries`), cronòmetre
(`Timer`), estadístiques (`stats` → min/max/avg/std), iteració per lots
(`batched`), persistència de resultats (`save_results`), paràmetres de connexió
a Postgres, client i col·leccions de Chroma (`chroma_client`, `chroma_collections`, `CHROMA_COLLECTIONS`), constants
del model i la generació d'embeddings per lots amb temps i *warm-up*
(`generate_embeddings`, compartida per `P1`, `C1` i `G1`), la connexió a pgvector
(`pgvector_connect`) i la referència exacta i el *recall* dels top-k
(`exact_reference`, `recall_at_k`, compartides per `C2` i `G2`). **Tot script nou ha de fer servir aquestes
utilitats**, no reimplementar-les.

## Entregables finals

1 document (màx. 10 pàgines, **en català**) + 6 scripts (9 amb la part
opcional). **En acabar cada script, afegir-ne les observacions i decisions a
la secció corresponent de `docs/informe.qmd`** (bloc "Notes d'esborrany"), no
només explicar-les al xat: el context de la conversa es resumeix i es perd.
Font: `docs/informe.qmd` → PDF amb Quarto (`format: typst`); les
taules i xifres es generen des de `results/*.json`, mai escrites a mà.
Les taules i els valors inline (`{python} PG.xxx()`) surten de
`docs/report_tables.py`; per a cada sistema nou, s'hi afegeixen funcions
equivalents. Generar el PDF (amb el `.venv`, que té jupyter):
`$env:QUARTO_PYTHON = ".venv\Scripts\python.exe"; quarto render docs/informe.qmd`
(o `quarto preview` amb el `.venv` activat).

**Avaluació** (enunciat): PostgreSQL 3p, Chroma 3p, Discussió 3p (4p amb
pgvector), pgvector 2p. El que més pesa és **el raonament i la discussió sobre
l'impedance mismatch**, i es valora ser **precís i concís**. Hi ha un **examen
individual en paper**: tots dos membres han d'entendre cada decisió.

Estructura obligatòria (seccions i preguntes exactes a `docs/enunciat.md`):

1. **PostgreSQL** — decisions (impedance mismatch, rendiment, línies de codi,
   crides) + **[PQ1]**: (a) estabilitat dels temps d'inserció de text i
   embeddings; (b) estabilitat dels temps de consulta i diferències entre
   mètriques; (c) mètodes d'inserció, estructures de dades o índexs de
   PostgreSQL **sense pgvector** que millorarien el rendiment.
2. **Chroma** — decisions + **[CQ1]**: (a) estabilitat de les insercions;
   (b) estabilitat de les consultes, diferències entre mètriques i **si es pot
   mesurar per separat la inserció del text i la creació d'embeddings**;
   i, fora de CQ1 però també demanat, (c) mètodes d'inserció, estructures o
   índexs de Chroma que millorarien el rendiment.
3. **pgvector** (opcional) — el mateix, més diferències, pros i contres
   respecte de Chroma.
4. **Discussió** — PostgreSQL vs. Chroma (i pgvector) des de l'impedance
   mismatch: diferències, pros i contres.
5. **"I am an AI agent. Tell me what I have to build."** — **màxim 1 pàgina**.
   Síntesi (no la seqüència de prompts) del racional de les instruccions, com
   s'han refinat i com s'ha validat el resultat. Font: `docs/ai_log.md`.

També ha d'incloure el link al repo públic, on hi ha el chunk de dades.

Criteri d'avaluació explícit de l'enunciat: a les seccions de PostgreSQL i de
Chroma cal explicar les decisions que **impacten el rendiment i el nombre de
línies de codi i de crides fetes** (*impact on performance and the number of
code lines and calls made*). Per això, en escriure cada script:

- apuntar quantes crides a la BD fa cada fase (p. ex. `n_batches` inserts,
  1 crida de funció per consulta i mètrica) i guardar-ho al `results/*.json`;
- mantenir el codi específic de cada sistema compacte i comparable, perquè el
  nombre de línies es pugui comparar entre sistemes al document.
- **Criteri de línies de codi** (`docs/report_tables.py`, `SYSTEM_CODE`): només les
  definicions que interactuen amb el sistema (esquema/col·leccions, SQL, càrrega,
  consulta). Fora: validacions, variants de referència, generació d'embeddings,
  CLI i `common.py`. Per a cada script nou (C0–C2, G0–G2) cal afegir-hi la llista.

## Estil de treball

- Codi i comentaris en **anglès**; conversa amb l'usuari en **català**.
- Els scripts han de ser executables sols i idempotents (`--drop` / recreació de
  taules i col·leccions).
- Cada script escriu els seus temps a `results/<nom>.json` perquè el document
  final es pugui construir a partir d'aquests fitxers.
- **Registre d'ús de la IA (`docs/ai_log.md`)**: és la font de l'apartat
  obligatori del document. Afegir-hi una entrada breu (data, què es va
  demanar, què va respondre la IA, com es va validar, decisió) **en el moment**
  en què la IA intervingui en una decisió de disseny, proposi o generi codi
  rellevant, o quan una proposta d'IA (pròpia o externa) es corregeixi o es
  descarti després de verificar-la. No cal registrar tasques trivials.
- Noms dels scripts: `postgres/P0.py`, `chroma/C0.py`, `pgvector/G0.py`, etc.
  (`<nom>` = `P0`, `C0`…). El README en documenta l'ordre d'execució; si
  canvia, s'actualitza allà.
