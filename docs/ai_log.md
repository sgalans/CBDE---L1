# Registre d'ús de la IA

Font per a l'apartat obligatori "Ús de la IA" del document final. Cada entrada
recull una decisió o tasca on ha intervingut la IA: **què es va demanar**,
**què va respondre** i **com es va validar**. Entrades breus, en ordre
cronològic, afegides en el moment (no reconstruïdes al final).

Eines:

- **Claude Code** (VS Code): assistent principal, amb accés al repo; llegeix
  fitxers, executa scripts i comprova resultats a la màquina.
- **Assistent conversacional extern**: revisor de les decisions de disseny.
- **`CLAUDE.md`**: context persistent (objectiu, decisions preses, indicacions
  del professor) que Claude Code llegeix a cada sessió, per no haver de
  repetir-lo a cada prompt i evitar propostes que contradiguin l'enunciat.

---

### Esquelet del repo i fase 1 (commit `08c8f10`)

- **Pendent de completar** per qui ho va fer: com es va generar l'esquelet,
  `common.py` i `prepare_corpus.py`, i amb quins prompts.

### 2026-09-25 — Preparació de l'entorn al portàtil

- **Demanat:** què cal instal·lar per treballar-hi.
- **IA:** va inspeccionar la màquina abans de respondre (Python, Docker, WSL2,
  port 5432, RAM, disc) i va llistar només el que faltava: Docker Desktop i
  Python 3.13.
- **Incidència:** en enganxar totes les comandes alhora, `activate` va fallar
  per l'*execution policy* de PowerShell i `pip` va instal·lar els paquets al
  Python 3.12 global. La IA ho va diagnosticar a partir de la sortida
  (`Defaulting to user installation`, wheels `cp312`) i va comprovar que el
  `.venv` era correcte però buit.
- **Validació:** script que importa totes les dependències dins el `.venv` i
  es connecta a PostgreSQL (versió 16.15, extensió `vector` disponible).
- **Resultat:** es va resoldre amb `Set-ExecutionPolicy -Scope CurrentUser
  RemoteSigned`. El README recomana comprovar el prompt `(.venv)` abans del
  `pip install` i executar les comandes d'una en una.

### 2026-09-25 — Execució i revisió de la fase 1 (corpus)

- **Demanat:** explicar què havia fet `prepare_corpus.py` i els fitxers nous.
- **IA:** va llegir el script i va inspeccionar els fitxers generats en comptes
  d'explicar-ho de memòria.
- **Validació:** 10.000 frases úniques, 500 chunks d'exactament 20, longitud
  40–392 caràcters, 10 consultes estratificades i reproduïbles (`seed = 42`),
  mirror utilitzat `rojagtap/bookcorpus`.
- **Observació derivada:** les consultes són frases del corpus, així que el
  top-2 ha d'excloure la pròpia frase.

### 2026-09-27 — Mètriques: L2 i cosinus són equivalents

- **Demanat (revisor extern):** revisar les decisions del `CLAUDE.md`.
- **IA externa:** `all-MiniLM-L6-v2` normalitza els vectors, i per a vectors
  unitaris ‖a−b‖² = 2 − 2·cos(a,b), de manera que els rànquings coincideixen.
  Proposava canviar el cosinus per L1.
- **Validació (Claude Code):** el pipeline del model acaba amb `Normalize`;
  normes entre 0,9999998 i 1,0000001. Mostra de 2.000 frases × 10 consultes:
  top-3 L2 = top-3 cosinus en 10/10; L1 diferia en 7/10.
- **Error detectat a la proposta:** Chroma només suporta `l2`, `ip` i `cosine`;
  amb L1 es trencaria "mateixes mètriques a tots els sistemes".
- **Decisió:** es mantenen L2 i cosinus; l'equivalència s'explica a
  [PQ1]/[CQ1] i es verifica als scripts de similitud. Sense L1 (tampoc com a
  tercera mètrica, per no complicar).

### 2026-09-27 — Chroma: col·leccions i separació text/embeddings

- **IA externa:** (1) la mètrica es fixa per col·lecció, calen dues;
  (2) cal generar els embeddings fora de Chroma i passar-los explícitament per
  poder mesurar-ne el temps per separat.
- **Validació (Claude Code):** experiment mínim amb `chromadb` 1.5.9:
  - `configuration={"hnsw": {"space": ...}}` es fixa en crear la col·lecció →
    (1) confirmat;
  - sense embeddings, la `DefaultEmbeddingFunction` els calcula dins l'`add()`
    → (2) confirmat;
  - **no previst per la IA externa:** amb `embedding_function=None`,
    `add(documents=...)` sense vectors llança `ValueError`. Chroma no permet
    inserir text sense vector.
- **Decisió:** dues col·leccions (`sentences_l2`, `sentences_cosine`) i
  embeddings propis. Es van plantejar dues opcions per a C0/C1 (A: `add()` de
  text + vector alhora; B: vectors de farciment + `update()`); l'equip va triar
  la **A** per ser l'ús natural de Chroma. L'asimetria respon [CQ1].

### 2026-09-27 — Revisió de buits del `CLAUDE.md`

- **IA externa:** va assenyalar cinc buits (imatge Docker, excloure la pròpia
  frase a P2, paràmetres de connexió, criteri "performance / línies de codi /
  crides", README).
- **Validació:** abans d'aplicar-los es van contrastar amb el repo. La imatge
  `pgvector/pgvector:pg16` **ja** era la del `docker-compose.yml` (el buit era
  només de documentació); els paràmetres de connexió es van comprovar a
  `common.pg_config()`.
- **Resultat:** tots cinc documentats al `CLAUDE.md`; README amb setup complet
  i ordre d'execució.

### 2026-09-27 — Revisió de C0/C1 contra l'enunciat literal

- **Context:** amb l'enunciat complet a `docs/enunciat.md`, Claude Code va
  contrastar-hi les decisions preses i va detectar que amb l'opció A, `C1`
  només generava embeddings, mentre que l'enunciat diu que ha de *"generate
  and store"*. Un error de disseny introduït per treballar amb un resum de
  l'enunciat en lloc del text original.
- **Proposta:** `C0` carrega el text amb la funció d'embeddings per defecte
  (el temps inclou per força el càlcul intern → resposta a [CQ1]); `C1`
  llegeix el text de la col·lecció, genera els embeddings amb el nostre model
  i els desa amb `update()`.
- **Validació:** prova amb `chromadb` 1.5.9: `add(documents)` amb la funció
  per defecte funciona; `update(embeddings)` substitueix els vectors;
  `query(where={"sentence_id": {"$ne": id}})` exclou la pròpia frase. Els
  vectors ONNX de Chroma i els de sentence-transformers difereixen com a màxim
  1,3·10⁻⁷ (l'afirmació anterior que podien ser "lleugerament diferents" era
  certa però irrellevant).
- **Decisió:** l'equip adopta la variant; substitueix l'opció A.

### 2026-09-27 — Segona revisió de buits (Docker, multiplataforma, lots)

- **IA externa:** cinc punts: (1) `cbde_pgvector` no es crea sola; (2) el
  README només té instruccions per a Windows; (3) els miralls CSV no consten
  al `CLAUDE.md`; (4) el grid de mides de lot no està fixat; (5) recordar que
  HNSW és cerca aproximada.
- **Validació (Claude Code):**
  - (1) **confirmat**: al contenidor només existien `cbde` i `postgres`. Es va
    afegir `docker/init/01-create-pgvector-db.sql` i es va provar en un
    contenidor temporal amb volum buit (es creen les dues BD i `vector` no
    s'activa a `cbde`). Detall no previst per la IA externa: els scripts
    d'init **només s'executen amb el volum buit**, així que al volum existent
    la BD es va crear a mà i el README explica com fer-ho.
  - (4) **incorrecte**: el grid ja estava fixat a `common.BATCH_SIZES`; només
    faltava documentar-lo com a decisió compartida.
  - (2), (3), (5): correctes, documentats. A (5) s'hi va afegir la idea de fer
    servir `P2` (exacte) com a referència per mesurar el *recall* de Chroma.

### 2026-09-27 — Eina per al document

- **IA externa:** Quarto, Typst, Overleaf o Jupyter.
- **Claude Code:** recomana Quarto amb sortida Typst (sense LaTeX), amb les
  taules generades des de `results/*.json` perquè no es copiïn a mà.
- **Decisió:** Quarto → Typst, document en català a `docs/informe.qmd`.
  L'enunciat original es desa a `docs/enunciat.md` i el `CLAUDE.md` hi remet
  com a font de veritat, perquè la IA contrasti les decisions amb el text
  literal i no amb resums.
