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

### P0 — Càrrega del text i anomalia de xarxa

- **Demanat:** escriure `P0` seguint el `CLAUDE.md` (grid de lots, idempotent,
  resultats a JSON).
- **IA:** una sola taula `sentences` (`INTEGER`, `SMALLINT`, `TEXT`), inserció
  amb `execute_values` (un `INSERT` multi-fila i un `commit` per lot) per a tot
  el grid, més `COPY` a la mida per defecte com a referència per a [PQ1](c).
- **Validació:** a la primera execució, a partir de 500 files cada lot trigava
  ~50 ms **independentment de la mida** (i `COPY` igual). La IA no va acceptar
  el resultat: una latència constant indica xarxa, no base de dades. Un
  experiment de latència per mida de missatge va mostrar una aturada de
  ~44 ms entre ~32 i ~70 KB **només** connectant per `localhost` (que a
  Windows es resol a `::1`); per `127.0.0.1` no hi era.
- **Decisió:** `common.pg_config()` es connecta a `127.0.0.1`. Amb el canvi,
  la corba és coherent (mida 1: ~20 s; 500–2000: ~0,1 s). Sense aquesta
  verificació, el document hauria atribuït a PostgreSQL un artefacte de Docker
  Desktop.
- **Mida de lot per defecte:** amb el grid (3 repeticions), la IA va proposar
  passar de 500 a 1000 (−28 % de temps; 2000 només guanya un 15 % més i deixa
  5 lots, massa pocs per a min/max/avg/std). L'equip ho va acceptar.

### P1 — Embeddings com a `REAL[]`

- **Demanat:** escriure `P1` (llegir de la BD, generar, emmagatzemar).
- **IA:** taula separada `sentence_embeddings` (INSERT en lloc d'UPDATE),
  tres fases cronometrades per separat i validació final dins de la BD.
- **Validació i correccions:**
  - La validació (norma calculada en SQL) va fallar amb `underflow`: elevar al
    quadrat un `float4` molt petit surt del rang i PostgreSQL llança error.
    Es calcula en `float8`; la mateixa regla s'aplicarà a `P2`.
  - L'emmagatzematge trigava ~8 s **independentment de la mida de lot** (≥ 50),
    cosa que indicava que el coll d'ampolla no eren les crides. Un perfil per
    passos d'un lot de 1000 va mostrar que el temps era la conversió
    float → text: psycopg2 genera 8,25 MB de SQL (`ARRAY[...]` amb 384.000
    literals `float8`) i el servidor l'ha d'analitzar. Enviar cada vector com
    un literal `'{...}'` ho redueix ~3× (8,7 s → 2,8 s).
- **Decisió:** `P1` fa servir el literal; l'adaptació per defecte es manté com
  a referència mesurada per a [PQ1]. Round-trip verificat bit a bit.

### P2 — Similitud dins de PostgreSQL

- **Demanat:** escriure `P2` complint la indicació del professor (distàncies
  calculades dins la BD).
- **IA:** funcions `LANGUAGE sql` per a les distàncies (en `float8`, aplicant
  la lliçó de l'underflow de P1) i una funció PL/pgSQL per al top-k que exclou
  la pròpia frase amb `WHERE`.
- **Validació:** el script contrasta, fora de la secció cronometrada, el
  resultat de la BD amb una força bruta en numpy (10/10 idèntic), comprova que
  el top-2 de les dues mètriques coincideix (10/10) i que es compleix
  L2² = 2·d_cos (error ≤ 1e-7). Així l'equivalència de mètriques decidida
  abans queda demostrada amb les dades reals, no només amb una mostra.
- **Nota:** numpy s'hi fa servir només com a oracle de validació; es va
  deixar explícit al codi per no contradir la indicació del professor.

### Redacció de [PQ1] i cosinus unitari

- **Correcció de l'usuari:** després d'acabar P0–P2, el document encara era
  ple de TODOs. La IA havia ajornat la redacció "fins a tenir les xifres
  oficials", però el disseny del document (xifres llegides dels JSON) feia
  innecessari esperar. Es va afegir al `CLAUDE.md` la regla d'escriure les
  observacions al document en acabar cada script.
- **IA:** `docs/report_tables.py` genera les taules i els valors inline a
  partir de `results/*.json`; el text de [PQ1] cita aquests valors en lloc de
  xifres escrites a mà. A proposta de la IA i amb l'acord de l'equip, `P2`
  mesura també `1 − a·b` (vàlid per a vectors unitaris) per demostrar amb
  dades una millora de [PQ1](c): mateix top-2 (10/10) i ~23 % més ràpid.
- **Validació:** el document es va renderitzar també a Markdown per revisar
  que cada valor inline coincidís amb els JSON. En tornar a mesurar, `COPY`
  va passar d'un 11 % (lots de 500) a un 25 % (lots de 1000) de guany; el
  text, que deia "només", es va reescriure sense prejutjar la magnitud.

### C0 — Disseny i una proposta de la IA descartada

- **IA:** client integrat (`PersistentClient`) amb telemetria desactivada, una
  sola funció d'embeddings ONNX compartida i escalfada abans de cronometrar, i
  una mesura de referència del temps d'embedding sol per saber quina part de
  l'`add()` és càlcul de vectors.
- **Incidència:** una prova "ràpida" va trigar més d'un minut (cada càrrega
  recalcula 10.000 embeddings) i l'usuari la va aturar. Per reduir el temps,
  la IA va proposar retallar el grid (treure la mida 1 o fer-la amb una
  mostra).
- **Correcció de l'usuari:** va preguntar si la IA seguia el `CLAUDE.md`. La
  proposta contradeia la decisió "grid únic; cap script defineix el seu propi
  grid", que garanteix que C0 sigui comparable amb P0. Es descarta: el grid
  complet es manté i s'executa en segon pla. Lliçó: contrastar cada proposta
  amb les decisions ja preses abans de presentar-la.

### Revisió externa de l'esborrany de PostgreSQL

- **IA externa (xat de Claude, amb l'informe):** cinc observacions.
- **Validació (Claude Code) contra el repo:**
  - (1) "El TODO de Chroma contradiu el CLAUDE.md": **incorrecte**, la IA
    externa tenia una versió antiga del `CLAUDE.md` (l'opció A ja s'havia
    substituït). Lliçó: una revisió externa només és fiable si rep el context
    actual.
  - (2) Justificació de la mida de lot: la dada "el CLAUDE.md diu 500" també
    era antiga, però la crítica de fons era **correcta**: "per tenir més lots
    per a les estadístiques" és un argument de mesura, no del sistema. Es
    reescriu amb el comportament del sistema (corba que s'aplana).
  - (3) Definir què és cada n i cada desviació: **correcte**, afegit a la
    metodologia i a les llegendes.
  - (4) Línies de codi esbiaixades (P2 inclou validacions): **correcte**. Es
    defineix un criteri explícit (`SYSTEM_CODE`: només codi que interactua
    amb el sistema) i es mostren les dues xifres. P2 passa de 217 a 71 línies.
  - (5) Definir *impedance mismatch*: l'equip decideix **no** incloure-la
    (l'enunciat no la demana).

### Llegendes de taula que no arribaven al PDF

- **IA externa:** va insistir dues vegades que el criteri de línies i el
  nombre de repeticions no s'explicaven enlloc.
- **Claude Code:** inicialment ho va considerar ja resolt, perquè les
  llegendes eren al `.qmd` i sortien a la renderització en Markdown que havia
  fet servir per validar. En comprovar la sortida Typst (la que genera el PDF)
  va veure que **Quarto descartava les llegendes** (`: caption`) de les
  taules impreses des d'una cel·la de Python.
- **Correcció:** les llegendes es generen com a paràgraf normal numerat
  ("**Taula N.** …") i es verifica sobre la sortida Typst.
- **Lliçó:** validar sobre el format que es lliura (PDF), no sobre un format
  intermedi. La crítica externa tenia raó encara que el seu diagnòstic ("no
  s'explica") no n'identifiqués la causa.

### Revisió externa del disseny de Chroma

- **IA externa:** (1) una sola càrrega a la mida oficial no permet respondre
  l'estabilitat de [CQ1] com a [PQ1]; (2) "la mètrica gairebé no afecta la
  inserció" no tenia dades; (3) "P0 + generació de P1" és una comparació
  aproximada (ONNX vs. PyTorch); (4) el filtre `where` té un cost no mesurat;
  (5) `update()` sobre HNSW no és gratuït.
- **Claude Code:** totes correctes. (2), (3) i (5) no requereixen executar
  res: (2) i (5) surten de la càrrega oficial i de la referència d'embedding
  ja previstes, (3) és text. Per a (1) va proposar primer una drecera (sumar
  la càrrega del grid a l'oficial, ~2,5 min) i després la va descartar per
  una solució més neta (3 càrregues oficials a L2, ~10 min més), avisant del
  cost. Per a (4), una variant de referència mesurada, com el cosinus unitari
  de P2.
- **Decisió de l'equip:** fer (1) i (4). La frase no suportada de (2) es
  treu fins tenir les dades.

### C1/C2 i redacció de [CQ1]: errors detectats en validar

- **Validació que falla amb raó:** `C1` exigia que els vectors desats fossin
  bit a bit iguals als generats, i a la col·lecció `cosine` no ho eren
  (≤ 1,5e-8). Abans de relaxar la comprovació, es va investigar: una primera
  prova no va ser concloent (regenerar embeddings amb lots diferents ja dona
  diferències de 1e-7), i es va refer comparant les dues col·leccions entre
  elles. Conclusió: l'espai `cosine` re-normalitza els vectors. Es manté la
  igualtat exacta a `l2` i una tolerància explícita a `cosine`.
- **Dada invàlida detectada per la IA:** en re-executar `C1`, la comparació
  ONNX vs. model va donar 0 perquè els vectors ja eren els nostres. Es va
  canviar el script perquè guardi `null` en lloc d'un 0 enganyós.
- **Error de càlcul propi:** la part d'embedding de l'`add()` sortia 32 %
  perquè dividia totals de 3 càrregues per totals d'1; en revisar les xifres
  abans de redactar-les es va corregir a mitjanes per lot (97 %).
- **Hipòtesi desmentida:** la IA esperava que `update()` fos més ràpid que
  l'`INSERT` de `P1` (Chroma rep numpy sense passar per text). Les dades
  diuen que costa el mateix; el text de [CQ1] ho diu així, sense forçar-ho.
- **Revisió de frases:** es van suavitzar tres afirmacions que anaven més
  enllà de les dades (efecte de la mètrica, causa del cost d'`update()`,
  "lots grans" com a millora, quan 2000 no millora 1000).

### Revisió externa de [CQ1]

- **IA externa:** va recalcular les xifres de la secció de Chroma (quasi
  totes correctes) i va trobar set problemes. El més greu: la conclusió
  "els lots importen menys a Chroma (11×) que a PostgreSQL (230×)"
  incomplia el criteri de comparació que el mateix document havia fixat
  (C0 ≡ P0 + generació de P1). Amb el criteri correcte, PostgreSQL només
  millora 2,5× i **la conclusió s'inverteix**.
- **Claude Code:** va verificar cada punt amb les dades abans d'aplicar-lo
  (els valors recalculats coincidien: 2,5×, 38,4 → 15,4 s, 137,8 ± 2,4 s,
  20–23×, 14×). Tots set eren correctes: notació ambigua de la identitat,
  *recall* sobregeneralitzat (només 20 veïns), "crida atòmica" sense font,
  llegenda ambigua entre grid i càrrega oficial, factor de consulta només
  vàlid per a L2 i "93 % més ràpid" ambigu.
- **Lliçó:** la IA que redacta tendeix a comparar amb la xifra més
  espectacular a mà (P0 sol) encara que contradigui el criteri declarat; una
  segona revisió independent ho va detectar.

### 2026-09-27 — Eina per al document

- **IA externa:** Quarto, Typst, Overleaf o Jupyter.
- **Claude Code:** recomana Quarto amb sortida Typst (sense LaTeX), amb les
  taules generades des de `results/*.json` perquè no es copiïn a mà.
- **Decisió:** Quarto → Typst, document en català a `docs/informe.qmd`.
  L'enunciat original es desa a `docs/enunciat.md` i el `CLAUDE.md` hi remet
  com a font de veritat, perquè la IA contrasti les decisions amb el text
  literal i no amb resums.
