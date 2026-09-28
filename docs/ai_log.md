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

## Guia per redactar l'apartat "I am an AI agent…" (pendent, es farà al final)

Recollida d'una revisió externa (2026-09-28), per no perdre-la:

- **Extensió:** màxim 1 pàgina per enunciat, però **objectiu ~¾ de pàgina**:
  la Discussió ja acaba a la pàgina 9 i el document no pot passar de 10.
- **Tres eixos que demana l'enunciat** (sintetitzar, no reproduir prompts):
  1. *Raonament de les instruccions:* el `CLAUDE.md` com a context
     persistent (decisions fixades, indicacions del professor, criteris de
     mesura) per no repetir-lo a cada sessió; l'enunciat literal a
     `docs/enunciat.md` com a font de veritat.
  2. *Com es van refinar:* decisions que van canviar en veure les dades o
     en contrastar-les amb l'enunciat: equivalència L2/cosinus documentada
     en lloc de canviar de mètrica; C0 amb la funció d'embeddings per
     defecte (després de detectar que l'opció A no complia "generate and
     store"); filtre de metadades mesurat contra k + 1; `localhost` vs.
     `127.0.0.1`; justificació de la mida de lot pel comportament del
     sistema.
  3. *Com es va validar:* comprovacions independents de la IA (vectors
     recuperats bit a bit, `EXPLAIN` dels plans, *recall* contra P2,
     identitat L2/cosinus a les dades, taula del top-2 que falla si els
     sistemes no coincideixen) i revisió creuada dels números amb un altre
     xat, **corregint també aquest xat quan s'equivocava** (p. ex. el 4 %
     que en realitat era 4,7 %; el "CLAUDE.md contradiu el TODO" basat en
     una versió antiga). És exactament el que valora l'enunciat: usar la IA
     amb criteri propi i no acceptar-ne el resultat sense comprovar-lo.
- **Advertència (revisió de Gemini, 2026-09-28):** va suggerir escriure que
  es va donar un *rol* a la IA ("enginyer expert"), que es va treballar per
  "fases estrictes" i que es va validar amb `EXPLAIN ANALYZE`. **No va ser
  així** (cap prompt de rol; es va fer servir `EXPLAIN`, no `ANALYZE`). A
  l'apartat només s'hi posa el que consta en aquest registre.

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

### Segona revisió externa de [CQ1]

- **IA externa:** (1) el 20× amaga que el 93 % del temps de Chroma és el
  filtre: la cerca sense filtre és ~275× més ràpida que P2, i cal descartar
  la xarxa; (2) "els lots ajuden més a Chroma" atribueix a la BD un efecte
  del model; (3) explicar el *recall* amb `ef_search`; (4) repetir C1.
- **Claude Code:** (1) correcte, amb dos matisos: la variant sense filtre
  inclou també el `get()`, i els 2,3 ms que la IA externa anomenava "viatge
  d'anada i tornada" són una crida sencera amb `COMMIT` (serveixen com a
  cota superior). (3) en lloc d'escriure `ef_search = 100` a mà, `C2` desa
  la configuració HNSW real al JSON. En redactar-ho, la mateixa IA va
  escriure dues xifres a mà ("93 %", "100 candidats") i una afirmació falsa
  ("cerca pràcticament exhaustiva": HNSW no recorre tots els vectors); totes
  tres es van detectar i corregir en revisar la sortida renderitzada.

- **Punt (4):** l'equip decideix repetir 3 vegades l'`update()` de `C1` (~20 s
  més) per simetria amb `P1`; la càrrega de cosinus de `C0` es manté en 1.

### 2026-09-28 — pgvector (G0–G2): disseny i revisió de coherència

- **Demanat:** fer la part opcional de pgvector.
- **IA — disseny:** `G0` reutilitza el codi de `P0` (el text no canvia amb
  pgvector); `G1` desa `vector(384)` amb l'adaptador oficial i construeix un
  índex HNSW per mètrica després de carregar; `G2` mesura la cerca exacta i
  l'aproximada sobre la mateixa taula, una crida SQL per consulta. Abans
  d'escriure `G1` es va llegir el codi de l'adaptador `pgvector.psycopg2`
  per saber què envia: un literal de text per vector, com `P1`, cosa que fa
  la comparació directa.
- **Correcció de l'usuari:** va demanar revisar que tot seguís el
  `CLAUDE.md` i l'enunciat. La IA havia (1) oblidat aquesta entrada de
  registre, (2) no havia documentat pgvector al `CLAUDE.md`, (3) havia
  duplicat codi (`load_exact_reference`/recall de `C2`, `read_sentences` de
  `P1`), (4) no havia afegit G0–G2 al criteri de línies i (5) desava les
  respostes de `G2` en un format diferent de `P2`/`C2`. Tot es va corregir:
  funcions compartides a `common.py` (referència exacta i *recall*),
  reutilització de `P1.read_sentences` (no a `common.py`, perquè el criteri
  de línies exclou `common.py` i hauria esbiaixat P1 respecte de C1), i
  format de respostes unificat.
- **Validacions afegides:** `EXPLAIN` per garantir que la configuració
  "hnsw" usa l'índex i l'"exacta" no; la configuració exacta ha de
  coincidir al 100 % amb `P2` o el script falla.

### 2026-09-28 — pgvector: execució i secció de l'informe

- **Error en executar:** `G2` va fallar amb `SHOW hnsw.ef_search` perquè els
  paràmetres de pgvector no existeixen a la sessió fins que es carrega la
  llibreria (primer ús del tipus `vector`). Es força la càrrega amb
  `SELECT NULL::vector` abans de llegir-lo.
- **Validació:** totes les comprovacions van passar: `EXPLAIN` (índex només a
  la configuració HNSW), configuració exacta = `P2` al 100 %, *recall* HNSW
  100 % amb `ef_search` = 40, vectors recuperats bit a bit iguals.
- **Resultat clau:** la cerca exacta de pgvector (mateixa força bruta que
  P2, distància en C) és ~90× més ràpida que P2: el cost de P2 era
  l'impedance mismatch, no l'algorisme. Els dos índexs HNSW ocupen més que
  la taula.
- **Redacció:** secció amb les mateixes preguntes que [PQ1]/[CQ1] i la de
  l'enunciat sobre les diferències amb Chroma; totes les xifres surten de
  `report_tables.py` (`GV.*`).

### 2026-09-28 — Revisió externa de la secció de pgvector

- **IA externa:** va recalcular les xifres (correctes) i va assenyalar:
  (1) falta la taula de línies i crides de pgvector ("the same
  information"); (2) a mida 1, `vector` surt +56 % més lent que `REAL[]`
  sense cap comentari; (3) la generació varia un 17 % entre scripts;
  (4) `G0` vs. `P0` és un 4 %, no un 5 %.
- **Claude Code:** (1) correcte, afegida. (4) **incorrecte**: la IA externa
  va calcular amb valors arrodonits de la taula; amb els del JSON surt 4,7 %
  → 5 %. (3) correcte, reconegut a la metodologia com a soroll entre
  execucions. (2) no es va acceptar "deu ser soroll" sense provar-ho: un
  experiment intercalant 2.000 insercions d'una fila a `REAL[]` i a
  `vector(384)` (mateixes condicions per a tots dos) va donar 2,15 vs.
  2,11 ms per fila i el mateix cost de preparació al client. La diferència
  de la taula és soroll de l'execució (durant `G1` l'usuari va llançar una
  previsualització del document); el text ho explica sense xifres fetes a
  mà, recolzant-se en la diferència entre `G0` i `P0`, que executen codi
  idèntic.

### 2026-09-28 — Discussió

- **Demanat:** escriure la Discussió seguint la proposta de la revisió
  externa (taula comparativa dels tres sistemes i gradació del mismatch, sense
  repetir la comparació pgvector–Chroma), fusionant-hi les taules de línies i
  crides, i afegir una taula amb el top-2 de cada consulta.
- **IA:** la taula del top-2 **comprova en generar-se** que els veïns són
  idèntics a P2, C2 i G2 i per a les dues mètriques; si no ho fossin, el
  document falla en lloc d'afirmar-ho. Les seccions conserven una frase amb
  les línies i crides clau, perquè l'enunciat ho demana a cada secció.
- **Autorevisió:** dues afirmacions del primer esborrany anaven més enllà de
  les dades ("dos ordres de magnitud més lentes", cert només respecte de
  pgvector amb HNSW; "creixen linealment", no mesurat) i es van reformular.

### 2026-09-28 — Revisió completa contra l'avaluació de l'enunciat

- **Demanat:** revisar l'informe sencer contra l'enunciat (especialment
  l'avaluació) i buscar dades o informació contradictòries.
- **Resultat:** cap xifra contradictòria (totes surten dels mateixos JSON).
  Es van corregir cinc problemes de redacció: dues referències a "la taula
  següent" que apuntaven a una altra taula; la pregunta d'estabilitat de les
  consultes de pgvector, plantejada però no resposta (i amb un CV alt a
  HNSW que calia explicar); un temps de construcció d'índex atribuït a tots
  dos índexs; una frase repetida (concisió, que l'enunciat avalua); i
  "Chroma elimina el desajust", massa fort davant de les pròpies dades (2
  crides per consulta).

### 2026-09-28 — "Fins a N vegades": el màxim que cap de les dues IA va trobar

- **Seqüència:** Claude Code va substituir "dos ordres de magnitud" per
  "fins a 242 vegades" (el factor de pgvector HNSW L2). La IA externa va
  notar que 242 no era el màxim i va proposar "fins a 290".
- **Validació:** en lloc de triar un número a mà, el rang es calcula sobre
  tots els índexs mesurats (pgvector HNSW i Chroma sense filtre, totes dues
  mètriques): surt **entre 242 i 367**. El màxim real (Chroma sense filtre,
  cosinus) no l'havia trobat cap de les dues IA. Lliçó: les xifres que
  resumeixen diverses taules s'han de calcular, no llegir-les a ull.

### 2026-09-28 — Revisió de Gemini

- **Acceptat, verificant-ho abans:** explicar d'on surt el màxim del rang
  (Chroma sense filtre); omplir el *recall* de les files sense filtre (els
  veïns són idèntics als filtrats, comprovat per C2); unitats explícites a la
  taula comparativa; explicar el `WHERE` amb HNSW **després de mirar el pla
  real amb `EXPLAIN`** (filtre sobre la sortida de l'índex).
- **Matisat:** atribuir la lentitud de l'ONNX a "l'embolcall o la
  configuració" era especulatiu; només s'afirma el que les dades suporten
  (mateix model, vectors iguals fins a 1e-7 → la diferència és de l'entorn
  d'execució).
- **Rebutjat:** els suggeriments per a l'apartat d'IA que no corresponen al
  que es va fer (rol, `EXPLAIN ANALYZE`).

### 2026-09-28 — Última ronda de revisions (Claude i Gemini)

- **Claude (xat):** tensió entre "els vectors ONNX coincideixen" i
  "C1 els substitueix perquè siguin els mateixos". Resolta indicant la
  tolerància (diferència màxima llegida de `C1.json`; es mostrarà la xifra
  quan `C0 → C1` s'executin seguits a les mesures oficials).
- **Gemini:** afirmava que el PDF mostrava `sentence_id > q` en lloc de `<>`.
  **Fals:** es van renderitzar les pàgines a PNG amb el compilador Typst i
  la pàgina mostra `<>` correctament; l'error era de l'extracció de text de
  l'eina que va llegir el PDF (el mateix passa amb els guions de partició).
  Lliçó: verificar sobre el format lliurat, i no acceptar una crítica sense
  reproduir-la.

### 2026-09-28 — Revisió de ChatGPT (rigor de les afirmacions)

- **Verificat abans de respondre:** la norma 1 és correcta (el model acaba
  amb `Normalize`; normes desades 1 ± 1,3e-7), i el repo és públic.
- **Troballa pròpia en fer-ho:** el link de l'informe porta a `main`, que no
  conté les dades ni els scripts (tot és a `develop`). Afegit a la llista de
  lliurament del `CLAUDE.md`.
- **Acceptat (8 canvis de redacció):** atribuir només una part del cost
  d'emmagatzemar vectors al mismatch (l'altra és inherent: 384 valors);
  treure "determinista" (la composició del lot canvia els valors ~1e-7);
  "cost de P2 no era l'algorisme" → quantificat (el 99 % del temps de P2);
  "el guany real" → "aïlla millor"; mateixos top-2 "en les 10 consultes",
  no garantia; "Chroma gairebé elimina" → "redueix fortament"; filtre "car"
  → quantificat; "pgvector és la millor opció" → conclusió descriptiva, dient
  què no s'ha avaluat; redacció dels vectors ONNX/PyTorch que semblava
  contradictòria.
- **No acceptat:** repetir a la Discussió limitacions ja explicades a les
  seccions (client integrat, C0 amb embeddings) i reestructurar les millores
  en "mesurat / no mesurat" (ja s'indica "mesurat" a cada una, i l'espai és
  just).

### 2026-09-28 — Mesures oficials

- **Preparació:** abans de llançar-les, la IA va comprovar l'estat de la
  màquina i va detectar que el portàtil se suspenia als 45 min (les mesures
  en duren ~70) i que Wallpaper Engine era el procés amb més CPU del
  sistema; l'usuari ho va corregir abans de començar.
- **Execució:** 65 min, totes les validacions correctes; la diferència
  ONNX vs. model es va poder mesurar (2·10⁻⁷) perquè `C0 → C1` van anar
  seguits.
- **Revisió de les interpretacions:** es van recalcular tots els valors que
  el text interpreta. Només una frase va deixar de ser certa ("a mida 1,
  `vector` surt més lent que `REAL[]`": ara és al revés); es va reescriure
  sense afirmar el sentit de la diferència, que és soroll segons la prova
  intercalada.

### 2026-09-27 — Eina per al document

- **IA externa:** Quarto, Typst, Overleaf o Jupyter.
- **Claude Code:** recomana Quarto amb sortida Typst (sense LaTeX), amb les
  taules generades des de `results/*.json` perquè no es copiïn a mà.
- **Decisió:** Quarto → Typst, document en català a `docs/informe.qmd`.
  L'enunciat original es desa a `docs/enunciat.md` i el `CLAUDE.md` hi remet
  com a font de veritat, perquè la IA contrasti les decisions amb el text
  literal i no amb resums.
