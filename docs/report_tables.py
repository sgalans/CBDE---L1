"""Tables and figures for docs/informe.qmd, built from results/*.json.

The report never contains hand-typed measurements: every number comes from the
JSON files written by the scripts, so re-running an experiment and re-rendering
the report keeps text and data consistent.

Tables are returned as Markdown strings (printed from a cell with
``output: asis``); numbers use a decimal comma, as the report is in Catalan.
"""

from __future__ import annotations

import itertools
import json
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import DEFAULT_BATCH_SIZE, METRICS, RESULTS_DIR, load_queries  # noqa: E402

METRIC_NAMES = {"l2": "L2", "cosine": "Cosinus", "cosine_unit": "Cosinus unitari (1 − a·b)",
                "l2_nofilter": "L2, sense filtre (k + 1)",
                "cosine_nofilter": "Cosinus, sense filtre (k + 1)"}


@cache
def load(name: str) -> dict:
    with (RESULTS_DIR / f"{name}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def num(x: float, decimals: int = 3) -> str:
    """Number in Catalan style: decimal comma, dot for thousands (2.754,5).

    Plain ASCII on purpose: Quarto escapes non-ASCII characters (such as a
    thin space) in inline ``{python}`` values.
    """
    text = f"{x:,.{decimals}f}"
    return text.replace(",", "#").replace(".", ",").replace("#", ".")


def ms(seconds: float, decimals: int = 1) -> str:
    return num(seconds * 1000, decimals)


def pct(x: float, decimals: int = 0) -> str:
    return num(100 * x, decimals) + " %"


def cv(s: dict) -> float:
    """Coefficient of variation (std / mean) of a stats dict."""
    return s["std"] / s["avg"] if s["avg"] else 0.0


def table(header: list[str], rows: list[list[str]], align: str, caption: str,
          widths: list[int] | None = None) -> str:
    """Markdown pipe table; ``align`` is one of l/r/c per column.

    ``widths`` are relative column widths. Pandoc only uses them when a row is
    wider than the page (it reads them from the number of dashes of the
    separator line); otherwise columns take their natural width.
    """
    widths = widths or [3] * len(align)

    def mark(a: str, w: int) -> str:
        dashes = "-" * max(w, 3)
        return {"l": ":" + dashes, "r": dashes + ":", "c": ":" + dashes + ":"}[a]

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(mark(a, w) for a, w in zip(align, widths)) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
        # A plain paragraph, not pandoc's ": caption" syntax: Quarto drops that
        # caption from tables printed by a code cell in the Typst/PDF output.
        f"**Taula {next(_table_number)}.** {caption}",
        "",
    ]
    return "\n".join(lines)


#: Tables are numbered in the order the report prints them.
_table_number = itertools.count(1)


def reset_table_numbers() -> None:
    """Restart numbering at 1. Called at the top of the report: Quarto may keep
    the Python kernel alive between renders, so module state would persist."""
    global _table_number
    _table_number = itertools.count(1)


def _by_size(grid: list[dict]) -> dict[int, dict]:
    return {g["batch_size"]: g for g in grid}


# --------------------------------------------------------------------------
# Data section
# --------------------------------------------------------------------------


def queries_table() -> str:
    rows = [[str(q["sentence_id"]), f"*{q['text']}*"] for q in load_queries()]
    return table(["ID", "Frase de consulta"], rows, "rl",
                 "Les 10 frases de consulta (`data/queries.json`), idèntiques als tres "
                 "sistemes. ID = `sentence_id`.",
                 widths=[6, 94])


# --------------------------------------------------------------------------
# PostgreSQL
# --------------------------------------------------------------------------


def pg_batch_grid_table() -> str:
    """Total load time of text (P0) and embeddings (P1) for every batch size."""
    p0, p1 = _by_size(load("P0")["grid"]), _by_size(load("P1")["grid"])
    rows = []
    for size in sorted(p0):
        a, b = p0[size], p1.get(size)
        rows.append([
            f"**{size}**" if size == DEFAULT_BATCH_SIZE else str(size),
            str(a["db_calls"]["statements"]),
            num(a["total_time"]["avg"]), num(a["total_time"]["std"]),
            num(b["total_time"]["avg"]) if b else "–",
            num(b["total_time"]["std"]) if b else "–",
        ])
    reps = load("P0")["grid"][0]["repeats"]
    return table(
        ["Mida de lot", "Lots (`INSERT`)", "Text: total (s)", "desv. entre càrregues",
         "Embeddings: total (s)", "desv. entre càrregues"],
        rows, "rrrrrr",
        f"Temps **total** de càrrega de les 10.000 frases per mida de lot: mitjana "
        f"i desviació entre {reps} càrregues completes. En negreta, la mida triada.")


def pg_insert_stats_table() -> str:
    """min/max/avg/std per batch at the chosen size: the statistics asked for."""
    p0, p1 = load("P0"), load("P1")
    text = _by_size(p0["grid"])[DEFAULT_BATCH_SIZE]["batch_time"]
    gen = p1["generate"]["batch_time"]
    store = _by_size(p1["grid"])[DEFAULT_BATCH_SIZE]["batch_time"]
    rows = []
    for label, s in (("Emmagatzematge del text (P0)", text),
                     ("Generació d'embeddings (P1)", gen),
                     ("Emmagatzematge d'embeddings (P1)", store)):
        rows.append([label, str(s["n"]), ms(s["min"]), ms(s["max"]), ms(s["avg"]),
                     ms(s["std"]), pct(cv(s))])
    reps = _by_size(p0["grid"])[DEFAULT_BATCH_SIZE]["repeats"]
    return table(
        ["Operació", "n", "mín (ms)", "màx (ms)", "mitjana (ms)", "desv. (ms)", "CV"],
        rows, "lrrrrrr",
        f"Temps **per lot** de {DEFAULT_BATCH_SIZE} frases; n = lots mesurats "
        f"(10 lots × {reps} càrregues per a l'emmagatzematge, 10 lots per a la generació). "
        f"CV = desviació / mitjana.",
        widths=[36, 6, 11, 11, 13, 12, 8])


def pg_insert_methods_table() -> str:
    """Alternative insertion methods measured as references."""
    p0, p1 = load("P0"), load("P1")
    ev = _by_size(p0["grid"])[p0["copy"]["batch_size"]]
    lit = _by_size(p1["grid"])[p1["adapt"]["batch_size"]]
    rows = [
        ["Text", "`INSERT` multi-fila (`execute_values`)", num(ev["total_time"]["avg"]), "1,00"],
        ["Text", "`COPY ... FROM STDIN`", num(p0["copy"]["total_time"]["avg"]),
         num(p0["copy"]["total_time"]["avg"] / ev["total_time"]["avg"], 2)],
        ["Embeddings", "Literal `'{...}'::real[]` (triat)", num(lit["total_time"]["avg"]), "1,00"],
        ["Embeddings", "`ARRAY[...]` per defecte de psycopg2", num(p1["adapt"]["total_time"]["avg"]),
         num(p1["adapt"]["total_time"]["avg"] / lit["total_time"]["avg"], 2)],
    ]
    return table(["Dades", "Mètode", "Total (s)", "Relatiu"], rows, "llrr",
                 f"Mètodes d'inserció alternatius, amb lots de {p0['copy']['batch_size']}.")


def pg_query_table() -> str:
    p2 = load("P2")
    rows = []
    for metric, t in p2["timing"].items():
        s = t["query_time"]
        rows.append([METRIC_NAMES.get(metric, metric), str(s["n"]), ms(s["min"]), ms(s["max"]),
                     ms(s["avg"]), ms(s["std"]), pct(cv(s))])
    return table(
        ["Mètrica", "n", "mín (ms)", "màx (ms)", "mitjana (ms)", "desv. (ms)", "CV"],
        rows, "lrrrrrr",
        f"Temps **per consulta** del top-{p2['k']}; n = 10 consultes × {p2['repeats']} rondes. "
        f"El cosinus unitari és una variant de referència, no una tercera mètrica.")


#: Criterion for comparing code size between systems: only the top-level
#: definitions that talk to the system (schema/collections, SQL, loading and
#: querying). Excluded everywhere: validation, reference-only variants,
#: embedding generation (identical for every system), CLI, printing and
#: common.py.
SYSTEM_CODE = {
    "postgres/P0.py": ["DDL", "INSERT_SQL", "recreate_table", "insert_values", "load"],
    "postgres/P1.py": ["DDL", "INSERT_SQL", "INSERT_TEMPLATE", "to_literal",
                       "read_sentences", "recreate_table", "store"],
    "postgres/P2.py": ["FUNCTIONS_SQL", "TOP_K_SQL", "install_functions", "top_k"],
    "chroma/C0.py": ["default_embedding_function", "recreate_collection", "load"],
    "chroma/C1.py": ["read_documents", "store"],
    "chroma/C2.py": ["top_k"],
    # G0 has no system code of its own: it reuses P0's table and loading code.
    "pgvector/G0.py": [],
    "pgvector/G1.py": ["DDL", "INSERT_SQL", "INDEXES", "recreate_table", "store",
                       "build_indexes"],
    "pgvector/G2.py": ["OPERATORS", "TOP_K_SQL", "CONFIGS", "top_k"],
}


def code_lines(script: str, names: list[str] | None = None) -> int:
    """Lines of code: no blank lines, comments or docstrings.

    With ``names``, only the lines of those top-level definitions (functions
    or assignments) are counted. Embedded SQL counts: it is code the system
    forced us to write.
    """
    import ast
    import io
    import tokenize

    source = (ROOT / script).read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstring_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node):
            doc = node.body[0]
            docstring_lines.update(range(doc.lineno, doc.end_lineno + 1))

    wanted: set[int] | None = None
    if names is not None:
        wanted = set()
        for node in tree.body:
            node_names = (
                [node.name] if isinstance(node, ast.FunctionDef)
                else [t.id for t in getattr(node, "targets", []) if isinstance(t, ast.Name)]
            )
            if set(node_names) & set(names):
                wanted.update(range(node.lineno, node.end_lineno + 1))

    code: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                        tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        code.update(line for line in range(tok.start[0], tok.end[0] + 1)
                    if line not in docstring_lines and (wanted is None or line in wanted))
    return len(code)


def system_lines(script: str) -> int:
    return code_lines(script, SYSTEM_CODE[script])


def pg_cost_table() -> str:
    """Code lines and database calls of each PostgreSQL script."""
    p0, p1 = load("P0"), load("P1")
    text_calls = _by_size(p0["grid"])[DEFAULT_BATCH_SIZE]["db_calls"]["statements"]
    emb_calls = _by_size(p1["grid"])[DEFAULT_BATCH_SIZE]["db_calls"]["statements"]
    rows = [
        ["P0 (text)", str(system_lines("postgres/P0.py")), str(code_lines("postgres/P0.py")),
         f"{text_calls} `INSERT` + {text_calls} `COMMIT`"],
        ["P1 (embeddings)", str(system_lines("postgres/P1.py")), str(code_lines("postgres/P1.py")),
         f"1 `SELECT` + {emb_calls} `INSERT` + {emb_calls} `COMMIT`"],
        ["P2 (similitud)", str(system_lines("postgres/P2.py")), str(code_lines("postgres/P2.py")),
         "1 crida a `top_k_similar` per consulta i mètrica"],
    ]
    return table(["Script", "Línies (sistema)", "Línies (total)", "Crides a la BD (càrrega oficial)"],
                 rows, "lrrl",
                 "Cost en codi i crides. *Línies (sistema)*: només el codi que parla amb la "
                 "BD (DDL, SQL, càrrega i consultes); sense validacions, variants de "
                 "referència, generació d'embeddings, línia d'ordres ni les utilitats de "
                 "`common.py`. *Línies (total)*: el script sencer. Mai comentaris ni docstrings.",
                 widths=[20, 14, 12, 54])


# --------------------------------------------------------------------------
# Inline values for the text
# --------------------------------------------------------------------------


class PG:
    """Named values used inline in the PostgreSQL section."""

    @staticmethod
    def _p0(size: int) -> float:
        return _by_size(load("P0")["grid"])[size]["total_time"]["avg"]

    @staticmethod
    def _p1(size: int) -> float:
        return _by_size(load("P1")["grid"])[size]["total_time"]["avg"]

    @classmethod
    def text_row_by_row(cls) -> str:
        return num(cls._p0(1), 1)

    @classmethod
    def text_default(cls) -> str:
        return num(cls._p0(DEFAULT_BATCH_SIZE), 2)

    @classmethod
    def text_speedup(cls) -> str:
        return num(cls._p0(1) / cls._p0(DEFAULT_BATCH_SIZE), 0)

    @classmethod
    def text_cost_per_call_ms(cls) -> str:
        return ms(cls._p0(1) / load("P0")["n_sentences"], 1)

    @classmethod
    def text_row_by_row_cv(cls) -> str:
        return pct(cv(_by_size(load("P0")["grid"])[1]["total_time"]))

    @classmethod
    def emb_row_by_row(cls) -> str:
        return num(cls._p1(1), 1)

    @classmethod
    def emb_default(cls) -> str:
        return num(cls._p1(DEFAULT_BATCH_SIZE), 2)

    @classmethod
    def emb_vs_text(cls) -> str:
        return num(cls._p1(DEFAULT_BATCH_SIZE) / cls._p0(DEFAULT_BATCH_SIZE), 0)

    @staticmethod
    def generate_total() -> str:
        return num(load("P1")["generate"]["batch_time"]["total"], 1)

    @staticmethod
    def read_time_ms() -> str:
        return ms(load("P1")["read"]["time"], 0)

    @staticmethod
    def copy_gain() -> str:
        p0 = load("P0")
        ev = _by_size(p0["grid"])[p0["copy"]["batch_size"]]["total_time"]["avg"]
        return pct(1 - p0["copy"]["total_time"]["avg"] / ev)

    @staticmethod
    def adapt_factor() -> str:
        p1 = load("P1")
        lit = _by_size(p1["grid"])[p1["adapt"]["batch_size"]]["total_time"]["avg"]
        return num(p1["adapt"]["total_time"]["avg"] / lit, 1)

    @staticmethod
    def query_avg_ms(metric: str) -> str:
        return ms(load("P2")["timing"][metric]["query_time"]["avg"], 0)

    @staticmethod
    def query_cv(metric: str) -> str:
        return pct(cv(load("P2")["timing"][metric]["query_time"]))

    @staticmethod
    def cosine_overhead() -> str:
        t = load("P2")["timing"]
        return pct(t["cosine"]["query_time"]["avg"] / t["l2"]["query_time"]["avg"] - 1)

    @staticmethod
    def unit_gain() -> str:
        t = load("P2")["timing"]
        return pct(1 - t["cosine_unit"]["query_time"]["avg"] / t["cosine"]["query_time"]["avg"])

    @staticmethod
    def same_topk() -> str:
        v = load("P2")["validation"]
        return f"{v['same_top_k_l2_vs_cosine']}/{v['queries']}"

    @staticmethod
    def identity_error() -> str:
        return sci(load("P2")["validation"]["max_abs_error_l2sq_vs_2cos"])


def sci(x: float) -> str:
    """Order of magnitude in Markdown: 1.3e-07 -> 1·10^-7^."""
    mantissa, exp = f"{x:.0e}".split("e")
    return f"{mantissa}·10^{int(exp)}^"


# --------------------------------------------------------------------------
# Chroma
# --------------------------------------------------------------------------

STATS_HEADER = ["Operació", "n", "mín (ms)", "màx (ms)", "mitjana (ms)", "desv. (ms)", "CV"]


def _stats_row(label: str, s: dict) -> list[str]:
    return [label, str(s["n"]), ms(s["min"]), ms(s["max"]), ms(s["avg"]), ms(s["std"]),
            pct(cv(s))]


def chroma_batch_grid_table() -> str:
    """Total C0 load time (text + ONNX embeddings inside add()) per batch size."""
    c0 = load("C0")
    p0 = _by_size(load("P0")["grid"])
    # Fair PostgreSQL counterpart of C0: storing the text (P0, same batch size)
    # plus generating the embeddings (P1), since C0's add() does both.
    generation = load("P1")["generate"]["batch_time"]["total"]
    rows = []
    for g in c0["grid"]:
        size = g["batch_size"]
        rows.append([
            f"**{size}**" if size == DEFAULT_BATCH_SIZE else str(size),
            str(g["db_calls"]["add"]),
            num(g["total_time"]["avg"], 1),
            ms(g["batch_time"]["avg"]),
            (num(g["total_time"]["avg"] / (p0[size]["total_time"]["avg"] + generation), 1)
             if size in p0 else "–"),
        ])
    return table(
        ["Mida de lot", "Crides `add()`", "Total (s)", "Per `add()` (ms)", "× PostgreSQL"],
        rows, "rrrrr",
        f"`C0`: temps **total** de càrrega de les 10.000 frases a la col·lecció L2 per mida "
        f"de lot, dins el grid ({c0['grid'][0]['repeats']} càrrega per mida; les 3 càrregues "
        f"oficials a la mida triada són a la Taula següent i al text). Inclou el càlcul dels "
        f"embeddings ONNX dins l'`add()`. *× PostgreSQL*: vegades el temps de fer el mateix a "
        f"PostgreSQL, és a dir, `P0` a la mateixa mida més la generació d'embeddings de `P1`. "
        f"En negreta, la mida triada.",
        widths=[14, 16, 14, 18, 14])


def chroma_insert_stats_table() -> str:
    """min/max/avg/std per batch at the chosen size for every Chroma insertion step."""
    c0, c1, p1 = load("C0"), load("C1"), load("P1")
    rows = [
        _stats_row(f"`add()` text + embedding ONNX, {METRIC_NAMES[m]} (C0)", o["batch_time"])
        for m, o in c0["official"].items()
    ]
    if "embedding_only" in c0:
        rows.append(_stats_row("Només embedding ONNX, referència (C0)",
                               c0["embedding_only"]["batch_time"]))
    rows.append(_stats_row("Generació d'embeddings, nostre model (C1)",
                           c1["generate"]["batch_time"]))
    rows += [_stats_row(f"`update()` embeddings, {METRIC_NAMES[m]} (C1)", s["batch_time"])
             for m, s in c1["store"].items()]
    rows.append(_stats_row("*Ref.: emmagatzematge d'embeddings a PostgreSQL (P1)*",
                           _by_size(p1["grid"])[DEFAULT_BATCH_SIZE]["batch_time"]))
    return table(
        STATS_HEADER, rows, "lrrrrrr",
        f"Temps **per lot** de {DEFAULT_BATCH_SIZE} frases a Chroma; n = lots mesurats "
        f"(10 per càrrega: {c0['official']['l2']['repeats']} càrregues a L2 i 1 a cosinus a "
        f"`C0`; a `C1`, 1 generació i {next(iter(c1['store'].values())).get('repeats', 1)} "
        f"`update()` complets per col·lecció; la fila *Només embedding* és una passada de la funció ONNX "
        f"sobre els mateixos 10 lots, sense desar res). L'última fila és la de PostgreSQL, "
        f"per comparar.",
        widths=[40, 5, 11, 11, 13, 12, 8])


def chroma_query_table() -> str:
    c2 = load("C2")
    recall = c2["validation"].get("recall_vs_P2", {})
    rows = []
    for metric, t in c2["timing"].items():
        s = t["query_time"]
        rows.append([METRIC_NAMES.get(metric, metric), str(s["n"]), ms(s["min"], 2),
                     ms(s["max"], 2), ms(s["avg"], 2), ms(s["std"], 2), pct(cv(s)),
                     pct(recall[metric]) if metric in recall else "–"])
    return table(
        ["Mètrica", "n", "mín (ms)", "màx (ms)", "mitjana (ms)", "desv. (ms)", "CV",
         "Recall vs P2"],
        rows, "lrrrrrrr",
        f"`C2`: temps **per consulta** del top-{c2['k']} (`get()` + `query()`); "
        f"n = 10 consultes × {c2['repeats']} rondes. *Recall*: part dels veïns exactes de "
        f"`P2` que retorna l'índex HNSW. Les files *sense filtre* són una variant de "
        f"referència: demanen k + 1 veïns i descarten la pròpia frase a Python.")


def chroma_cost_table() -> str:
    """Code lines and calls of each Chroma script."""
    c0, c1 = load("C0"), load("C1")
    adds = next(iter(c0["official"].values()))["db_calls"]["add"]
    updates = next(iter(c1["store"].values()))["db_calls"]["update"]
    n_coll = len(c0["official"])
    rows = [
        ["C0 (text)", str(system_lines("chroma/C0.py")), str(code_lines("chroma/C0.py")),
         f"{adds} `add()` × {n_coll} col·leccions"],
        ["C1 (embeddings)", str(system_lines("chroma/C1.py")), str(code_lines("chroma/C1.py")),
         f"1 `get()` + {updates} `update()` × {n_coll} col·leccions"],
        ["C2 (similitud)", str(system_lines("chroma/C2.py")), str(code_lines("chroma/C2.py")),
         "2 crides (`get()` + `query()`) per consulta i mètrica"],
    ]
    return table(["Script", "Línies (sistema)", "Línies (total)", "Crides a Chroma (càrrega oficial)"],
                 rows, "lrrl",
                 "Cost en codi i crides a Chroma, amb el mateix criteri de línies que a "
                 "PostgreSQL. No hi ha cap `COMMIT`: Chroma no té transaccions explícites.",
                 widths=[20, 14, 12, 54])


class CH:
    """Named values used inline in the Chroma section."""

    @staticmethod
    def _grid(size: int) -> dict:
        return _by_size(load("C0")["grid"])[size]

    @classmethod
    def load_row_by_row_min(cls) -> str:
        return num(cls._grid(1)["total_time"]["avg"] / 60, 1)

    @classmethod
    def load_default(cls) -> str:
        return num(cls._grid(DEFAULT_BATCH_SIZE)["total_time"]["avg"], 1)

    @classmethod
    def load_speedup(cls) -> str:
        return num(cls._grid(1)["total_time"]["avg"]
                   / cls._grid(DEFAULT_BATCH_SIZE)["total_time"]["avg"], 0)

    @classmethod
    def add_row_by_row_ms(cls) -> str:
        return ms(cls._grid(1)["batch_time"]["avg"], 0)

    @classmethod
    def add_vs_pg_row(cls) -> str:
        """How many times more a 1-row add() costs than a 1-row INSERT (P0)."""
        pg = _by_size(load("P0")["grid"])[1]["batch_time"]["avg"]
        return num(cls._grid(1)["batch_time"]["avg"] / pg, 0)

    @staticmethod
    def embedding_share() -> str:
        """Share of an add() batch spent computing ONNX embeddings (per-batch means)."""
        c0 = load("C0")
        add = c0["official"]["l2"]["batch_time"]["avg"]
        return pct(c0["embedding_only"]["batch_time"]["avg"] / add)

    @staticmethod
    def add_store_ms() -> str:
        """Per batch: add() minus ONNX embedding = what storing costs inside add()."""
        c0 = load("C0")
        return ms(c0["official"]["l2"]["batch_time"]["avg"]
                  - c0["embedding_only"]["batch_time"]["avg"], 0)

    @staticmethod
    def update_ms(metric: str = "l2") -> str:
        return ms(load("C1")["store"][metric]["batch_time"]["avg"], 0)

    @staticmethod
    def filter_same() -> str:
        """Answers identical with and without the where filter (queries x metrics)."""
        v = load("C2")["validation"]
        return f"{v['same_top_k_filter_vs_nofilter']}/{2 * v['queries']}"

    @staticmethod
    def cosine_changed() -> str:
        """Vectors that the cosine collection stored with a (rounding) change."""
        return num(load("C1")["validation"]["vectors_changed"]["cosine"], 0)

    @staticmethod
    def cosine_change() -> str:
        return sci(load("C1")["validation"]["roundtrip_max_abs_error"]["cosine"])

    @staticmethod
    def pg_equiv_speedup() -> str:
        """Same row-by-row vs. default ratio for PostgreSQL's equivalent of C0:
        P0 at that batch size plus P1's (always batched) generation."""
        grid = _by_size(load("P0")["grid"])
        gen = load("P1")["generate"]["batch_time"]["total"]
        return num((grid[1]["total_time"]["avg"] + gen)
                   / (grid[DEFAULT_BATCH_SIZE]["total_time"]["avg"] + gen), 1)

    @staticmethod
    def pg_equiv(size: int) -> str:
        grid = _by_size(load("P0")["grid"])
        return num(grid[size]["total_time"]["avg"] + load("P1")["generate"]["batch_time"]["total"], 1)

    @staticmethod
    def hnsw(param: str) -> str:
        """An HNSW parameter as used by the l2 collection (recorded by C2)."""
        return str(load("C2")["hnsw"]["l2"][param])

    @staticmethod
    def index_speedup(metric: str = "l2") -> str:
        """P2 brute force vs. C2 without the where filter (get + index search)."""
        c2 = load("C2")["timing"][f"{metric}_nofilter"]["query_time"]["avg"]
        p2 = load("P2")["timing"][metric]["query_time"]["avg"]
        return num(p2 / c2, 0)

    @staticmethod
    def nofilter_ms(metric: str = "l2") -> str:
        return ms(load("C2")["timing"][f"{metric}_nofilter"]["query_time"]["avg"], 2)

    @staticmethod
    def pg_call_share() -> str:
        """Upper bound of PostgreSQL's per-call overhead (a whole INSERT + COMMIT
        of P0) as a share of a P2 query."""
        call = _by_size(load("P0")["grid"])[1]["batch_time"]["avg"]
        return pct(call / load("P2")["timing"]["l2"]["query_time"]["avg"])

    @staticmethod
    def official_load_avg() -> str:
        return num(load("C0")["official"]["l2"]["total_time"]["avg"], 1)

    @staticmethod
    def official_load_std() -> str:
        return num(load("C0")["official"]["l2"]["total_time"]["std"], 1)

    @staticmethod
    def nofilter_factor(metric: str = "l2") -> str:
        """How many times faster the query is without the where filter."""
        t = load("C2")["timing"]
        return num(t[metric]["query_time"]["avg"] / t[f"{metric}_nofilter"]["query_time"]["avg"], 0)

    @staticmethod
    def official_load_cv() -> str:
        """Variation of the total time across the official l2 loads."""
        return pct(cv(load("C0")["official"]["l2"]["total_time"]))

    @staticmethod
    def metric_insert_diff() -> str:
        """Relative difference of the per-batch add() time, cosine vs. l2."""
        o = load("C0")["official"]
        return pct(abs(o["cosine"]["batch_time"]["avg"] / o["l2"]["batch_time"]["avg"] - 1))

    @staticmethod
    def onnx_vs_model() -> str:
        """How many times slower Chroma's ONNX embedding is than our model."""
        onnx = load("C0")["embedding_only"]["batch_time"]["total"]
        ours = load("C1")["generate"]["batch_time"]["total"]
        return num(onnx / ours, 1)

    @staticmethod
    def onnx_diff() -> str:
        diff = load("C1")["validation"]["onnx_vs_model_max_abs_diff"]
        return sci(diff) if diff is not None else "(no mesurat)"

    @staticmethod
    def store_total(metric: str = "l2") -> str:
        return num(load("C1")["store"][metric]["batch_time"]["total"], 2)

    @staticmethod
    def store_vs_pg() -> str:
        """How many times faster update() is than P1's INSERT of REAL[] (per batch)."""
        chroma = load("C1")["store"]["l2"]["batch_time"]["avg"]
        pg = _by_size(load("P1")["grid"])[DEFAULT_BATCH_SIZE]["batch_time"]["avg"]
        return num(pg / chroma, 1)

    @staticmethod
    def read_time_ms() -> str:
        return ms(load("C1")["read"]["time"], 0)

    @staticmethod
    def query_avg_ms(metric: str) -> str:
        return ms(load("C2")["timing"][metric]["query_time"]["avg"], 1)

    @staticmethod
    def query_cv(metric: str) -> str:
        return pct(cv(load("C2")["timing"][metric]["query_time"]))

    @staticmethod
    def query_speedup(metric: str) -> str:
        """How many times faster C2 is than P2 for the same metric."""
        c2 = load("C2")["timing"][metric]["query_time"]["avg"]
        p2 = load("P2")["timing"][metric]["query_time"]["avg"]
        return num(p2 / c2, 0)

    @staticmethod
    def recall(metric: str) -> str:
        return pct(load("C2")["validation"]["recall_vs_P2"][metric])

    @staticmethod
    def nofilter_gain(metric: str = "l2") -> str:
        """Time saved by asking k + 1 neighbours instead of the where filter."""
        t = load("C2")["timing"]
        return pct(1 - t[f"{metric}_nofilter"]["query_time"]["avg"] / t[metric]["query_time"]["avg"])

    @staticmethod
    def same_topk() -> str:
        v = load("C2")["validation"]
        return f"{v['same_top_k_l2_vs_cosine']}/{v['queries']}"

    @staticmethod
    def identity_error() -> str:
        return sci(load("C2")["validation"]["max_abs_error_l2sq_vs_2cos"])


# --------------------------------------------------------------------------
# pgvector
# --------------------------------------------------------------------------


def mb(n_bytes: int) -> str:
    return num(n_bytes / 1e6, 1)


def pgv_batch_grid_table() -> str:
    """Total load times of G0 (text) and G1 (vector) per batch size, next to P1."""
    g0, g1 = _by_size(load("G0")["grid"]), _by_size(load("G1")["grid"])
    p1 = _by_size(load("P1")["grid"])
    rows = []
    for size in sorted(g0):
        rows.append([
            f"**{size}**" if size == DEFAULT_BATCH_SIZE else str(size),
            num(g0[size]["total_time"]["avg"]), num(g0[size]["total_time"]["std"]),
            num(g1[size]["total_time"]["avg"]), num(g1[size]["total_time"]["std"]),
            num(p1[size]["total_time"]["avg"]) if size in p1 else "–",
        ])
    reps = load("G0")["grid"][0]["repeats"]
    return table(
        ["Mida de lot", "Text (G0): total (s)", "desv. entre càrregues",
         "`vector` (G1): total (s)", "desv. entre càrregues", "*Ref.: `REAL[]` (P1)*"],
        rows, "rrrrrr",
        f"pgvector: temps **total** de càrrega de les 10.000 frases per mida de lot "
        f"(mitjana i desviació entre {reps} càrregues). L'última columna és `P1`, per "
        f"comparar. En negreta, la mida triada.",
        widths=[12, 17, 16, 17, 16, 17])


def pgv_stats_table() -> str:
    g0, g1 = load("G0"), load("G1")
    rows = [
        _stats_row("Emmagatzematge del text (G0)",
                   _by_size(g0["grid"])[DEFAULT_BATCH_SIZE]["batch_time"]),
        _stats_row("Generació d'embeddings (G1)", g1["generate"]["batch_time"]),
        _stats_row("Emmagatzematge de `vector(384)` (G1)",
                   _by_size(g1["grid"])[DEFAULT_BATCH_SIZE]["batch_time"]),
    ]
    rows += [_stats_row(f"Construcció de l'índex HNSW, {METRIC_NAMES[m]} (G1)",
                        g1["index"][m]["build_time"]) for m in METRICS]
    reps = _by_size(g0["grid"])[DEFAULT_BATCH_SIZE]["repeats"]
    return table(
        STATS_HEADER, rows, "lrrrrrr",
        f"pgvector: temps **per lot** de {DEFAULT_BATCH_SIZE} frases (10 lots × {reps} "
        f"càrregues; generació, 10 lots) i per **construcció completa** de cada índex "
        f"({reps} construccions).",
        widths=[40, 5, 11, 11, 13, 12, 8])


def pgv_query_table() -> str:
    g2, p2 = load("G2"), load("P2")
    names = {"exact": "Exacta (sense índex)", "hnsw": "HNSW"}
    rows = []
    for config, r in g2["configs"].items():
        recall = r["validation"].get("recall_vs_P2", {})
        for m in METRICS:
            s = r["timing"][m]["query_time"]
            rows.append([f"{names[config]}, {METRIC_NAMES[m]}", str(s["n"]), ms(s["min"], 2),
                         ms(s["max"], 2), ms(s["avg"], 2), ms(s["std"], 2), pct(cv(s)),
                         num(p2["timing"][m]["query_time"]["avg"] / s["avg"], 0),
                         pct(recall[m]) if m in recall else "–"])
    return table(
        ["Configuració", "n", "mín (ms)", "màx (ms)", "mitjana (ms)", "desv. (ms)", "CV",
         "× P2", "Recall vs P2"],
        rows, "lrrrrrrrr",
        f"`G2`: temps **per consulta** del top-{g2['k']} (una sola crida SQL); n = 10 "
        f"consultes × {g2['repeats']} rondes. *× P2*: vegades més ràpid que la força bruta "
        f"en SQL de `P2`.",
        widths=[28, 5, 10, 10, 12, 10, 7, 7, 11])


class GV:
    """Named values used inline in the pgvector section."""

    @staticmethod
    def _grid(name: str, size: int = DEFAULT_BATCH_SIZE) -> dict:
        return _by_size(load(name)["grid"])[size]

    @classmethod
    def text_vs_p0(cls) -> str:
        """Relative difference of the text load, G0 vs. P0, at the chosen size."""
        g, p = cls._grid("G0")["total_time"]["avg"], cls._grid("P0")["total_time"]["avg"]
        return pct(abs(g / p - 1))

    @classmethod
    def store_total(cls) -> str:
        return num(cls._grid("G1")["total_time"]["avg"], 2)

    @classmethod
    def store_gain_vs_p1(cls) -> str:
        g, p = cls._grid("G1")["total_time"]["avg"], cls._grid("P1")["total_time"]["avg"]
        return pct(1 - g / p)

    @staticmethod
    def index_build(metric: str = "l2") -> str:
        return num(load("G1")["index"][metric]["build_time"]["avg"], 2)

    @staticmethod
    def index_mb(metric: str = "l2") -> str:
        return mb(load("G1")["index"][metric]["index_bytes"])

    @staticmethod
    def table_mb() -> str:
        return mb(load("G1")["index"]["table_bytes"])

    @staticmethod
    def query_ms(config: str, metric: str = "l2") -> str:
        return ms(load("G2")["configs"][config]["timing"][metric]["query_time"]["avg"], 1)

    @staticmethod
    def query_cv(config: str, metric: str = "l2") -> str:
        return pct(cv(load("G2")["configs"][config]["timing"][metric]["query_time"]))

    @staticmethod
    def speedup_vs_p2(config: str, metric: str = "l2") -> str:
        g = load("G2")["configs"][config]["timing"][metric]["query_time"]["avg"]
        return num(load("P2")["timing"][metric]["query_time"]["avg"] / g, 0)

    @staticmethod
    def vs_chroma(metric: str = "l2") -> str:
        """How many times faster G2 with HNSW is than C2 (both with the exclusion)."""
        g = load("G2")["configs"]["hnsw"]["timing"][metric]["query_time"]["avg"]
        return num(load("C2")["timing"][metric]["query_time"]["avg"] / g, 0)

    @staticmethod
    def ef_search() -> str:
        return str(load("G2")["hnsw_ef_search"])

    @staticmethod
    def recall(config: str, metric: str = "l2") -> str:
        return pct(load("G2")["configs"][config]["validation"]["recall_vs_P2"][metric])

    @staticmethod
    def same_topk() -> str:
        v = load("G2")["configs"]["exact"]["validation"]
        return f"{v['same_top_k_l2_vs_cosine']}/10"

    @staticmethod
    def identity_error() -> str:
        return sci(load("G2")["configs"]["exact"]["validation"]["max_abs_error_l2sq_vs_2cos"])

    @staticmethod
    def lines(script: str) -> str:
        return str(system_lines(script))


def pgv_cost_table() -> str:
    """Code lines and calls of each pgvector script, as Tables for P and C."""
    g0, g1 = load("G0"), load("G1")
    text_calls = g0["official"]["db_calls"]["statements"]
    emb_calls = g1["official"]["db_calls"]["statements"]
    rows = [
        ["G0 (text)", "0 (codi de `P0`)", str(code_lines("pgvector/G0.py")),
         f"{text_calls} `INSERT` + {text_calls} `COMMIT`"],
        ["G1 (embeddings)", str(system_lines("pgvector/G1.py")), str(code_lines("pgvector/G1.py")),
         f"1 `SELECT` + {emb_calls} `INSERT` + {emb_calls} `COMMIT` + 2 `CREATE INDEX`"],
        ["G2 (similitud)", str(system_lines("pgvector/G2.py")), str(code_lines("pgvector/G2.py")),
         "1 consulta SQL per consulta i mètrica"],
    ]
    return table(["Script", "Línies (sistema)", "Línies (total)", "Crides a la BD (càrrega oficial)"],
                 rows, "lrrl",
                 "Cost en codi i crides de pgvector, amb el mateix criteri de línies que a "
                 "PostgreSQL i Chroma.",
                 widths=[20, 16, 12, 52])


def generation_spread() -> str:
    """Range of the per-batch embedding generation time across P1, C1 and G1:
    same model, same machine, different runs."""
    avgs = [load(n)["generate"]["batch_time"]["avg"] for n in ("P1", "C1", "G1")]
    return f"{ms(min(avgs), 0)} i {ms(max(avgs), 0)} ms"


class GV2:
    """Extra inline values for the pgvector section."""

    @staticmethod
    def row_by_row(name: str) -> str:
        return num(_by_size(load(name)["grid"])[1]["total_time"]["avg"], 1)

    @staticmethod
    def same_code_spread() -> str:
        """Row-by-row text load of G0 vs. P0: identical code, different runs."""
        g = _by_size(load("G0")["grid"])[1]["total_time"]["avg"]
        p = _by_size(load("P0")["grid"])[1]["total_time"]["avg"]
        return pct(abs(g / p - 1))
