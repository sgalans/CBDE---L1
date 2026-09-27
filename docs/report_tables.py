"""Tables and figures for docs/informe.qmd, built from results/*.json.

The report never contains hand-typed measurements: every number comes from the
JSON files written by the scripts, so re-running an experiment and re-rendering
the report keeps text and data consistent.

Tables are returned as Markdown strings (printed from a cell with
``output: asis``); numbers use a decimal comma, as the report is in Catalan.
"""

from __future__ import annotations

import json
import sys
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import DEFAULT_BATCH_SIZE, RESULTS_DIR, load_queries  # noqa: E402

METRIC_NAMES = {"l2": "L2", "cosine": "Cosinus", "cosine_unit": "Cosinus unitari (1 − a·b)"}


@cache
def load(name: str) -> dict:
    with (RESULTS_DIR / f"{name}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def num(x: float, decimals: int = 3) -> str:
    """Number with a decimal comma and thin-space thousands (Catalan style)."""
    text = f"{x:,.{decimals}f}"
    return text.replace(",", " ").replace(".", ",")


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
        f": {caption}",
        "",
    ]
    return "\n".join(lines)


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
        ["Mida de lot", "Lots (`INSERT`)", "Text: total (s)", "desv.",
         "Embeddings: total (s)", "desv."],
        rows, "rrrrrr",
        f"Temps total de càrrega de les 10.000 frases per mida de lot "
        f"(mitjana i desviació de {reps} càrregues completes). "
        f"En negreta, la mida triada.")


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
    return table(
        ["Operació", "n", "mín (ms)", "màx (ms)", "mitjana (ms)", "desv. (ms)", "CV"],
        rows, "lrrrrrr",
        f"Temps per lot de {DEFAULT_BATCH_SIZE} frases. CV = desviació / mitjana.",
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
        f"Temps del top-{p2['k']} per consulta ({p2['repeats']} rondes × 10 consultes). "
        f"El cosinus unitari és una variant de referència, no una tercera mètrica.")


def code_lines(script: str) -> int:
    """Lines of code of a script: no blank lines, comments or docstrings.

    Embedded SQL counts: it is code the system forced us to write.
    """
    import ast
    import io
    import tokenize

    source = (ROOT / script).read_text(encoding="utf-8")
    docstring_lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef)) and ast.get_docstring(node):
            doc = node.body[0]
            docstring_lines.update(range(doc.lineno, doc.end_lineno + 1))
    code: set[int] = set()
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                        tokenize.DEDENT, tokenize.ENDMARKER):
            continue
        code.update(line for line in range(tok.start[0], tok.end[0] + 1)
                    if line not in docstring_lines)
    return len(code)


def pg_cost_table() -> str:
    """Code lines and database calls of each PostgreSQL script."""
    p0, p1 = load("P0"), load("P1")
    text_calls = _by_size(p0["grid"])[DEFAULT_BATCH_SIZE]["db_calls"]["statements"]
    emb_calls = _by_size(p1["grid"])[DEFAULT_BATCH_SIZE]["db_calls"]["statements"]
    rows = [
        ["P0 (text)", str(code_lines("postgres/P0.py")),
         f"{text_calls} `INSERT` + {text_calls} `COMMIT`"],
        ["P1 (embeddings)", str(code_lines("postgres/P1.py")),
         f"1 `SELECT` + {emb_calls} `INSERT` + {emb_calls} `COMMIT`"],
        ["P2 (similitud)", str(code_lines("postgres/P2.py")),
         "1 crida a `top_k_similar` per consulta i mètrica"],
    ]
    return table(["Script", "Línies de codi", "Crides a la BD (càrrega oficial)"], rows, "lrl",
                 "Cost en codi i crides. Les línies inclouen el SQL i les comprovacions; "
                 "no inclouen comentaris ni docstrings.",
                 widths=[22, 16, 62])


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
        e = load("P2")["validation"]["max_abs_error_l2sq_vs_2cos"]
        mantissa, exp = f"{e:.0e}".split("e")
        return f"{mantissa}·10^{int(exp)}^"
