"""Build the Group 11 poster (A2 portrait) from the results of ``python main.py``.

Usage (from this folder):
    python build_poster.py                 # reads results/ next to this file, writes poster.html
    python build_poster.py --pdf           # also prints poster.pdf and poster.png with Microsoft Edge

The text of the poster is in poster_template.html; this script only fills in the numbers,
tables and charts (inline SVG), so that every number on the poster comes from the result files.
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LOGS = ["BPIC11_f1", "BPIC11_f2", "BPIC11_f3"]
STRATEGIES = {"apriori": ("A", "#2a78d6"), "impressed": ("B", "#eb6834")}
INK, INK2, MUTED, HAIR, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SIZE_FILLS = {1: "#d9d8d2", 2: "#a3a19a", 3: "#5f5e5a"}  # neutral ordinal ramp for itemset size
SIZE_TEXT = {1: INK, 2: INK, 3: "#ffffff"}
FONT = 'font-family="Segoe UI, system-ui, sans-serif"'
PT = 0.3528  # 1 pt in mm; the SVG user unit is 1 mm when the chart is as wide as its viewBox


def esc(text) -> str:
    return html.escape(str(text))


def svg(width: float, height: float, body: str) -> str:
    return (f'<svg class="chart" viewBox="0 0 {width:.1f} {height:.1f}" xmlns="http://www.w3.org/2000/svg" '
            f'{FONT}>{body}</svg>')


def text(x, y, s, size=11, fill=INK, anchor="start", weight=400, extra="") -> str:
    return (f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size * PT:.2f}" fill="{fill}" text-anchor="{anchor}" '
            f'font-weight="{weight}" {extra}>{esc(s)}</text>')


def nice_step(value: float, max_ticks: int = 5) -> float:
    """Tick step of 1, 2 or 5 times a power of ten, so that 0..value needs at most ``max_ticks`` steps."""
    if value <= 0:
        return 1.0
    raw = value / max_ticks
    power = 10 ** np.floor(np.log10(raw))
    for step in (1, 2, 5, 10):
        if raw <= step * power:
            return step * power
    return 10 * power


def nice_max(value: float, max_ticks: int = 5) -> float:
    step = nice_step(value, max_ticks)
    return float(np.ceil(value / step - 1e-9) * step)


def ticks(maximum: float, max_ticks: int = 5) -> list[float]:
    step = nice_step(maximum, max_ticks)
    return [round(i * step, 10) for i in range(int(round(maximum / step)) + 1)]


def fmt(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".") if value else "0"


# ---------------------------------------------------------------- loading


def load(results: Path) -> dict:
    summary = pd.read_csv(results / "summary.csv")
    data = {"summary": summary, "importance": {}, "ranking": {}}
    for dataset in summary["dataset"].unique():
        for strategy in STRATEGIES:
            wide = results / dataset / f"{strategy}_location_importance.csv"
            if wide.exists():
                data["importance"][(dataset, strategy)] = pd.read_csv(wide)
                data["ranking"][(dataset, strategy)] = pd.read_csv(results / dataset / f"{strategy}_summary.csv")
    data["simulated"] = simulated_counts(results / "simulated")
    return data


def simulated_counts(folder: Path) -> pd.DataFrame:
    """Per scenario and strategy: in how many logs {x}, {p, q} and {y} were selected."""
    rows = []
    for run in sorted(p for p in folder.iterdir() if p.is_dir()):
        scenario = run.name.rsplit("_seed", 1)[0]
        for strategy in STRATEGIES:
            itemsets = {frozenset(i) for i in json.loads((run / f"{strategy}_itemsets.json").read_text())}
            rows.append({"scenario": scenario, "strategy": strategy, "itemsets": len(itemsets),
                         "x": frozenset(["x"]) in itemsets, "pq": frozenset(["p", "q"]) in itemsets,
                         "y": frozenset(["y"]) in itemsets})
    runs = pd.DataFrame(rows)
    noise = pd.read_csv(folder / "summary_simulated.csv").groupby(["scenario", "strategy"])["noise_only_share"].mean()
    counts = runs.groupby(["scenario", "strategy"]).agg(logs=("x", "size"), x=("x", "sum"), pq=("pq", "sum"),
                                                        y=("y", "sum"), itemsets=("itemsets", "mean"))
    return counts.join(noise).reset_index()


def log_stats(path: Path, case="case:concept:name") -> dict:
    frame = pd.read_csv(path)
    lengths = frame.groupby(case).size()
    labels = frame.groupby(case)["label"].first()
    return {"cases": len(lengths), "activities": frame["concept:name"].nunique(),
            "max_length": int(lengths.max()), "positive": float((labels == 1).mean())}


# ---------------------------------------------------------------- static pieces


def trace_rows() -> str:
    """The six sepsis cases of [4, Table 1]; LA marked, its position-2 occurrences outlined."""
    cases = [
        ("RG LA LE CR ER ST IL IA NC ER CR LE CR", "Negative"),
        ("RG ER ST IL LE CR IA LA IC DI", "Positive"),
        ("RG LA ER ST LE CR IA NC CR LE CR LE CR", "Negative"),
        ("RG LE LE LA ER ST CR IA IC LE LA NC DI", "Positive"),
        ("RG ER ST LE CR IL IA LA NC CR LE", "Positive"),
        ("RG LA CR ER ST LE IL IA NC DI", "Negative"),
    ]
    rows = []
    for number, (trace, outcome) in enumerate(cases, start=1):
        tokens = []
        for position, activity in enumerate(trace.split(), start=1):
            css = "tok la2" if activity == "LA" and position == 2 else "tok la" if activity == "LA" else "tok"
            tokens.append(f'<span class="{css}">{activity}</span>')
        out_class = ' class="out neg"' if outcome == "Negative" else ' class="out"'
        rows.append(f'<tr><td class="case">{number}</td><td>{"".join(tokens)}</td><td{out_class}>{outcome}</td></tr>')
    return "\n".join(rows)


def course_map() -> str:
    """Event log -> looking back (discovery, conformance) / looking ahead (PPM, explanation)."""
    w, h = 112, 52
    navy, red = "#283a8f", "#7f3129"
    box = lambda x, y, bw, bh, fill, stroke, sw=0.35: (
        f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="1.6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    arrow = lambda x1, y1, x2, y2: (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{MUTED}" stroke-width="0.45" marker-end="url(#ah)"/>')
    body = ['<defs><marker id="ah" viewBox="0 0 6 6" refX="5" refY="3" markerWidth="4" markerHeight="4" orient="auto">'
            f'<path d="M0,0 L6,3 L0,6 z" fill="{MUTED}"/></marker></defs>']
    body.append(box(1, 18, 22, 14, "#f6f5f1", AXIS))
    body.append(text(12, 24.2, "Event", 11, INK, "middle", 650) + text(12, 28.6, "log", 11, INK, "middle", 650))
    body.append(arrow(23.5, 22, 33, 10))
    body.append(arrow(23.5, 28, 33, 40))
    # looking back
    body.append(box(33.5, 1, 77, 18, "#f6f5f1", AXIS))
    body.append(text(36, 6.3, "Looking back", 10.5, INK2, "start", 650))
    body.append(text(36, 11.2, "discovery, conformance checking", 10.5, INK))
    body.append(text(36, 15.6, "weeks 2–7 [3]", 9.5, MUTED))
    # looking ahead
    body.append(box(33.5, 23, 77, 28, "#ffffff", navy, 0.5))
    body.append(text(36, 28.3, "Looking ahead", 10.5, navy, "start", 650))
    body.append(text(36, 33.0, "predict / act on running cases (weeks 8–9)", 10.5, INK))
    body.append(box(36, 35.5, 33, 13, "#f6f5f1", AXIS, 0.3))
    body.append(text(52.5, 40.6, "groups 7, 9, 10", 9.5, INK2, "middle"))
    body.append(text(52.5, 45.0, "predict / act", 9.5, MUTED, "middle"))
    body.append(box(72, 35.5, 36, 13, "#fbf3f1", red, 0.6))
    body.append(text(90, 40.6, "group 11: explain", 9.5, red, "middle", 650))
    body.append(text(90, 45.0, "the prediction (xPPM)", 9.5, red, "middle"))
    return svg(w, h, "".join(body))


def arrow_svg() -> str:
    return ('<svg viewBox="0 0 10 10"><path d="M1,5 H8 M5,2 L8.5,5 L5,8" fill="none" stroke="#283a8f" '
            'stroke-width="1.3" stroke-linecap="round" stroke-linejoin="round"/></svg>')


def tokens(seq: str, highlight: set[int], positions: bool = False) -> str:
    out = []
    for i, activity in enumerate(seq.split(), start=1):
        css = "ptok hl" if i in highlight else "ptok"
        sup = f"<sup>{i}</sup>" if positions else ""
        out.append(f'<span class="{css}">{sup}{activity}</span>')
    return "".join(out)


# ---------------------------------------------------------------- results pieces


def data_rows(summary: pd.DataFrame, results: Path, code: Path | None) -> str:
    rows = []
    max_length = {"BPIC11_f1": 36, "BPIC11_f2": 40, "BPIC11_f3": 31}  # printed by main.py for each log
    for log in LOGS:
        row = summary[(summary.dataset == log)].iloc[0]
        rows.append(f"<tr><td>BPIC11 {log[-2:]}</td><td>{row.cases}</td><td>{row.activities}</td>"
                    f"<td>{max_length[log]}</td><td>{row.positive_rate:.0%}</td><td>E3</td></tr>")
    synthetic = summary[summary.dataset == "synthetic"].iloc[0]
    synthetic_length = log_stats(code / "data/synthetic/synthetic.csv")["max_length"] if code else 7
    rows.append(f"<tr><td>Synthetic + 2 controls</td><td>{synthetic.cases}</td><td>{synthetic.activities}</td>"
                f"<td>{synthetic_length}</td><td>{synthetic.positive_rate:.0%}</td><td>E1</td></tr>")
    simulated = [log_stats(p) for p in sorted((results / "simulated").glob("*.csv"))
                 if p.name.startswith("SIM_")]
    if simulated:
        cases = simulated[0]["cases"]
        activities = max(s["activities"] for s in simulated)
        length = max(s["max_length"] for s in simulated)
        positive = np.mean([s["positive"] for s in simulated])
        rows.append(f"<tr><td>Simulated, 2 × 10 logs</td><td>{cases}</td><td>{activities}</td>"
                    f"<td>{length}</td><td>{positive:.0%}</td><td>E2</td></tr>")
    return "\n".join(rows)


def chart_e1(data: dict) -> tuple[str, str]:
    """Bars: mean F1 drop of the top itemset in the synthetic log and in both controls (strategy A)."""
    summary = data["summary"]
    entries = []
    labels = {"synthetic": "Synthetic log", "shuffled_within_case": "Control: within case",
              "shuffled_outcomes": "Control: outcomes"}
    for dataset, label in labels.items():
        ranking = data["ranking"][(dataset, "apriori")]
        top = ranking.iloc[0]
        test_f1 = summary[(summary.dataset == dataset) & (summary.strategy == "apriori")].test_f1.iloc[0]
        entries.append((label, top.itemset, float(top["mean"]), test_f1))
    w, row_h, left, right = 92, 9.5, 36, 13
    h = 6 + row_h * len(entries) + 9
    x_max = nice_max(max(e[2] for e in entries))
    scale = lambda v: left + (w - left - right) * v / x_max
    body = []
    for t in ticks(x_max):
        x = scale(t)
        body.append(f'<line x1="{x:.2f}" y1="3" x2="{x:.2f}" y2="{h - 8:.2f}" stroke="{HAIR}" stroke-width="0.25"/>')
        body.append(text(x, h - 3.5, fmt(t), 10, MUTED, "middle"))
    for i, (label, itemset, mean, test_f1) in enumerate(entries):
        y = 5 + i * row_h
        body.append(text(left - 2.5, y + 3.2, label, 11, INK, "end", 650 if i == 0 else 400))
        body.append(text(left - 2.5, y + 7.2, f"{{{itemset}}} · test F1 {test_f1:.2f}", 9.5, MUTED, "end"))
        bar = max(scale(mean) - left, 0)
        if bar > 0.4:
            body.append(f'<path d="M{left},{y + 1.2} h{bar - 1:.2f} a1,1 0 0 1 1,1 v3.6 a1,1 0 0 1 -1,1 h{-(bar - 1):.2f} z" '
                        f'fill="#2a78d6"/>')
        body.append(text(left + bar + 1.5, y + 4.6, f"{mean:.3f}", 10.5, INK2))
    body.append(f'<line x1="{left}" y1="3" x2="{left}" y2="{h - 8:.2f}" stroke="{AXIS}" stroke-width="0.35"/>')
    synthetic, within, outcomes = entries
    # Largest effect in either direction of any itemset (both strategies) in the two controls.
    noise = max(abs(data["ranking"][(d, s)]["mean"]).max()
                for d in ("shuffled_within_case", "shuffled_outcomes") for s in STRATEGIES)
    finding = (f"Moving <b>B</b> lowers F1 by <b>{synthetic[2]:.2f}</b> on average; in the controls no itemset moves "
               f"it by more than ±{noise:.2f}. The location is learnable (test F1 {synthetic[3]:.2f} vs. "
               f"{within[3]:.2f} / {outcomes[3]:.2f}). Only A tests B; IMPresseD does not select it. "
               f"With 15 cases (3 test cases per fold) E1 is a sanity check.")
    return svg(w, h, "".join(body)), finding


def table_e2(data: dict) -> tuple[str, str]:
    counts = data["simulated"]
    names = {"SIM_rare_pair": "Rare pair<br>(30&nbsp;%)", "SIM_frequent_pair": "Frequent<br>pair (60&nbsp;%)"}
    rows = ['<table class="e2"><tr><th>Pair</th><th></th><th>{x}<br>location</th>'
            '<th>{p,q}<br>location</th><th>{y}<br>presence</th><th>Pure<br>noise</th></tr>']
    for scenario in ["SIM_rare_pair", "SIM_frequent_pair"]:
        for strategy, (letter, color) in STRATEGIES.items():
            row = counts[(counts.scenario == scenario) & (counts.strategy == strategy)].iloc[0]
            cell = lambda v: f'<td class="zero">{v}/{row.logs}</td>' if v == 0 else f"<td>{v}/{row.logs}</td>"
            first = f"<td rowspan=2>{names[scenario]}</td>" if strategy == "apriori" else ""
            rows.append(f'<tr>{first}<td><span class="dot {"a" if letter == "A" else "b"}"></span>{letter}</td>'
                        f"{cell(int(row.x))}{cell(int(row.pq))}{cell(int(row.y))}"
                        f"<td>{row.noise_only_share:.0%}</td></tr>")
    rows.append("</table>")
    rare_b = counts[(counts.scenario == "SIM_rare_pair") & (counts.strategy == "impressed")].iloc[0]
    finding = (f"Both select the distractor {{y}} in every log; A always selects {{x}} (it keeps every activity), "
               f"B misses it in {rare_b.logs - int(rare_b.x)} of {rare_b.logs} rare-pair logs. <b>No strategy ever selects "
               f"{{p, q}}</b>: A ranks by frequency (its top 10 is full of start/end activities), and B only extends "
               f"patterns whose <i>count</i> says something about the outcome, which p and q do not.")
    return "\n".join(rows), finding


def chart_e3_sizes(summary: pd.DataFrame) -> str:
    w, row_h, left, right = 96, 5.6, 16, 9
    rows = [(log, strategy) for log in LOGS for strategy in STRATEGIES]
    h = 2 + row_h * len(rows) + 2 * 2.2 + 7
    totals = {key: int(summary[(summary.dataset == key[0]) & (summary.strategy == key[1])].itemsets.iloc[0])
              for key in rows}
    x_max = nice_max(max(totals.values()))
    scale = lambda v: (w - left - right) * v / x_max
    body, y = [], 2
    for t in ticks(x_max):
        x = left + scale(t)
        body.append(f'<line x1="{x:.2f}" y1="1" x2="{x:.2f}" y2="{h - 7:.2f}" stroke="{HAIR}" stroke-width="0.25"/>')
        body.append(text(x, h - 2.8, fmt(t), 10, MUTED, "middle"))
    for i, (log, strategy) in enumerate(rows):
        if i and strategy == "apriori":
            y += 2.2
        row = summary[(summary.dataset == log) & (summary.strategy == strategy)].iloc[0]
        letter, color = STRATEGIES[strategy]
        if strategy == "apriori":
            body.append(text(1, y + row_h + 0.9, f"f{log[-1]}", 11.5, INK, "start", 650))
        body.append(f'<circle cx="{left - 6.5}" cy="{y + row_h / 2 - 0.3:.2f}" r="1.4" fill="{color}"/>')
        body.append(text(left - 4.3, y + row_h / 2 + 1.1, letter, 10.5, INK, "start", 650))
        x = left
        for size in (1, 2, 3):
            count = int(row.get(f"size{size}", 0) if not pd.isna(row.get(f"size{size}", 0)) else 0)
            width = scale(count)
            if width <= 0:
                continue
            seg = max(width - 0.5, 0.2)  # 0.5 mm gap between segments
            body.append(f'<rect x="{x:.2f}" y="{y + 0.6:.2f}" width="{seg:.2f}" height="{row_h - 1.6:.2f}" rx="0.6" '
                        f'fill="{SIZE_FILLS[size]}"/>')
            if seg > 7:
                body.append(text(x + seg / 2, y + row_h / 2 + 1.0, count, 9.5, SIZE_TEXT[size], "middle", 600))
            x += width
        body.append(text(x + 1.2, y + row_h / 2 + 1.0, totals[(log, strategy)], 10, INK2))
        y += row_h
    return svg(w, h, "".join(body))


def size_key() -> str:
    return "".join(f'<span style="white-space:nowrap"><span class="sw" style="background:{SIZE_FILLS[s]}"></span>{s}</span>'
                   for s in (1, 2, 3)) + " activities"


def chart_e3_top(data: dict, top_n: int = 3) -> tuple[str, list[dict]]:
    """Per log a small multiple: top itemsets of A and B, mean (dot) and 25-75 % range (bar)."""
    w, row_h, left, right, split = 125, 4.3, 52, 4, 1.4
    panels, tops = [], []
    for log in LOGS:
        entries = []
        for strategy in STRATEGIES:
            ranking = data["ranking"][(log, strategy)].head(top_n)
            wide = data["importance"][(log, strategy)]
            for _, r in ranking.iterrows():
                values = wide[r.itemset].to_numpy()
                entries.append((strategy, r.itemset, float(r["mean"]), *np.percentile(values, [25, 75])))
            tops.append({"log": log, "strategy": strategy, "itemset": ranking.iloc[0].itemset,
                         "mean": float(ranking.iloc[0]["mean"])})
        x_max = nice_max(max(max(e[2], e[4]) for e in entries), 4)
        x_min = min(0.0, min(e[3] for e in entries))
        scale = lambda v: left + (w - left - right) * (v - x_min) / (x_max - x_min)
        top = 6.0
        bottom = top + row_h * len(entries) + split
        h = bottom + 5
        body = [text(0.5, 3.8, f"BPIC11 f{log[-1]}", 12, INK, "start", 650)]
        for t in ticks(x_max, 4):
            x = scale(t)
            body.append(f'<line x1="{x:.2f}" y1="{top:.2f}" x2="{x:.2f}" y2="{bottom:.2f}" stroke="{HAIR}" stroke-width="0.25"/>')
            body.append(text(x, bottom + 3.6, fmt(t), 9.5, MUTED, "middle"))
        body.append(f'<line x1="{scale(0):.2f}" y1="{top:.2f}" x2="{scale(0):.2f}" y2="{bottom:.2f}" stroke="{AXIS}" stroke-width="0.35"/>')
        for i, (strategy, itemset, mean, q25, q75) in enumerate(entries):
            letter, color = STRATEGIES[strategy]
            cy = top + i * row_h + row_h / 2 + (split if i >= top_n else 0)
            label = itemset if len(itemset) <= 30 else itemset[:29] + "…"
            body.append(text(left - 2, cy + 1.2, label, 9.5, INK, "end"))
            body.append(f'<line x1="{scale(q25):.2f}" y1="{cy:.2f}" x2="{max(scale(q75), scale(q25) + 0.3):.2f}" y2="{cy:.2f}" '
                        f'stroke="{color}" stroke-opacity="0.35" stroke-width="2.4" stroke-linecap="round"/>')
            body.append(f'<circle cx="{scale(mean):.2f}" cy="{cy:.2f}" r="1.3" fill="{color}" stroke="#fff" stroke-width="0.4"/>')
        panels.append(svg(w, h, "".join(body)))
    return "".join(f"<div>{p}</div>" for p in panels), tops


# ---------------------------------------------------------------- main


def build(results: Path, code: Path | None) -> str:
    data = load(results)
    summary = data["summary"]
    template = (HERE / "poster_template.html").read_text(encoding="utf-8")
    e1_chart, e1_text = chart_e1(data)
    e2_table, e2_text = table_e2(data)
    e3_top_chart, tops = chart_e3_top(data)
    replacements = {
        "TRACE_ROWS": trace_rows(),
        "COURSE_MAP": course_map(),
        "ARROW": arrow_svg(),
        "PERM_ORIG": tokens("a b c d e", set(), positions=True),
        "PERM_SINGLE": tokens("c a b d e", {1}),
        "PERM_SET": tokens("a c b e d", {3, 5}),
        "DATA_ROWS": data_rows(summary, results, code),
        "CHART_E1": e1_chart, "E1_TEXT": e1_text,
        "TABLE_E2": e2_table, "E2_TEXT": e2_text,
        "CHART_E3_SIZES": chart_e3_sizes(summary),
        "SIZE_KEY": size_key(),
        "CHART_E3_TOP": e3_top_chart,
        "TOP_N": "3",
        "E3_TEXT": e3_text(summary, data, tops),
        "E3_TOP_TEXT": e3_top_text(summary, data, tops),
        "E4_TEXT": e4_text(results),
        "TAKEAWAY": takeaway(),
        "RUN_NOTE": "Code: github.com/HoyngM/group-11-pm",
    }
    for key, value in replacements.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def e3_text(summary: pd.DataFrame, data: dict, tops: list[dict]) -> str:
    """Filled in by hand after reading the results; numbers are taken from the result files."""
    jaccard = summary[(summary.strategy == "impressed") & summary.dataset.isin(LOGS)]["multi_itemset_jaccard_vs_apriori"]
    return (f"A tests mostly single activities, B mostly pairs and triples. Their multi-activity itemsets hardly "
            f"overlap (Jaccard {jaccard.min():.2f}–{jaccard.max():.2f}).")


def e4_text(results: Path) -> str:
    """Sensitivity of A: support 0.2 and top 25 against the default 0.5 and top 10."""
    e4 = pd.read_csv(results / "summary.csv")
    if "apriori_support0.2_top25" not in set(e4.strategy):
        return "E4 not run."
    same, changed, extra = [], [], []
    for log in LOGS:
        base = e4[(e4.dataset == log) & (e4.strategy == "apriori")].iloc[0]
        wide = e4[(e4.dataset == log) & (e4.strategy == "apriori_support0.2_top25")].iloc[0]
        extra.append(int(wide.itemsets - base.itemsets))
        if base.top_itemset == wide.top_itemset:
            same.append(f"f{log[-1]} ({base.top_itemset}, {base.top_importance:.2f})")
        else:
            changed.append((f"f{log[-1]}", max(base.top_importance, wide.top_importance)))
    text = (f"<b>E4 · Sensitivity of A.</b> Support 0.2 and top 25 add {min(extra)}–{max(extra)} multi-activity "
            f"itemsets per log. The top itemset stays the same in {' and '.join(same)}")
    if changed:
        text += "; it changes in " + " and ".join(f"{log}, where every drop is below {value:.2f}"
                                                  for log, value in changed)
    return text + "."


def e3_top_text(summary: pd.DataFrame, data: dict, tops: list[dict]) -> str:
    """Interpretation of the top itemsets; every number is read from the result files."""
    top = {(t["log"], t["strategy"]): t for t in tops}
    a2, a3 = top[("BPIC11_f2", "apriori")], top[("BPIC11_f3", "apriori")]
    b2, b3 = top[("BPIC11_f2", "impressed")], top[("BPIC11_f3", "impressed")]
    f1_max = max(top[("BPIC11_f1", s)]["mean"] for s in STRATEGIES)
    # f2: how many of B's best itemsets contain A's top activity, and what B measures for it alone.
    ranking_b2 = data["ranking"][("BPIC11_f2", "impressed")]
    contains = ranking_b2["itemset"].str.split(", ").apply(lambda items: a2["itemset"] in items)
    leading = int(contains.cummin().sum())  # length of the run of top itemsets that contain it
    alone = ranking_b2.loc[ranking_b2["itemset"] == a2["itemset"], "mean"]
    alone_text = f", alone {alone.iloc[0]:.3f}" if len(alone) else ""
    b3_has_single = a3["itemset"] in set(data["ranking"][("BPIC11_f3", "impressed")]["itemset"])
    f3_text = (f"B does not select it as a single activity; its best itemset reaches only {b3['mean']:.3f}."
               if not b3_has_single else f"B's best: {b3['itemset']} ({b3['mean']:.3f}).")
    return (f"<b>f2:</b> both point to <b>{a2['itemset']}</b>. A tests it alone ({a2['mean']:.3f}); B's {leading} best "
            f"itemsets all contain it (best {b2['mean']:.3f}{alone_text}), so the added activities change little. "
            f"<b>f3:</b> A's top is <b>{a3['itemset']}</b> ({a3['mean']:.3f}); {f3_text} "
            f"<b>f1</b> (label on occurrence only): every mean drop ≤ {f1_max:.3f}, location hardly matters.")


def takeaway() -> str:
    return ("<p>The importance step finds a planted location (E1), but only what is selected can be found.</p>"
            "<p>A and B test different itemsets; on BPIC11 the strongest location effects are single activities, "
            "which A always tests and B can miss (f3).</p>")


def render(html_path: Path) -> None:
    edge = shutil.which("msedge") or r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
    url = html_path.resolve().as_uri()
    common = [edge, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--hide-scrollbars"]
    subprocess.run([*common, f"--print-to-pdf={html_path.with_suffix('.pdf')}", url], check=True)
    subprocess.run([*common, "--window-size=1588,2246", "--force-device-scale-factor=1",
                    f"--screenshot={html_path.with_suffix('.png')}", url], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--results", type=Path, default=HERE / "results", help="results folder of main.py")
    parser.add_argument("--code", type=Path, default=None, help="Project/code folder (for the synthetic log)")
    parser.add_argument("--pdf", action="store_true", help="also write poster.pdf and poster.png with Edge")
    arguments = parser.parse_args()
    out = HERE / "poster.html"
    out.write_text(build(arguments.results, arguments.code), encoding="utf-8")
    print(f"wrote {out}")
    if arguments.pdf:
        render(out)
        print("wrote poster.pdf and poster.png")


if __name__ == "__main__":
    main()
