#!/usr/bin/env python3
"""Summarizes the raw CSV files that VectorStoreBenchmark writes, one table per results directory.

  summarize.py <results dir>...      e.g. summarize.py benchmark-results/2026-10-14-*

Recomputes everything from the raw files (latency percentiles, recall@10, insert rate), checks
that they are complete and consistent, prints a Markdown table, and writes summary.csv into each
directory. Needs only the Python standard library.
"""

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

SEARCHES = ["k10", "k100", "k1000", "k10_filtered"]


def percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile, as the benchmark computes it."""
    return sorted_values[math.ceil(p * len(sorted_values)) - 1]


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize(directory: Path) -> list[dict]:
    rows = []
    for run_file in sorted(directory.glob("*-run.csv")):
        prefix = run_file.name.removesuffix("-run.csv")
        (run,) = read_csv(run_file)
        searches = read_csv(directory / f"{prefix}-searches.csv")
        inserts = read_csv(directory / f"{prefix}-inserts.csv")
        corpus = int(run["corpus"])

        inserted = sum(int(r["rows"]) for r in inserts)
        insert_ns = sum(int(r["latency_ns"]) for r in inserts)
        if inserted != corpus or insert_ns != int(run["insert_ns"]):
            raise SystemExit(f"{prefix}: inserts.csv does not add up to run.csv")

        row = {
            "store": run["store"],
            "corpus": corpus,
            "device": run["device"],
            "insert_per_s": round(corpus / (insert_ns / 1e9), 1),
        }
        by_search = defaultdict(list)
        for s in searches:
            if len(s["ids"].split()) != int(s["returned"]):
                raise SystemExit(f"{prefix}: query {s['query']} lists {len(s['ids'].split())} ids, returned {s['returned']}")
            by_search[s["search"]].append(s)
        for name in SEARCHES:
            results = by_search[name]
            latencies = sorted(int(s["latency_ns"]) / 1e6 for s in results)
            row[f"{name}_p50_ms"] = round(percentile(latencies, 0.5), 3)
            row[f"{name}_p99_ms"] = round(percentile(latencies, 0.99), 3)
            row[f"{name}_recall_at_10"] = round(sum(int(s["hits_at_10"]) for s in results) / (10 * len(results)), 4)
        row["get_100_ids_ms"] = round(int(run["get_100_ids_ns"]) / 1e6, 3)
        if run.get("reopen_and_get_all_ids_ns"):
            row["reopen_ms"] = round(int(run["reopen_and_get_all_ids_ns"]) / 1e6, 3)
        if run.get("disk_bytes"):
            row["disk_mb"] = round(int(run["disk_bytes"]) / 1e6, 1)
        row["embeddings_exact"] = run["embeddings_exact"]
        rows.append(row)
    rows.sort(key=lambda r: (r["corpus"], r["store"]))
    return rows


def print_markdown(directory: Path, rows: list[dict]) -> None:
    print(f"## {directory.name}\n")
    print(f"Device: {rows[0]['device']}\n")
    print("| store | corpus | insert/s | k10 p50 ms | k10 recall | k100 p50 ms | k100 recall | k1000 p50 ms "
          "| filtered k10 p50 ms | filtered recall | reopen ms | disk MB |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(f"| {r['store']} | {r['corpus']:,} | {r['insert_per_s']:,.0f} | {r['k10_p50_ms']} | {r['k10_recall_at_10']} "
              f"| {r['k100_p50_ms']} | {r['k100_recall_at_10']} | {r['k1000_p50_ms']} | {r['k10_filtered_p50_ms']} "
              f"| {r['k10_filtered_recall_at_10']} | {r.get('reopen_ms', '')} | {r.get('disk_mb', '')} |")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    for arg in sys.argv[1:]:
        directory = Path(arg)
        rows = summarize(directory)
        if not rows:
            continue
        with open(directory / "summary.csv", "w", newline="", encoding="utf-8") as f:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print_markdown(directory, rows)


if __name__ == "__main__":
    main()
