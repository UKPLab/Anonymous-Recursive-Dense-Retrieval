#!/usr/bin/env python3
"""Aggregate the outputs of reproduce/run.py and compare them with the paper values of the configuration.

    python reproduce/aggregate.py --config reproduce/configs/multihop_tree.yaml --results results/

Suite scores follow the paper's conventions:

* **Multi-hop**: the mean over queries of Recall@k over gold groups. Queries without gold are skipped.
* **NanoBEIR and BRIGHT**: the task macro.
* **TopiOCQA**: the query mean.
* **ToolRet**: the macro over its 35 query groups.
* **CoIR**: the macro over its ten datasets. Each dataset is first averaged over its subsplits, which covers
  the six CodeSearchNet-* and the six CodeSearchNet-ccr-* languages.

A (suite, task) counts only when all of its shards are present, so partial runs are never reported as results.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import yaml

MULTI_HOP = ("musique", "browsecomp", "fanoutqa", "frames")
METRIC_INDEX = {"recall@5": 0, "recall@10": 1, "recall@100": 2}


def load(results_dir, name):
    """{(suite, task): merged result}, only for complete shard sets."""
    by_task = defaultdict(dict)
    for path in glob.glob(os.path.join(results_dir, f"{name}-*.json")):
        d = json.load(open(path))
        by_task[(d["suite"], d["task"])][d["shard"]] = d
    merged, incomplete = {}, []
    for key, shards in by_task.items():
        n = next(iter(shards.values()))["num_shards"]
        if sorted(shards) != list(range(n)):
            incomplete.append((key, sorted(shards), n))
            continue
        parts = [shards[i] for i in range(n)]
        out = {"suite": key[0], "task": key[1], "parent": parts[0].get("parent"), "qids": [], "groups": [], "runs": {}}
        for p in parts:
            out["qids"] += p["qids"]
            out["groups"] += p["groups"] or [None] * len(p["qids"])
            for k, v in p["runs"].items():
                out["runs"].setdefault(k, []).extend(v["values"])
        merged[key] = out
    return merged, incomplete


def suite_score(suite, tasks, key, metric):
    """Score of one run/readout on one suite, or None if the suite has no values for it."""
    idx = METRIC_INDEX.get(metric)
    rows = []  # (task, parent, group, value)
    for t in tasks:
        vals = t["runs"].get(key)
        if vals is None:
            continue
        for q, g, v in zip(t["qids"], t["groups"], vals):
            if v is None:
                continue
            x = v[idx] if isinstance(v, list) else v
            if x is None or (isinstance(x, float) and np.isnan(x)):
                continue
            rows.append((t["task"], t["parent"], g, float(x)))
    if not rows:
        return None
    if suite in MULTI_HOP or suite == "topiocqa":
        return 100 * float(np.mean([r[3] for r in rows]))
    if suite == "toolret":
        by = defaultdict(list)
        for _, _, g, v in rows:
            by[g].append(v)
        return 100 * float(np.mean([np.mean(v) for v in by.values()]))
    by = defaultdict(list)
    for t, p, _, v in rows:
        by[(p or t, t)].append(v)
    if suite == "coir":
        ds = defaultdict(list)
        for (p, _), v in by.items():
            ds[p].append(np.mean(v))
        return 100 * float(np.mean([np.mean(v) for v in ds.values()]))
    return 100 * float(np.mean([np.mean(v) for v in by.values()]))  # task macro (NanoBEIR, BRIGHT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--results", default="results")
    ap.add_argument("--tolerance", type=float, default=0.3, help="max |ours - paper| in points to count as reproduced")
    ap.add_argument("--out", default=None, help="write the comparison as JSON")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    merged, incomplete = load(args.results, cfg["name"])
    for key, have, n in incomplete:
        print(f"[incomplete] {key}: shards {have} of {n} — skipped")
    by_suite = defaultdict(list)
    for (suite, _), t in merged.items():
        by_suite[suite].append(t)
    metric = cfg["expected"].get("metric", "recall@5")
    suites = cfg["suites"]
    report, ok, total = {}, 0, 0
    print(f"{'run/readout':28s} " + " ".join(f"{s[:10]:>17s}" for s in suites) + f" {'mean':>17s}")
    for key, exp in cfg["expected"].items():
        if key == "metric":
            continue
        ours = {s: suite_score(s, by_suite.get(s, []), key, metric) for s in suites}
        row, cells = {}, []
        for s in suites + ["mean"]:
            if s == "mean":
                have = [ours[x] for x in suites]
                o = float(np.mean(have)) if all(v is not None for v in have) else None
            else:
                o = ours[s]
            p = exp.get(s)
            d = None if (o is None or p is None) else o - p
            if d is not None:
                total += 1
                ok += abs(d) <= args.tolerance
            row[s] = {"ours": o, "paper": p, "delta": d}
            cells.append("—".rjust(17) if o is None else (f"{o:6.2f}" + ("" if p is None else f" ({d:+.2f})")).rjust(17))
        report[key] = {"source": exp.get("source"), **row}
        print(f"{key:28s} " + " ".join(cells))
    print(f"\n{ok}/{total} values within ±{args.tolerance} of the paper")
    if args.out:
        json.dump({"config": cfg["name"], "tolerance": args.tolerance, "incomplete": incomplete, "results": report},
                  open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
