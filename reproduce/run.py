#!/usr/bin/env python3
"""Reproduce a table of the paper from a YAML configuration.

One traversal is run per search configuration; every readout of that search reads the same saved
traversal (as in the paper's Appendix F). Results are compared with the paper values in the config.

    python reproduce/run.py --config reproduce/configs/table1_multihop.yaml \
        --model RDR-8B --data-root data/ --out results/ [--shard 0 --num-shards 8] [--exact]

Single-hop suites load from the Hugging Face Hub (``pip install recursive-dense-retrieval[eval]``);
multi-hop suites read the prepared files described in ``reproduce/README.md``.
"""
import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import rdr  # noqa: E402
from rdr import DenseIndex, RecursiveRetriever, make_readout, make_search  # noqa: E402
from rdr.evaluation import benchmarks as B  # noqa: E402
from rdr.evaluation.metrics import ndcg_at_k, recall_at_k  # noqa: E402
from rdr.readout.base import backfill  # noqa: E402
from rdr.types import Ranking  # noqa: E402

MULTI_HOP = ("musique", "browsecomp", "fanoutqa", "frames")


def load_tasks(suite: str, data_root: str, hf_cache: str) -> List[B.Task]:
    if suite in MULTI_HOP:
        return B.load_multihop(suite, os.path.join(data_root, suite))
    if suite == "topiocqa":
        return B.load_topiocqa(os.path.join(data_root, "topiocqa_1m"))
    return B.SINGLE_HOP[suite](hf_cache)


def build_readout(spec):
    spec = dict(spec)
    bf = spec.pop("backfill_root_ranking", False)
    ro = make_readout(spec)
    if not bf:
        return ro

    class _Bf(rdr.Readout):
        name = "backfilled"
        needs_embeddings = getattr(ro, "needs_embeddings", False)

        def __call__(self, t):
            r = ro(t)
            return Ranking.from_order(backfill(r.ids.tolist(), t.root.ids[:200].tolist()))

    return _Bf()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", default="RDR-8B")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--hf-cache", default=None)
    ap.add_argument("--index-cache", default="index_cache", help="document embeddings are cached here")
    ap.add_argument("--out", default="results")
    ap.add_argument("--suites", default=None, help="comma-separated subset of the config's suites")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--exact", action="store_true", help="paper-exact batching (slower)")
    ap.add_argument("--encode-device", default=None)
    ap.add_argument("--index-device", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    suites = args.suites.split(",") if args.suites else cfg["suites"]
    exact_grouping = "query" if cfg["prompts"] == "multi-hop" else "slot"
    enc = rdr.Encoder(args.model, query_max_length=cfg.get("query_max_length", 4096),
                      document_max_length=cfg.get("document_max_length", 512), batch_size=8,
                      sort_by_length=not args.exact, devices=[args.encode_device] if args.encode_device else None)
    retr = RecursiveRetriever(enc, prompts=cfg["prompts"], query_batch_size=(10 ** 9 if args.exact else 32),
                              index_device=args.index_device, encode_grouping=exact_grouping if args.exact else "level",
                              inflight=1 if args.exact else 2)
    os.makedirs(args.out, exist_ok=True)
    for suite in suites:
        for task in load_tasks(suite, args.data_root, args.hf_cache):
            if cfg.get("query_sample", {}).get(suite):
                task = task.sample(cfg["query_sample"][suite], seed=cfg.get("query_sample_seed"))
            qsel = list(range(args.shard, len(task.query_ids), args.num_shards))
            qids = [task.query_ids[i] for i in qsel]
            cache = os.path.join(args.index_cache, os.path.basename(args.model), suite, task.cache_key or task.name)
            index = DenseIndex.build(task.corpus_texts, enc, ids=task.corpus_ids, device=args.index_device,
                                     cache_dir=cache, show_progress_bar=True)
            row = {task.corpus_ids[i]: i for i in range(len(task.corpus_ids))}
            excl = [[row[d] for d in task.excluded_ids.get(q, ()) if d in row] or None for q in qids]
            instr = [task.query_instructions[i] for i in qsel]
            queries = [task.query_texts[i] for i in qsel]
            result = {"suite": suite, "task": task.name, "parent": task.parent, "qids": qids,
                      "groups": [task.query_groups.get(q) for q in qids] if task.query_groups else None,
                      "shard": args.shard, "num_shards": args.num_shards, "runs": {}}
            for run in cfg["runs"]:
                search = make_search(run["search"])
                readouts = {name: build_readout(spec) for name, spec in run["readouts"].items()}
                views = sorted({v for r in readouts.values() for v in rdr.retriever._views_of(r)})
                needs = any(getattr(r, "needs_embeddings", False) for r in readouts.values())
                t0 = time.time()
                travs = retr.traverse(queries, index, search=search, instruction=instr, exclude=excl, views=views,
                                      keep_embeddings=needs)
                secs = time.time() - t0
                for name, ro in readouts.items():
                    ranks = retr.read(travs, index, ro)
                    vals = []
                    for q, r in zip(qids, ranks):
                        ranked = [task.corpus_ids[i] for i in r.ids.tolist()]
                        if suite in MULTI_HOP:
                            g = task.gold_groups.get(q, [])
                            vals.append([recall_at_k(ranked, g, k) for k in (5, 10, 100)] if g else None)
                        else:
                            vals.append(ndcg_at_k(ranked, task.qrels.get(q, {}), 10))
                    result["runs"][f"{run['name']}/{name}"] = {"values": vals, "seconds": secs}
                del travs
            tag = f"{cfg['name']}-{suite}-{task.name}-sh{args.shard}of{args.num_shards}.json"
            json.dump(result, open(os.path.join(args.out, tag), "w"))
            print("[reproduce] wrote", tag, flush=True)


if __name__ == "__main__":
    main()
