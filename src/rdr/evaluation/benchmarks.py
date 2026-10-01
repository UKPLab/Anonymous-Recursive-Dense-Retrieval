"""Benchmark loaders of the paper's evaluation (requires the ``eval`` extra: ``pip install recursive-dense-retrieval[eval]``).

Single-hop suites load from the Hugging Face Hub as in the paper's evaluation code
(same repositories, splits, revisions, text joins and instruction strings). Multi-hop suites read the
prepared files described in ``reproduce/README.md``.

Every loader returns a list of :class:`Task`.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from .. import instructions as I


@dataclass
class Task:
    """One evaluation task (a corpus with its queries).

    Attributes:
        name: Task name (e.g. ``Bright-biology``, ``NanoSciFact``, ``musique``).
        corpus_ids: Document ids.
        corpus_texts: Document texts, aligned with ``corpus_ids``.
        query_ids: Query ids.
        query_texts: Query texts (TopiOCQA: the serialized conversation).
        query_instructions: Root instruction per query.
        qrels: ``{query id: {doc id: relevance}}``.
        excluded_ids: ``{query id: doc ids that must not be retrieved}`` (BRIGHT).
        query_groups: ``{query id: group}`` for group macro averages (ToolRet's 35 groups).
        gold_groups: Multi-hop gold, ``{query id: [set of doc ids, …]}``; a group counts as found when
            any member is retrieved.
        parent: Dataset of a subsplit in the official macro average (CoIR).
        cache_key: Stable name for the document-embedding cache.
    """

    name: str
    corpus_ids: List[str]
    corpus_texts: List[str]
    query_ids: List[str]
    query_texts: List[str]
    query_instructions: List[str]
    qrels: Dict[str, Dict[str, int]] = field(default_factory=dict)
    excluded_ids: Dict[str, Set[str]] = field(default_factory=dict)
    query_groups: Dict[str, str] = field(default_factory=dict)
    gold_groups: Dict[str, List[Set[str]]] = field(default_factory=dict)
    parent: Optional[str] = None
    cache_key: Optional[str] = None

    def sample(self, limit: int, seed: Optional[int] = None) -> "Task":
        """A deterministic sample of ``limit`` queries, as in the paper's evaluation code.

        With ``seed`` (the paper's CoIR sample uses ``20260924``): ``numpy.random.default_rng([seed,
        crc32(task name)]).choice(n, limit, replace=False)``, sorted. Without a seed: evenly spread,
        ``i * n // limit``.

        Args:
            limit: Queries to keep (CoIR: up to 1,000 per subsplit, 17,901 in total).
            seed: Sampling seed, or ``None`` for the evenly spread sample.
        """
        n = len(self.query_ids)
        if not limit or n <= limit:
            return self
        if seed is not None:
            import zlib

            import numpy as np

            rng = np.random.default_rng([int(seed), zlib.crc32(self.name.encode())])
            pick = sorted(set(int(x) for x in rng.choice(n, limit, replace=False)))
        else:
            pick = sorted(set((i * n) // limit for i in range(limit)))
        return Task(self.name, self.corpus_ids, self.corpus_texts, [self.query_ids[i] for i in pick],
                    [self.query_texts[i] for i in pick], [self.query_instructions[i] for i in pick], self.qrels,
                    self.excluded_ids, self.query_groups, self.gold_groups, self.parent, self.cache_key)


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _splits(ds) -> Dict[str, Any]:
    from datasets import DatasetDict

    return dict(ds) if isinstance(ds, DatasetDict) else {"all": ds}


def _join_title_text(row: Mapping[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or row.get("content") or "").strip()
    return f"{title} {text}".strip()


# --------------------------------------------------------------------------- single-hop
def load_nanobeir(cache_dir: Optional[str] = None, selected: Optional[Set[str]] = None) -> List[Task]:
    """NanoBEIR (13 tasks) from ``sentence-transformers/NanoBEIR-en``; queries without qrels are dropped.

    Args:
        cache_dir: Hugging Face datasets cache directory.
        selected: Only load these tasks.
    """
    repo = "sentence-transformers/NanoBEIR-en"
    corpora = _splits(_load_dataset(repo, "corpus", cache_dir=cache_dir))
    queries = _splits(_load_dataset(repo, "queries", cache_dir=cache_dir))
    qrels = _splits(_load_dataset(repo, "qrels", cache_dir=cache_dir))
    out = []
    for split, instruction in I.NANOBEIR_ROOTS.items():
        if selected and split not in selected:
            continue
        rels: Dict[str, Dict[str, int]] = {}
        for row in qrels[split]:
            rels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = int(row.get("score", 1))
        qmap = {str(r["_id"]): str(r["text"]) for r in queries[split]}
        qids = [str(r["_id"]) for r in queries[split] if str(r["_id"]) in rels]
        out.append(Task(split, [str(r["_id"]) for r in corpora[split]], [str(r["text"]) for r in corpora[split]],
                        qids, [qmap[q] for q in qids], [instruction] * len(qids), rels, cache_key=split))
    return out


BRIGHT_REVISION = "c26703e6600d97c579ee2985f16cf307db13ed85"


def load_bright(cache_dir: Optional[str] = None, selected: Optional[Set[str]] = None,
                instructions: Optional[Mapping[str, str]] = None) -> List[Task]:
    """BRIGHT (12 domains) from ``mteb/BRIGHT`` at the pinned revision, with its excluded ids.

    Args:
        cache_dir: Hugging Face datasets cache directory.
        selected: Only load these tasks.
        instructions: Root instruction per domain (default: :data:`rdr.instructions.BRIGHT_ROOTS`).
    """
    instructions = instructions or I.BRIGHT_ROOTS
    out = []
    for domain, instruction in instructions.items():
        name = f"Bright-{domain}"
        if selected and domain not in selected and name not in selected:
            continue
        docs = _load_dataset("mteb/BRIGHT", "documents", split=domain, cache_dir=cache_dir, revision=BRIGHT_REVISION)
        ex = _load_dataset("mteb/BRIGHT", "examples", split=domain, cache_dir=cache_dir, revision=BRIGHT_REVISION)
        rels, excl = {}, {}
        for row in ex:
            qid = str(row["id"])
            rels[qid] = {str(d): 1 for d in row["gold_ids"]}
            ids = {str(d) for d in (row.get("excluded_ids") or []) if str(d) != "N/A"}
            if ids:
                excl[qid] = ids
        out.append(Task(name, [str(r["id"]) for r in docs], [str(r["content"]) for r in docs],
                        [str(r["id"]) for r in ex], [str(r["query"]) for r in ex], [instruction] * len(ex), rels,
                        excluded_ids=excl, cache_key=f"bright-{domain}"))
    return out


COIR_TASKS = {
    "codetrans-dl": ["codetrans-dl"],
    "codetrans-contest": ["codetrans-contest"],
    "codesearchnet-ccr": [f"CodeSearchNet-ccr-{l}" for l in ("go", "java", "javascript", "ruby", "python", "php")],
    "codesearchnet": [f"CodeSearchNet-{l}" for l in ("go", "java", "javascript", "ruby", "python", "php")],
    "cosqa": ["cosqa"], "apps": ["apps"], "synthetic-text2sql": ["synthetic-text2sql"],
    "stackoverflow-qa": ["stackoverflow-qa"], "codefeedback-st": ["codefeedback-st"], "codefeedback-mt": ["codefeedback-mt"],
}


def load_coir(cache_dir: Optional[str] = None, selected: Optional[Set[str]] = None) -> List[Task]:
    """CoIR (10 datasets, 20 subsplits); ``parent`` is the dataset of the official 10-dataset macro.

    Args:
        cache_dir: Hugging Face datasets cache directory.
        selected: Only load these tasks.
    """
    out = []
    for parent, leaves in COIR_TASKS.items():
        for leaf in leaves:
            if selected and leaf not in selected and parent not in selected:
                continue
            instruction = I.root_for("coir", leaf)
            qc = _load_dataset(f"CoIR-Retrieval/{leaf}-queries-corpus", cache_dir=cache_dir)
            rel_ds = _load_dataset(f"CoIR-Retrieval/{leaf}-qrels", cache_dir=cache_dir)["test"]
            rels: Dict[str, Dict[str, int]] = {}
            for row in rel_ds:
                rels.setdefault(str(row["query_id"]), {})[str(row["corpus_id"])] = int(row["score"])
            qmap = {str(r["_id"]): str(r["text"]) for r in qc["queries"]}
            qids = [q for q in qmap if q in rels]
            out.append(Task(leaf, [str(r["_id"]) for r in qc["corpus"]], [_join_title_text(r) for r in qc["corpus"]],
                            qids, [qmap[q] for q in qids], [instruction] * len(qids), rels, parent=parent,
                            cache_key=leaf))
    return out


TOOLRET_TASKS = {
    **{t: "code" for t in ("craft-math-algebra", "craft-tabmwp", "craft-vqa", "gorilla-huggingface", "gorilla-pytorch",
                           "gorilla-tensor", "toolink")},
    **{t: "web" for t in ("apibank", "apigen", "mnms", "reversechain", "rotbench", "t-eval-dialog", "t-eval-step",
                          "taskbench-daily", "toolace", "toolbench", "toolemu", "tooleyes", "toollens", "ultratool",
                          "autotools-food", "autotools-music", "autotools-weather", "restgpt-spotify",
                          "restgpt-tmdb")},
    **{t: "customized" for t in ("appbench", "gpt4tools", "gta", "taskbench-huggingface", "taskbench-multimedia",
                                 "metatool", "tool-be-honest", "toolalpaca", "toolbench-sam")},
}


def load_toolret(cache_dir: Optional[str] = None) -> List[Task]:
    """ToolRet: one tool corpus, 35 query groups (macro-averaged), per-query instructions.

    Args:
        cache_dir: Hugging Face datasets cache directory.
    """
    from datasets import concatenate_datasets

    tools = concatenate_datasets([_load_dataset("mangopy/ToolRet-Tools", c, cache_dir=cache_dir)["tools"]
                                  for c in ("web", "code", "customized")])
    qids, texts, instr, rels, groups, seen = [], [], [], {}, {}, set()
    for task, cat in TOOLRET_TASKS.items():
        for row in _load_dataset("mangopy/ToolRet-Queries", task, cache_dir=cache_dir)["queries"]:
            raw = str(row["id"])
            qid = raw if raw not in seen else f"{task}::{raw}"
            seen.add(qid)
            labels = row["labels"]
            labels = json.loads(labels) if isinstance(labels, str) else labels
            qids.append(qid)
            texts.append(str(row["query"]))
            instr.append(str(row.get("instruction") or I.TOOLRET_FALLBACK))
            rels[qid] = {str(x["id"]): int(x["relevance"]) for x in labels}
            groups[qid] = f"{cat}::{task}"
    return [Task("ToolRet", [str(r["id"]) for r in tools], [str(r["documentation"]) for r in tools], qids, texts, instr,
                 rels, query_groups=groups, cache_key="toolret-all-tools")]


def load_topiocqa(data_dir: str) -> List[Task]:
    """TopiOCQA with the paper's 1M-passage corpus (``queries.json`` + ``corpus.jsonl`` in ``data_dir``).

    Args:
        data_dir: Directory with the prepared files (see ``reproduce/README.md``).
    """
    queries = json.load(open(os.path.join(data_dir, "queries.json")))
    ids, texts = [], []
    with open(os.path.join(data_dir, "corpus.jsonl")) as fh:
        for line in fh:
            d = json.loads(line)
            ids.append(d["id"])
            t, x = (d.get("title") or "").strip(), (d.get("text") or "").strip()
            texts.append(f"{t}\n{x}" if t else x)
    qids = [q["qid"] for q in queries]
    return [Task("topiocqa1m", ids, texts, qids, [q["conversation"] for q in queries],
                 [I.TOPIOCQA_ROOT] * len(queries), {q["qid"]: {g: 1 for g in q["gold"]} for q in queries},
                 cache_key="topiocqa1m")]


# ---------------------------------------------------------------------------- multi-hop
def _norm(text: str) -> str:
    return " ".join(str(text).split())


def load_multihop(name: str, data_dir: str) -> List[Task]:
    """MuSiQue (support), BrowseComp+ (evidence), FanOutQA and FRAMES (source groups) from prepared files.

    Args:
        name: ``musique``, ``browsecomp``, ``fanoutqa`` or ``frames``.
        data_dir: Directory with the prepared files (see ``reproduce/README.md``).
    """
    name = name.lower()
    if name in ("browsecomp", "browsecomp_plus", "browsecomp+"):
        doc_ids = [str(x) for x in json.load(open(os.path.join(data_dir, "doc_ids.json")))]
        doc_texts = json.load(open(os.path.join(data_dir, "doc_texts.json")))
        queries = {}
        for line in open(os.path.join(data_dir, "queries.tsv")):
            qid, q = line.rstrip("\n").split("\t", 1)
            queries[qid] = q
        ev = {q: set() for q in queries}
        for line in open(os.path.join(data_dir, "qrel_evidence.txt")):
            f = line.split()
            if len(f) >= 3 and f[0] in ev:
                ev[f[0]].add(f[2])
        qids = list(queries)
        return [Task("browsecomp", doc_ids, doc_texts, qids, [queries[q] for q in qids], [I.MULTIHOP_ROOT] * len(qids),
                     {q: {d: 1 for d in ev[q]} for q in qids}, gold_groups={q: [{d} for d in ev[q]] for q in qids})]
    if name == "frames":
        samples = json.load(open(os.path.join(data_dir, "frames.json")))
        corpus = json.load(open(os.path.join(data_dir, "frames_corpus.json")))["docs"]
        qids = [str(i) for i in range(len(samples))]
        gold = {q: [set(map(str, d)) for d in s["groups"].values()] for q, s in zip(qids, samples)}
        return [Task("frames", [str(x["id"]) for x in corpus], [x["text"] for x in corpus], qids,
                     [s["question"] for s in samples], [I.MULTIHOP_ROOT] * len(qids), gold_groups=gold)]
    if name == "fanoutqa":
        docs = [json.loads(line) for line in open(os.path.join(data_dir, "fanoutqa_docs.jsonl")) if line.strip()]
        official = json.load(open(os.path.join(data_dir, "fanoutqa_dev_corpus.json")))
        t2o: Dict[str, Set[str]] = {}
        for i, row in enumerate(official):
            oid = f"fanoutqa_{i:06d}"
            title, text = str(row.get("title", "")), str(row.get("text", ""))
            for cand in (text, f"{title}\n{text}", f"{title} {text}"):
                t2o.setdefault(_norm(cand), set()).add(oid)
        doc_ids = [next(iter(t2o[_norm(r["text"])])) for r in docs]
        qids, qs = [], []
        for line in open(os.path.join(data_dir, "queries.tsv")):
            qid, q = line.rstrip("\n").split("\t", 1)
            qids.append(qid)
            qs.append(q)
        groups: Dict[str, Dict[str, Set[str]]] = {q: {} for q in qids}
        with open(os.path.join(data_dir, "qrel_source_groups.tsv")) as h:
            next(h)
            for line in h:
                qid, gid, did, rel = line.rstrip("\n").split("\t")[:4]
                if int(rel) > 0 and qid in groups:
                    groups[qid].setdefault(gid, set()).add(did)
        return [Task("fanoutqa", doc_ids, [r["text"] for r in docs], qids, qs, [I.MULTIHOP_ROOT] * len(qids),
                     gold_groups={q: list(groups[q].values()) for q in qids})]
    if name == "musique":
        samples = json.load(open(os.path.join(data_dir, "musique.json")))
        corpus = json.load(open(os.path.join(data_dir, "musique_corpus.json")))
        texts = [f"{x['title']}\n{x.get('paragraph_text', x.get('text', ''))}" for x in corpus]
        doc_ids = ["chunk-" + hashlib.md5(x.encode()).hexdigest() for x in texts]
        by_text = {_norm(t): d for d, t in zip(doc_ids, texts)}
        qids, qs, gold = [], [], {}
        for i, s in enumerate(samples):
            q = str(s.get("id", i))
            ids = set()
            for p in s.get("paragraphs", []):
                if p.get("is_supporting", True) is not False:
                    t = f"{p['title']}\n{p.get('text', p.get('paragraph_text', ''))}"
                    if _norm(t) in by_text:
                        ids.add(by_text[_norm(t)])
            qids.append(q)
            qs.append(s["question"])
            gold[q] = [{d} for d in ids]
        return [Task("musique", doc_ids, texts, qids, qs, [I.MULTIHOP_ROOT] * len(qids), gold_groups=gold)]
    raise KeyError(f"unknown multi-hop suite {name!r}")


SINGLE_HOP = {"nanobeir": load_nanobeir, "bright": load_bright, "coir": load_coir, "toolret": load_toolret}
