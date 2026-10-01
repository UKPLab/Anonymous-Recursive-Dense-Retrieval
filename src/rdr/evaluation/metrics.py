"""Retrieval metrics used in the paper.

* Multi-hop suites report Recall@k over *gold groups*: a group counts as found when any of its
  passages is in the top-k (MuSiQue ``support`` and BrowseComp+ ``evidence`` are one passage per
  group; FanOutQA and FRAMES ``source_group`` group all chunks of a source page).
* Single-hop suites report nDCG@10 with binary relevance (qrels > 0), as in the evaluation code.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Union

Gold = Union[Sequence[Iterable], Mapping[str, Iterable]]


def gold_groups(gold: Gold) -> List[Set]:
    """Normalize gold labels to a list of non-empty sets (one per group)."""
    if isinstance(gold, Mapping):
        groups = [set(v) for v in gold.values()]
    else:
        groups = [set(g) if isinstance(g, (set, frozenset, list, tuple)) else {g} for g in gold]
    return [g for g in groups if g]


def recall_at_k(ranked: Sequence, gold: Gold, k: int) -> float:
    """Fraction of gold groups with at least one member in the top-``k``.

    Args:
        ranked: Ranked document ids (best first).
        gold: Gold documents: a flat collection (each document is its own group) or a list of groups,
            where any member of a group credits it (source-group metric of FanOutQA and FRAMES).
        k: Cutoff.
    """
    groups = gold_groups(gold)
    if not groups:
        return float("nan")
    top = set(ranked[:k])
    return sum(1 for g in groups if top & g) / len(groups)


def ndcg_at_k(ranked: Sequence, qrels: Mapping, k: int = 10, graded: bool = False) -> float:
    """nDCG@k. ``graded=False`` (default) uses binary gains for every ``rel > 0``, as the paper's
    evaluation does; ``graded=True`` uses the relevance value as a linear gain (pytrec_eval style).

    Args:
        ranked: Ranked document ids (best first).
        qrels: ``{doc_id: relevance}`` of one query; entries with ``relevance <= 0`` are ignored.
        k: Cutoff.
        graded: Use graded gains instead of binary gains.
    """
    rel = {d: float(r) for d, r in qrels.items() if float(r) > 0}
    if not rel:
        return float("nan")
    gain = (lambda d: rel.get(d, 0.0)) if graded else (lambda d: 1.0 if d in rel else 0.0)
    dcg = sum(gain(d) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted(rel.values(), reverse=True) if graded else [1.0] * len(rel)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def alpha_ndcg_at_k(ranked: Sequence, aspects: Mapping, k: int = 10, alpha: float = 0.5) -> float:
    """alpha-nDCG@k (novelty-aware; BRIGHT-Pro). ``aspects`` maps doc -> iterable of aspect ids.

    The ideal ranking is computed greedily, the standard approximation.

    Args:
        ranked: Ranked document ids (best first).
        aspects: ``{doc_id: aspect ids}`` of one query.
        k: Cutoff.
        alpha: Redundancy penalty per repeated aspect.
    """
    def gains(order):
        seen: Dict = {}
        out = []
        for d in order[:k]:
            g = 0.0
            for a in aspects.get(d, ()):
                g += (1 - alpha) ** seen.get(a, 0)
                seen[a] = seen.get(a, 0) + 1
            out.append(g)
        return out

    def dcg(gs):
        return sum(g / math.log2(i + 2) for i, g in enumerate(gs))

    pool = [d for d in aspects if aspects[d]]
    if not pool:
        return float("nan")
    ideal, seen = [], {}
    remaining = list(pool)
    for _ in range(min(k, len(remaining))):
        best, best_g = None, -1.0
        for d in remaining:
            g = sum((1 - alpha) ** seen.get(a, 0) for a in aspects[d])
            if g > best_g:
                best, best_g = d, g
        ideal.append(best)
        remaining.remove(best)
        for a in aspects[best]:
            seen[a] = seen.get(a, 0) + 1
    i = dcg(gains(ideal))
    return dcg(gains(list(ranked))) / i if i > 0 else 0.0


def macro(values: Mapping[str, Sequence[float]]) -> float:
    """Mean over groups of the per-group means (task/group macro average).

    Args:
        values: ``{group: per-query values}``.
    """
    means = [sum(v) / len(v) for v in values.values() if len(v)]
    return sum(means) / len(means) if means else float("nan")
