"""Classical fusion rules applied to the states of one traversal (ablations of Appendix F / Table 21).

All rules read the same per-state candidate lists (top-K, raw cosine) as the global readout; only the
aggregation differs:

* ``RRF``: reciprocal rank fusion ``sum_s 1/(k + rank_s(d))`` (flat or level-balanced).
* ``CombMNZ``: sparse-softmax sum times the number of states that retain the document.
* ``MaxOverStates``: ``max_s p_s(d)``.
* ``Borda``: ``sum_s (K - rank_s(d) + 1)``.
* ``PostOrderRRF``: COR-style — every node's list is fused with its children's fused lists by RRF,
  recursively from the leaves up to the root.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..types import Ranking, State, Traversal
from .base import Readout, accumulate, state_weights


def _level_groups(traversal: Traversal, view: str, include_root: bool = True) -> Dict[int, List[State]]:
    by_level = traversal.levels(view)
    if not include_root:
        by_level = {h: s for h, s in by_level.items() if h != 1}
    return by_level


class RRF(Readout):
    """Reciprocal rank fusion of the states' ranked lists, ``1/(k + rank)`` with ranks from 1.

    Args:
        k: RRF constant (60, as in Cormack et al.).
        level_balanced: Weight each state by ``1 / (L |F_h|)`` as the global readout does; ``False`` sums
            all states with weight one.
        top_k: Ranks of each state that are fused.
        output_top_k: Length of the fused ranking (``None`` keeps every candidate).
        view: Which view of the traversal to read (``"primary"`` or e.g. ``"recovery"``).
    """

    name = "rrf"

    def __init__(self, k: int = 60, level_balanced: bool = False, top_k: int = 200,
                 output_top_k: Optional[int] = 200, view: str = "primary") -> None:
        self.k = k
        self.level_balanced = level_balanced
        self.top_k = top_k
        self.output_top_k = output_top_k
        self.view = view

    def _rr(self, s: State):
        ids = s.ids[: self.top_k]
        return ids, 1.0 / (self.k + np.arange(1, ids.shape[0] + 1, dtype=np.float64))

    def __call__(self, traversal: Traversal) -> Ranking:
        by_level = _level_groups(traversal, self.view)
        pairs, weights = [], []
        if self.level_balanced:
            L = len(by_level)
            for ss in by_level.values():
                for s in ss:
                    pairs.append(self._rr(s))
                    weights.append(1.0 / (L * len(ss)))
        else:
            for ss in by_level.values():
                for s in ss:
                    pairs.append(self._rr(s))
                    weights.append(1.0)
        return Ranking.from_scores(accumulate(pairs, weights), self.output_top_k)


class Borda(Readout):
    """Borda count over the states' top-K lists: ``K - rank + 1`` points per state.

    Args:
        top_k: ``K``, ranks of each state that receive points.
        output_top_k: Length of the fused ranking (``None`` keeps every candidate).
        view: Which view of the traversal to read (``"primary"`` or e.g. ``"recovery"``).
    """

    name = "borda"

    def __init__(self, top_k: int = 200, output_top_k: Optional[int] = 200, view: str = "primary") -> None:
        self.top_k = top_k
        self.output_top_k = output_top_k
        self.view = view

    def __call__(self, traversal: Traversal) -> Ranking:
        pairs = []
        for ss in _level_groups(traversal, self.view).values():
            for s in ss:
                ids = s.ids[: self.top_k]
                K = ids.shape[0]
                pairs.append((ids, (K - np.arange(K)).astype(np.float64)))
        return Ranking.from_scores(accumulate(pairs), self.output_top_k)


class CombMNZ(Readout):
    """``(sum of normalized scores) x (number of states whose top-K contains the document)``.

    ``level_balanced=True`` multiplies the global readout by the count (the paper's overlap-sensitive
    ablation); ``False`` multiplies the flat sum.

    Args:
        level_balanced: Base score is the level-balanced global readout (``True``) or the flat mean.
        temperature: Softmax temperature of the per-state distributions.
        top_k: Candidates retained per state; also defines the count.
        normalization: Per-state score map, as in :class:`GlobalReadout`.
        output_top_k: Length of the fused ranking (``None`` keeps every candidate).
        view: Which view of the traversal to read (``"primary"`` or e.g. ``"recovery"``).
    """

    name = "combmnz"

    def __init__(self, level_balanced: bool = True, temperature: float = 0.05, top_k: int = 200,
                 normalization: str = "softmax", output_top_k: Optional[int] = 200, view: str = "primary") -> None:
        self.level_balanced = level_balanced
        self.temperature = temperature
        self.top_k = top_k
        self.normalization = normalization
        self.output_top_k = output_top_k
        self.view = view

    def __call__(self, traversal: Traversal) -> Ranking:
        by_level = _level_groups(traversal, self.view)
        pairs, weights, counts = [], [], []
        L = len(by_level)
        n_all = sum(len(ss) for ss in by_level.values())
        for ss in by_level.values():
            for s in ss:
                ids, w = state_weights(s, self.normalization, self.temperature, self.top_k)
                pairs.append((ids, w))
                counts.append((ids, np.ones_like(w)))
                weights.append(1.0 / (L * len(ss)) if self.level_balanced else 1.0 / max(1, n_all))
        total = accumulate(pairs, weights)
        cnt = accumulate(counts)
        return Ranking.from_scores({d: v * cnt[d] for d, v in total.items()}, self.output_top_k)


class MaxOverStates(Readout):
    """``max_s p_s(d)``: the best normalized score any state gives the document.

    Args:
        temperature: Softmax temperature of the per-state distributions.
        top_k: Candidates retained per state.
        normalization: Per-state score map, as in :class:`GlobalReadout`.
        output_top_k: Length of the fused ranking (``None`` keeps every candidate).
        view: Which view of the traversal to read (``"primary"`` or e.g. ``"recovery"``).
    """

    name = "max"

    def __init__(self, temperature: float = 0.05, top_k: int = 200, normalization: str = "softmax",
                 output_top_k: Optional[int] = 200, view: str = "primary") -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.normalization = normalization
        self.output_top_k = output_top_k
        self.view = view

    def __call__(self, traversal: Traversal) -> Ranking:
        best: Dict[int, float] = {}
        for ss in _level_groups(traversal, self.view).values():
            for s in ss:
                ids, w = state_weights(s, self.normalization, self.temperature, self.top_k)
                for d, v in zip(ids.tolist(), w.tolist()):
                    if v > best.get(d, -np.inf):
                        best[d] = v
        return Ranking.from_scores(best, self.output_top_k)


class PostOrderRRF(Readout):
    """COR-style post-order aggregation: each node's list is RRF-fused with its children's fused
    lists (recursively, leaves first) and the root's fused list is the output.

    Args:
        k: RRF constant; COR's ``1/(60 + r + 1)`` with ranks ``r`` from 0 equals ``k=60`` with ranks from 1.
        top_k: Ranks of each node's own list that enter the fusion.
        output_top_k: Length of the fused ranking (``None`` keeps every candidate).
    """

    name = "rrf_postorder"

    def __init__(self, k: int = 60, top_k: int = 200, output_top_k: Optional[int] = 200) -> None:
        self.k = k
        self.top_k = top_k
        self.output_top_k = output_top_k

    def __call__(self, traversal: Traversal) -> Ranking:
        states = traversal.states
        children: Dict[int, List[int]] = {}
        for i, s in enumerate(states):
            if s.parent >= 0 and s.ids is not None:
                children.setdefault(s.parent, []).append(i)

        def fused(i: int) -> List[int]:
            own = states[i].ids[: self.top_k].tolist()
            lists = [own] + [fused(c) for c in children.get(i, [])]
            if len(lists) == 1:
                return own
            sc: Dict[int, float] = {}
            for lst in lists:
                for r, d in enumerate(lst, start=1):
                    sc[d] = sc.get(d, 0.0) + 1.0 / (self.k + r)
            return Ranking.from_scores(sc, self.top_k).ids.tolist()

        order = fused(0)
        return Ranking.from_order(order[: self.output_top_k] if self.output_top_k else order)
