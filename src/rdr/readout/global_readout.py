"""The global (complete-child) readout — Equation 2 of the paper.

Every state contributes a sparse softmax over its top-K candidates; states are averaged within a
level and the non-empty levels are averaged with equal weight::

    p_s(d) = 1[d in C_s] exp(l_s(d)/tau) / sum_{u in C_s} exp(l_s(u)/tau)
    r(d)   = 1/L * sum_h 1/|F_h| * sum_{s in F_h} p_s(d)

A candidate ranks high when many states across levels retain and score it highly, including
candidates that no state selected to extend the search.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

from ..types import Ranking, Traversal
from .base import Readout, accumulate, backfill, state_weights


class GlobalReadout(Readout):
    """Level-balanced aggregation of sparse per-state score distributions (RDR's default readout).

    Args:
        temperature: Softmax temperature ``tau`` over raw cosine scores (paper: 0.05).
        top_k: Candidates retained per state, ``|C_s|`` (paper: 200).
        level_balanced: Average states within each level, then average non-empty levels (paper). If
            ``False`` every state gets equal weight (flat CombSUM).
        include_root: Set ``False`` for the "without root" ablation (levels >= 2 only).
        levels: Only aggregate these levels (prefix readouts such as ``RDR(3,3)`` of an ``RDR(4,3)`` run).
        normalization: Per-state score map: ``"softmax"`` (paper), ``"raw"``, ``"minmax"``, ``"minmax_full"``.
        restrict_to_selected: Readout ``B``: keep only documents the search selected (``E``), same scores.
        backfill_root: After ``restrict_to_selected``, append the root ranking (single-hop protocol).
        output_top_k: Final sparse cut of the aggregated scores (paper: 200; ``None`` keeps everything).
        view: Which view of the traversal to read (``"primary"`` or e.g. ``"recovery"``).
    """

    name = "global"

    def __init__(
        self,
        temperature: float = 0.05,
        top_k: int = 200,
        level_balanced: bool = True,
        include_root: bool = True,
        levels: Optional[Iterable[int]] = None,
        normalization: str = "softmax",
        restrict_to_selected: bool = False,
        backfill_root: bool = False,
        output_top_k: Optional[int] = 200,
        view: str = "primary",
    ) -> None:
        self.temperature = temperature
        self.top_k = top_k
        self.level_balanced = level_balanced
        self.include_root = include_root
        self.levels = None if levels is None else sorted(set(levels))
        self.normalization = normalization
        self.restrict_to_selected = restrict_to_selected
        self.backfill_root = backfill_root
        self.output_top_k = output_top_k
        self.view = view

    def scores(self, traversal: Traversal) -> dict:
        by_level = traversal.levels(self.view)
        if self.view != "primary" and self.include_root and 1 not in by_level:
            # auxiliary views re-encode non-root states only; they are read together with the root
            by_level = {1: [traversal.root], **by_level}
        if not self.include_root:
            by_level = {h: s for h, s in by_level.items() if h != 1}
        if self.levels is not None:
            by_level = {h: s for h, s in by_level.items() if h in self.levels}
        pairs, weights = [], []
        if self.level_balanced:
            nonempty = [h for h, ss in by_level.items() if ss]
            L = len(nonempty)
            for h in nonempty:
                ss = by_level[h]
                for s in ss:
                    pairs.append(state_weights(s, self.normalization, self.temperature, self.top_k))
                    weights.append(1.0 / (L * len(ss)))
        else:
            allstates = [s for ss in by_level.values() for s in ss]
            for s in allstates:
                pairs.append(state_weights(s, self.normalization, self.temperature, self.top_k))
                weights.append(1.0 / max(1, len(allstates)))
        return accumulate(pairs, weights)

    def __call__(self, traversal: Traversal) -> Ranking:
        sc = self.scores(traversal)
        ranking = Ranking.from_scores(sc, self.output_top_k)
        if self.restrict_to_selected:
            E = set(traversal.selected_documents())
            keep = np.array([i in E for i in ranking.ids.tolist()], dtype=bool)
            ranking = Ranking(ranking.ids[keep], ranking.scores[keep])
            if self.backfill_root:
                order = backfill(ranking.ids.tolist(), traversal.root.ids.tolist())
                ranking = Ranking.from_order(order)
        return ranking


class RootReadout(Readout):
    """The root query alone, ``RDR(1,0)`` / ``q0``: its raw cosine ranking.

    Args:
        top_k: Cut the root's candidate list (``None`` keeps the search's ``candidate_k``).
    """

    name = "root"

    def __init__(self, top_k: Optional[int] = None) -> None:
        self.top_k = top_k

    def __call__(self, traversal: Traversal) -> Ranking:
        r = traversal.root
        ids, sc = (r.ids, r.scores) if self.top_k is None else (r.ids[: self.top_k], r.scores[: self.top_k])
        return Ranking(ids, sc)
