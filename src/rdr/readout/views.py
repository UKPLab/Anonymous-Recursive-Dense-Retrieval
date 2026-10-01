"""Fusing rankings of several views of one traversal (Appendix E.1.1).

The optional recovery view re-encodes the non-root states of a continuation tree with the recovery
instruction; it never changes which states were expanded. Its readout is fused with the primary
readout by z-agreement: each view is standardized on its own scored support and a document missing
from a view receives that view's minimum standardized score.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from ..types import Ranking, Traversal
from .base import Readout


def z_standardize(scores: Dict[int, float]) -> Dict[int, float]:
    """``z_v(d) = (r_v(d) - mu_v) / sigma_v`` on the view's own support (sample std; ``sigma = 0 -> 1``)."""
    vals = np.array([v for v in scores.values() if np.isfinite(v)], dtype=np.float64)
    if vals.size < 2:
        return {d: 0.0 for d in scores}
    mu = vals.mean()
    sd = vals.std(ddof=1)
    sd = 1.0 if sd == 0 else sd
    return {d: (v - mu) / sd for d, v in scores.items() if np.isfinite(v)}


def z_agreement(views: Sequence[Dict[int, float]], weights: Optional[Sequence[float]] = None) -> Dict[int, float]:
    """Equal-weight (or weighted) sum of per-view z-scores over the union of supports (Eq. 5–9)."""
    zs = [z_standardize(v) for v in views]
    weights = [1.0] * len(zs) if weights is None else list(weights)
    support = set().union(*[set(z) for z in zs]) if zs else set()
    mins = [min(z.values()) if z else 0.0 for z in zs]
    out = {}
    for d in support:
        out[d] = sum(w * z.get(d, m) for z, m, w in zip(zs, mins, weights))
    return out


class ZAgreement(Readout):
    """Fuse the readouts of two (or more) views of the same traversal by z-agreement (Eq. 5–9).

    Each view is cut to its top ``view_top_k`` entries and, on chunked corpora, pooled per document
    (``group_reduce`` per view: ``"max"`` or ``"last"``) before standardization, as in the paper.

    Args:
        *readouts: Two or more readouts, usually of different views (e.g. ``GlobalReadout()`` and
            ``recovery_view_readout()``).
        weights: Weight of each view's z-scores (default: equal weights, as in the paper).
        group_reduce: Chunk-to-document pooling per view on chunked corpora, ``"max"`` or ``"last"``
            (default: ``"max"`` for every view).
        view_top_k: Entries of each view that are standardized (paper: 200).
        output_top_k: Length of the fused ranking (``None`` keeps the union of supports).

    Example — Table 1, "+ recovery (z-agreement)"::

        ZAgreement(GlobalReadout(), recovery_view_readout())
    """

    name = "z_agreement"

    def __init__(self, *readouts: Readout, weights: Optional[Sequence[float]] = None,
                 group_reduce: Optional[Sequence[str]] = None, view_top_k: Optional[int] = 200,
                 output_top_k: Optional[int] = None) -> None:
        if len(readouts) < 2:
            raise ValueError("ZAgreement needs at least two readouts")
        self.readouts = list(readouts)
        self.weights = weights
        self.group_reduce = list(group_reduce) if group_reduce is not None else ["max"] * len(readouts)
        self.view_top_k = view_top_k
        self.output_top_k = output_top_k

    def __call__(self, traversal: Traversal) -> Ranking:
        from .base import group_max

        groups = traversal.meta.get("groups")
        views, rep = [], {}
        for r, mode in zip(self.readouts, self.group_reduce):
            rk = r(traversal)
            if groups is not None:
                rk = group_max(rk, groups, top_k=self.view_top_k, reduce=mode)
                d = {}
                for i, sc in zip(rk.ids.tolist(), rk.scores.tolist()):
                    g = int(groups[i])
                    d[g] = sc
                    rep.setdefault(g, i)
            else:
                rk = rk if self.view_top_k is None else rk.top(self.view_top_k)
                d = rk.as_dict()
                for i in d:
                    rep.setdefault(i, i)
            views.append(d)
        fused = z_agreement(views, self.weights)
        return Ranking.from_scores({rep[k]: v for k, v in fused.items()}, self.output_top_k)


def recovery_view_readout(view: str = "recovery") -> Readout:
    """The recovery view read out alone, as in the paper's evaluation code (Table 20, "recovery alone"): every
    recovery state contributes a sparse softmax of its full-column min–max scores, summed without level
    balancing and without the root.

    Args:
        view: Name of the recovery view (``SearchEngine`` views are named ``"recovery"`` by default).
    """
    from .global_readout import GlobalReadout

    return GlobalReadout(view=view, include_root=False, level_balanced=False, normalization="softmax_minmax_full")


def recovery_fusion(view: str = "recovery") -> "ZAgreement":
    """``+ recovery (z-agreement)`` of Table 1: the primary global readout fused with the recovery view.

    Args:
        view: Name of the recovery view.
    """
    from .global_readout import GlobalReadout

    return ZAgreement(GlobalReadout(), recovery_view_readout(view), group_reduce=("max", "last"))


class MinMaxCombMNZ(Readout):
    """Min–max CombMNZ of two readouts: ``(m(a) + lam * m_2(a)) * #views retaining a in their top-K``.

    Kept for comparison with earlier versions of the recovery fusion (``lam = 0.5``, ``K = 100``).

    Args:
        first: Readout of the first view.
        second: Readout of the second view.
        lam: Weight of the second view's min–max scores.
        k: Entries of each view that are normalized and counted.
        output_top_k: Length of the fused ranking.
    """

    name = "minmax_combmnz"

    def __init__(self, first: Readout, second: Readout, lam: float = 0.5, k: int = 100,
                 output_top_k: Optional[int] = None) -> None:
        self.first, self.second, self.lam, self.k, self.output_top_k = first, second, lam, k, output_top_k

    @staticmethod
    def _mm(r: Ranking, k: int) -> Dict[int, float]:
        r = r.top(k)
        if not len(r):
            return {}
        lo, hi = float(r.scores.min()), float(r.scores.max())
        return {int(d): (float(s) - lo) / (hi - lo + 1e-12) for d, s in zip(r.ids, r.scores)}

    def __call__(self, traversal: Traversal) -> Ranking:
        a = self._mm(self.first(traversal), self.k)
        b = self._mm(self.second(traversal), self.k)
        out = {}
        for d in set(a) | set(b):
            out[d] = (a.get(d, 0.0) + self.lam * b.get(d, 0.0)) * ((d in a) + (d in b))
        return Ranking.from_scores(out, self.output_top_k)
