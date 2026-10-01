"""Readout interface and the per-state score normalizations.

A readout maps one :class:`~rdr.types.Traversal` to a :class:`~rdr.types.Ranking`. It only reads the
saved states (their top-K candidates and raw cosine scores); it never encodes or searches.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..types import Ranking, State, Traversal


# --------------------------------------------------------------------- normalizations
def sparse_softmax(scores: np.ndarray, temperature: float = 0.05) -> np.ndarray:
    """``p_s(d) = exp(l_s(d)/tau) / sum_{u in C_s} exp(l_s(u)/tau)`` over the retained candidates."""
    x = np.asarray(scores, dtype=np.float64) / float(temperature)
    x = x - x.max() if x.size else x
    e = np.exp(x)
    return e / e.sum() if e.size else e


def minmax(scores: np.ndarray, lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    """Min–max normalization. ``lo``/``hi`` default to the candidates' own range; pass the full-column
    range (``state.meta['min'/'max']``) to reproduce the legacy GRITHopper/HippoRAG trees."""
    s = np.asarray(scores, dtype=np.float64)
    if not s.size:
        return s
    lo = float(s.min()) if lo is None else float(lo)
    hi = float(s.max()) if hi is None else float(hi)
    return (s - lo) / (hi - lo + 1e-12)


def state_weights(state: State, normalization: str = "softmax", temperature: float = 0.05,
                  top_k: Optional[int] = 200) -> tuple:
    """Candidate ids and their normalized weights for one state.

    ``normalization``: ``"softmax"`` (the paper's sparse softmax, default), ``"raw"`` (raw cosine),
    ``"minmax"`` (over the retained candidates), ``"minmax_full"`` (over the whole index column) or
    ``"softmax_minmax_full"`` (sparse softmax of full-column min–max scores; the recovery view).
    """
    ids, sc = state.ids, state.scores
    if top_k is not None:
        ids, sc = ids[:top_k], sc[:top_k]
    finite = np.isfinite(sc)
    ids, sc = ids[finite], sc[finite]
    if normalization == "softmax":
        w = sparse_softmax(sc, temperature)
    elif normalization == "raw":
        w = np.asarray(sc, dtype=np.float64)
    elif normalization == "minmax":
        w = minmax(sc)
    elif normalization == "minmax_full":
        w = minmax(sc, state.meta.get("min"), state.meta.get("max"))
    elif normalization == "softmax_minmax_full":
        w = sparse_softmax(minmax(sc, state.meta.get("min"), state.meta.get("max")), temperature)
    else:
        raise ValueError(f"unknown normalization {normalization!r}")
    return ids, w


def accumulate(pairs: Sequence[tuple], weights: Optional[Sequence[float]] = None) -> Dict[int, float]:
    """Sum ``(ids, values)`` pairs into one ``{doc: score}`` map (optionally weighting each pair)."""
    if not pairs:
        return {}
    ids = np.concatenate([p[0] for p in pairs])
    if weights is None:
        vals = np.concatenate([p[1] for p in pairs])
    else:
        vals = np.concatenate([np.asarray(p[1], dtype=np.float64) * w for p, w in zip(pairs, weights)])
    if not ids.size:
        return {}
    uniq, inv = np.unique(ids, return_inverse=True)
    tot = np.bincount(inv, weights=vals, minlength=uniq.shape[0])
    return {int(u): float(t) for u, t in zip(uniq, tot)}


def backfill(primary: Sequence[int], filler: Sequence[int]) -> List[int]:
    """``primary`` in order, then ``filler`` in its order, duplicates skipped (root backfill)."""
    seen, out = set(), []
    for d in list(primary) + list(filler):
        if d not in seen:
            seen.add(d)
            out.append(int(d))
    return out


def group_max(ranking: Ranking, groups: Optional[np.ndarray], top_k: Optional[int] = 200,
              reduce: str = "max") -> Ranking:
    """Chunk → document output processing: keep the top ``top_k`` chunks, then pool per group.

    ``reduce="max"`` (paper) keeps each group's best chunk; ``"last"`` keeps the lowest-ranked chunk
    within the cut, which follows the paper's recovery-view bookkeeping. Returned ids are
    the kept chunk of each group, so they still index the corpus. Without ``groups`` the ranking is only
    cut to ``top_k``.
    """
    r = ranking if top_k is None else ranking.top(top_k)
    if groups is None:
        return r
    best: Dict[int, tuple] = {}
    for i, s in zip(r.ids.tolist(), r.scores.tolist()):
        g = int(groups[i])
        if reduce == "last" or g not in best or s > best[g][1]:
            best[g] = (i, s)
    items = sorted(best.values(), key=lambda t: (-t[1], t[0]))
    return Ranking([i for i, _ in items], [s for _, s in items])


class Readout(ABC):
    """Turns the states of one traversal into a ranking."""

    #: short name used by presets and the docs
    name: str = "readout"
    #: realized readouts rank only the documents a search selected; they must be requested explicitly
    realized: bool = False

    @abstractmethod
    def __call__(self, traversal: Traversal) -> Ranking:  # pragma: no cover - interface
        ...

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({fields})"
