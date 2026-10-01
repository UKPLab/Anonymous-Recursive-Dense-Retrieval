"""Readouts that need the encoder or the index again (they add encoder calls or lookups).

* :class:`RootInterpolation` — DC-PRF's readout: search with ``normalize(lam z_root + (1-lam) z_last)``.
* :class:`Assembly` — set-conditioned selection of ``n`` documents from the tree's top-``pool`` (Eq. 10):
  ``n`` pool-restricted encoder calls, no additional full-index searches.
* :class:`StateReadout` — the ranking of one specific state (e.g. the joint-feedback state).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from ..types import ASSEMBLY, Ranking, Traversal
from .base import Readout, backfill
from .global_readout import GlobalReadout


class ModelReadout(Readout):
    """A readout that receives a context with ``encoder``, ``index`` and ``renderer``.

    ``batch(travs, ctx)`` is called once per query batch so encoder calls stay batched.
    """

    needs_embeddings: bool = False

    def __call__(self, traversal: Traversal) -> Ranking:  # pragma: no cover - use batch()
        raise RuntimeError(f"{type(self).__name__} needs the retriever context; call it through RecursiveRetriever")

    def batch(self, travs: Sequence[Traversal], ctx) -> List[Ranking]:  # pragma: no cover - interface
        raise NotImplementedError


class StateReadout(Readout):
    """Ranking of the state at ``level`` (default: the deepest) — e.g. joint feedback, or the terminal
    state of a chain. With several states at that level the one with the best path score is used.

    Args:
        level: Level whose state is read (1 = root; ``None`` = the deepest encoded level).
    """

    name = "state"

    def __init__(self, level: Optional[int] = None) -> None:
        self.level = level

    def __call__(self, traversal: Traversal) -> Ranking:
        lvl = self.level or traversal.depth()
        cands = [s for s in traversal.states if s.level == lvl and s.ids is not None]
        s = max(cands, key=lambda st: st.path_score)
        return Ranking(s.ids, s.scores)


class RootInterpolation(ModelReadout):
    """DC-PRF's readout on a chain: one extra lookup with ``normalize(lam * z_root + (1 - lam) * z_last)``.

    Args:
        lam: Root weight (DC-PRF: 0.8).
        top_k: Candidates returned.
    """

    name = "root_interpolation"
    needs_embeddings = True

    def __init__(self, lam: float = 0.8, top_k: int = 200) -> None:
        self.lam = lam
        self.top_k = top_k

    def batch(self, travs, ctx):
        vecs = []
        for t in travs:
            z0 = t.root.embedding
            last = max((s for s in t.states if s.embedding is not None), key=lambda s: (s.level, s.path_score))
            v = self.lam * z0.float() + (1.0 - self.lam) * last.embedding.float()
            vecs.append(torch.nn.functional.normalize(v, p=2, dim=0))
        q = torch.stack(vecs)
        excl = [t.meta.get("exclude") for t in travs]
        out = ctx.index.search(q, self.top_k, exclude=excl if any(e is not None for e in excl) else None)
        return [Ranking(out.ids[i], out.scores[i]) for i in range(len(travs))]


class Assembly(ModelReadout):
    """Set-conditioned selection (Appendix I.1, Eq. 10).

    Starting from ``S_0 = {}``, ``n`` times: render ``[iota_set; q; ctx(S_j)]``, encode it and add the
    pool document with the highest cosine not yet in ``S``. The pool is the base readout's top-``pool``.

    Args:
        base: Readout that forms the pool (default: the global readout).
        pool: Pool size (paper: 50).
        n: Documents selected (paper: 10).
        role: Rendering role of the set-conditioned state (``"assembly"``).
    """

    name = "assembly"

    def __init__(self, base: Optional[Readout] = None, pool: int = 50, n: int = 10, role: str = ASSEMBLY) -> None:
        self.base = base or GlobalReadout()
        self.pool = pool
        self.n = n
        self.role = role

    def batch(self, travs, ctx):
        base = [self.base(t) for t in travs]
        pools = [r.ids[: self.pool].tolist() for r in base]
        chosen: List[List[int]] = [[] for _ in travs]
        for _ in range(self.n):
            rows = [i for i, p in enumerate(pools) if len(chosen[i]) < len(p)]
            if not rows:
                break
            texts = [ctx.renderer.render(travs[i].query, [ctx.index.text(d) for d in chosen[i]], self.role,
                                         instruction=travs[i].meta.get("root_instruction"),
                                         level=len(chosen[i]) + 1) for i in rows]
            q = ctx.encoder.encode_queries(texts)
            for r, i in enumerate(rows):
                left = [d for d in pools[i] if d not in chosen[i]]
                sc = ctx.index.score(q[r : r + 1], [left])[0]
                chosen[i].append(left[int(np.argmax(sc))])  # first maximum in pool order, as in the paper
        return [Ranking.from_order(backfill(chosen[i], base[i].ids.tolist())) for i in range(len(travs))]
