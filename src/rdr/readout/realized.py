"""Realized (path) readouts: rank only the documents a search selected, each by its own path.

These are the outputs of prior recursive retrievers (MDR, BeamDR, beam retrieval, GRITHopper-style
trees). RDR keeps them for comparison; the default readout of every search strategy is
:class:`~rdr.readout.GlobalReadout`, so a realized readout has to be requested explicitly.

``E`` is the realized set: every document some state selected (``traversal.proposals``), including the
selections of the deepest states and the paths a beam pruned.
"""

from __future__ import annotations

from typing import Dict, List

from ..types import Ranking, Traversal
from .base import Readout, backfill


class PathProduct(Readout):
    """``score(d) = max over selections of d of prod_i l_i`` — the product of the selecting scores along
    the path (raw cosines, or min–max scores for ``TreeSearch(path_score="minmax")``).

    Raw cosines are below one, so a path product can only shrink along a path: the root's picks lead the
    list.

    Args:
        backfill_root: Complete the list with the root ranking (documents not yet listed, in root order),
            as the single-hop protocol does before nDCG@10.
    """

    name = "path_product"
    realized = True

    def __init__(self, backfill_root: bool = False) -> None:
        self.backfill_root = backfill_root

    def __call__(self, traversal: Traversal) -> Ranking:
        best: Dict[int, float] = {}
        for p in traversal.proposals:
            if p.doc not in best or p.cum > best[p.doc]:
                best[p.doc] = p.cum
        r = Ranking.from_scores(best)
        if self.backfill_root:
            r = Ranking.from_order(backfill(r.ids.tolist(), traversal.root.ids.tolist()))
        return r


class ChainOrder(Readout):
    """Complete chains in hop order, as returned by beam retrievers.

    The complete paths (the selections of the deepest states) are sorted by ``score`` and emitted one
    after another, each in hop order, duplicates skipped:

    * ``"logprob"`` — sum of the states' log sparse-softmax probabilities (BeamDR's probability chain),
    * ``"sum"`` — sum of raw cosines (MDR's sequence score),
    * ``"product"`` — product of raw cosines.

    Ties break by the path.

    Args:
        score: Path score that orders the complete chains: ``"logprob"``, ``"sum"`` or ``"product"``.
        backfill_root: Complete the list with the root ranking (documents not yet listed, in root order),
            as the single-hop protocol does before nDCG@10.
    """

    name = "chain_order"
    realized = True

    def __init__(self, score: str = "logprob", backfill_root: bool = False) -> None:
        if score not in ("logprob", "sum", "product"):
            raise ValueError("score must be 'logprob', 'sum' or 'product'")
        self.score = score
        self.backfill_root = backfill_root

    def _key(self, p) -> float:
        return p.logp if self.score == "logprob" else (p.path_sum if self.score == "sum" else p.cum)

    def __call__(self, traversal: Traversal) -> Ranking:
        if not traversal.proposals:
            return Ranking.from_order(traversal.root.ids.tolist() if self.backfill_root else [])
        deepest = max(p.level for p in traversal.proposals)
        final = [p for p in traversal.proposals if p.level == deepest]
        final.sort(key=lambda p: (-self._key(p), p.path))
        order = backfill([d for p in final for d in p.path], [])
        if self.backfill_root:
            order = backfill(order, traversal.root.ids.tolist())
        return Ranking.from_order(order)


class SequenceSum(ChainOrder):
    """MDR's sequence score: complete chains ordered by the sum of raw cosines (``ChainOrder("sum")``).

    Args:
        backfill_root: Complete the list with the root ranking (documents not yet listed, in root order),
            as the single-hop protocol does before nDCG@10.
    """

    name = "sequence_sum"

    def __init__(self, backfill_root: bool = False) -> None:
        super().__init__(score="sum", backfill_root=backfill_root)


class ProbabilityChain(ChainOrder):
    """BeamDR's probability chain: complete chains ordered by the product of state softmax probabilities
    (``ChainOrder("logprob")``).

    Args:
        backfill_root: Complete the list with the root ranking (documents not yet listed, in root order),
            as the single-hop protocol does before nDCG@10.
    """

    name = "probability_chain"

    def __init__(self, backfill_root: bool = False) -> None:
        super().__init__(score="logprob", backfill_root=backfill_root)


class TerminalState(Readout):
    """The last state of a chain alone (its raw cosine ranking). With several deepest states the one with
    the best path score is used."""

    name = "terminal"

    def __init__(self) -> None:
        pass

    def __call__(self, traversal: Traversal) -> Ranking:
        depth = traversal.depth()
        cands = [s for s in traversal.states if s.level == depth and s.ids is not None]
        s = max(cands, key=lambda st: st.path_score)
        return Ranking(s.ids, s.scores)


class SelectedSet(Readout):
    """The realized set ``E`` in selection order (TreeHop-style union output).

    Args:
        backfill_root: Complete the list with the root ranking (documents not yet listed, in root order),
            as the single-hop protocol does before nDCG@10.
    """

    name = "selected_set"
    realized = True

    def __init__(self, backfill_root: bool = False) -> None:
        self.backfill_root = backfill_root

    def __call__(self, traversal: Traversal) -> Ranking:
        order: List[int] = traversal.selected_documents()
        if self.backfill_root:
            order = backfill(order, traversal.root.ids.tolist())
        return Ranking.from_order(order)
