"""The search families of the paper under one interface (Figure 2, Tables 3, 30, 33).

========================  ================================================================  =====================
Strategy                  States                                                            As in
========================  ================================================================  =====================
``DenseSearch()``         the root only, ``RDR(1,0)``                                       single-vector search
``TreeSearch(H, b)``      every state selects its ``b`` best candidates                     RDR (primary)
``ChainSearch(H)``        one successor per state, ``TreeSearch(H, 1)``                     DC-PRF chain
``BreadthSearch(m)``      root + ``m`` independently observed candidates, ``RDR(2, m)``       shallow breadth
``BeamSearch(W, H)``      the ``W`` best paths by path product survive each level           MDR / BeamDR
``DocAsQuerySearch(H,b)`` tree whose children search with the selected document's vector    COR (no LLM aspects)
``JointFeedback(m)``      root + one state observing the root's top-``m`` jointly             ANCE-PRF
========================  ================================================================  =====================
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from ..types import DOC_QUERY, FEEDBACK, Traversal
from .base import DEDUP_MODES, Proposal, SearchStrategy, StateSpec, state_probabilities, top_candidates


class TreeSearch(SearchStrategy):
    """``RDR(H, b)``: from the root, every state selects up to ``b`` candidates; each selection becomes a
    child that appends the document to its parent's observations.

    Args:
        depth: ``H``, encoded levels including the root (paper: 4).
        branching: ``b``, selections per state (paper: 3).
        root_branching: Selections of the root (defaults to ``branching``).
        dedup: ``"level"`` (merge per level, exclude earlier levels; multi-hop), ``"none"`` (single-hop)
            or ``"sequential"`` (legacy). See :mod:`rdr.search.base`.
        beam: Keep only the ``beam`` best children per level by cumulative path score (beam search).
        path_score: ``"raw"`` (product of raw cosines; paper) or ``"minmax"`` (product of full-column
            min–max normalized scores, the legacy GRITHopper/HippoRAG tree). Only affects path scores.
    """

    name = "tree"

    def __init__(self, depth: int = 4, branching: int = 3, root_branching: Optional[int] = None,
                 dedup: str = "level", beam: Optional[int] = None, path_score: str = "raw") -> None:
        if dedup not in DEDUP_MODES:
            raise ValueError(f"dedup must be one of {DEDUP_MODES}")
        if path_score not in ("raw", "minmax"):
            raise ValueError("path_score must be 'raw' or 'minmax'")
        self.depth = depth
        self.branching = branching
        self.root_branching = root_branching
        self.dedup = dedup
        self.beam = beam
        self.path_score = path_score

    @property
    def needs_range(self) -> bool:
        return self.path_score == "minmax"

    def _b(self, level: int) -> int:
        return self.root_branching if (level == 1 and self.root_branching is not None) else self.branching

    def _proposal(self, traversal: Traversal, si: int, d: int, v: float, probs) -> Proposal:
        s = traversal.states[si]
        p = probs.get(d, 0.0) if probs is not None else 0.0
        if self.path_score == "minmax":
            lo, hi = s.meta["min"], s.meta["max"]
            v = (v - lo) / (hi - lo + 1e-12)
        return Proposal(doc=d, parent=si, level=s.level, path=s.path + (d,), score=v, cum=s.path_score * v,
                        path_sum=s.meta.get("path_sum", 0.0) + v,
                        logp=s.meta.get("path_logp", 0.0) + math.log(max(p, 1e-300)))

    def select(self, traversal: Traversal, level: int, states: List[int]) -> List[StateSpec]:
        b = self._b(level)
        used = traversal.meta.setdefault("used", set())
        props: List[Proposal] = []
        if self.dedup == "level":
            best: Dict[int, Proposal] = {}
            for si in states:
                s = traversal.states[si]
                probs = state_probabilities(s)
                for d, v in top_candidates(s, b, used):
                    p = self._proposal(traversal, si, d, v, probs)
                    if d not in best or p.cum > best[d].cum:
                        best[d] = p
            used.update(best)
            props = list(best.values())
        else:
            for si in states:
                s = traversal.states[si]
                probs = state_probabilities(s)
                picks = top_candidates(s, b, used if self.dedup == "sequential" else set())
                if self.dedup == "sequential":
                    used.update(d for d, _ in picks)
                props.extend(self._proposal(traversal, si, d, v, probs) for d, v in picks)
        base = len(traversal.proposals)
        traversal.proposals.extend(props)
        kept = list(range(len(props)))
        if self.beam is not None and len(kept) > self.beam:
            kept = sorted(kept, key=lambda k: -props[k].cum)[: self.beam]
        return [self._spec(props[k], base + k) for k in kept]

    def _spec(self, p: Proposal, pi: int) -> StateSpec:
        return StateSpec(parent=p.parent, path=p.path, role=self.role, path_score=p.cum, path_sum=p.path_sum,
                         path_logp=p.logp, meta={"proposal": pi})

    def max_states(self) -> int:
        total, frontier = 1, 1
        for h in range(1, self.depth):
            frontier = frontier * self._b(h)
            if self.beam is not None:
                frontier = min(frontier, self.beam)
            total += frontier
        return total


class DenseSearch(TreeSearch):
    """Plain dense retrieval: one encoder call, one lookup (``RDR(1,0)``, ``q0``)."""

    name = "dense"

    def __init__(self) -> None:
        super().__init__(depth=1, branching=0)


class ChainSearch(TreeSearch):
    """A single chain of ``H`` states: ``TreeSearch(H, branching=1)``.

    Args:
        depth: ``H``, encoded states of the chain including the root (paper: 4).
        dedup: Selection rule, as in :class:`TreeSearch`; with one state per level ``"level"`` and
            ``"sequential"`` coincide (the chain never revisits a selected document).
    """

    name = "chain"

    def __init__(self, depth: int = 4, dedup: str = "level") -> None:
        super().__init__(depth=depth, branching=1, dedup=dedup)


class BreadthSearch(TreeSearch):
    """Shallow breadth ``RDR(2, m)``: the root and ``m`` independently observed candidates (Eq. 4),
    ``1 + m`` encodings in two dependent rounds.

    Args:
        width: ``m``, the root's top-``m`` candidates; each is observed by one level-2 state.
        dedup: Selection rule, as in :class:`TreeSearch` (only the root selects, so the modes coincide).
    """

    name = "breadth"

    def __init__(self, width: int = 8, dedup: str = "level") -> None:
        super().__init__(depth=2, branching=width, dedup=dedup)
        self.width = width


class BeamSearch(TreeSearch):
    """Beam search over observation paths, as in MDR / BeamDR (Appendix H.1).

    The root selects ``W`` candidates. At every level each path proposes its ``W`` best candidates not
    selected before; proposals of one document keep the path with the highest product of raw cosines,
    and the ``W`` best paths survive. Every proposed document counts as selected. A width-``W`` beam has
    at most ``1 + (H-1) W`` states (``W = 13`` matches the 40-state ceiling of ``RDR(4, 3)``).

    The default readout stays :class:`~rdr.readout.GlobalReadout`; the beam retrievers' own outputs are
    :class:`~rdr.readout.ProbabilityChain` (BeamDR) and :class:`~rdr.readout.SequenceSum` (MDR).

    Args:
        width: ``W``, surviving paths per level; also the root's selections.
        depth: ``H``, encoded levels including the root.
        proposals: Candidates each path proposes per level (defaults to ``width``).
    """

    name = "beam"

    def __init__(self, width: int = 5, depth: int = 4, proposals: Optional[int] = None) -> None:
        super().__init__(depth=depth, branching=proposals or width, root_branching=width, dedup="level", beam=width)
        self.width = width


class DocAsQuerySearch(TreeSearch):
    """Document-as-query expansion (COR-style, without COR's LLM aspect queries): same geometry,
    eligibility and deduplication as :class:`TreeSearch`, but each child searches with the selected
    document's own index vector instead of an encoded state (Table 32).

    Read it out with :class:`~rdr.readout.PostOrderRRF` for COR's own aggregation, or with the default
    global readout.

    Args:
        depth: ``H``, levels including the root (the root is still an encoded query).
        branching: ``b``, documents each state promotes to queries.
        root_branching: Selections of the root (defaults to ``branching``).
        dedup: Selection rule, as in :class:`TreeSearch`.
        beam: Optional beam width, as in :class:`TreeSearch`.
        path_score: ``"raw"`` or ``"minmax"`` path scores, as in :class:`TreeSearch`.
    """

    name = "doc_as_query"

    def _spec(self, p, pi):
        spec = super()._spec(p, pi)
        spec.role = DOC_QUERY
        spec.embedding_from_document = p.doc
        return spec


class JointFeedback(TreeSearch):
    """Pseudo-relevance feedback as in ANCE-PRF: the query and the root's top-``m`` documents are encoded
    jointly into one second query vector, read out alone (``StateReadout(level=2)``).

    Args:
        m: Feedback documents, the root's top-``m``, observed together by the second state (paper: 3).
    """

    name = "joint_feedback"

    def __init__(self, m: int = 3) -> None:
        super().__init__(depth=2, branching=0)
        self.m = m

    def select(self, traversal, level, states):
        if level != 1:
            return []
        root = traversal.states[states[0]]
        docs = tuple(int(d) for d in root.ids[: self.m].tolist())
        return [StateSpec(parent=states[0], path=docs, role=FEEDBACK)]

    def max_states(self) -> int:
        return 2
