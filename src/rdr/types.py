"""Core data structures shared by search strategies and readouts.

A *state* is one call of the encoder: an instruction, the query and an ordered list of
observed documents (Equation 1 of the paper). Every state is searched against the index
and keeps its top-K candidates with their raw cosine scores. A *traversal* is the set of
states one search strategy produced for one query. Readouts turn a traversal into a
ranking; they never re-encode or re-search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = "root"
CONTINUATION = "continuation"
RECOVERY = "recovery"
DOC_QUERY = "doc_query"          # child state = the selected document's own index embedding (COR-style)
FEEDBACK = "feedback"            # jointly encoded pseudo-relevance feedback state (ANCE-PRF-style)
INTERPOLATED = "interpolated"    # root-interpolated state (DC-PRF-style)
ASSEMBLY = "assembly"            # set-conditioned selection state


@dataclass
class State:
    """One encoder call and its search result.

    Attributes:
        level: Tree level, counting the root as level 1 (a level-``h`` state has observed ``h-1`` documents).
        parent: Index of the parent state inside the traversal (``-1`` for the root).
        path: Observed documents as index positions, in the order they were appended.
        role: Which instruction rendered the state (``root``, ``continuation``, ``recovery`` …).
        ids: Top-K candidate index positions, sorted by ``(-score, index)``.
        scores: Raw cosine scores aligned with ``ids`` (float32).
        path_score: Product of the raw cosine scores along ``path`` (1.0 for the root). Used by realized
            (path) readouts only.
        text: The rendered state text (kept only when ``keep_text=True``).
        embedding: The state's query vector (kept only when a strategy or readout needs it).
        meta: Free-form strategy information (e.g. the beam slot).
    """

    level: int
    parent: int
    path: Tuple[int, ...]
    role: str = ROOT
    ids: Optional[np.ndarray] = None
    scores: Optional[np.ndarray] = None
    path_score: float = 1.0
    text: Optional[str] = None
    embedding: Any = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_root(self) -> bool:
        return self.parent < 0

    @property
    def last(self) -> Optional[int]:
        return self.path[-1] if self.path else None


@dataclass
class Traversal:
    """All states one search strategy produced for one query.

    ``states`` holds the primary view (the states that drive the search). Additional views of the
    same traversal, e.g. the recovery re-encoding of every non-root state, live in ``views`` under
    their own name and never change which states were expanded.
    """

    query: str
    query_index: int
    states: List[State] = field(default_factory=list)
    views: Dict[str, List[State]] = field(default_factory=dict)
    proposals: List[Any] = field(default_factory=list)
    """Every selection in order (:class:`rdr.search.base.Proposal`), including those of the deepest
    level and those a beam pruned: the realized set ``E`` of path readouts."""
    meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    @property
    def root(self) -> State:
        return self.states[0]

    def view(self, name: str = "primary") -> List[State]:
        if name in ("primary", None):
            return self.states
        return self.views[name]

    def levels(self, view: str = "primary") -> Dict[int, List[State]]:
        out: Dict[int, List[State]] = {}
        for s in self.view(view):
            if s.ids is None:
                continue
            out.setdefault(s.level, []).append(s)
        return dict(sorted(out.items()))

    def selected_documents(self) -> List[int]:
        """The realized set ``E``: every document some state selected, in first-selection order."""
        seen, out = set(), []
        for p in self.proposals:
            if p.doc not in seen:
                seen.add(p.doc)
                out.append(p.doc)
        return out

    def depth(self) -> int:
        return max((s.level for s in self.states), default=0)

    def num_states(self, view: str = "primary") -> int:
        return sum(1 for s in self.view(view) if s.ids is not None)


@dataclass
class Ranking:
    """A ranked list of index positions with scores (descending)."""

    ids: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        self.ids = np.asarray(self.ids, dtype=np.int64)
        self.scores = np.asarray(self.scores, dtype=np.float64)

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def top(self, k: int) -> "Ranking":
        return Ranking(self.ids[:k], self.scores[:k])

    def as_dict(self) -> Dict[int, float]:
        return {int(i): float(s) for i, s in zip(self.ids, self.scores)}

    @staticmethod
    def from_scores(scores: Dict[int, float], k: Optional[int] = None) -> "Ranking":
        """Sort a ``{doc: score}`` map by ``(-score, doc)`` (deterministic tie-break by index)."""
        if not scores:
            return Ranking(np.zeros(0, dtype=np.int64), np.zeros(0))
        items = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        if k is not None:
            items = items[:k]
        ids = np.fromiter((i for i, _ in items), dtype=np.int64, count=len(items))
        sc = np.fromiter((s for _, s in items), dtype=np.float64, count=len(items))
        return Ranking(ids, sc)

    @staticmethod
    def from_order(order: Sequence[int]) -> "Ranking":
        """A ranking given only an order (scores are ``n, n-1, …``)."""
        n = len(order)
        return Ranking(np.asarray(order, dtype=np.int64), np.arange(n, 0, -1, dtype=np.float64))


@dataclass
class SearchHit:
    """One returned result, sentence-transformers ``semantic_search`` style."""

    corpus_id: Any
    score: float
    index: int

    def as_dict(self) -> Dict[str, Any]:
        return {"corpus_id": self.corpus_id, "score": self.score, "index": self.index}


def unique_in_order(xs: Iterable[int]) -> List[int]:
    seen, out = set(), []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
