"""Search strategies: which states are formed.

After every level has been encoded and searched, the strategy *selects* documents from each new
state (the proposals). Selected documents become children one level deeper, until the maximum depth
``H``; the selections made by the deepest states are still recorded (realized readouts rank them)
but are not encoded. The engine (:mod:`rdr.search.engine`) runs one level at a time for a whole batch
of queries. Strategies never produce the final ranking; that is the readout's job.

Selection rules (``dedup``)
---------------------------
``"level"`` (multi-hop default)
    Every state of a level proposes its ``b`` best candidates among the documents not selected at an
    earlier level. Proposals of the same document are merged into one child, keeping the path with the
    highest cumulative score. This is why a multi-hop ``RDR(4,3)`` realizes about 23 of its 40 states.
``"none"`` (single-hop default)
    Every state selects its own ``b`` best candidates; nothing is excluded or merged, so a complete
    tree always has ``1 + b + b^2 + …`` states.
``"sequential"``
    States select in frontier order and each selection is excluded for all later states
    (the legacy GRITHopper/HippoRAG tree).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..types import CONTINUATION, ROOT, State, Traversal

DEDUP_MODES = ("level", "none", "sequential")


@dataclass
class StateSpec:
    """A state to create: its parent, observation path and instruction role."""

    parent: int
    path: Tuple[int, ...]
    role: str = CONTINUATION
    path_score: float = 1.0
    path_sum: float = 0.0
    path_logp: float = 0.0
    embedding_from_document: Optional[int] = None  # document-as-query: reuse this document's index vector
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Proposal:
    """One selection: ``doc`` selected by state ``parent`` at ``level`` (the parent's level)."""

    doc: int
    parent: int
    level: int
    path: Tuple[int, ...]
    score: float        # the selecting state's raw cosine for doc
    cum: float          # product of raw cosines along path
    path_sum: float     # sum of raw cosines along path (MDR)
    logp: float         # sum of log sparse-softmax probabilities along path (BeamDR)
    expanded: bool = False


def state_probabilities(state: State, temperature: float = 0.05, top_k: int = 200) -> Dict[int, float]:
    """Sparse softmax of a state's top-K candidates (used for BeamDR-style probability chains)."""
    ids, sc = state.ids[:top_k], state.scores[:top_k].astype(np.float64)
    fin = np.isfinite(sc)
    ids, sc = ids[fin], sc[fin]
    if not sc.size:
        return {}
    e = np.exp((sc - sc.max()) / temperature)
    p = e / e.sum()
    return {int(d): float(x) for d, x in zip(ids, p)}


def top_candidates(state: State, n: int, blocked: Set[int]) -> List[Tuple[int, float]]:
    """The ``n`` best candidates of a state (ids are ``(-score, index)`` sorted) not in ``blocked``."""
    out = []
    for d, v in zip(state.ids.tolist(), state.scores.tolist()):
        if d in blocked or not math.isfinite(v):
            continue
        out.append((int(d), float(v)))
        if len(out) == n:
            break
    return out


class SearchStrategy(ABC):
    """Base class of all search strategies. ``depth`` is ``H``: encoded levels including the root."""

    name: str = "search"
    depth: int = 1
    #: role (instruction) of context-bearing states; set by the retriever from its prompt config
    role: str = CONTINUATION

    def roots(self, traversal: Traversal) -> List[StateSpec]:
        return [StateSpec(parent=-1, path=(), role=ROOT)]

    @abstractmethod
    def select(self, traversal: Traversal, level: int, states: List[int]) -> List[StateSpec]:
        """Select from the ``states`` (indices into ``traversal.states``) searched at ``level``, record the
        proposals in ``traversal.proposals`` and return the children to encode at ``level + 1``."""

    def max_states(self) -> int:  # pragma: no cover - informational
        return 0

    def __repr__(self) -> str:
        fields = ", ".join(f"{k}={v!r}" for k, v in vars(self).items() if not k.startswith("_"))
        return f"{type(self).__name__}({fields})"
