"""Level-synchronous execution of a search strategy for a batch of queries.

At each level the engine renders every new state of every query in the batch, encodes them in one
call, searches them in one batched lookup and lets the strategy select the next level. With separate
encode and index devices, several query batches run concurrently: one batch encodes on the encoder
device while another searches on the index device (``RecursiveRetriever(..., inflight=2)``).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..prompts import RECOVERY_VIEW
from ..types import State, Traversal
from .base import SearchStrategy, StateSpec


@dataclass
class ViewSpec:
    """An auxiliary view: re-encode the realized selections with another role; the search is unchanged.

    Args:
        name: View name (``traversal.views[name]``).
        role: Rendering role of the view states (``"recovery_view"``: the last observation is marked
            incorrect, Appendix E.1).
        source: ``"proposals"`` (every realized selection, the paper's recovery view) or ``"states"``
            (every non-root primary state).
        record_range: Record each view state's full-column score range (needed by ``minmax_full``).
        eligible: Optional filter on the source items.
    """

    name: str = "recovery"
    role: str = RECOVERY_VIEW
    source: str = "proposals"
    record_range: bool = True
    eligible: Optional[Callable] = None


class SearchEngine:
    """Runs strategies against an index with an encoder and a state renderer.

    Args:
        encoder: Object with ``encode_queries(texts)`` (see :class:`rdr.encoder.Encoder`).
        index: :class:`rdr.index.DenseIndex`.
        renderer: :class:`rdr.prompts.StateRenderer`.
        candidate_k: Candidates kept per state (the readout's ``|C_s|``; paper: 200).
        keep_text: Keep each state's rendered text (debugging, visualization).
        keep_embeddings: Keep state vectors (root interpolation needs them).
        record_range: Record every primary state's full-column score range (``minmax_full``).
        encode_grouping: How new states are grouped into encoder calls: ``"level"`` (all states of the
            level at once; fastest), ``"query"`` (one call per query, frontier order; the multi-hop
            protocol) or ``"slot"`` (slot-major across queries; the single-hop protocol). With an
            encoder that does not sort by length, ``"query"``/``"slot"`` reproduce the paper's batches.
    """

    def __init__(self, encoder, index, renderer, candidate_k: int = 200, keep_text: bool = False,
                 keep_embeddings: bool = False, record_range: bool = False, encode_grouping: str = "level") -> None:
        if encode_grouping not in ("level", "query", "slot"):
            raise ValueError("encode_grouping must be 'level', 'query' or 'slot'")
        self.encode_grouping = encode_grouping
        self.encoder = encoder
        self.index = index
        self.renderer = renderer
        self.candidate_k = candidate_k
        self.keep_text = keep_text
        self.keep_embeddings = keep_embeddings
        self.record_range = record_range
        self.encode_lock = threading.Lock()
        self.search_lock = threading.Lock()
        self.stats = {"encoded_states": 0, "doc_query_states": 0, "levels": 0}

    # ------------------------------------------------------------------ helpers
    def _texts(self, travs: Sequence[Traversal], items: Sequence[Tuple[int, StateSpec]]) -> List[str]:
        out = []
        for qi, sp in items:
            t = travs[qi]
            obs = [self.index.text(d) for d in sp.path]
            out.append(self.renderer.render(t.query, obs, sp.role, instruction=t.meta.get("root_instruction"),
                                            level=len(sp.path) + 1))
        return out

    def _embed(self, travs, items) -> Tuple[torch.Tensor, List[Optional[str]]]:
        enc_idx = [k for k, (_, sp) in enumerate(items) if sp.embedding_from_document is None]
        doc_idx = [k for k, (_, sp) in enumerate(items) if sp.embedding_from_document is not None]
        texts: List[Optional[str]] = [None] * len(items)
        vecs = torch.empty((len(items), self.index.dim), dtype=self.index.dtype, device=self.index.device)
        if enc_idx:
            tx = self._texts(travs, [items[k] for k in enc_idx])
            with self.encode_lock:
                e = self._encode_grouped(tx, [items[k][0] for k in enc_idx])
            # the producers cast query vectors to the index dtype before the matmul (fp16 on GPU)
            vecs[torch.as_tensor(enc_idx, device=self.index.device)] = e.to(self.index.device, dtype=self.index.dtype)
            for k, t in zip(enc_idx, tx):
                texts[k] = t
            self.stats["encoded_states"] += len(enc_idx)
        if doc_idx:
            pos = [items[k][1].embedding_from_document for k in doc_idx]
            vecs[torch.as_tensor(doc_idx, device=self.index.device)] = self.index.vectors(pos)
            self.stats["doc_query_states"] += len(doc_idx)
        return vecs, texts

    def _encode_grouped(self, texts: List[str], owners: List[int]) -> torch.Tensor:
        if self.encode_grouping == "level" or len(set(owners)) <= 1:
            return self.encoder.encode_queries(texts)
        if self.encode_grouping == "query":
            order = sorted(range(len(texts)), key=lambda k: owners[k])  # stable: frontier order per query
            parts, groups, start = [], [], 0
            while start < len(order):
                q = owners[order[start]]
                end = start
                while end < len(order) and owners[order[end]] == q:
                    end += 1
                groups.append(order[start:end])
                start = end
            embs = [self.encoder.encode_queries([texts[k] for k in g]) for g in groups]
            out = torch.empty((len(texts), embs[0].shape[1]), dtype=embs[0].dtype, device=embs[0].device)
            for g, e in zip(groups, embs):
                out[torch.as_tensor(g, device=e.device)] = e
            return out
        # "slot": the j-th new state of every query in one encoder call, then the (j+1)-th, ... (one call per slot,
        # as the paper's single-hop code: batches never mix two slots)
        slot, seen = [], {}
        for q in owners:
            slot.append(seen.get(q, 0))
            seen[q] = seen.get(q, 0) + 1
        out = None
        for j in sorted(set(slot)):
            ks = sorted((k for k in range(len(texts)) if slot[k] == j), key=lambda k: owners[k])
            e = self.encoder.encode_queries([texts[k] for k in ks])
            if out is None:
                out = torch.empty((len(texts), e.shape[1]), dtype=e.dtype, device=e.device)
            out[torch.as_tensor(ks, device=e.device)] = e
        return out

    def _search(self, vecs: torch.Tensor, rows_exclude, record_range: bool):
        with self.search_lock:
            out = self.index.search(vecs, self.candidate_k, exclude=rows_exclude, return_range=record_range)
        return out.ids, out.scores, out.ranges

    def _materialize(self, travs, items, level, record_range) -> List[Tuple[int, int]]:
        vecs, texts = self._embed(travs, items)
        excl = [travs[qi].meta.get("exclude") for qi, _ in items]
        ids, scores, rng = self._search(vecs, excl if any(e is not None for e in excl) else None, record_range)
        placed = []
        for k, (qi, sp) in enumerate(items):
            meta = dict(sp.meta)
            meta["path_sum"] = sp.path_sum
            meta["path_logp"] = sp.path_logp
            if rng is not None:
                meta["min"], meta["max"] = float(rng[k, 0]), float(rng[k, 1])
            st = State(level=level, parent=sp.parent, path=tuple(sp.path), role=sp.role, ids=ids[k], scores=scores[k],
                       path_score=sp.path_score, meta=meta,
                       text=texts[k] if self.keep_text else None,
                       embedding=vecs[k].detach().cpu() if self.keep_embeddings else None)
            t = travs[qi]
            t.states.append(st)
            pi = sp.meta.get("proposal")
            if pi is not None:
                t.proposals[pi].expanded = True
            placed.append((qi, len(t.states) - 1))
        return placed

    # ---------------------------------------------------------------------- run
    def run(
        self,
        strategy: SearchStrategy,
        queries: Sequence[str],
        instructions: Optional[Sequence[Optional[str]]] = None,
        exclude: Optional[Sequence[Optional[Sequence[int]]]] = None,
        views: Sequence[ViewSpec] = (),
        offset: int = 0,
    ) -> List[Traversal]:
        """Build one traversal per query (level-synchronous over the whole batch)."""
        travs = []
        for i, q in enumerate(queries):
            meta = {"root_instruction": instructions[i] if instructions is not None else None,
                    "exclude": exclude[i] if exclude is not None else None}
            travs.append(Traversal(query=q, query_index=offset + i, meta=meta))
        items = [(qi, sp) for qi, t in enumerate(travs) for sp in strategy.roots(t)]
        level = 1
        record = self.record_range or getattr(strategy, "needs_range", False)
        while items:
            placed = self._materialize(travs, items, level, record)
            self.stats["levels"] = max(self.stats["levels"], level)
            by_q: Dict[int, List[int]] = {}
            for qi, si in placed:
                by_q.setdefault(qi, []).append(si)
            # selection happens at every level, the deepest one included (its selections are recorded
            # for realized readouts but not encoded)
            nxt = [(qi, sp) for qi, sis in by_q.items() for sp in strategy.select(travs[qi], level, sis)]
            if level >= strategy.depth:
                break
            items = nxt
            level += 1
        for view in views:
            self._run_view(travs, view)
        return travs

    def _run_view(self, travs: List[Traversal], view: ViewSpec) -> None:
        items = []
        for qi, t in enumerate(travs):
            t.views.setdefault(view.name, [])
            if view.source == "proposals":
                src = [(p.parent, p.path, p.cum, p.level + 1) for p in t.proposals]
            else:
                src = [(s.parent, s.path, s.path_score, s.level) for s in t.states if s.parent >= 0]
            for parent, path, cum, lvl in src:
                if view.eligible is not None and not view.eligible(t, path):
                    continue
                items.append((qi, StateSpec(parent=parent, path=path, role=view.role, path_score=cum,
                                            meta={"level": lvl})))
        if not items:
            return
        vecs, texts = self._embed(travs, items)
        excl = [travs[qi].meta.get("exclude") for qi, _ in items]
        ids, scores, rng = self._search(vecs, excl if any(e is not None for e in excl) else None, view.record_range)
        for k, (qi, sp) in enumerate(items):
            meta = {}
            if rng is not None:
                meta["min"], meta["max"] = float(rng[k, 0]), float(rng[k, 1])
            travs[qi].views[view.name].append(
                State(level=sp.meta["level"], parent=sp.parent, path=sp.path, role=view.role, ids=ids[k],
                      scores=scores[k], path_score=sp.path_score, meta=meta,
                      text=texts[k] if self.keep_text else None))
