"""``RecursiveRetriever``: model + prompts + index + search + readout.

Example::

    from rdr import RecursiveRetriever, TreeSearch, BeamSearch, GlobalReadout

    retriever = RecursiveRetriever("RDR-8B", prompts="multi-hop")
    index = retriever.build_index(corpus)                       # documents are encoded raw
    hits = retriever.search(["Who is the uncle of Clio Goldsmith?"], index, top_k=10)
    hits = retriever.search(queries, index, search=BeamSearch(width=5))   # beam, still global readout

Every search strategy is read out with the global readout unless a readout is passed explicitly;
realized path readouts (``PathProduct``, ``SequenceSum``, ``ChainOrder``) must be requested by name.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import torch

from .encoder import Encoder
from .index import DenseIndex
from .prompts import PromptConfig, StateRenderer
from .readout.base import Readout, group_max
from .readout.global_readout import GlobalReadout
from .readout.model_based import ModelReadout
from .registry import make_readout, make_search
from .search.base import SearchStrategy
from .search.engine import SearchEngine, ViewSpec
from .search.strategies import TreeSearch
from .types import RECOVERY, Ranking, Traversal


@dataclass
class _Ctx:
    encoder: Any
    index: Any
    renderer: Any


def _views_of(readout: Readout) -> List[str]:
    """Names of auxiliary views a readout (or its sub-readouts) reads."""
    out = []
    v = getattr(readout, "view", None)
    if isinstance(v, str) and v != "primary":
        out.append(v)
    for attr in ("readouts", "base", "first", "second"):
        sub = getattr(readout, attr, None)
        subs = sub if isinstance(sub, (list, tuple)) else [sub]
        for s in subs:
            if isinstance(s, Readout):
                out.extend(_views_of(s))
    return list(dict.fromkeys(out))


class RecursiveRetriever:
    """Recursive dense retrieval with one instruction-conditioned encoder.

    Args:
        model: A ``SentenceTransformer``, a model name/path, or an :class:`~rdr.encoder.Encoder`.
        prompts: A :class:`~rdr.prompts.PromptConfig` or a preset name (``"multi-hop"``, ``"single-hop"``,
            ``"single-hop-continuation"``, ``"bright-pro"``).
        search: Default search strategy (``TreeSearch(4, 3)``) or a name (``"tree"``, ``"beam"`` …).
        readout: Default readout (``GlobalReadout()``) or a name.
        candidate_k: Candidates kept per state (paper: 200).
        query_max_length: Token budget of a rendered state (paper and training: 4096).
        document_max_length: Token budget of an indexed document (paper: 512 for the multi-hop passages;
            the single-hop corpora were encoded with up to 8192 tokens).
        batch_size: Encoder batch size.
        query_batch_size: Queries processed together (level-synchronous batching).
        encode_device: Device of the encoder (or a list for data-parallel encoding).
        index_device: Device of the document index and the exact search (e.g. a second GPU).
        inflight: Query batches in flight; with separate encode/index devices, ``inflight >= 2`` overlaps
            encoding of one batch with the search of another.
        torch_dtype: Weight dtype when ``model`` is a name or path (RDR-8B: ``"bfloat16"``).
        encode_grouping: ``"level"`` (fastest), ``"query"`` or ``"slot"`` (the paper's batch composition, see
            :class:`rdr.search.engine.SearchEngine`).
    """

    def __init__(
        self,
        model,
        prompts: Union[str, PromptConfig] = "multi-hop",
        search: Union[str, SearchStrategy, None] = None,
        readout: Union[str, Readout, None] = None,
        candidate_k: int = 200,
        query_max_length: int = 4096,
        document_max_length: int = 512,
        batch_size: int = 8,
        query_batch_size: int = 32,
        encode_device: Union[str, Sequence[str], None] = None,
        index_device: Optional[str] = None,
        inflight: int = 2,
        torch_dtype: Union[str, torch.dtype, None] = "bfloat16",
        encode_grouping: str = "level",
    ) -> None:
        if isinstance(model, Encoder) or (hasattr(model, "encode_queries") and hasattr(model, "encode_documents")):
            self.encoder = model  # any object with encode_queries/encode_documents/truncate
        else:
            devices = list(encode_device) if isinstance(encode_device, (list, tuple)) else (
                [encode_device] if encode_device is not None else None)
            if isinstance(model, str):
                from .encoder import load_sentence_transformer

                model = load_sentence_transformer(model, device=devices[0] if devices else None,
                                                  torch_dtype=torch_dtype, max_seq_length=query_max_length)
            self.encoder = Encoder(model, query_max_length=query_max_length,
                                   document_max_length=document_max_length, batch_size=batch_size,
                                   devices=devices)
        self.prompts = PromptConfig.preset(prompts) if isinstance(prompts, str) else prompts
        self.renderer = StateRenderer(self.prompts, truncate=self.encoder.truncate)
        self.search_strategy = make_search(search) if search is not None else TreeSearch(4, 3)
        self.readout = make_readout(readout) if readout is not None else GlobalReadout()
        self.candidate_k = candidate_k
        self.query_batch_size = query_batch_size
        self.index_device = index_device
        self.inflight = max(1, inflight)
        self.encode_grouping = encode_grouping

    # --------------------------------------------------------------------- index
    def build_index(self, corpus: Union[Sequence[str], Dict[Any, str]], ids: Optional[Sequence[Any]] = None,
                    groups: Optional[Sequence[Any]] = None, cache_dir: Optional[str] = None,
                    show_progress_bar: bool = True) -> DenseIndex:
        """Encode a corpus (list of texts or ``{id: text}``) into a :class:`DenseIndex` on ``index_device``.

        Args:
            corpus: Document texts, or a ``{corpus_id: text}`` mapping.
            ids: Corpus ids returned in the hits (default: the mapping's keys, or ``0 … N-1``).
            groups: Optional document key per entry for chunked corpora; readouts then max-pool chunk
                scores per document (the paper's multi-hop output processing).
            cache_dir: Directory for the document embeddings. If it already holds embeddings they are
                loaded instead of encoding again; the cache is not validated against the texts or the
                model, so use one directory per corpus and model.
            show_progress_bar: Show a progress bar while encoding.
        """
        if isinstance(corpus, dict):
            ids = list(corpus.keys()) if ids is None else ids
            texts = list(corpus.values())
        else:
            texts = list(corpus)
        return DenseIndex.build(texts, self.encoder, ids=ids, groups=groups, device=self.index_device,
                                show_progress_bar=show_progress_bar, cache_dir=cache_dir)

    # -------------------------------------------------------------------- search
    def traverse(
        self,
        queries: Sequence[str],
        index: DenseIndex,
        search: Union[str, SearchStrategy, None] = None,
        instruction: Union[str, Sequence[Optional[str]], None] = None,
        exclude: Optional[Sequence[Optional[Sequence[int]]]] = None,
        views: Sequence[Union[str, ViewSpec]] = (),
        keep_text: bool = False,
        keep_embeddings: bool = False,
        show_progress_bar: bool = False,
    ) -> List[Traversal]:
        """Run the search and return the traversals (all states with their candidates).

        A traversal can be read out any number of times with :meth:`read`, so readouts are compared on
        identical states (as in the paper's Appendix F).

        Args:
            queries: Query texts.
            index: The document index.
            search: Search strategy for this call (default: the retriever's ``search``).
            instruction: Root instruction, one for all queries or one per query (benchmark instructions);
                ``None`` uses the prompt configuration's ``root``.
            exclude: Per query, index positions that may never be retrieved (e.g. BRIGHT's excluded ids);
                see :meth:`DenseIndex.positions`.
            views: Extra views to encode over the same traversal, e.g. ``["recovery"]``; readouts that
                need a view request it automatically in :meth:`search`.
            keep_text: Keep each state's rendered text in the traversal.
            keep_embeddings: Keep each state's query vector (needed by :class:`RootInterpolation`).
            show_progress_bar: Show a progress bar over query batches.
        """
        strategy = make_search(search) if search is not None else self.search_strategy
        strategy.role = self.prompts.primary_role
        n = len(queries)
        instr = [instruction] * n if (instruction is None or isinstance(instruction, str)) else list(instruction)
        vspecs = [ViewSpec(name=v) if isinstance(v, str) else v for v in views]
        engine = SearchEngine(self.encoder, index, self.renderer, candidate_k=self.candidate_k,
                              keep_text=keep_text, keep_embeddings=keep_embeddings,
                              encode_grouping=self.encode_grouping)
        batches = [range(a, min(n, a + self.query_batch_size)) for a in range(0, n, self.query_batch_size)]

        def run(b: range) -> List[Traversal]:
            return engine.run(strategy, [queries[i] for i in b], [instr[i] for i in b],
                              [exclude[i] for i in b] if exclude is not None else None, vspecs, offset=b.start)

        out: List[Traversal] = []
        if self.inflight > 1 and len(batches) > 1:
            with ThreadPoolExecutor(max_workers=self.inflight) as pool:
                for res in pool.map(run, batches):
                    out.extend(res)
        else:
            it = batches
            if show_progress_bar:
                from tqdm.auto import tqdm

                it = tqdm(batches, desc="search")
            for b in it:
                out.extend(run(b))
        return out

    def read(self, traversals: Sequence[Traversal], index: DenseIndex,
             readout: Union[str, Readout, None] = None) -> List[Ranking]:
        """Apply a readout to saved traversals (no further encoding, except for model-based readouts).

        Args:
            traversals: Output of :meth:`traverse`.
            index: The index the traversals were built on.
            readout: Readout for this call (default: the retriever's ``readout``).
        """
        ro = make_readout(readout) if readout is not None else self.readout
        groups = index.group_index()
        for t in traversals:
            t.meta["groups"] = groups
        if isinstance(ro, ModelReadout):
            ctx = _Ctx(self.encoder, index, self.renderer)
            rankings: List[Ranking] = []
            for a in range(0, len(traversals), self.query_batch_size):
                rankings.extend(ro.batch(traversals[a : a + self.query_batch_size], ctx))
        else:
            rankings = [ro(t) for t in traversals]
        if groups is not None:
            rankings = [group_max(r, groups, top_k=getattr(ro, "output_top_k", 200) or None) for r in rankings]
        return rankings

    def search(
        self,
        queries: Union[str, Sequence[str]],
        index: DenseIndex,
        top_k: int = 10,
        search: Union[str, SearchStrategy, None] = None,
        readout: Union[str, Readout, None] = None,
        instruction: Union[str, Sequence[Optional[str]], None] = None,
        exclude: Optional[Sequence[Optional[Sequence[int]]]] = None,
        return_traversals: bool = False,
        show_progress_bar: bool = False,
    ):
        """Search and read out. Returns ``semantic_search``-style hits: one list of
        ``{"corpus_id", "score", "index"}`` dicts per query (and the traversals if requested).

        Args:
            queries: One query or a list of queries.
            index: The document index (:meth:`build_index`).
            top_k: Hits returned per query.
            search: Search strategy for this call, an instance or a name (default: the retriever's).
            readout: Readout for this call, an instance or a name (default: the retriever's
                :class:`GlobalReadout`). Views it needs (e.g. the recovery view) are encoded automatically.
            instruction: Root instruction, one for all queries or one per query.
            exclude: Per query, index positions that may never be retrieved.
            return_traversals: Also return the traversals, to read them out again with :meth:`read`.
            show_progress_bar: Show a progress bar over query batches.
        """
        single = isinstance(queries, str)
        qs = [queries] if single else list(queries)
        ro = make_readout(readout) if readout is not None else self.readout
        views = _views_of(ro)
        travs = self.traverse(qs, index, search=search, instruction=instruction, exclude=exclude, views=views,
                              keep_embeddings=getattr(ro, "needs_embeddings", False),
                              show_progress_bar=show_progress_bar)
        rankings = self.read(travs, index, ro)
        hits = [[{"corpus_id": index.ids[i], "score": float(s), "index": int(i)}
                 for i, s in zip(r.ids[:top_k].tolist(), r.scores[:top_k].tolist())] for r in rankings]
        out = hits[0] if single else hits
        return (out, travs) if return_traversals else out
