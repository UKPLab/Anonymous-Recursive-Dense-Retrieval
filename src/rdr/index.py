"""Exact dense index over cached document vectors.

The index is searched with raw cosine scores (document and query vectors are L2-normalized), as
in the paper: document vectors are encoded once and never change during the query-side search. Top-K results are ordered by ``(-score, index)`` so that ties break
deterministically by corpus position.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

Exclusion = Optional[Sequence[Optional[Sequence[int]]]]


def _as_device(device: Union[str, torch.device, None]) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def sort_by_score_then_index(vals: np.ndarray, idx: np.ndarray):
    """Row-wise sort by descending score, ties by ascending index (stable two-pass sort)."""
    o1 = np.argsort(idx, axis=1, kind="stable")
    idx1 = np.take_along_axis(idx, o1, axis=1)
    val1 = np.take_along_axis(vals, o1, axis=1)
    o2 = np.argsort(-val1, axis=1, kind="stable")
    return np.take_along_axis(val1, o2, axis=1), np.take_along_axis(idx1, o2, axis=1)


@dataclass
class SearchOutput:
    ids: np.ndarray                      # int64 [m, k]
    scores: np.ndarray                   # float32 [m, k]
    ranges: Optional[np.ndarray] = None  # float32 [m, 2]: full-column (min, max), if requested


class DenseIndex:
    """Exact inner-product (= cosine) search over normalized document embeddings.

    Args:
        embeddings: ``[N, d]`` document embeddings. They are L2-normalized on load unless
            ``normalized=True`` says they already are.
        texts: Document texts, needed to render observed documents into later query states.
        ids: External document ids returned to the user (default: ``range(N)``).
        groups: Optional document key per entry for chunked corpora. Readouts can max-pool
            chunk scores per group (the paper's multi-hop output processing).
        device: Where the vectors live and the search runs (``"cuda:1"``, ``"cpu"`` …).
        dtype: Storage/compute dtype. ``float16`` on GPU (as in the paper), ``float32`` on CPU.
        normalized: The embeddings are already L2-normalized; keep their bits as given.
        query_chunk_size: Queries scored per matmul (default: sized to a ~512 MB score block).
    """

    def __init__(
        self,
        embeddings: Union[torch.Tensor, np.ndarray],
        texts: Optional[Sequence[str]] = None,
        ids: Optional[Sequence[Any]] = None,
        groups: Optional[Sequence[Any]] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
        normalized: bool = False,
        query_chunk_size: Optional[int] = None,
    ) -> None:
        self.device = _as_device(device)
        if dtype is None:
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        emb = torch.as_tensor(embeddings)
        if not normalized:
            emb = torch.nn.functional.normalize(emb.float(), p=2, dim=1)
        self.embeddings = emb.to(self.device, dtype=dtype).contiguous()
        self.dtype = dtype
        n = self.embeddings.shape[0]
        self.texts = list(texts) if texts is not None else None
        self.ids = list(ids) if ids is not None else list(range(n))
        if len(self.ids) != n or (self.texts is not None and len(self.texts) != n):
            raise ValueError("embeddings, texts and ids must have the same length")
        self.groups = list(groups) if groups is not None else None
        self._group_index: Optional[np.ndarray] = None
        self.query_chunk_size = query_chunk_size

    # ----------------------------------------------------------------- basics
    def __len__(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def positions(self, ids: Sequence[Any]) -> List[int]:
        """Index positions of corpus ids (unknown ids are skipped), e.g. for ``exclude=``.

        Args:
            ids: Corpus ids.
        """
        if getattr(self, "_row", None) is None:
            self._row = {d: i for i, d in enumerate(self.ids)}
        return [self._row[d] for d in ids if d in self._row]

    def text(self, i: int) -> str:
        if self.texts is None:
            raise ValueError("this index was built without document texts; they are needed to render observations")
        return self.texts[i]

    def vectors(self, positions: Sequence[int]) -> torch.Tensor:
        """Index vectors of the given positions (used by document-as-query expansion)."""
        return self.embeddings[torch.as_tensor(list(positions), device=self.device, dtype=torch.long)]

    def group_index(self) -> Optional[np.ndarray]:
        """Integer group code per entry (for chunk → document max-pooling), or ``None``."""
        if self.groups is None:
            return None
        if self._group_index is None:
            codes: Dict[Any, int] = {}
            self._group_index = np.fromiter((codes.setdefault(g, len(codes)) for g in self.groups), dtype=np.int64)
        return self._group_index

    # ----------------------------------------------------------------- search
    def _chunk(self, m: int) -> int:
        if self.query_chunk_size:
            return self.query_chunk_size
        budget = 512 * 1024 * 1024  # bytes of the score matrix per chunk
        per_row = max(1, len(self) * torch.finfo(self.dtype).bits // 8)
        return max(1, min(m, budget // per_row))

    @torch.inference_mode()
    def search(
        self,
        queries: Union[torch.Tensor, np.ndarray],
        k: int,
        exclude: Exclusion = None,
        tie_slack: int = 16,
        return_range: bool = False,
    ) -> SearchOutput:
        """Top-``k`` search for a batch of query vectors.

        Args:
            queries: ``[m, d]`` normalized query vectors (any device/dtype).
            k: Number of candidates per query.
            exclude: Optional per-row index positions that are not eligible (benchmark exclusions).
                They are removed *before* the top-K cut, i.e. before any per-state softmax.
            tie_slack: Extra candidates fetched so that ties at the cut are ordered by index.
            return_range: Also return each row's full-column minimum and maximum score (before exclusions),
                used by min–max normalizations.
        """
        q = torch.as_tensor(queries)
        if q.ndim == 1:
            q = q[None, :]
        q = q.to(self.device, dtype=self.dtype)
        m, n = q.shape[0], len(self)
        k = min(k, n)
        k2 = min(n, k + tie_slack)
        out_ids = np.empty((m, k), dtype=np.int64)
        out_sc = np.empty((m, k), dtype=np.float32)
        out_rng = np.empty((m, 2), dtype=np.float32) if return_range else None
        step = self._chunk(m)
        for a in range(0, m, step):
            b = min(m, a + step)
            sims = q[a:b] @ self.embeddings.T  # [rows, N] in the index dtype
            if return_range:
                out_rng[a:b, 0] = sims.amin(1).float().cpu().numpy()
                out_rng[a:b, 1] = sims.amax(1).float().cpu().numpy()
            if exclude is not None:
                rows, cols = [], []
                for r in range(a, b):
                    ex = exclude[r] if r < len(exclude) else None
                    if ex is not None and len(ex):
                        cols.extend(int(c) for c in ex)
                        rows.extend([r - a] * len(ex))
                if rows:
                    sims[torch.as_tensor(rows, device=self.device), torch.as_tensor(cols, device=self.device)] = float("-inf")
            vals, idx = torch.topk(sims, k2, dim=1, largest=True, sorted=True)
            vals = vals.float().cpu().numpy()
            idx = idx.cpu().numpy().astype(np.int64)
            vals, idx = sort_by_score_then_index(vals, idx)
            out_ids[a:b] = idx[:, :k]
            out_sc[a:b] = vals[:, :k]
        return SearchOutput(out_ids, out_sc, out_rng)

    @torch.inference_mode()
    def score(self, queries: Union[torch.Tensor, np.ndarray], positions: Sequence[Sequence[int]]) -> List[np.ndarray]:
        """Raw cosine of each query row against an explicit candidate list (pool-restricted steps)."""
        q = torch.as_tensor(queries).to(self.device, dtype=self.dtype)
        out = []
        for r, pos in enumerate(positions):
            v = self.vectors(pos)
            out.append((v @ q[r]).float().cpu().numpy())
        return out

    # ------------------------------------------------------------ build/save
    @classmethod
    def build(
        cls,
        texts: Sequence[str],
        encoder,
        ids: Optional[Sequence[Any]] = None,
        groups: Optional[Sequence[Any]] = None,
        device: Union[str, torch.device, None] = None,
        dtype: Optional[torch.dtype] = None,
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,
        cache_dir: Optional[str] = None,
    ) -> "DenseIndex":
        """Encode ``texts`` with ``encoder.encode_documents`` and build an index.

        Args:
            texts: Document texts.
            encoder: An :class:`~rdr.encoder.Encoder` (anything with ``encode_documents``).
            ids: Corpus ids returned to the user (default: ``range(N)``).
            groups: Optional document key per entry for chunked corpora.
            device: Where the vectors live and the search runs.
            dtype: Storage/compute dtype (default: float16 on GPU, float32 on CPU).
            batch_size: Document batch size (default: the encoder's).
            show_progress_bar: Show a progress bar while encoding.
            cache_dir: If it already holds embeddings, they are loaded instead of encoding (not validated
                against the texts or the model); otherwise the new embeddings are saved there.
        """
        if cache_dir is not None and os.path.exists(os.path.join(cache_dir, "embeddings.npy")):
            return cls.load(cache_dir, device=device, dtype=dtype, texts=texts)
        emb = encoder.encode_documents(list(texts), batch_size=batch_size, show_progress_bar=show_progress_bar)
        index = cls(emb, texts=texts, ids=ids, groups=groups, device=device, dtype=dtype, normalized=True)
        if cache_dir is not None:
            index.save(cache_dir)
        return index

    def save(self, path: str) -> None:
        """Save embeddings (fp16), ids, groups and texts to a directory.

        Args:
            path: Target directory (created if needed).
        """
        os.makedirs(path, exist_ok=True)
        np.save(os.path.join(path, "embeddings.npy"), self.embeddings.float().cpu().numpy().astype(np.float16))
        meta = {"ids": [str(i) for i in self.ids], "groups": self.groups, "n": len(self), "dim": self.dim}
        with open(os.path.join(path, "index.json"), "w") as f:
            json.dump(meta, f)
        if self.texts is not None:
            with open(os.path.join(path, "texts.jsonl"), "w") as f:
                for t in self.texts:
                    f.write(json.dumps(t) + "\n")

    @classmethod
    def load(cls, path: str, device=None, dtype=None, texts: Optional[Sequence[str]] = None) -> "DenseIndex":
        """Load an index written by :meth:`save` (or by ``build(..., cache_dir=...)``).

        Args:
            path: Directory of the saved index.
            device: Where the vectors live and the search runs.
            dtype: Storage/compute dtype (default: float16 on GPU, float32 on CPU).
            texts: Document texts, if they were not saved with the index.

        Saved ids are strings; pass your own ``ids`` to the constructor if you need other types.
        """
        emb = np.load(os.path.join(path, "embeddings.npy"), mmap_mode="r")
        meta = {}
        mp = os.path.join(path, "index.json")
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
        if texts is None and os.path.exists(os.path.join(path, "texts.jsonl")):
            with open(os.path.join(path, "texts.jsonl")) as f:
                texts = [json.loads(line) for line in f]
        # Saved vectors are already normalized: keep them bit-exact (re-normalizing fp16 vectors in fp32
        # would move the last bit of some scores and can flip near-ties).
        return cls(torch.from_numpy(np.ascontiguousarray(emb)), texts=texts, ids=meta.get("ids"),
                   groups=meta.get("groups"), device=device, dtype=dtype, normalized=True)
