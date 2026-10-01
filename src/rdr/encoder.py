"""Encoders: a thin, sentence-transformers based wrapper around the query/document encoder.

RDR states are rendered to complete strings by :mod:`rdr.prompts` (instruction, query and observed
documents), so the encoder never adds a prompt itself: :meth:`Encoder.encode_queries` encodes the
rendered state texts as they are, :meth:`Encoder.encode_documents` encodes documents with the
model's document contract (raw text for RDR-8B).
"""

from __future__ import annotations

import inspect
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

Device = Union[str, torch.device]


def _resolve_device(device: Optional[Device]) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def has_sentence_transformers_config(path: str) -> bool:
    """Whether a local folder or a Hub repository carries a sentence-transformers configuration."""
    if os.path.isdir(path):
        return os.path.exists(os.path.join(path, "modules.json"))
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(path, "modules.json")  # works offline when cached
        return True
    except Exception:
        return False


def embedding_dimension(model) -> int:
    """Output dimension of a SentenceTransformer or module (sentence-transformers 3–6)."""
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension", "get_word_embedding_dimension"):
        f = getattr(model, name, None)
        if f is not None:
            return int(f())
    raise AttributeError(f"cannot read the embedding dimension of {type(model).__name__}")


def restore_float32_rotary(model) -> int:
    """Recompute rotary (RoPE) inverse frequencies in float32; returns the number of fixed modules.

    ``SentenceTransformer.__init__`` casts the whole module to the weights' dtype (``self.to(dtype)``), which
    also turns the non-persistent RoPE buffer ``inv_freq`` into bfloat16; transformers keeps it in float32.
    The rounded frequencies shift every embedding (cosine ~0.9998 to the reference, fp16 score shifts of
    1e-3 to 5e-3). The frequencies are recomputed on CPU, as at model init, which reproduces the loaded buffer
    bit for bit.
    """
    n = 0
    for mod in model.modules():
        inv = getattr(mod, "inv_freq", None)
        if not isinstance(inv, torch.Tensor) or inv.dtype == torch.float32 or not hasattr(mod, "config"):
            continue
        fn = getattr(mod, "rope_init_fn", None)  # transformers 4.x
        if fn is None:  # transformers 5.x
            rope_type = getattr(mod, "rope_type", "default")
            if rope_type == "default":
                fn = mod.compute_default_rope_parameters
            else:
                from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

                fn = ROPE_INIT_FUNCTIONS[rope_type]
        new, _ = fn(mod.config, "cpu")
        mod.inv_freq = new.to(device=inv.device, dtype=torch.float32)
        if isinstance(getattr(mod, "original_inv_freq", None), torch.Tensor):
            mod.original_inv_freq = mod.inv_freq.clone()
        n += 1
    return n


def _ends_with_normalize(model) -> bool:
    """Whether a SentenceTransformer pipeline already L2-normalizes its output (last module ``Normalize``)."""
    try:
        from sentence_transformers.models import Normalize
    except Exception:  # pragma: no cover
        return False
    mods = list(model.children()) if hasattr(model, "children") else []
    return bool(mods) and isinstance(mods[-1], Normalize)


def load_sentence_transformer(
    model_name_or_path: str,
    device: Optional[Device] = None,
    torch_dtype: Union[str, torch.dtype, None] = "bfloat16",
    max_seq_length: int = 4096,
    pooling: str = "lasttoken",
    trust_remote_code: bool = False,
    **kwargs,
):
    """Load a :class:`~sentence_transformers.SentenceTransformer`.

    Hub models and local folders with a sentence-transformers config (``modules.json``) load directly.
    A plain Hugging Face checkpoint (e.g. a training checkpoint of Qwen3-Embedding) is wrapped as
    ``Transformer -> Pooling(lasttoken) -> Normalize`` with left padding, which is the Qwen3-Embedding
    contract RDR-8B was trained with.

    Args:
        model_name_or_path: Hub name (``"RDR-8B"``) or local path.
        device: Device to load the model on (default: CUDA if available, else CPU).
        torch_dtype: Weight dtype (RDR-8B: ``"bfloat16"``).
        max_seq_length: Token limit of the model's inputs.
        pooling: Pooling of a wrapped plain checkpoint (``"lasttoken"`` for Qwen3-based models).
        trust_remote_code: Passed to sentence-transformers / transformers.
        **kwargs: Further ``SentenceTransformer`` arguments.
    """
    from sentence_transformers import SentenceTransformer, models

    dev = str(_resolve_device(device))
    dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
    model_kwargs = {"torch_dtype": dtype} if dtype is not None else {}
    if has_sentence_transformers_config(model_name_or_path):
        st = SentenceTransformer(model_name_or_path, device=dev, model_kwargs=model_kwargs,
                                 trust_remote_code=trust_remote_code, **kwargs)
        st.max_seq_length = max_seq_length
        restore_float32_rotary(st)
        return st
    params = inspect.signature(models.Transformer.__init__).parameters
    if "model_kwargs" in params:  # sentence-transformers >= 6
        # text only: with the inferred "message" modality, sentence-transformers 6 renders every input through
        # the tokenizer's chat template ("<|im_start|>user\n...<|im_end|>"), which breaks the RDR-8B contract
        word = models.Transformer(model_name_or_path, max_seq_length=max_seq_length, model_kwargs=model_kwargs,
                                  processor_kwargs={"padding_side": "left"},
                                  modality_config={"text": {"method": "forward", "method_output_name": "last_hidden_state"}},
                                  module_output_name="token_embeddings")
    else:  # sentence-transformers 3 - 5
        word = models.Transformer(model_name_or_path, max_seq_length=max_seq_length, model_args=model_kwargs,
                                  tokenizer_args={"padding_side": "left"})
    pool = models.Pooling(embedding_dimension(word), pooling_mode=pooling)
    st = SentenceTransformer(modules=[word, pool, models.Normalize()], device=dev)
    restore_float32_rotary(st)
    return st


class Encoder:
    """Query/document encoder built on a :class:`~sentence_transformers.SentenceTransformer`.

    Args:
        model: A ``SentenceTransformer`` (or a model name/path, loaded with :func:`load_sentence_transformer`).
        query_max_length: Token budget of a rendered query state (instruction + query + observations).
        document_max_length: Token budget of an indexed document.
        batch_size: Batch size for rendered query states.
        document_batch_size: Batch size for documents (default: ``max(batch_size, 32)``).
        document_prompt: Optional prompt prepended to documents (``None`` = raw documents, RDR-8B's contract).
        devices: Optional list of devices for data-parallel encoding (one replica per device).
        torch_dtype: Weight dtype when ``model`` is a name or path (RDR-8B: ``"bfloat16"``).
        sort_by_length: Let sentence-transformers sort texts by length for faster batching (default). Set
            ``False`` to encode in the given order with fixed consecutive batches, the batch composition of
            the paper's runs (bf16 embeddings depend slightly on which texts share a batch).
    """

    def __init__(
        self,
        model,
        query_max_length: int = 4096,
        document_max_length: int = 512,
        batch_size: int = 8,
        document_batch_size: Optional[int] = None,
        document_prompt: Optional[str] = None,
        devices: Optional[Sequence[Device]] = None,
        torch_dtype: Union[str, torch.dtype, None] = "bfloat16",
        sort_by_length: bool = True,
    ) -> None:
        if isinstance(model, str):
            model = load_sentence_transformer(model, device=(devices[0] if devices else None),
                                              torch_dtype=torch_dtype, max_seq_length=query_max_length)
        self.model = model
        if hasattr(model, "modules"):
            restore_float32_rotary(model)  # also for user-built SentenceTransformers (see the function)
        self.query_max_length = query_max_length
        self.document_max_length = document_max_length
        self.batch_size = batch_size
        self.document_batch_size = document_batch_size or max(batch_size, 32)
        self.document_prompt = document_prompt
        self.sort_by_length = sort_by_length
        self._replicas = None
        if devices is not None and len(devices) > 1:
            self._replicas = _Replicas(model, devices)

    # ---------------------------------------------------------------- properties
    @property
    def tokenizer(self):
        return self.model.tokenizer

    @property
    def device(self) -> torch.device:
        return torch.device(self.model.device)

    @property
    def dim(self) -> int:
        return embedding_dimension(self.model)

    # ------------------------------------------------------------------- encode
    def _encode(self, texts: List[str], max_length: int, batch_size: int, prompt: Optional[str] = None,
                show_progress_bar: bool = False) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.dim)
        if self._replicas is not None:
            return self._replicas.encode(texts, max_length, batch_size, prompt)
        if not self.sort_by_length:
            return self._encode_ordered(texts, max_length, batch_size, prompt)
        # length-sorted batches (longest first, as sentence-transformers' encode) through the same tokenization and
        # normalization as the ordered path
        return encode_sorted(self.model, texts, max_length, batch_size, prompt)

    def _encode_ordered(self, texts: List[str], max_length: int, batch_size: int,
                        prompt: Optional[str] = None) -> torch.Tensor:
        """Encode in the given order with consecutive batches (no length sorting); see :func:`encode_texts`."""
        return encode_texts(self.model, texts, max_length, batch_size, prompt)

    def encode_queries(self, texts: Sequence[str], batch_size: Optional[int] = None,
                       show_progress_bar: bool = False) -> torch.Tensor:
        """Encode rendered states (they already contain the instruction).

        Args:
            texts: Texts to encode.
            batch_size: Override of the encoder's batch size.
            show_progress_bar: Show a progress bar.

        Returns:
            L2-normalized embeddings, ``[len(texts), dim]``.
        """
        return self._encode(list(texts), self.query_max_length, batch_size or self.batch_size,
                            show_progress_bar=show_progress_bar)

    def encode_documents(self, texts: Sequence[str], batch_size: Optional[int] = None,
                         show_progress_bar: bool = False) -> torch.Tensor:
        """Encode documents for the index (raw text unless ``document_prompt`` is set).

        Args:
            texts: Texts to encode.
            batch_size: Override of the encoder's batch size.
            show_progress_bar: Show a progress bar.

        Returns:
            L2-normalized embeddings, ``[len(texts), dim]``.
        """
        return self._encode(list(texts), self.document_max_length, batch_size or self.document_batch_size,
                            prompt=self.document_prompt, show_progress_bar=show_progress_bar)

    # --------------------------------------------------------------- utilities
    def truncate(self, text: str, max_tokens: Optional[int]) -> str:
        """Cap a document at ``max_tokens`` tokens (the paper's 512-token cap per injected document)."""
        if not max_tokens:
            return text
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return self.tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


@torch.inference_mode()
def encode_texts(model, texts: List[str], max_length: int, batch_size: int, prompt: Optional[str] = None) -> torch.Tensor:
    """Encode texts in the given order with consecutive batches.

    Texts are tokenized with the Hugging Face tokenizer as they are: sentence-transformers' ``tokenize``
    strips every text (and version 6 may apply a chat template), the paper's encoder does neither. The
    pipeline's ``Normalize`` module already normalizes; a second bf16 normalization would rescale ~25% of the
    vectors by 1/0.99609375, so it is applied only to pipelines without ``Normalize``.
    """
    from sentence_transformers.util import batch_to_device

    tok = model.tokenizer
    normalized = _ends_with_normalize(model)
    out = []
    for a in range(0, len(texts), batch_size):
        chunk = texts[a : a + batch_size]
        if prompt:
            chunk = [prompt + t for t in chunk]
        feats = dict(tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt"))
        emb = model(batch_to_device(feats, model.device))["sentence_embedding"]
        out.append(emb if normalized else torch.nn.functional.normalize(emb, p=2, dim=1))
    return torch.cat(out)


def encode_sorted(model, texts: List[str], max_length: int, batch_size: int, prompt: Optional[str] = None) -> torch.Tensor:
    """:func:`encode_texts` on length-sorted batches (longest first), results in the input order."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    emb = encode_texts(model, [texts[i] for i in order], max_length, batch_size, prompt)
    out = torch.empty_like(emb)
    out[torch.as_tensor(order, device=emb.device)] = emb
    return out


class _Replicas:
    """Data-parallel encoding over several devices with one model replica per device (threaded)."""

    def __init__(self, model, devices: Sequence[Device]) -> None:
        import copy
        from concurrent.futures import ThreadPoolExecutor

        self.devices = [torch.device(d) for d in devices]
        self.models = [model]
        model.to(self.devices[0])
        for d in self.devices[1:]:
            m = copy.deepcopy(model).to(d)
            self.models.append(m)
        self.pool = ThreadPoolExecutor(max_workers=len(self.devices))

    def encode(self, texts: List[str], max_length: int, batch_size: int, prompt: Optional[str]) -> torch.Tensor:
        n = len(self.models)
        shards = [list(range(i, len(texts), n)) for i in range(n)]

        def run(i):
            idx = shards[i]
            if not idx:
                return None
            e = encode_sorted(self.models[i], [texts[j] for j in idx], max_length, batch_size, prompt)
            return e.to(self.devices[0])

        parts = list(self.pool.map(run, range(n)))
        out = torch.empty((len(texts), parts[0].shape[1]), dtype=parts[0].dtype, device=self.devices[0])
        for i, p in enumerate(parts):
            if p is not None:
                out[torch.as_tensor(shards[i], device=self.devices[0])] = p
        return out


def to_numpy(x) -> np.ndarray:
    return x.float().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
