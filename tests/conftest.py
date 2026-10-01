import hashlib

import numpy as np
import pytest
import torch


class FakeEncoder:
    """Deterministic bag-of-words hashing encoder: similar texts get similar vectors."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.calls = 0
        self.query_max_length = 4096
        self.document_max_length = 512

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in text.lower().replace("\n", " ").split():
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 + (h >> 8) % 3
        n = np.linalg.norm(v)
        return v / n if n else v

    def encode_queries(self, texts, batch_size=None, show_progress_bar=False):
        self.calls += len(texts)
        return torch.tensor(np.stack([self._vec(t) for t in texts]))

    def encode_documents(self, texts, batch_size=None, show_progress_bar=False):
        return torch.tensor(np.stack([self._vec(t) for t in texts]))

    def truncate(self, text, n):
        return " ".join(text.split()[:n]) if n else text


CORPUS = [
    "Clio Goldsmith is a French actress born 16 June 1957 daughter of Teddy Goldsmith",
    "Edward Goldsmith known as Teddy Goldsmith was a British environmentalist brother of James Goldsmith",
    "James Goldsmith was a financier and uncle of Clio Goldsmith",
    "The Ecologist magazine was founded by Edward Goldsmith",
    "Paris is the capital of France",
    "The Eiffel Tower is in Paris",
    "Gradient descent optimizes differentiable functions",
    "A binary search tree keeps keys sorted",
    "The uncle of a person is the brother of a parent",
    "Actresses in French cinema of the 1970s",
] + [f"filler document number {i} about topic {i % 7}" for i in range(40)]


@pytest.fixture
def fake_retriever():
    from rdr import RecursiveRetriever

    enc = FakeEncoder()
    r = RecursiveRetriever(enc, prompts="multi-hop", query_batch_size=2, inflight=1)
    index = r.build_index(CORPUS, show_progress_bar=False)
    return r, index, enc
