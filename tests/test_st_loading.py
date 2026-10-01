"""Loading a plain Hugging Face checkpoint wraps it as Transformer -> Pooling(lasttoken) -> Normalize with left
padding (the Qwen3-Embedding contract of RDR-8B), across sentence-transformers versions.

Needs a small local checkpoint: RDR_TEST_HF_MODEL=distilbert-base-uncased HF_HUB_OFFLINE=1 pytest tests/test_st_loading.py
"""
import os

import pytest

MODEL = os.environ.get("RDR_TEST_HF_MODEL")


@pytest.mark.skipif(not MODEL, reason="set RDR_TEST_HF_MODEL to a small local checkpoint without modules.json")
def test_plain_checkpoint_is_wrapped_with_lasttoken_pooling_and_left_padding():
    from rdr.encoder import has_sentence_transformers_config, load_sentence_transformer

    assert not has_sentence_transformers_config(MODEL)
    st = load_sentence_transformer(MODEL, device="cpu", torch_dtype="float32", max_seq_length=128)
    names = [type(m).__name__ for m in st]
    assert names == ["Transformer", "Pooling", "Normalize"]
    pooling = st[1]
    mode = getattr(pooling, "pooling_mode", None) or pooling.get_pooling_mode_str()
    assert "lasttoken" in str(mode)
    assert st.tokenizer.padding_side == "left"
