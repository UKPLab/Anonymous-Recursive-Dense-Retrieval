import numpy as np
import pytest

import rdr
from rdr import (BeamSearch, BreadthSearch, ChainSearch, DenseSearch, DocAsQuerySearch, GlobalReadout,
                 JointFeedback, PathProduct, TreeSearch)
from rdr.readout import (RRF, Assembly, Borda, ChainOrder, CombMNZ, MaxOverStates, PostOrderRRF,
                         ProbabilityChain, RootInterpolation, SequenceSum, StateReadout, TerminalState, ZAgreement,
                         recovery_fusion, recovery_view_readout)

QUERIES = ["Who is the uncle of Clio Goldsmith?", "Where is the Eiffel Tower?", "What does gradient descent do?"]


def test_tree_state_counts(fake_retriever):
    r, index, _ = fake_retriever
    travs = r.traverse(QUERIES, index, search=TreeSearch(4, 3, dedup="none"))
    assert all(t.num_states() == 40 for t in travs)
    assert all(len(t.proposals) == 3 + 9 + 27 + 81 for t in travs)  # the deepest selections are recorded too
    travs = r.traverse(QUERIES, index, search=TreeSearch(4, 3, dedup="level"))
    for t in travs:
        docs = [p.doc for p in t.proposals]
        assert len(docs) == len(set(docs))  # each document selected at most once
        assert t.num_states() <= 40


def test_all_searches_and_default_global(fake_retriever):
    r, index, _ = fake_retriever
    for s in [DenseSearch(), TreeSearch(4, 3), ChainSearch(4), BreadthSearch(8), BeamSearch(5, 4),
              DocAsQuerySearch(4, 3), JointFeedback(3)]:
        hits = r.search(QUERIES, index, search=s, top_k=5)
        assert len(hits) == 3 and all(len(h) == 5 for h in hits), s
        assert all(h[0]["score"] >= h[-1]["score"] for h in hits)


def test_beam_state_budget(fake_retriever):
    r, index, _ = fake_retriever
    for W in (5, 13):
        travs = r.traverse(QUERIES, index, search=BeamSearch(W, 4))
        assert all(t.num_states() <= 1 + 3 * W for t in travs)


@pytest.mark.parametrize("readout", [
    GlobalReadout(), GlobalReadout(level_balanced=False), GlobalReadout(include_root=False),
    GlobalReadout(restrict_to_selected=True, backfill_root=True), PathProduct(), PathProduct(backfill_root=True),
    SequenceSum(), ProbabilityChain(), ChainOrder("product"), TerminalState(), RRF(), RRF(level_balanced=True, k=1),
    Borda(), CombMNZ(), CombMNZ(level_balanced=False), MaxOverStates(), PostOrderRRF(), StateReadout(),
])
def test_readouts(fake_retriever, readout):
    r, index, _ = fake_retriever
    travs = r.traverse(QUERIES, index, search=TreeSearch(4, 3))
    for rk in r.read(travs, index, readout):
        assert len(rk) > 0
        assert np.all(np.diff(rk.scores) <= 1e-12)


def test_model_readouts_and_views(fake_retriever):
    r, index, _ = fake_retriever
    hits = r.search(QUERIES, index, search=ChainSearch(4), readout=RootInterpolation(0.8), top_k=3)
    assert len(hits[0]) == 3
    hits = r.search(QUERIES, index, readout=Assembly(pool=20, n=5), top_k=5)
    assert len(hits[0]) == 5
    hits, travs = r.search(QUERIES, index, readout=recovery_fusion(), top_k=5, return_traversals=True)
    assert len(hits[0]) == 5
    t = travs[0]
    assert len(t.views["recovery"]) == len(t.proposals)  # one recovery state per realized selection
    hits = r.search(QUERIES, index, readout=recovery_view_readout(), top_k=5)
    assert len(hits[0]) == 5


def test_global_readout_math():
    from rdr.types import State, Traversal

    t = Traversal("q", 0)
    t.states.append(State(1, -1, (), ids=np.array([0, 1]), scores=np.array([0.9, 0.8], dtype=np.float32)))
    t.states.append(State(2, 0, (0,), ids=np.array([1, 2]), scores=np.array([0.7, 0.7], dtype=np.float32)))
    t.states.append(State(2, 0, (1,), ids=np.array([2, 3]), scores=np.array([0.9, 0.5], dtype=np.float32)))
    r = GlobalReadout(temperature=0.05)(t).as_dict()
    e = np.exp
    p_root = {0: e(0.9 / .05) / (e(0.9 / .05) + e(0.8 / .05)), 1: e(0.8 / .05) / (e(0.9 / .05) + e(0.8 / .05))}
    s2 = {1: 0.5, 2: 0.5}
    s3 = {2: e(0.9 / .05) / (e(0.9 / .05) + e(0.5 / .05)), 3: e(0.5 / .05) / (e(0.9 / .05) + e(0.5 / .05))}
    exp = {d: 0.5 * p_root.get(d, 0) + 0.5 * 0.5 * (s2.get(d, 0) + s3.get(d, 0)) for d in range(4)}
    for d in range(4):
        assert abs(r[d] - exp[d]) < 1e-6


def test_multihop_rendering_matches_producer():
    from rdr.prompts import PromptConfig, StateRenderer

    I = rdr.instructions
    mh = StateRenderer(PromptConfig.preset("multi-hop"))
    assert mh.render("Q?", [], "root") == "Instruct: " + I.MULTIHOP_ROOT + "\nQuery:Q?"
    assert mh.render("Q?", ["D1", "D2"], "continuation") == (
        "Instruct: " + I.CONTINUATION + "\nQuery:Q?\nDocument 1: D1\nDocument 2: D2")
    # recovery view: ancestors as documents, the last observation marked incorrect
    assert mh.render("Q?", ["D1"], "recovery_view") == (
        "Instruct: " + I.RECOVERY_MULTIHOP + "\nQuery:Q?\nRetrieved (incorrect):\nD1")
    assert mh.render("Q?", ["D1", "D2"], "recovery_view") == (
        "Instruct: " + I.RECOVERY_MULTIHOP + "\nQuery:Q?\nDocument 1: D1\nRetrieved (incorrect):\nD2")
    # joint feedback: root instruction, root's top documents in one state
    assert mh.render("Q?", ["A", "B", "C"], "feedback") == (
        "Instruct: " + I.MULTIHOP_ROOT + "\nQuery:Q?\nDocument 1: A\nDocument 2: B\nDocument 3: C")
    # assembly: the empty set is the root query
    assert mh.render("Q?", [], "assembly") == "Instruct: " + I.MULTIHOP_ROOT + "\nQuery:Q?"


def test_singlehop_rendering_matches_producer():
    from rdr.prompts import PromptConfig, StateRenderer

    I = rdr.instructions
    ins = "Represent this biology post for searching relevant passages: "
    sh = StateRenderer(PromptConfig.preset("single-hop"))
    assert sh.render(" Q? ", [], "root", instruction=ins) == (
        "Instruct: Represent this biology post for searching relevant passages:\nQuery: Q?")
    assert sh.render("Q?", [" D1 "], "recovery", instruction=ins) == (
        "Instruct: " + ins.strip() + "\nQuery: Q?\n" + I.RECOVERY + "\nRetrieved Context:\nDocument 1: D1")
    assert sh.render("Q?", ["D1", "D2"], "recovery", instruction=ins) == (
        "Instruct: " + ins.strip() + "\nQuery: Q?\n" + I.RECOVERY + "\nIncorrect document 1: D1\nIncorrect document 2: D2")
    assert sh.render("Q?", ["A", "B", "C"], "feedback", instruction=ins) == (
        "Instruct: " + ins.strip() + "\nQuery: Q?\nRetrieved Context:\nDocument 1: A\nDocument 2: B\nDocument 3: C")


def test_preset_instruction_override_merges():
    from rdr import PromptConfig
    cfg = PromptConfig.preset("single-hop", instructions={"recovery": "X"})
    assert cfg.instructions["recovery"] == "X"
    assert cfg.instructions["continuation"] == PromptConfig.preset("single-hop").instructions["continuation"]
