"""Recursive Dense Retrieval (RDR).

One instruction-conditioned encoder, recursive search (tree, chain, beam, breadth, …) and a global
readout over all search states. See https://anonymous.4open.science/w/Recursive-Dense-Retrieval/ for the docs.
"""

from . import evaluation, instructions
from .encoder import Encoder, load_sentence_transformer
from .index import DenseIndex
from .prompts import ContextLayout, PromptConfig, StateRenderer
from .readout import (RRF, Assembly, Borda, ChainOrder, CombMNZ, GlobalReadout, MaxOverStates, PathProduct,
                      PostOrderRRF, ProbabilityChain, Readout, RootInterpolation, RootReadout, SelectedSet,
                      SequenceSum, StateReadout, TerminalState, ZAgreement, recovery_fusion, recovery_view_readout)
from .registry import make_readout, make_search
from .retriever import RecursiveRetriever
from .search import (BeamSearch, BreadthSearch, ChainSearch, DenseSearch, DocAsQuerySearch, JointFeedback,
                     SearchStrategy, TreeSearch)
from .types import Ranking, State, Traversal

__version__ = "0.1.0"

__all__ = [
    "RecursiveRetriever", "Encoder", "load_sentence_transformer", "DenseIndex", "PromptConfig", "StateRenderer",
    "ContextLayout", "instructions", "SearchStrategy", "DenseSearch", "TreeSearch", "ChainSearch", "BreadthSearch",
    "BeamSearch", "DocAsQuerySearch", "JointFeedback", "Readout", "GlobalReadout", "RootReadout", "PathProduct",
    "SequenceSum", "ProbabilityChain", "ChainOrder", "SelectedSet", "recovery_fusion", "recovery_view_readout",
    "TerminalState", "StateReadout", "RRF", "PostOrderRRF", "Borda", "CombMNZ",
    "MaxOverStates", "ZAgreement", "RootInterpolation", "Assembly", "make_search", "make_readout", "Ranking",
    "State", "Traversal", "__version__",
]
