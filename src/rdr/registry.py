"""Name -> component registry, so configurations can be given as strings or dicts.

``make_search("tree")``, ``make_search({"name": "beam", "width": 13})``,
``make_readout("global")``, ``make_readout({"name": "rrf", "k": 60, "level_balanced": True})``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Union

from .readout.base import Readout
from .readout.fusion import RRF, Borda, CombMNZ, MaxOverStates, PostOrderRRF
from .readout.global_readout import GlobalReadout, RootReadout
from .readout.model_based import Assembly, RootInterpolation, StateReadout
from .readout.realized import ChainOrder, PathProduct, ProbabilityChain, SelectedSet, SequenceSum, TerminalState
from .readout.views import MinMaxCombMNZ, ZAgreement, recovery_fusion, recovery_view_readout
from .search.base import SearchStrategy
from .search.strategies import (BeamSearch, BreadthSearch, ChainSearch, DenseSearch, DocAsQuerySearch,
                                JointFeedback, TreeSearch)

SEARCHES: Dict[str, Callable[..., SearchStrategy]] = {
    "dense": DenseSearch,
    "root": DenseSearch,
    "tree": TreeSearch,
    "chain": ChainSearch,
    "breadth": BreadthSearch,
    "beam": BeamSearch,
    "doc_as_query": DocAsQuerySearch,
    "joint_feedback": JointFeedback,
}

READOUTS: Dict[str, Callable[..., Readout]] = {
    "global": GlobalReadout,
    "combsum": lambda **kw: GlobalReadout(level_balanced=False, **kw),
    "root": RootReadout,
    "path_product": PathProduct,
    "sequence_sum": SequenceSum,
    "probability_chain": ProbabilityChain,
    "chain_order": ChainOrder,
    "selected_set": SelectedSet,
    "terminal": TerminalState,
    "state": StateReadout,
    "rrf": RRF,
    "rrf_level": lambda **kw: RRF(level_balanced=True, **kw),
    "rrf_postorder": PostOrderRRF,
    "borda": Borda,
    "combmnz": CombMNZ,
    "max": MaxOverStates,
    "root_interpolation": RootInterpolation,
    "assembly": Assembly,
    "z_agreement": lambda **kw: recovery_fusion(**kw),
    "recovery_fusion": lambda **kw: recovery_fusion(**kw),
    "recovery": lambda **kw: recovery_view_readout(**kw),
    "path_recovery_fusion": lambda **kw: ZAgreement(PathProduct(), recovery_view_readout(), group_reduce=("max", "last"),
                                                    **kw),
    "minmax_combmnz": lambda **kw: MinMaxCombMNZ(GlobalReadout(), recovery_view_readout(), **kw),
}


def _make(table: Dict[str, Callable[..., Any]], spec, kind: str):
    if spec is None or not isinstance(spec, (str, dict)):
        return spec
    if isinstance(spec, str):
        name, kwargs = spec, {}
    else:
        kwargs = dict(spec)
        name = kwargs.pop("name")
    key = name.lower().replace("-", "_")
    if key not in table:
        raise KeyError(f"unknown {kind} {name!r}; available: {sorted(table)}")
    return table[key](**kwargs)


def make_search(spec: Union[str, dict, SearchStrategy, None]) -> SearchStrategy:
    """A search strategy from a name or a dict, e.g. ``make_search({"name": "beam", "width": 13})``.

    Args:
        spec: A name from :data:`SEARCHES`, a dict with ``name`` plus constructor arguments, or an
            instance (returned unchanged).
    """
    return _make(SEARCHES, spec, "search")


def make_readout(spec: Union[str, dict, Readout, None]) -> Readout:
    """A readout from a name or a dict, e.g. ``make_readout({"name": "rrf", "k": 60})``.

    Args:
        spec: A name from :data:`READOUTS`, a dict with ``name`` plus constructor arguments, or an
            instance (returned unchanged).
    """
    return _make(READOUTS, spec, "readout")
