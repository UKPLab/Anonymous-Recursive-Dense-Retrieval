#!/usr/bin/env python3
"""Export the public API of `rdr` (signatures, defaults, docstrings, parameter descriptions) as JSON.

The website's API reference is generated from this file, so the documentation cannot drift from the code::

    python scripts/export_api.py > ../website/public/data/api.json
"""
import inspect
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import rdr  # noqa: E402
from rdr import instructions as I  # noqa: E402
from rdr import prompts as P  # noqa: E402
from rdr import readout as R  # noqa: E402
from rdr import search as S  # noqa: E402
from rdr.evaluation import benchmarks as BM  # noqa: E402

GROUPS = [
    ("Model & prompts", [rdr.RecursiveRetriever, rdr.Encoder, rdr.load_sentence_transformer, P.PromptConfig,
                         P.ContextLayout, P.StateRenderer, I.get, I.root_for]),
    ("Configuration by name", [rdr.make_search, rdr.make_readout]),
    ("Index", [rdr.DenseIndex]),
    ("Search strategies", [S.DenseSearch, S.TreeSearch, S.ChainSearch, S.BreadthSearch, S.BeamSearch,
                           S.DocAsQuerySearch, S.JointFeedback]),
    ("Readout: global", [R.GlobalReadout, R.RootReadout]),
    ("Readout: realized paths", [R.PathProduct, R.SequenceSum, R.ProbabilityChain, R.ChainOrder, R.TerminalState,
                                 R.SelectedSet]),
    ("Readout: single state & model-based", [R.StateReadout, R.RootInterpolation, R.Assembly]),
    ("Readout: fusion over states", [R.RRF, R.PostOrderRRF, R.Borda, R.CombMNZ, R.MaxOverStates]),
    ("Readout: views", [R.ZAgreement, R.recovery_fusion, R.recovery_view_readout]),
    ("Evaluation", [rdr.evaluation.recall_at_k, rdr.evaluation.ndcg_at_k, rdr.evaluation.alpha_ndcg_at_k,
                    rdr.evaluation.macro]),
    ("Benchmarks", [BM.Task, BM.load_nanobeir, BM.load_bright, BM.load_coir, BM.load_toolret, BM.load_topiocqa,
                    BM.load_multihop]),
]

_ARG_RE = re.compile(r"^\s{0,12}\*{0,2}(\w+)(?:\s*\(([^)]*)\))?:\s*(.*)$")


def parse_docstring(doc: str):
    """Split a Google-style docstring into summary, body and ``Args``/``Attributes`` descriptions."""
    doc = inspect.cleandoc(doc or "")
    params, lines, section, current = {}, [], None, None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped in ("Args:", "Arguments:", "Attributes:", "Parameters:"):
            section, current = "args", None
            continue
        if stripped in ("Returns:", "Example:", "Examples:", "Example::", "Raises:"):
            section, current = None, None
            lines.append(line)
            continue
        if section == "args" and stripped and not line[:1].isspace():
            section, current = None, None  # a dedented line ends the section
        if section == "args":
            m = _ARG_RE.match(line)
            if m and (len(line) - len(line.lstrip())) <= 4:
                current = m.group(1)
                params[current] = m.group(3).strip()
                continue
            if current and stripped:
                params[current] += " " + stripped
                continue
            if not stripped:
                continue
            section = None
        lines.append(line)
    text = "\n".join(lines).strip()
    summary = text.split("\n\n")[0].replace("\n", " ") if text else ""
    return summary, text, params


def _params(sig, pdesc):
    out = []
    for name, p in sig.parameters.items():
        if name in ("self", "cls") or p.kind in (p.VAR_KEYWORD,):
            continue
        default = None if p.default is inspect._empty else repr(p.default)
        ann = None if p.annotation is inspect._empty else (p.annotation if isinstance(p.annotation, str)
                                                              else getattr(p.annotation, "__name__", str(p.annotation)))
        out.append({"name": ("*" + name) if p.kind == p.VAR_POSITIONAL else name, "default": default,
                    "annotation": ann, "description": pdesc.get(name, "")})
    return out


def describe(obj):
    kind = "class" if inspect.isclass(obj) else "function"
    target = obj.__init__ if inspect.isclass(obj) else obj
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):
        sig = None
    summary, body, pdesc = parse_docstring(obj.__doc__)
    if inspect.isclass(obj) and obj.__init__.__doc__ and obj.__init__ is not object.__init__:
        _, _, extra = parse_docstring(obj.__init__.__doc__)
        pdesc = {**extra, **pdesc}
    params = _params(sig, pdesc) if sig is not None else []
    methods = []
    if inspect.isclass(obj):
        for mname in ("search", "traverse", "read", "build_index", "encode_queries", "encode_documents", "render",
                      "preset", "build", "positions", "save", "load"):
            m = obj.__dict__.get(mname)
            if m is None:
                continue
            f = m.__func__ if isinstance(m, (staticmethod, classmethod)) else m
            try:
                fsig = inspect.signature(f)
                msig = str(fsig)
            except (TypeError, ValueError):
                fsig, msig = None, "(...)"
            ms, mb, mp = parse_docstring(f.__doc__)
            ret = None
            if fsig is not None and fsig.return_annotation is not inspect._empty:
                ret = fsig.return_annotation if isinstance(fsig.return_annotation, str) else str(fsig.return_annotation)
            methods.append({"name": mname, "signature": msig, "summary": ms, "doc": mb,
                            "params": _params(fsig, mp) if fsig is not None else [], "returns": ret})
    src = inspect.getsourcefile(obj) or ""
    rel = src.split("/src/")[-1] if "/src/" in src else src
    return {"name": obj.__name__, "kind": kind, "module": obj.__module__, "qualname": f"{obj.__module__}.{obj.__name__}",
            "signature": f"{obj.__name__}{sig}" if sig is not None else obj.__name__, "summary": summary, "doc": body,
            "params": params, "methods": methods, "source": rel,
            "name_key": getattr(obj, "name", None) if inspect.isclass(obj) else None,
            "realized": bool(getattr(obj, "realized", False)) if inspect.isclass(obj) else False}


def main():
    out = {"package": "rdr", "version": rdr.__version__, "groups": []}
    for title, objs in GROUPS:
        out["groups"].append({"title": title, "items": [describe(o) for o in objs]})
    from rdr import registry as REG

    aliases = {"combsum": "GlobalReadout(level_balanced=False)", "rrf_level": "RRF(level_balanced=True)",
               "z_agreement": "recovery_fusion()", "recovery_fusion": "recovery_fusion()",
               "recovery": "recovery_view_readout()",
               "path_recovery_fusion": 'ZAgreement(PathProduct(), recovery_view_readout(), group_reduce=("max", "last"))',
               "minmax_combmnz": "MinMaxCombMNZ(GlobalReadout(), recovery_view_readout())"}

    def _target(key, f):
        if key in aliases and getattr(f, "__name__", "") == "<lambda>":
            return aliases[key]
        return getattr(f, "__name__", None)

    out["registry"] = {"searches": {k: _target(k, v) for k, v in REG.SEARCHES.items()},
                       "readouts": {k: _target(k, v) for k, v in REG.READOUTS.items()}}
    out["instructions"] = {k: v for k, v in I.INSTRUCTIONS.items()}
    out["instructions_roots"] = {"bright": I.BRIGHT_ROOTS, "nanobeir": I.NANOBEIR_ROOTS, "coir": I.COIR_ROOTS}
    out["prompt_presets"] = {}
    for name in ("multi-hop", "single-hop", "single-hop-continuation", "bright-pro"):
        cfg = P.PromptConfig.preset(name)
        rnd = P.StateRenderer(cfg)
        root = cfg.root or "{benchmark instruction}"
        out["prompt_presets"][name] = {
            "primary_role": cfg.primary_role, "wrap": cfg.wrap, "strip": cfg.strip,
            "observation_max_tokens": cfg.observation_max_tokens,
            "examples": {
                "root": rnd.render("{query}", [], "root", instruction=root),
                "level2": rnd.render("{query}", ["{document 1}"], cfg.primary_role, instruction=root),
                "level3": rnd.render("{query}", ["{document 1}", "{document 2}"], cfg.primary_role, instruction=root),
                "recovery_view": rnd.render("{query}", ["{document 1}", "{document 2}"], "recovery_view",
                                            instruction=root) if name == "multi-hop" else None,
            },
        }
    json.dump(out, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
