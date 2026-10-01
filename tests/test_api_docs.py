"""The website's API reference is generated from scripts/export_api.py: every documented item must be
importable under the path the docs print, and every parameter must have a description."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _export():
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_api.py")], capture_output=True, text=True,
                         check=True)
    return json.loads(out.stdout)


def _public_path(module: str, name: str):
    if module == "rdr.instructions":
        return "rdr.instructions", name
    if module == "rdr.evaluation.benchmarks":
        return "rdr.evaluation.benchmarks", name
    if module.startswith("rdr.evaluation"):
        return "rdr.evaluation", name
    return "rdr", name


def test_documented_items_are_importable_and_complete():
    api = _export()
    missing = []
    for group in api["groups"]:
        for it in group["items"]:
            mod, name = _public_path(it["module"], it["name"])
            assert hasattr(importlib.import_module(mod), name), f"{mod}.{name} is documented but not importable"
            missing += [f"{it['name']}({p['name']})" for p in it["params"] if not p["description"]]
            for m in it["methods"]:
                missing += [f"{it['name']}.{m['name']}({p['name']})" for p in m["params"] if not p["description"]]
    assert not missing, f"parameters without a description: {missing}"


def test_registry_names_resolve():
    from rdr import make_readout, make_search
    from rdr.registry import READOUTS, SEARCHES

    for name in SEARCHES:
        make_search(name)
    for name in READOUTS:
        make_readout(name)
