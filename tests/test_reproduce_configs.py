"""Every configuration in reproduce/configs builds, and every expected value names a run/readout of it."""
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

from rdr import make_readout, make_search  # noqa: E402

CONFIGS = sorted((Path(__file__).resolve().parents[1] / "reproduce" / "configs").glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.stem for p in CONFIGS])
def test_config_builds(path):
    cfg = yaml.safe_load(path.read_text())
    runs = {r["name"]: r for r in cfg["runs"]}
    for r in runs.values():
        make_search(r["search"])
        for spec in r["readouts"].values():
            make_readout(spec)
    for key in cfg["expected"]:
        if key == "metric":
            continue
        run, readout = key.split("/")
        assert run in runs and readout in runs[run]["readouts"], f"{path.name}: {key} has no run/readout"
