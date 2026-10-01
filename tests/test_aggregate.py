"""reproduce/aggregate.py: shard merging and the paper's suite conventions."""
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("yaml")
spec = importlib.util.spec_from_file_location("aggregate", Path(__file__).resolve().parents[1] / "reproduce" / "aggregate.py")
agg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agg)


def _write(tmp, name, suite, task, shard, n, qids, values, parent=None, groups=None, key="tree/global"):
    d = {"suite": suite, "task": task, "parent": parent, "qids": qids, "groups": groups, "shard": shard,
         "num_shards": n, "runs": {key: {"values": values, "seconds": 0.0}}}
    (tmp / f"{name}-{suite}-{task}-sh{shard}of{n}.json").write_text(json.dumps(d))


def test_incomplete_shards_are_skipped(tmp_path):
    _write(tmp_path, "x", "musique", "musique", 0, 2, ["q1"], [[1.0, 1.0, 1.0]])
    merged, incomplete = agg.load(str(tmp_path), "x")
    assert not merged and incomplete


def test_suite_conventions(tmp_path):
    _write(tmp_path, "x", "coir", "CodeSearchNet-go", 0, 1, ["a", "b"], [1.0, 0.0], parent="codesearchnet")
    _write(tmp_path, "x", "coir", "CodeSearchNet-java", 0, 1, ["c"], [1.0], parent="codesearchnet")
    _write(tmp_path, "x", "coir", "cosqa", 0, 1, ["d"], [0.0], parent="cosqa")
    _write(tmp_path, "x", "toolret", "ToolRet", 0, 1, ["e", "f", "g"], [1.0, 0.0, 0.0], groups=["g1", "g2", "g2"])
    merged, _ = agg.load(str(tmp_path), "x")
    tasks = lambda s: [t for (su, _), t in merged.items() if su == s]  # noqa: E731
    # CodeSearchNet = mean(0.5, 1.0) = 0.75; cosqa = 0 -> macro 37.5
    assert agg.suite_score("coir", tasks("coir"), "tree/global", "ndcg@10") == pytest.approx(37.5)
    # groups: g1 = 1.0, g2 = 0.0 -> 50
    assert agg.suite_score("toolret", tasks("toolret"), "tree/global", "ndcg@10") == pytest.approx(50.0)
