import json
from pathlib import Path

from gorgon.benchmarks.jsonl import append_jsonl


def test_append_jsonl_writes_line(tmp_path: Path) -> None:
    target = tmp_path / "bench.jsonl"
    payload = {"name": "run-1", "value": 3}

    append_jsonl(target, payload)

    line = target.read_text().strip()
    assert json.loads(line) == payload
