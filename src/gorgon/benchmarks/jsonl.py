from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    line = json.dumps(payload, sort_keys=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
