from __future__ import annotations

from datetime import datetime


def current_date() -> str:
    return datetime.today().isoformat().split("T")[0]
