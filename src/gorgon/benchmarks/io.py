from __future__ import annotations

from pathlib import Path


def append_report_section(path: Path, section: str) -> None:
    content = path.read_text() if path.exists() else ""
    if content and not content.endswith("\n"):
        content += "\n"
    if content and not content.endswith("\n\n"):
        content += "\n"

    path.write_text(content + section)
