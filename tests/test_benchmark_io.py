from pathlib import Path

from gorgon.benchmarks.io import append_report_section


def test_append_report_section(tmp_path: Path) -> None:
    target = tmp_path / "bench.md"
    target.write_text("# Benchmark Results\n\n## Existing\n- Old: 1\n")

    append_report_section(target, "## Phase 5: End-to-end Llama-3-8B (2026-01-30)\n- New: 2\n")

    content = target.read_text()
    assert content.startswith("# Benchmark Results")
    assert "## Existing" in content
    assert "## Phase 5: End-to-end Llama-3-8B (2026-01-30)" in content
