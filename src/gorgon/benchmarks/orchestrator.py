from __future__ import annotations

from gorgon.benchmarks.report import BenchmarkReport, render_phase5_section


def format_phase5_report(report: BenchmarkReport) -> str:
    return render_phase5_section(report)
