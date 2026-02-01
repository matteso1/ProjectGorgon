from __future__ import annotations

from gorgon.benchmarks.metrics import compute_speedup
from gorgon.benchmarks.report import (
    BenchmarkConfigSummary,
    BenchmarkReport,
    BenchmarkRun,
    SystemInfo,
)


def make_report(
    timestamp: str,
    system: SystemInfo,
    config: BenchmarkConfigSummary,
    baseline: BenchmarkRun,
    speculative: BenchmarkRun,
) -> BenchmarkReport:
    speedup = compute_speedup(baseline.tokens_per_second, speculative.tokens_per_second)
    return BenchmarkReport(
        timestamp=timestamp,
        system=system,
        config=config,
        baseline=baseline,
        speculative=speculative,
        speedup=speedup,
    )
