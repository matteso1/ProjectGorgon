from __future__ import annotations

from gorgon.benchmarks.config import BenchmarkConfig
from gorgon.benchmarks.entrypoint import build_config_summary
from gorgon.benchmarks.pipeline import make_report
from gorgon.benchmarks.report import BenchmarkRun, SystemInfo


def make_dry_report(
    system: SystemInfo, config: BenchmarkConfig, timestamp: str
) -> "BenchmarkReport":
    baseline = BenchmarkRun(
        name="baseline",
        token_count=100,
        elapsed_s=10.0,
        tokens_per_second=10.0,
        acceptance_rate=None,
    )
    speculative = BenchmarkRun(
        name="speculative",
        token_count=100,
        elapsed_s=5.0,
        tokens_per_second=20.0,
        acceptance_rate=0.5,
    )
    return make_report(
        timestamp=timestamp,
        system=system,
        config=build_config_summary(config),
        baseline=baseline,
        speculative=speculative,
    )
