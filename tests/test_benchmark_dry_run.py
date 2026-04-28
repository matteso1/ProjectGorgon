from gorgon.benchmarks.config import BenchmarkConfig
from gorgon.benchmarks.dry_run import make_dry_report
from gorgon.benchmarks.report import SystemInfo


def test_make_dry_report() -> None:
    system = SystemInfo(
        platform="Linux",
        torch_version="2.10.0",
        cuda_version="12.8",
        gpu="RTX 4090",
    )
    config = BenchmarkConfig()

    report = make_dry_report(system, config, timestamp="2026-01-30")

    assert report.timestamp == "2026-01-30"
    assert report.baseline.tokens_per_second == 10.0
    assert report.speculative.tokens_per_second == 20.0
    assert report.speculative.acceptance_rate == 0.5
    assert report.speedup == 2.0
