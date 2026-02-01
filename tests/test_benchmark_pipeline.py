from gorgon.benchmarks.pipeline import make_report
from gorgon.benchmarks.report import BenchmarkConfigSummary, BenchmarkRun, SystemInfo


def test_make_report_computes_speedup() -> None:
    system = SystemInfo(
        platform="Linux",
        torch_version="2.10.0",
        cuda_version="12.8",
        gpu="RTX 4090",
    )
    config = BenchmarkConfigSummary(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        max_new_tokens=128,
        prompt_max_length=256,
        warmup_steps=2,
        num_trials=5,
        seed=0,
        num_medusa_heads=4,
        top_k=4,
    )
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

    report = make_report("2026-01-30", system, config, baseline, speculative)

    assert report.timestamp == "2026-01-30"
    assert report.speedup == 2.0
