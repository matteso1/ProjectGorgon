from gorgon.benchmarks.report import (
    BenchmarkConfigSummary,
    BenchmarkReport,
    BenchmarkRun,
    SystemInfo,
    render_phase5_section,
)


def test_render_phase5_section() -> None:
    report = BenchmarkReport(
        timestamp="2026-01-30",
        system=SystemInfo(
            platform="Linux",
            torch_version="2.10.0",
            cuda_version="12.8",
            gpu="RTX 4090",
        ),
        config=BenchmarkConfigSummary(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=128,
            prompt_max_length=256,
            warmup_steps=2,
            num_trials=5,
            seed=0,
            num_medusa_heads=4,
            top_k=4,
        ),
        baseline=BenchmarkRun(
            name="baseline",
            token_count=200,
            elapsed_s=20.0,
            tokens_per_second=10.0,
            acceptance_rate=None,
        ),
        speculative=BenchmarkRun(
            name="speculative",
            token_count=200,
            elapsed_s=8.0,
            tokens_per_second=25.0,
            acceptance_rate=0.5,
        ),
        speedup=2.5,
    )

    section = render_phase5_section(report)

    assert "## Phase 5: End-to-end Llama-3-8B (2026-01-30)" in section
    assert "- Platform: Linux" in section
    assert "- Torch: 2.10.0" in section
    assert "- CUDA: 12.8" in section
    assert "- GPU: RTX 4090" in section

    assert "- Model: meta-llama/Meta-Llama-3-8B-Instruct" in section
    assert "- Max new tokens: 128" in section
    assert "- Prompt max length: 256" in section
    assert "- Trials: 5" in section
    assert "- Warmup: 2" in section
    assert "- Seed: 0" in section
    assert "- Medusa heads: 4" in section
    assert "- Top-k: 4" in section

    assert "### Baseline (autoregressive)" in section
    assert "- Tokens: 200" in section
    assert "- Elapsed (s): 20.00" in section
    assert "- Tokens/s: 10.00" in section
    assert "- Acceptance rate: n/a" in section

    assert "### Speculative (Medusa)" in section
    assert "- Tokens: 200" in section
    assert "- Elapsed (s): 8.00" in section
    assert "- Tokens/s: 25.00" in section
    assert "- Acceptance rate: 0.50" in section

    assert "### Speedup" in section
    assert "- Speculative vs baseline: 2.50x" in section
