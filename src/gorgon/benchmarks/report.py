from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SystemInfo:
    platform: str
    torch_version: str
    cuda_version: str
    gpu: str


@dataclass
class BenchmarkConfigSummary:
    model_name: str
    max_new_tokens: int
    prompt_max_length: int
    warmup_steps: int
    num_trials: int
    seed: int
    num_medusa_heads: int
    top_k: int


@dataclass
class BenchmarkRun:
    name: str
    token_count: int
    elapsed_s: float
    tokens_per_second: float
    acceptance_rate: Optional[float]


@dataclass
class BenchmarkReport:
    timestamp: str
    system: SystemInfo
    config: BenchmarkConfigSummary
    baseline: BenchmarkRun
    speculative: BenchmarkRun
    speedup: float


def report_to_dict(report: BenchmarkReport) -> dict:
    return {
        "timestamp": report.timestamp,
        "system": {
            "platform": report.system.platform,
            "torch": report.system.torch_version,
            "cuda": report.system.cuda_version,
            "gpu": report.system.gpu,
        },
        "config": {
            "model": report.config.model_name,
            "max_new_tokens": report.config.max_new_tokens,
            "prompt_max_length": report.config.prompt_max_length,
            "warmup_steps": report.config.warmup_steps,
            "num_trials": report.config.num_trials,
            "seed": report.config.seed,
            "num_medusa_heads": report.config.num_medusa_heads,
            "top_k": report.config.top_k,
        },
        "baseline": {
            "tokens": report.baseline.token_count,
            "elapsed_s": report.baseline.elapsed_s,
            "tokens_per_second": report.baseline.tokens_per_second,
            "acceptance_rate": report.baseline.acceptance_rate,
        },
        "speculative": {
            "tokens": report.speculative.token_count,
            "elapsed_s": report.speculative.elapsed_s,
            "tokens_per_second": report.speculative.tokens_per_second,
            "acceptance_rate": report.speculative.acceptance_rate,
        },
        "speedup": report.speedup,
    }


def render_phase5_section(report: BenchmarkReport) -> str:
    acceptance = (
        f"{report.speculative.acceptance_rate:.2f}"
        if report.speculative.acceptance_rate is not None
        else "n/a"
    )

    lines = [
        f"## Phase 5: End-to-end Llama-3-8B ({report.timestamp})",
        "",
        "### System",
        f"- Platform: {report.system.platform}",
        f"- Torch: {report.system.torch_version}",
        f"- CUDA: {report.system.cuda_version}",
        f"- GPU: {report.system.gpu}",
        "",
        "### Config",
        f"- Model: {report.config.model_name}",
        f"- Max new tokens: {report.config.max_new_tokens}",
        f"- Prompt max length: {report.config.prompt_max_length}",
        f"- Trials: {report.config.num_trials}",
        f"- Warmup: {report.config.warmup_steps}",
        f"- Seed: {report.config.seed}",
        f"- Medusa heads: {report.config.num_medusa_heads}",
        f"- Top-k: {report.config.top_k}",
        "",
        "### Baseline (autoregressive)",
        f"- Tokens: {report.baseline.token_count}",
        f"- Elapsed (s): {report.baseline.elapsed_s:.2f}",
        f"- Tokens/s: {report.baseline.tokens_per_second:.2f}",
        "- Acceptance rate: n/a",
        "",
        "### Speculative (Medusa)",
        f"- Tokens: {report.speculative.token_count}",
        f"- Elapsed (s): {report.speculative.elapsed_s:.2f}",
        f"- Tokens/s: {report.speculative.tokens_per_second:.2f}",
        f"- Acceptance rate: {acceptance}",
        "",
        "### Speedup",
        f"- Speculative vs baseline: {report.speedup:.2f}x",
    ]
    return "\n".join(lines)
