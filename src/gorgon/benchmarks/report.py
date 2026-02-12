from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


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
    mean_accepted_length: Optional[float] = None
    per_head_acceptance: Optional[List[float]] = None
    tree_utilization: Optional[float] = None
    time_breakdown: Optional[Dict[str, float]] = None


@dataclass
class BenchmarkReport:
    timestamp: str
    system: SystemInfo
    config: BenchmarkConfigSummary
    baseline: BenchmarkRun
    speculative: BenchmarkRun
    speedup: float


def _run_to_dict(run: BenchmarkRun) -> dict:
    d: dict = {
        "tokens": run.token_count,
        "elapsed_s": run.elapsed_s,
        "tokens_per_second": run.tokens_per_second,
        "acceptance_rate": run.acceptance_rate,
    }
    if run.mean_accepted_length is not None:
        d["mean_accepted_length"] = run.mean_accepted_length
    if run.per_head_acceptance is not None:
        d["per_head_acceptance"] = run.per_head_acceptance
    if run.tree_utilization is not None:
        d["tree_utilization"] = run.tree_utilization
    if run.time_breakdown is not None:
        d["time_breakdown"] = run.time_breakdown
    return d


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
        "baseline": _run_to_dict(report.baseline),
        "speculative": _run_to_dict(report.speculative),
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
    ]

    if report.speculative.mean_accepted_length is not None:
        lines.append(
            f"- Mean accepted length (tau): {report.speculative.mean_accepted_length:.2f}"
        )
    if report.speculative.tree_utilization is not None:
        lines.append(
            f"- Tree utilization: {report.speculative.tree_utilization:.2%}"
        )
    if report.speculative.per_head_acceptance is not None:
        rates_str = ", ".join(
            f"H{i}={r:.2%}" for i, r in enumerate(report.speculative.per_head_acceptance)
        )
        lines.append(f"- Per-head acceptance: {rates_str}")

    lines += [
        "",
        "### Speedup",
        f"- Speculative vs baseline: {report.speedup:.2f}x",
    ]
    return "\n".join(lines)
