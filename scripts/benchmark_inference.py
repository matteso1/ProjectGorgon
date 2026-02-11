from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from gorgon.benchmarks.cli import parse_args
from gorgon.benchmarks.config import BenchmarkConfig
from gorgon.benchmarks.dry_run import make_dry_report
from gorgon.benchmarks.entrypoint import build_config_summary
from gorgon.benchmarks.jsonl import append_jsonl
from gorgon.benchmarks.pipeline import make_report
from gorgon.benchmarks.runner import ensure_hf_token, infer_device, select_head_train_device
from gorgon.benchmarks.runner_core import run_benchmark_trials
from gorgon.benchmarks.system import gather_system_info
from gorgon.benchmarks.time_utils import current_date
from gorgon.benchmarks.report import report_to_dict
from gorgon.models.backbone import load_backbone_4bit
from gorgon.training.medusa_distill import distill_heads_last_token


def main() -> None:
    args = parse_args(sys.argv[1:])
    config = BenchmarkConfig(
        model_name=args.model_name,
        max_new_tokens=args.max_new_tokens,
        prompt_max_length=args.prompt_max_length,
        warmup_steps=args.warmup_steps,
        num_trials=args.num_trials,
        seed=args.seed,
        num_medusa_heads=args.num_medusa_heads,
        top_k=args.top_k,
    )
    system = gather_system_info()
    device = infer_device(args.device)

    timestamp = current_date() if not args.dry_run else "dry-run"
    if args.dry_run:
        report = make_dry_report(system, config, timestamp=timestamp)
    else:
        hf_token = ensure_hf_token()
        model, tokenizer, heads = load_backbone_4bit(
            model_name=config.model_name,
            num_heads=config.num_medusa_heads,
            token=hf_token,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        if args.head_train_steps > 0:
            train_device = select_head_train_device(device)
            distill_heads_last_token(
                backbone=model,
                heads=heads,
                tokenizer=tokenizer,
                prompts=config.prompts,
                steps=args.head_train_steps,
                lr=args.head_train_lr,
                device=train_device,
            )
        baseline, speculative = run_benchmark_trials(
            model=model,
            tokenizer=tokenizer,
            heads=heads,
            config=config,
            device=device,
        )
        report = make_report(
            timestamp=timestamp,
            system=system,
            config=build_config_summary(config),
            baseline=baseline,
            speculative=speculative,
        )

    jsonl_path = Path(args.report_path).with_suffix(".jsonl")
    append_jsonl(jsonl_path, report_to_dict(report))


if __name__ == "__main__":
    main()
