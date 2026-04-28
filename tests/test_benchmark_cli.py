from gorgon.benchmarks.cli import parse_args


def test_parse_args_defaults() -> None:
    args = parse_args([])

    assert args.device == "auto"
    assert args.model_name == "meta-llama/Meta-Llama-3-8B-Instruct"
    assert args.max_new_tokens == 128
    assert args.prompt_max_length == 256
    assert args.warmup_steps == 2
    assert args.num_trials == 5
    assert args.seed == 0
    assert args.num_medusa_heads == 4
    assert args.top_k == 4
    assert args.dry_run is False
    assert args.head_train_steps == 0
    assert args.head_train_lr == 0.001
    assert args.report_path.endswith("reports/benchmark.md")


def test_parse_args_dry_run_flag() -> None:
    args = parse_args(["--dry-run"])
    assert args.dry_run is True


def test_parse_args_head_train_overrides() -> None:
    args = parse_args(["--head-train-steps", "5", "--head-train-lr", "0.01"])
    assert args.head_train_steps == 5
    assert args.head_train_lr == 0.01
