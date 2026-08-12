#!/usr/bin/env python3
"""Run the reproducible evaluator training, comparison, and benchmark pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "data" / "lichess_evaluations_4m.csv"
DEFAULT_MODELS = ROOT_DIR / "models"
STAGES = ("ridge", "neural", "compare", "benchmark")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Ridge/MLP training, comparison, and strength benchmarking."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        default=list(STAGES),
        help="pipeline stages to run in order (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-cp", type=_positive_int, default=1500)
    parser.add_argument("--ridge-alpha", type=_positive_float, default=10.0)
    parser.add_argument("--epochs", type=_positive_int, default=20)
    parser.add_argument("--batch-size", type=_positive_int, default=2048)
    parser.add_argument("--learning-rate", type=_positive_float, default=1e-3)
    parser.add_argument("--expected-positions", type=_positive_int, default=4_000_000)
    parser.add_argument("--benchmark-depth", type=_positive_int, default=5)
    parser.add_argument("--benchmark-time-limit", type=_positive_float, default=0.1)
    parser.add_argument("--benchmark-max-plies", type=_positive_int, default=100)
    parser.add_argument("--benchmark-openings", type=_positive_int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without running them",
    )
    return parser


def build_commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    input_path = args.input.resolve()
    models_dir = args.models_dir.resolve()
    ridge_model = models_dir / "learned_evaluator.json"
    neural_model = models_dir / "neural_evaluator.pt"
    comparison = models_dir / "evaluator_comparison.json"
    benchmark = models_dir / "strength_benchmark.json"
    overwrite = ["--overwrite"] if args.overwrite else []

    commands = {
        "ridge": [
            sys.executable,
            str(ROOT_DIR / "scripts" / "train_evaluator.py"),
            "--input",
            str(input_path),
            "--output",
            str(ridge_model),
            "--alpha",
            str(args.ridge_alpha),
            "--clip-cp",
            str(args.clip_cp),
            "--seed",
            str(args.seed),
            *overwrite,
        ],
        "neural": [
            sys.executable,
            str(ROOT_DIR / "scripts" / "train_neural_eval.py"),
            "--input",
            str(input_path),
            "--output",
            str(neural_model),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--lr",
            str(args.learning_rate),
            "--clip-cp",
            str(args.clip_cp),
            "--seed",
            str(args.seed),
            *overwrite,
        ],
        "compare": [
            sys.executable,
            str(ROOT_DIR / "scripts" / "compare_evaluators.py"),
            "--input",
            str(input_path),
            "--ridge-model",
            str(ridge_model),
            "--neural-model",
            str(neural_model),
            "--output",
            str(comparison),
            "--clip-cp",
            str(args.clip_cp),
            "--seed",
            str(args.seed),
            *overwrite,
        ],
        "benchmark": [
            sys.executable,
            str(ROOT_DIR / "scripts" / "benchmark_evaluators.py"),
            "--first",
            "ridge",
            "--second",
            "neural",
            "--output",
            str(benchmark),
            "--depth",
            str(args.benchmark_depth),
            "--time-limit",
            str(args.benchmark_time_limit),
            "--max-plies",
            str(args.benchmark_max_plies),
            "--openings",
            str(args.benchmark_openings),
            "--expected-positions",
            str(args.expected_positions),
            *overwrite,
        ],
    }
    return [(stage, commands[stage]) for stage in args.stages]


def main() -> None:
    args = build_parser().parse_args()
    stages_requiring_data = {"ridge", "neural", "compare"}
    needs_training_data = bool(stages_requiring_data.intersection(args.stages))
    if needs_training_data and not args.input.is_file() and not args.dry_run:
        raise SystemExit(f"training CSV not found: {args.input}")

    for stage, command in build_commands(args):
        print(f"\n[{stage}] {' '.join(command)}", flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT_DIR, check=True)


if __name__ == "__main__":
    main()
