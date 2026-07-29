#!/usr/bin/env python3
"""Compare all evaluators on the same held-out game-level test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Callable

import chess
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from board.evaluation import evaluate_handcrafted_board  # noqa: E402
from board.learned_evaluation import (  # noqa: E402
    evaluate_learned_board,
    load_model as load_ridge_model,
)
from board.neural_evaluation import (  # noqa: E402
    evaluate_neural_board,
    load_neural_model,
)
from scripts.training_common import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    dataset_metadata,
    load_split_rows,
    regression_metrics,
)


def evaluate_many(
    boards: list[chess.Board],
    evaluator: Callable[[chess.Board], int],
) -> tuple[np.ndarray, float]:
    start = perf_counter()
    predictions = np.asarray([evaluator(board) for board in boards], dtype=np.float64)
    elapsed = perf_counter() - start
    positions_per_second = len(boards) / max(elapsed, 1e-9)
    return predictions, positions_per_second


def compare(
    input_path: Path,
    ridge_model_path: Path,
    neural_model_path: Path,
    output_path: Path,
    *,
    clip_cp: int = 1500,
    seed: int = DEFAULT_RANDOM_SEED,
    overwrite: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"comparison already exists: {output_path} (use --overwrite)"
        )

    dataset = load_split_rows(input_path, clip_cp=clip_cp, seed=seed)
    test_rows = dataset.splits["test"]
    boards = [chess.Board(row.fen) for row in test_rows]
    labels = np.asarray([row.score for row in test_rows], dtype=np.float64)

    load_ridge_model(ridge_model_path)
    load_neural_model(neural_model_path)
    evaluators = {
        "handcrafted": evaluate_handcrafted_board,
        "ridge": lambda board: evaluate_learned_board(board, ridge_model_path),
        "neural": lambda board: evaluate_neural_board(board, neural_model_path),
    }

    results: dict[str, dict[str, float]] = {}
    for name, evaluator in evaluators.items():
        predictions, positions_per_second = evaluate_many(boards, evaluator)
        results[name] = {
            **regression_metrics(labels, predictions),
            "positions_per_second": positions_per_second,
        }
        print(
            f"{name}: MAE={results[name]['mae_cp']:.1f} cp, "
            f"RMSE={results[name]['rmse_cp']:.1f} cp, "
            f"R2={results[name]['r2']:.3f}, "
            f"speed={positions_per_second:,.0f} positions/s"
        )

    lowest_mae_evaluator = min(results, key=lambda name: results[name]["mae_cp"])
    artifact: dict[str, object] = {
        "format_version": 1,
        "comparison_split": "test",
        "selection_rule": "lowest_test_mae_cp",
        "lowest_mae_evaluator": lowest_mae_evaluator,
        "dataset": dataset_metadata(
            dataset,
            input_path=input_path,
            clip_cp=clip_cp,
            seed=seed,
        ),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Lowest-MAE evaluator: {lowest_mae_evaluator}")
    print(f"Comparison written to {output_path}")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare handcrafted, Ridge, and neural evaluators."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--ridge-model",
        type=Path,
        default=Path("models/learned_evaluator.json"),
    )
    parser.add_argument(
        "--neural-model",
        type=Path,
        default=Path("models/neural_evaluator.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/evaluator_comparison.json"),
    )
    parser.add_argument("--clip-cp", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        compare(
            args.input,
            args.ridge_model,
            args.neural_model,
            args.output,
            clip_cp=args.clip_cp,
            seed=args.seed,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
