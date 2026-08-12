#!/usr/bin/env python3
"""Run a fixed-time, color-swapped evaluator strength benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

import chess

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from board.evaluators import EVALUATOR_CHOICES, Evaluator, resolve_evaluator  # noqa: E402
from board.learned_evaluation import (  # noqa: E402
    DEFAULT_MODEL_PATH as DEFAULT_RIDGE_MODEL_PATH,
)
from board.neural_evaluation import (  # noqa: E402
    DEFAULT_MODEL_PATH as DEFAULT_NEURAL_MODEL_PATH,
)
from engine import analyze_position  # noqa: E402


DEFAULT_OPENINGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Open Game", ("e2e4", "e7e5", "g1f3", "b8c6")),
    ("Sicilian Defense", ("e2e4", "c7c5", "g1f3", "d7d6")),
    ("Queen's Gambit", ("d2d4", "d7d5", "c2c4", "e7e6")),
    ("English Opening", ("c2c4", "e7e5", "b1c3", "g8f6")),
    ("Reti Opening", ("g1f3", "d7d5", "g2g3", "c7c5")),
    ("French Defense", ("e2e4", "e7e6", "d2d4", "d7d5")),
)


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT_DIR))
    except ValueError:
        return path.name


def _model_metadata(name: str) -> dict[str, object]:
    if name == "handcrafted":
        return {"model_type": "handcrafted", "artifact_path": None}

    if name == "ridge":
        model_path = DEFAULT_RIDGE_MODEL_PATH
        artifact = json.loads(model_path.read_text(encoding="utf-8"))
    elif name == "neural":
        import torch

        model_path = DEFAULT_NEURAL_MODEL_PATH
        artifact = torch.load(model_path, map_location="cpu", weights_only=True)
    else:
        raise ValueError(f"unknown evaluator: {name}")

    dataset = artifact.get("dataset")
    if not isinstance(dataset, dict):
        raise RuntimeError(f"{name} model does not contain dataset metadata")
    metrics = artifact.get("metrics")
    test_metrics = metrics.get("test") if isinstance(metrics, dict) else None
    return {
        "model_type": artifact.get("model_type"),
        "artifact_path": _display_path(model_path),
        "artifact_sha256": _file_sha256(model_path),
        "feature_encoding": artifact.get("feature_encoding"),
        "dataset": dataset,
        "test_metrics": test_metrics,
    }


def _validate_model_datasets(
    metadata: dict[str, dict[str, object]],
    expected_positions: int | None,
) -> dict[str, object] | None:
    learned_datasets = [
        details["dataset"]
        for details in metadata.values()
        if isinstance(details.get("dataset"), dict)
    ]
    if not learned_datasets:
        return None

    dataset_keys = {
        (
            dataset.get("source_sha256"),
            dataset.get("total_positions"),
            dataset.get("split_strategy"),
            dataset.get("random_seed"),
        )
        for dataset in learned_datasets
    }
    if len(dataset_keys) != 1:
        raise RuntimeError("benchmark models were not trained on the same dataset")

    dataset = dict(learned_datasets[0])
    if (
        expected_positions is not None
        and dataset.get("total_positions") != expected_positions
    ):
        raise RuntimeError(
            "benchmark model contains "
            f"{dataset.get('total_positions')!r} positions; "
            f"expected {expected_positions}"
        )
    return dataset


def play_game(
    *,
    opening_name: str,
    opening_moves: tuple[str, ...],
    white_name: str,
    white_evaluator: Evaluator,
    black_name: str,
    black_evaluator: Evaluator,
    depth: int,
    time_limit: float,
    max_plies: int,
) -> dict[str, object]:
    board = chess.Board()
    for move_text in opening_moves:
        move = chess.Move.from_uci(move_text)
        if move not in board.legal_moves:
            raise ValueError(f"illegal opening move {move_text} in {opening_name}")
        board.push(move)

    searched_moves: list[str] = []
    depth_totals = {white_name: 0, black_name: 0}
    time_totals = {white_name: 0.0, black_name: 0.0}
    move_counts = {white_name: 0, black_name: 0}

    while (
        len(searched_moves) < max_plies
        and not board.is_game_over(claim_draw=True)
    ):
        if board.turn == chess.WHITE:
            evaluator_name = white_name
            evaluator = white_evaluator
        else:
            evaluator_name = black_name
            evaluator = black_evaluator

        result = analyze_position(
            board,
            depth=depth,
            time_limit=time_limit,
            evaluator=evaluator,
        )
        if result.best_move is None:
            break

        searched_moves.append(result.best_move.uci())
        depth_totals[evaluator_name] += result.completed_depth
        time_totals[evaluator_name] += result.elapsed_seconds
        move_counts[evaluator_name] += 1
        board.push(result.best_move)

    outcome = board.outcome(claim_draw=True)
    result_text = outcome.result() if outcome is not None else "*"
    termination = (
        outcome.termination.name if outcome is not None else "MAX_PLIES"
    )
    average_depth = {
        name: (
            depth_totals[name] / move_counts[name] if move_counts[name] else 0.0
        )
        for name in (white_name, black_name)
    }
    average_move_time = {
        name: (
            time_totals[name] / move_counts[name] if move_counts[name] else 0.0
        )
        for name in (white_name, black_name)
    }

    return {
        "opening": opening_name,
        "opening_moves": list(opening_moves),
        "white_evaluator": white_name,
        "black_evaluator": black_name,
        "result": result_text,
        "termination": termination,
        "searched_plies": len(searched_moves),
        "moves": searched_moves,
        "final_fen": board.fen(),
        "average_completed_depth": average_depth,
        "average_move_time_seconds": average_move_time,
    }


def _summarize(
    games: list[dict[str, object]],
    evaluator_names: tuple[str, str],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {
        name: {
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "unresolved": 0,
            "points": 0.0,
            "average_completed_depth": 0.0,
            "average_move_time_seconds": 0.0,
        }
        for name in evaluator_names
    }
    depth_samples = {name: [] for name in evaluator_names}
    time_samples = {name: [] for name in evaluator_names}

    for game in games:
        white_name = str(game["white_evaluator"])
        black_name = str(game["black_evaluator"])
        result = game["result"]
        if result == "1-0":
            summary[white_name]["wins"] += 1
            summary[white_name]["points"] += 1.0
            summary[black_name]["losses"] += 1
        elif result == "0-1":
            summary[black_name]["wins"] += 1
            summary[black_name]["points"] += 1.0
            summary[white_name]["losses"] += 1
        elif result == "1/2-1/2":
            for name in (white_name, black_name):
                summary[name]["draws"] += 1
                summary[name]["points"] += 0.5
        else:
            summary[white_name]["unresolved"] += 1
            summary[black_name]["unresolved"] += 1

        for name in evaluator_names:
            depth_samples[name].append(game["average_completed_depth"][name])
            time_samples[name].append(game["average_move_time_seconds"][name])

    for name in evaluator_names:
        summary[name]["average_completed_depth"] = sum(
            depth_samples[name]
        ) / len(depth_samples[name])
        summary[name]["average_move_time_seconds"] = sum(
            time_samples[name]
        ) / len(time_samples[name])
    return summary


def benchmark(
    first_name: str,
    second_name: str,
    output_path: Path,
    *,
    depth: int = 5,
    time_limit: float = 0.1,
    max_plies: int = 120,
    opening_count: int = len(DEFAULT_OPENINGS),
    expected_positions: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if first_name == second_name:
        raise ValueError("benchmark evaluators must be different")
    if opening_count > len(DEFAULT_OPENINGS):
        raise ValueError(f"opening_count cannot exceed {len(DEFAULT_OPENINGS)}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"benchmark already exists: {output_path} (use --overwrite)"
        )

    first = resolve_evaluator(first_name)
    second = resolve_evaluator(second_name)
    if first.selected != first_name or second.selected != second_name:
        raise RuntimeError("requested benchmark evaluator used a fallback")

    model_metadata = {
        name: _model_metadata(name) for name in (first_name, second_name)
    }
    dataset = _validate_model_datasets(model_metadata, expected_positions)

    games: list[dict[str, object]] = []
    start = monotonic()
    for opening_name, opening_moves in DEFAULT_OPENINGS[:opening_count]:
        pairings = (
            (first_name, first.evaluate, second_name, second.evaluate),
            (second_name, second.evaluate, first_name, first.evaluate),
        )
        for white_name, white_evaluator, black_name, black_evaluator in pairings:
            game = play_game(
                opening_name=opening_name,
                opening_moves=opening_moves,
                white_name=white_name,
                white_evaluator=white_evaluator,
                black_name=black_name,
                black_evaluator=black_evaluator,
                depth=depth,
                time_limit=time_limit,
                max_plies=max_plies,
            )
            games.append(game)
            print(
                f"{opening_name}: {white_name} vs {black_name} "
                f"{game['result']} ({game['termination']})"
            )

    summary = _summarize(games, (first_name, second_name))
    completed_games = sum(game["result"] != "*" for game in games)
    if completed_games:
        recommended_default = max(
            (first_name, second_name),
            key=lambda name: summary[name]["points"],
        )
        if summary[first_name]["points"] == summary[second_name]["points"]:
            recommended_default = "inconclusive"
    else:
        recommended_default = "inconclusive"

    artifact: dict[str, object] = {
        "format_version": 2,
        "benchmark_type": "fixed_time_color_swapped_self_play",
        "settings": {
            "evaluators": [first_name, second_name],
            "depth": depth,
            "time_limit_seconds": time_limit,
            "max_plies_after_opening": max_plies,
            "opening_count": opening_count,
            "game_count": len(games),
        },
        "dataset": dataset,
        "model_metadata": model_metadata,
        "completed_games": completed_games,
        "recommended_default": recommended_default,
        "summary": summary,
        "games": games,
        "elapsed_seconds": monotonic() - start,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Recommended default: {recommended_default}")
    print(f"Benchmark written to {output_path}")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare evaluator playing strength with color-swapped games."
    )
    parser.add_argument("--first", choices=EVALUATOR_CHOICES, default="ridge")
    parser.add_argument("--second", choices=EVALUATOR_CHOICES, default="neural")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/strength_benchmark.json"),
    )
    parser.add_argument("--depth", type=_positive_int, default=5)
    parser.add_argument("--time-limit", type=_positive_float, default=0.1)
    parser.add_argument("--max-plies", type=_positive_int, default=120)
    parser.add_argument(
        "--openings",
        type=_positive_int,
        default=len(DEFAULT_OPENINGS),
    )
    parser.add_argument(
        "--expected-positions",
        type=_positive_int,
        help="fail unless learned models contain this many training positions",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        benchmark(
            args.first,
            args.second,
            args.output,
            depth=args.depth,
            time_limit=args.time_limit,
            max_plies=args.max_plies,
            opening_count=args.openings,
            expected_positions=args.expected_positions,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
