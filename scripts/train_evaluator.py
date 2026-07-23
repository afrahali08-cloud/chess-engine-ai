#!/usr/bin/env python3
"""Train a lightweight position evaluator from extracted Lichess scores."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from board.learned_features import FEATURE_COUNT, fen_features


SPLIT_NAMES = ("train", "validation", "test")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def split_for_game(game_id: str) -> str:
    """Assign a complete game to a deterministic 80/10/10 split."""
    bucket = hashlib.sha256(game_id.encode("utf-8")).digest()[0] % 10
    if bucket < 8:
        return "train"
    if bucket == 8:
        return "validation"
    return "test"


def load_rows(
    input_path: Path, *, clip_cp: int
) -> dict[str, list[tuple[str, float]]]:
    if not input_path.is_file():
        raise FileNotFoundError(f"training CSV not found: {input_path}")

    split_rows: dict[str, list[tuple[str, float]]] = {
        name: [] for name in SPLIT_NAMES
    }

    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        required_fields = {"game_id", "fen", "cp"}
        if not required_fields.issubset(reader.fieldnames or []):
            raise ValueError("training CSV must contain game_id, fen, and cp columns")

        for row in reader:
            split_name = split_for_game(row["game_id"])
            score = float(np.clip(int(row["cp"]), -clip_cp, clip_cp))
            split_rows[split_name].append((row["fen"], score))

    empty_splits = [name for name, rows in split_rows.items() if not rows]
    if empty_splits:
        raise ValueError(f"training CSV produced empty splits: {', '.join(empty_splits)}")

    return split_rows


def build_matrix(rows: list[tuple[str, float]]) -> tuple[csr_matrix, np.ndarray]:
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    labels = np.empty(len(rows), dtype=np.float32)

    for row_index, (fen, score) in enumerate(rows):
        labels[row_index] = score
        for feature_index, feature_value in fen_features(fen):
            matrix_rows.append(row_index)
            matrix_columns.append(feature_index)
            matrix_values.append(feature_value)

    matrix = csr_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(rows), FEATURE_COUNT),
        dtype=np.float32,
    )
    return matrix, labels


def evaluate_model(model: Ridge, matrix: csr_matrix, labels: np.ndarray) -> dict[str, float]:
    predictions = model.predict(matrix)
    return {
        "mae_cp": float(mean_absolute_error(labels, predictions)),
        "rmse_cp": float(np.sqrt(mean_squared_error(labels, predictions))),
        "r2": float(r2_score(labels, predictions)),
    }


def train(
    input_path: Path,
    output_path: Path,
    *,
    alpha: float,
    clip_cp: int,
    overwrite: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"model already exists: {output_path} (use --overwrite to replace it)"
        )

    split_rows = load_rows(input_path, clip_cp=clip_cp)
    matrices = {}
    labels = {}

    for split_name in SPLIT_NAMES:
        print(f"Encoding {split_name}: {len(split_rows[split_name]):,} positions")
        matrices[split_name], labels[split_name] = build_matrix(split_rows[split_name])

    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(matrices["train"], labels["train"])

    metrics = {
        split_name: evaluate_model(model, matrices[split_name], labels[split_name])
        for split_name in SPLIT_NAMES
    }
    artifact: dict[str, object] = {
        "format_version": 1,
        "model_type": "ridge",
        "feature_encoding": "symmetric_piece_square_v1",
        "feature_count": FEATURE_COUNT,
        "label": "white_centipawns",
        "clip_cp": clip_cp,
        "alpha": alpha,
        "intercept": float(model.intercept_),
        "coefficients": [float(value) for value in model.coef_],
        "split_counts": {
            split_name: len(split_rows[split_name]) for split_name in SPLIT_NAMES
        },
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Ridge position evaluator from extracted Lichess data."
    )
    parser.add_argument("--input", required=True, type=Path, help="training CSV")
    parser.add_argument("--output", required=True, type=Path, help="model JSON")
    parser.add_argument(
        "--alpha",
        type=_positive_float,
        default=10.0,
        help="Ridge regularization strength (default: 10.0)",
    )
    parser.add_argument(
        "--clip-cp",
        type=_positive_int,
        default=1500,
        help="clip training labels to +/- this score (default: 1500)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the model file if it already exists",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        artifact = train(
            args.input,
            args.output,
            alpha=args.alpha,
            clip_cp=args.clip_cp,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))

    print("Training complete")
    for split_name, split_metrics in artifact["metrics"].items():
        print(
            f"{split_name}: MAE={split_metrics['mae_cp']:.1f} cp, "
            f"RMSE={split_metrics['rmse_cp']:.1f} cp, R2={split_metrics['r2']:.3f}"
        )
    print(f"Model written to {args.output}")


if __name__ == "__main__":
    main()
