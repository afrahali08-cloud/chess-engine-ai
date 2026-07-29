#!/usr/bin/env python3
"""Train a Ridge position evaluator from extracted chess scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from board.learned_features import (  # noqa: E402
    FEATURE_COUNT,
    FEATURE_ENCODING,
    fen_features,
)
from scripts.training_common import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    SPLIT_NAMES,
    LoadedDataset,
    PositionRow,
    dataset_metadata,
    load_split_rows,
    regression_metrics,
    split_for_game,
)


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


def build_matrix(rows: list[PositionRow]) -> tuple[csr_matrix, np.ndarray]:
    matrix_rows: list[int] = []
    matrix_columns: list[int] = []
    matrix_values: list[float] = []
    labels = np.empty(len(rows), dtype=np.float32)

    for row_index, row in enumerate(rows):
        labels[row_index] = row.score
        for feature_index, feature_value in fen_features(row.fen):
            matrix_rows.append(row_index)
            matrix_columns.append(feature_index)
            matrix_values.append(feature_value)

    matrix = csr_matrix(
        (matrix_values, (matrix_rows, matrix_columns)),
        shape=(len(rows), FEATURE_COUNT),
        dtype=np.float32,
    )
    return matrix, labels


def train(
    input_path: Path,
    output_path: Path,
    *,
    alpha: float,
    clip_cp: int,
    seed: int = DEFAULT_RANDOM_SEED,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"model already exists: {output_path} (use --overwrite to replace it)"
        )

    dataset: LoadedDataset = load_split_rows(
        input_path,
        clip_cp=clip_cp,
        seed=seed,
        limit=limit,
    )
    matrices: dict[str, csr_matrix] = {}
    labels: dict[str, np.ndarray] = {}

    for split_name in SPLIT_NAMES:
        rows = dataset.splits[split_name]
        print(f"Encoding {split_name}: {len(rows):,} positions")
        matrices[split_name], labels[split_name] = build_matrix(rows)

    model = Ridge(alpha=alpha, solver="lsqr")
    model.fit(matrices["train"], labels["train"])

    metrics = {
        split_name: regression_metrics(
            labels[split_name],
            model.predict(matrices[split_name]),
        )
        for split_name in SPLIT_NAMES
    }
    artifact: dict[str, object] = {
        "format_version": 2,
        "model_type": "ridge",
        "feature_encoding": FEATURE_ENCODING,
        "feature_count": FEATURE_COUNT,
        "label": "white_centipawns",
        "clip_cp": clip_cp,
        "alpha": alpha,
        "intercept": float(model.intercept_),
        "coefficients": [float(value) for value in model.coef_],
        "dataset": dataset_metadata(
            dataset,
            input_path=input_path,
            clip_cp=clip_cp,
            seed=seed,
        ),
        "training": {
            "algorithm": "ridge",
            "alpha": alpha,
            "solver": "lsqr",
            "random_seed": seed,
        },
        "split_counts": dataset.split_counts,
        "metrics": metrics,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a Ridge evaluator with game-level data splits."
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
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"deterministic split seed (default: {DEFAULT_RANDOM_SEED})",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        help="optional maximum number of CSV rows",
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
            seed=args.seed,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))

    print("Training complete")
    for split_name, split_metrics in artifact["metrics"].items():
        print(
            f"{split_name}: MAE={split_metrics['mae_cp']:.1f} cp, "
            f"RMSE={split_metrics['rmse_cp']:.1f} cp, "
            f"R2={split_metrics['r2']:.3f}"
        )
    print(f"Model written to {args.output}")


if __name__ == "__main__":
    main()
