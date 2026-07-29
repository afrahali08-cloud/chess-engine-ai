"""Shared dataset splitting and regression metrics for evaluator training."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_RANDOM_SEED = 42
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_STRATEGY = "sha256_game_id_80_10_10_v1"


@dataclass(frozen=True)
class PositionRow:
    game_id: str
    fen: str
    score: float


@dataclass(frozen=True)
class LoadedDataset:
    splits: dict[str, list[PositionRow]]
    source_sha256: str

    @property
    def split_counts(self) -> dict[str, int]:
        return {name: len(self.splits[name]) for name in SPLIT_NAMES}

    @property
    def total_positions(self) -> int:
        return sum(self.split_counts.values())


def split_for_game(game_id: str, seed: int = DEFAULT_RANDOM_SEED) -> str:
    """Assign every position from a game to one deterministic 80/10/10 split."""
    digest = hashlib.sha256(f"{seed}:{game_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big") % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def load_split_rows(
    input_path: Path,
    *,
    clip_cp: int,
    seed: int = DEFAULT_RANDOM_SEED,
    limit: int | None = None,
) -> LoadedDataset:
    """Load scored positions and keep complete games in a single data split."""
    if not input_path.is_file():
        raise FileNotFoundError(f"training CSV not found: {input_path}")
    if clip_cp <= 0:
        raise ValueError("clip_cp must be greater than zero")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be greater than zero")

    splits: dict[str, list[PositionRow]] = {name: [] for name in SPLIT_NAMES}
    source_hash = hashlib.sha256()

    with input_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        required_fields = {"game_id", "fen", "cp"}
        if not required_fields.issubset(reader.fieldnames or []):
            raise ValueError("training CSV must contain game_id, fen, and cp columns")

        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break

            game_id = row["game_id"].strip()
            fen = row["fen"].strip()
            if not game_id or not fen:
                raise ValueError(f"row {index + 2} has an empty game_id or fen")

            try:
                raw_score = float(row["cp"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"row {index + 2} has an invalid cp score") from error

            score = float(np.clip(raw_score, -clip_cp, clip_cp))
            split_name = split_for_game(game_id, seed)
            splits[split_name].append(PositionRow(game_id, fen, score))
            source_hash.update(
                f"{game_id}\0{fen}\0{raw_score:.12g}\n".encode("utf-8")
            )

    empty_splits = [name for name, rows in splits.items() if not rows]
    if empty_splits:
        raise ValueError(f"training CSV produced empty splits: {', '.join(empty_splits)}")

    return LoadedDataset(splits=splits, source_sha256=source_hash.hexdigest())


def regression_metrics(
    labels: np.ndarray | list[float],
    predictions: np.ndarray | list[float],
) -> dict[str, float]:
    """Return the same regression metrics for every evaluator."""
    expected = np.asarray(labels, dtype=np.float64)
    actual = np.asarray(predictions, dtype=np.float64)
    if expected.shape != actual.shape or expected.size == 0:
        raise ValueError("labels and predictions must be non-empty and have equal shape")

    errors = actual - expected
    residual_sum = float(np.sum(errors**2))
    centered_sum = float(np.sum((expected - expected.mean()) ** 2))
    r2 = 0.0 if centered_sum == 0.0 else 1.0 - residual_sum / centered_sum

    return {
        "mae_cp": float(np.mean(np.abs(errors))),
        "rmse_cp": float(np.sqrt(np.mean(errors**2))),
        "r2": r2,
    }


def dataset_metadata(
    dataset: LoadedDataset,
    *,
    input_path: Path,
    clip_cp: int,
    seed: int,
) -> dict[str, object]:
    return {
        "source_file": input_path.name,
        "source_sha256": dataset.source_sha256,
        "total_positions": dataset.total_positions,
        "split_counts": dataset.split_counts,
        "split_strategy": SPLIT_STRATEGY,
        "random_seed": seed,
        "clip_cp": clip_cp,
    }
