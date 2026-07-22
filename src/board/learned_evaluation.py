"""Runtime inference for the learned chess evaluation model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chess

try:
    from .learned_features import FEATURE_COUNT, board_features
except ImportError:
    from learned_features import FEATURE_COUNT, board_features


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "learned_evaluator.json"
)


class LearnedModelError(ValueError):
    """Raised when a learned model file is incompatible or malformed."""


@dataclass(frozen=True)
class LearnedModel:
    intercept: float
    coefficients: tuple[float, ...]
    clip_cp: int


@lru_cache(maxsize=None)
def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> LearnedModel:
    """Load and validate a model once per process."""
    try:
        artifact = json.loads(model_path.read_text(encoding="utf-8"))
        coefficients = tuple(float(value) for value in artifact["coefficients"])
        intercept = float(artifact["intercept"])
        clip_cp = int(artifact["clip_cp"])
    except (
        AttributeError,
        KeyError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise LearnedModelError(f"invalid learned model: {model_path}") from error

    if artifact.get("feature_encoding") != "symmetric_piece_square_v1":
        raise LearnedModelError("unsupported learned model feature encoding")
    if len(coefficients) != FEATURE_COUNT:
        raise LearnedModelError(
            f"expected {FEATURE_COUNT} coefficients, found {len(coefficients)}"
        )
    if clip_cp <= 0:
        raise LearnedModelError("model clip_cp must be greater than zero")

    return LearnedModel(intercept, coefficients, clip_cp)


def evaluate_learned_board(
    board: chess.Board, model_path: Path = DEFAULT_MODEL_PATH
) -> int:
    """Predict a white-relative centipawn score for a non-terminal board."""
    model = load_model(model_path)
    score = model.intercept

    for feature_index, feature_value in board_features(board):
        score += model.coefficients[feature_index] * feature_value

    score = max(-model.clip_cp, min(model.clip_cp, score))
    return int(round(score))
