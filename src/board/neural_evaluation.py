"""Runtime inference for the neural chess evaluator."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chess
import torch

from .neural_features import FEATURE_ENCODING, TOTAL_FEATURES, board_to_features
from .neural_model import EvalNet


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "neural_evaluator.pt"
)


class NeuralModelError(ValueError):
    """Raised when a neural model checkpoint is incompatible or malformed."""


@lru_cache(maxsize=None)
def _load_model_cached(model_path: Path) -> tuple[EvalNet, int]:
    try:
        checkpoint = torch.load(
            model_path,
            map_location="cpu",
            weights_only=True,
        )
        input_size = int(checkpoint["input_size"])
        clip_cp = int(checkpoint["clip_cp"])
        feature_encoding = checkpoint["feature_encoding"]
        model_state = checkpoint["model_state"]
    except FileNotFoundError:
        raise
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise NeuralModelError(f"invalid neural model: {model_path}") from error

    if input_size != TOTAL_FEATURES:
        raise NeuralModelError(
            f"expected {TOTAL_FEATURES} features, model contains {input_size}"
        )
    if feature_encoding != FEATURE_ENCODING:
        raise NeuralModelError(
            f"expected feature encoding {FEATURE_ENCODING}, "
            f"found {feature_encoding}"
        )
    if clip_cp <= 0:
        raise NeuralModelError("model clip_cp must be greater than zero")

    model = EvalNet(input_size=input_size)
    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise NeuralModelError(f"incompatible neural weights: {model_path}") from error
    model.eval()
    return model, clip_cp


def load_neural_model(
    model_path: Path = DEFAULT_MODEL_PATH,
) -> tuple[EvalNet, int]:
    """Load and cache a validated neural evaluator."""
    return _load_model_cached(Path(model_path))


def evaluate_neural_board(
    board: chess.Board,
    model_path: Path = DEFAULT_MODEL_PATH,
) -> int:
    """Predict a white-relative centipawn score for a board."""
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if (
        board.is_stalemate()
        or board.is_insufficient_material()
        or board.is_seventyfive_moves()
        or board.is_fivefold_repetition()
    ):
        return 0

    model, clip_cp = load_neural_model(model_path)
    features = torch.from_numpy(board_to_features(board)).unsqueeze(0)

    with torch.inference_mode():
        score = float(model(features).item())

    score = max(-clip_cp, min(clip_cp, score))
    return int(round(score))
