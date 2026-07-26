from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chess
import torch
import torch.nn as nn

try:
    from .neural_features import TOTAL_FEATURES, board_to_features
except ImportError:
    from neural_features import TOTAL_FEATURES, board_to_features


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "models" / "neural_evaluator.pt"
)


class EvalNet(nn.Module):
    def __init__(self, input_size: int = TOTAL_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


@lru_cache(maxsize=1)
def _load_model(model_path: Path):
    """Load model once and cache it — called millions of times per game."""
    checkpoint = torch.load(model_path, map_location="cpu")
    model = EvalNet(input_size=checkpoint["input_size"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint["clip_cp"]


def evaluate_neural_board(
    board: chess.Board,
    model_path: Path = DEFAULT_MODEL_PATH
) -> int:
    """
    Predict centipawn eval for a position using the neural network.
    
    Positive = white winning, negative = black winning.
    Returns integer centipawns, same interface as evaluate_board().
    """
    # handle terminal states before calling model
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    model, clip_cp = _load_model(model_path)

    features = board_to_features(board)
    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        score = model(x).item()

    score = max(-clip_cp, min(clip_cp, score))
    score = score*2.5
    score = max(-clip_cp, min(clip_cp, score))
    return int(round(score))