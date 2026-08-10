"""The immutable snapshot the renderer draws.

``app`` builds one of these per frame from the session, the selection state, and
the worker status. Keeping it a plain dataclass means a full-frame render test
needs no window and no live game.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .interaction import EMPTY, SelectionState


@dataclass(frozen=True)
class CoachView:
    classification: str
    centipawn_loss: int
    played_san: str
    best_san: str
    explanation: str
    search_depth: int
    used_static_fallback: bool = False


@dataclass(frozen=True)
class MoveRow:
    number: int
    white: str | None
    black: str | None
    white_classification: str | None = None
    black_classification: str | None = None


@dataclass(frozen=True)
class ViewModel:
    board: chess.Board
    flipped: bool = False
    selection: SelectionState = EMPTY
    last_move: chess.Move | None = None
    now: float = 0.0

    # header / status
    evaluator_name: str = "handcrafted"
    requested_evaluator: str = "handcrafted"
    depth: int = 4
    time_limit: float = 5.0
    coach_enabled: bool = False
    human_color: chess.Color = chess.WHITE

    thinking: bool = False
    thinking_label: str = ""
    status_text: str = ""
    status_color: str = "dim"  # "dim" | "text" | "accent" | "warn" | "error"

    eval_cp: float | None = None
    eval_is_stale: bool = False
    # Animated bar fill. None means "derive it from eval_cp", which is what a
    # static render (or a test) wants.
    eval_fill: float | None = None

    coach: CoachView | None = None
    coach_pending: bool = False
    moves: tuple[MoveRow, ...] = ()
    move_scroll: int = 0
    result_text: str | None = None

    widgets: tuple = field(default_factory=tuple)
