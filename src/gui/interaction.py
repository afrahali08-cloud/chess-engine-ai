"""Click-to-select / click-to-move state machine.

Pure logic: it takes a board and a square index and returns the next state plus
an optional move. No pygame, no session mutation, so it is fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping

import chess


ERROR_FLASH_SECONDS = 0.26
PROMOTION_OPTIONS = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)


@dataclass(frozen=True)
class PromotionPrompt:
    from_square: int
    to_square: int
    options: tuple[int, ...] = PROMOTION_OPTIONS


@dataclass(frozen=True)
class SelectionState:
    selected: int | None = None
    targets: Mapping[int, tuple[chess.Move, ...]] = field(default_factory=dict)
    promotion: PromotionPrompt | None = None
    error_square: int | None = None
    error_expires_at: float = 0.0

    def is_target(self, square: int) -> bool:
        return square in self.targets

    def error_active(self, now: float) -> bool:
        return self.error_square is not None and now < self.error_expires_at


EMPTY = SelectionState()


def legal_targets(
    board: chess.Board,
    from_square: int,
) -> dict[int, tuple[chess.Move, ...]]:
    """Map each reachable square to the legal moves landing there.

    More than one move per square only happens for promotions.
    """
    grouped: dict[int, list[chess.Move]] = {}
    for move in board.legal_moves:
        if move.from_square == from_square:
            grouped.setdefault(move.to_square, []).append(move)
    return {square: tuple(moves) for square, moves in grouped.items()}


def _select(board: chess.Board, square: int, now: float) -> SelectionState:
    targets = legal_targets(board, square)
    if not targets:
        # A piece that cannot legally move (usually pinned) is worth flagging.
        return SelectionState(
            error_square=square,
            error_expires_at=now + ERROR_FLASH_SECONDS,
        )
    return SelectionState(selected=square, targets=targets)


def clear(state: SelectionState) -> SelectionState:
    """Drop the selection and any promotion prompt, keeping an active flash."""
    return SelectionState(
        error_square=state.error_square,
        error_expires_at=state.error_expires_at,
    )


def expire_error(state: SelectionState, now: float) -> SelectionState:
    if state.error_square is not None and now >= state.error_expires_at:
        return replace(state, error_square=None, error_expires_at=0.0)
    return state


def click_square(
    state: SelectionState,
    board: chess.Board,
    square: int,
    *,
    human_color: chess.Color | None = None,
    now: float = 0.0,
    allow_input: bool = True,
) -> tuple[SelectionState, chess.Move | None]:
    """Advance the selection for a click on ``square``.

    Returns the next state and a move to play, if the click completed one.
    """
    if not allow_input:
        return state, None

    # While the promotion picker is open it owns all input; any board click
    # cancels it.
    if state.promotion is not None:
        return clear(state), None

    movable = human_color is None or board.turn == human_color
    piece = board.piece_at(square)
    own_piece = piece is not None and piece.color == board.turn and movable

    if state.selected is None:
        if own_piece:
            return _select(board, square, now), None
        return state, None

    if square == state.selected:
        return clear(state), None

    moves = state.targets.get(square)
    if moves:
        if len(moves) == 1:
            return clear(state), moves[0]
        return (
            replace(
                state,
                promotion=PromotionPrompt(
                    from_square=state.selected,
                    to_square=square,
                    options=tuple(
                        move.promotion
                        for move in moves
                        if move.promotion is not None
                    )
                    or PROMOTION_OPTIONS,
                ),
            ),
            None,
        )

    if own_piece:
        return _select(board, square, now), None

    return (
        SelectionState(
            error_square=square,
            error_expires_at=now + ERROR_FLASH_SECONDS,
        ),
        None,
    )


def click_promotion(
    state: SelectionState,
    piece_type: int | None,
) -> tuple[SelectionState, chess.Move | None]:
    """Resolve an open promotion prompt. ``None`` cancels it."""
    prompt = state.promotion
    if prompt is None:
        return state, None
    if piece_type is None or piece_type not in prompt.options:
        return clear(state), None
    return (
        clear(state),
        chess.Move(prompt.from_square, prompt.to_square, promotion=piece_type),
    )
