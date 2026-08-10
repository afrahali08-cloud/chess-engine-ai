"""Headless game state shared by the terminal and windowed frontends.

The terminal frontend computes and applies a move back to back. The GUI submits
the computation to a worker thread and applies the result frames later. Both
need the same board bookkeeping, so this module splits the two halves:

* ``*_job`` methods snapshot the board on the calling thread and return a
  zero-argument callable that is safe to run anywhere.
* ``apply_*`` methods mutate the session and must run on the owning thread.

The search in :mod:`engine` keeps a module-level transposition table that it
clears on entry, so only one job may run at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable

import chess

try:
    from .board.evaluators import (
        DEFAULT_EVALUATOR,
        EvaluatorSelection,
        resolve_evaluator,
    )
    from .coach import MoveAnalysis, analyze_move
    from .engine import SearchResult, analyze_position
except ImportError:
    from board.evaluators import (
        DEFAULT_EVALUATOR,
        EvaluatorSelection,
        resolve_evaluator,
    )
    from coach import MoveAnalysis, analyze_move
    from engine import SearchResult, analyze_position


@dataclass(frozen=True)
class SessionConfig:
    evaluator: str = DEFAULT_EVALUATOR
    depth: int = 4
    time_limit: float = 5.0
    coach: bool = False
    coach_depth: int = 4
    coach_time_limit: float = 1.0
    coach_line_time_limit: float = 1.5  # extra budget for the why-lines
    human_color: chess.Color = chess.WHITE

    def replace(self, **changes) -> "SessionConfig":
        return replace(self, **changes)


@dataclass(frozen=True)
class MoveRecord:
    ply: int
    move: chess.Move
    san: str
    color: chess.Color
    by_engine: bool
    score: float | None = None
    analysis: MoveAnalysis | None = None


@dataclass(frozen=True)
class MovePair:
    number: int
    white: str | None
    black: str | None
    white_analysis: MoveAnalysis | None = None
    black_analysis: MoveAnalysis | None = None


class GameSession:
    """Board, move history, and the compute/apply seam around the search."""

    def __init__(
        self,
        config: SessionConfig | None = None,
        *,
        selection: EvaluatorSelection | None = None,
    ) -> None:
        self.config = config or SessionConfig()
        self.board = chess.Board()
        self._history: list[MoveRecord] = []
        self.selection = (
            selection
            if selection is not None
            else resolve_evaluator(self.config.evaluator)
        )

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    @property
    def history(self) -> tuple[MoveRecord, ...]:
        return tuple(self._history)

    def is_game_over(self) -> bool:
        return self.board.is_game_over()

    def human_to_move(self) -> bool:
        return not self.is_game_over() and self.board.turn == self.config.human_color

    def engine_to_move(self) -> bool:
        return not self.is_game_over() and self.board.turn != self.config.human_color

    def san_history(self) -> tuple[str, ...]:
        return tuple(record.san for record in self._history)

    def move_pairs(self) -> tuple[MovePair, ...]:
        """Group the history into numbered ``(white, black)`` rows.

        Replaces the SAN replay loop the terminal summary used to do inline and
        also drives the GUI move list.
        """
        pairs: list[MovePair] = []
        for record in self._history:
            if record.color == chess.WHITE or not pairs or pairs[-1].black is not None:
                pairs.append(
                    MovePair(
                        number=len(pairs) + 1,
                        white=record.san if record.color == chess.WHITE else None,
                        black=record.san if record.color == chess.BLACK else None,
                        white_analysis=(
                            record.analysis if record.color == chess.WHITE else None
                        ),
                        black_analysis=(
                            record.analysis if record.color == chess.BLACK else None
                        ),
                    )
                )
            else:
                pairs[-1] = replace(
                    pairs[-1],
                    black=record.san,
                    black_analysis=record.analysis,
                )
        return tuple(pairs)

    def result_text(self) -> str | None:
        """Human-readable outcome, or ``None`` while the game is in progress."""
        if not self.board.is_game_over():
            return None
        outcome = self.board.outcome()
        if outcome is None:
            return "Game over"
        if outcome.winner is None:
            return f"Draw - {outcome.termination.name}"
        winner = "White" if outcome.winner == chess.WHITE else "Black"
        if outcome.termination == chess.Termination.CHECKMATE:
            return f"Checkmate - {winner} wins"
        return f"{winner} wins - {outcome.termination.name}"

    def last_move(self) -> chess.Move | None:
        return self.board.move_stack[-1] if self.board.move_stack else None

    # ------------------------------------------------------------------
    # compute: bound callables that never touch self.board
    # ------------------------------------------------------------------
    def engine_job(self) -> Callable[[], SearchResult]:
        """Snapshot the position now and return a callable safe for a worker."""
        snapshot = self.board.copy()
        depth = self.config.depth
        time_limit = self.config.time_limit
        evaluate = self.selection.evaluate

        def run() -> SearchResult:
            return analyze_position(
                snapshot,
                depth=depth,
                time_limit=time_limit,
                evaluator=evaluate,
            )

        return run

    def coach_job(
        self,
        board_before: chess.Board,
        move: chess.Move,
    ) -> Callable[[], MoveAnalysis]:
        """``board_before`` must be a copy taken before the move was applied."""
        snapshot = board_before.copy()
        depth = self.config.coach_depth
        time_limit = self.config.coach_time_limit
        line_time_limit = self.config.coach_line_time_limit
        evaluate = self.selection.evaluate

        def run() -> MoveAnalysis:
            return analyze_move(
                snapshot,
                move,
                depth=depth,
                time_limit=time_limit,
                evaluator=evaluate,
                line_time_limit=line_time_limit,
            )

        return run

    def evaluator_job(self, name: str) -> Callable[[], EvaluatorSelection]:
        """Resolve an evaluator and warm it, so the first search pays no import."""

        def run() -> EvaluatorSelection:
            selection = resolve_evaluator(name)
            selection.evaluate(chess.Board())
            return selection

        return run

    # ------------------------------------------------------------------
    # apply: fast, mutating, owning thread only
    # ------------------------------------------------------------------
    def apply_move(
        self,
        move: chess.Move,
        *,
        by_engine: bool,
        score: float | None = None,
        analysis: MoveAnalysis | None = None,
    ) -> MoveRecord:
        if move not in self.board.legal_moves:
            raise ValueError(f"illegal move: {move}")
        record = MoveRecord(
            ply=len(self.board.move_stack),
            move=move,
            san=self.board.san(move),
            color=self.board.turn,
            by_engine=by_engine,
            score=score,
            analysis=analysis,
        )
        self.board.push(move)
        self._history.append(record)
        return record

    def attach_analysis(self, ply: int, analysis: MoveAnalysis) -> bool:
        """Late-bind coach output onto an already applied move."""
        for index, record in enumerate(self._history):
            if record.ply == ply:
                self._history[index] = replace(record, analysis=analysis)
                return True
        return False

    def undo(self, plies: int = 1) -> int:
        """Pop up to ``plies`` moves. Returns how many were actually popped."""
        popped = 0
        for _ in range(max(0, plies)):
            if not self.board.move_stack:
                break
            self.board.pop()
            if self._history:
                self._history.pop()
            popped += 1
        return popped

    def undo_to_human_turn(self) -> int:
        """Take back enough moves that it is the human's turn again."""
        if not self.board.move_stack:
            return 0
        # After the engine has replied it takes two pops; if the human has just
        # moved and the engine has not answered yet, one is enough.
        plies = 2 if self.board.turn == self.config.human_color else 1
        return self.undo(plies)

    def reset(self, *, config: SessionConfig | None = None) -> None:
        if config is not None:
            self.config = config
        self.board = chess.Board()
        self._history = []

    def set_config(self, config: SessionConfig) -> None:
        self.config = config

    def set_selection(self, selection: EvaluatorSelection) -> None:
        self.selection = selection

    # ------------------------------------------------------------------
    # blocking convenience used by the terminal frontend
    # ------------------------------------------------------------------
    def play_engine_move(self) -> MoveRecord | None:
        result = self.engine_job()()
        if result.best_move is None:
            return None
        return self.apply_move(
            result.best_move,
            by_engine=True,
            score=result.best_score,
        )

    def play_human_move(
        self,
        move: chess.Move,
    ) -> tuple[MoveRecord, MoveAnalysis | None]:
        analysis = None
        if self.config.coach:
            analysis = self.coach_job(self.board, move)()
        record = self.apply_move(move, by_engine=False, analysis=analysis)
        return record, analysis
