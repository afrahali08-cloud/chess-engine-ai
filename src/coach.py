"""Move-quality coaching based on root-search centipawn loss."""

from __future__ import annotations

from dataclasses import dataclass

import chess

try:
    from .board.evaluators import DEFAULT_EVALUATOR, Evaluator, resolve_evaluator
    from .engine import SearchResult, analyze_position
except ImportError:
    from board.evaluators import DEFAULT_EVALUATOR, Evaluator, resolve_evaluator
    from engine import SearchResult, analyze_position


@dataclass(frozen=True)
class MoveAnalysis:
    played_move: chess.Move
    best_move: chess.Move
    played_san: str
    best_san: str
    played_score: float
    best_score: float
    centipawn_loss: int
    classification: str
    explanation: str
    search_depth: int
    used_static_fallback: bool


def classify_centipawn_loss(centipawn_loss: float) -> str:
    """Convert non-negative evaluation loss into a coaching label."""
    loss = max(0.0, centipawn_loss)
    if loss <= 10:
        return "Best"
    if loss <= 30:
        return "Excellent"
    if loss <= 80:
        return "Good"
    if loss <= 150:
        return "Inaccuracy"
    if loss <= 300:
        return "Mistake"
    return "Blunder"


def calculate_centipawn_loss(
    turn: chess.Color,
    *,
    best_score: float,
    played_score: float,
) -> int:
    """Calculate score loss from the perspective of the side that moved."""
    raw_loss = (
        best_score - played_score
        if turn == chess.WHITE
        else played_score - best_score
    )
    return int(round(max(0.0, raw_loss)))


def _resolve_evaluator(evaluator: Evaluator | str | None) -> Evaluator:
    if evaluator is None:
        return resolve_evaluator(DEFAULT_EVALUATOR).evaluate
    if isinstance(evaluator, str):
        return resolve_evaluator(evaluator).evaluate
    if callable(evaluator):
        return evaluator
    raise TypeError("evaluator must be a name or callable")


def _static_move_score(
    board: chess.Board,
    move: chess.Move,
    evaluator: Evaluator,
) -> float:
    board.push(move)
    try:
        return float(evaluator(board))
    finally:
        board.pop()


def _move_description(board: chess.Board, move: chess.Move) -> str:
    details: list[str] = []
    if board.is_capture(move):
        captured_piece = board.piece_at(move.to_square)
        if captured_piece is None and board.is_en_passant(move):
            details.append("captures a pawn")
        elif captured_piece is not None:
            details.append(f"captures a {chess.piece_name(captured_piece.piece_type)}")
    if move.promotion:
        details.append(f"promotes to a {chess.piece_name(move.promotion)}")

    board.push(move)
    try:
        if board.is_checkmate():
            details.append("delivers checkmate")
        elif board.is_check():
            details.append("gives check")
    finally:
        board.pop()

    if not details:
        return "improves the engine evaluation"
    if len(details) == 1:
        return details[0]
    return ", ".join(details[:-1]) + f" and {details[-1]}"


def _explain(
    board: chess.Board,
    *,
    played_move: chess.Move,
    best_move: chess.Move,
    played_san: str,
    best_san: str,
    classification: str,
    centipawn_loss: int,
) -> str:
    if played_move == best_move:
        return f"{played_san} matches the engine's top choice."
    if classification == "Best":
        return (
            f"{played_san} is within 0.10 pawns of the top choice, "
            f"{best_san}."
        )

    best_description = _move_description(board, best_move)
    return (
        f"The engine preferred {best_san}, which {best_description}. "
        f"{played_san} gives up about {centipawn_loss / 100:.2f} pawns "
        "of evaluation."
    )


def analyze_move(
    board: chess.Board,
    played_move: chess.Move,
    *,
    depth: int = 4,
    time_limit: float = 1.0,
    evaluator: Evaluator | str | None = None,
) -> MoveAnalysis:
    """Compare a legal move with every alternative from one shared search."""
    if played_move not in board.legal_moves:
        raise ValueError(f"illegal move: {played_move}")

    evaluate = _resolve_evaluator(evaluator)
    turn = board.turn
    played_san = board.san(played_move)
    result: SearchResult = analyze_position(
        board,
        depth=depth,
        time_limit=time_limit,
        evaluator=evaluate,
    )
    if result.best_move is None:
        raise ValueError("cannot coach a position with no legal moves")

    best_move = result.best_move
    best_san = board.san(best_move)
    used_static_fallback = not result.move_scores
    if result.move_scores:
        best_score = result.best_score
        played_score = result.move_scores[played_move]
    else:
        best_score = _static_move_score(board, best_move, evaluate)
        played_score = _static_move_score(board, played_move, evaluate)

    centipawn_loss = calculate_centipawn_loss(
        turn,
        best_score=best_score,
        played_score=played_score,
    )
    classification = classify_centipawn_loss(centipawn_loss)
    explanation = _explain(
        board,
        played_move=played_move,
        best_move=best_move,
        played_san=played_san,
        best_san=best_san,
        classification=classification,
        centipawn_loss=centipawn_loss,
    )
    return MoveAnalysis(
        played_move=played_move,
        best_move=best_move,
        played_san=played_san,
        best_san=best_san,
        played_score=played_score,
        best_score=best_score,
        centipawn_loss=centipawn_loss,
        classification=classification,
        explanation=explanation,
        search_depth=result.completed_depth,
        used_static_fallback=used_static_fallback,
    )


def format_move_analysis(analysis: MoveAnalysis) -> str:
    return "\n".join(
        (
            (
                f"  Coach: {analysis.classification} "
                f"({analysis.centipawn_loss / 100:.2f} pawn loss)"
            ),
            f"  Best move: {analysis.best_san}",
            f"  Why: {analysis.explanation}",
        )
    )
