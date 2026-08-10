"""Move-quality coaching based on root-search centipawn loss."""

from __future__ import annotations

from dataclasses import dataclass

import chess

try:
    from .board.evaluators import DEFAULT_EVALUATOR, Evaluator, resolve_evaluator
    from .engine import SearchResult, analyze_position
    from .tactics import (
        describe_material_swing,
        describe_move_purpose,
        hangs_after,
        material_balance,
    )
except ImportError:
    from board.evaluators import DEFAULT_EVALUATOR, Evaluator, resolve_evaluator
    from engine import SearchResult, analyze_position
    from tactics import (
        describe_material_swing,
        describe_move_purpose,
        hangs_after,
        material_balance,
    )


# How far to look for the punishment, by how bad the move was. A blunder needs
# a longer line to explain than an inaccuracy.
REFUTATION_PLIES = {
    "Best": 0,
    "Excellent": 0,
    "Good": 2,
    "Inaccuracy": 3,
    "Mistake": 5,
    "Blunder": 6,
}
BEST_LINE_PLIES = 1  # what the engine expected in reply to its own choice
DEFAULT_LINE_TIME_LIMIT = 1.5  # total, split across the plies of a line
MIN_PLY_TIME = 0.05  # measured: already enough to find a hanging-queen capture


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
    reason: str = ""
    refutation_san: tuple[str, ...] = ()
    best_line_san: tuple[str, ...] = ()
    material_swing: str | None = None
    mate_for_opponent: bool = False
    missed_mate: bool = False


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


@dataclass(frozen=True)
class Line:
    """A short continuation, in SAN, plus what it does to material."""

    san: tuple[str, ...]
    ends_in_mate: bool
    material_swing: str | None


def build_line(
    board: chess.Board,
    *,
    plies: int,
    time_limit: float,
    evaluator: Evaluator,
    mover: chess.Color,
) -> Line:
    """Play out the engine's preferred continuation from ``board``.

    Re-searches each position rather than reading a principal variation out of
    the search, which keeps ``engine.py`` untouched. The board is restored
    before returning, whatever happens.
    """
    if plies <= 0:
        return Line(san=(), ends_in_mate=False, material_swing=None)

    per_ply = max(MIN_PLY_TIME, time_limit / plies)
    start_material = material_balance(board)
    san: list[str] = []
    material_after: list[int] = []
    pushed = 0
    try:
        for _ in range(plies):
            if board.is_game_over():
                break
            result = analyze_position(
                board,
                depth=4,
                time_limit=per_ply,
                evaluator=evaluator,
            )
            # Never push a move without checking it belongs to this position:
            # an illegal push corrupts the board for every later ply.
            if result.best_move is None or result.best_move not in board.legal_moves:
                break
            san.append(board.san(result.best_move))
            board.push(result.best_move)
            pushed += 1
            material_after.append(material_balance(board))
            if board.is_checkmate():
                break
        ends_in_mate = board.is_checkmate()
        final_material = material_balance(board)
    finally:
        for _ in range(pushed):
            board.pop()

    cut = _cut_index(material_after, start_material, ends_in_mate, len(san))
    # Measure the swing where the line is cut, so the figure always describes
    # the moves actually shown.
    if ends_in_mate:
        swing = None  # a mate line is about the mate, not the material traded
    elif cut:
        swing = describe_material_swing(
            start_material, material_after[cut - 1], mover
        )
    else:
        swing = describe_material_swing(start_material, final_material, mover)

    return Line(
        san=tuple(san[:cut]),
        ends_in_mate=ends_in_mate,
        material_swing=swing,
    )


def _cut_index(
    material_after: list[int],
    start_material: int,
    ends_in_mate: bool,
    length: int,
) -> int:
    """Where to cut a line so it makes its point and stops.

    "Qg5 Nxg5" explains the blunder; the quiet moves the search played after it
    only bury the point, and letting the line run also made the material figure
    describe captures the reader never saw. Cut at the first material change,
    then keep going only while an exchange is still resolving. Mate lines are
    kept whole, since every move in them is forcing.
    """
    if ends_in_mate or not material_after:
        return length
    first_change = next(
        (
            index
            for index, material in enumerate(material_after)
            if material != start_material
        ),
        None,
    )
    if first_change is None:
        return min(2, length)  # nothing swung: show just the expected reply
    cut = first_change + 1
    while cut < len(material_after) and material_after[cut] != material_after[cut - 1]:
        cut += 1
    return cut


def format_line(prefix_san: str, continuation: tuple[str, ...]) -> str:
    return " ".join((prefix_san, *continuation))


def _reason(
    board: chess.Board,
    *,
    played_move: chess.Move,
    played_san: str,
    best_san: str,
    classification: str,
    refutation: Line,
    hanging: tuple[int, int] | None,
) -> str:
    """One sentence on what is wrong with the played move.

    Only reports what was computed. When the search finds no forcing punishment
    the sentence says exactly that instead of inventing a positional story.
    """
    if refutation.ends_in_mate:
        return f"{played_san} allows a forced mate."
    if hanging is not None:
        _square, piece_type = hanging
        square_name = chess.square_name(_square)
        return (
            f"{played_san} leaves your {chess.piece_name(piece_type)} "
            f"on {square_name} en prise."
        )
    if refutation.material_swing is not None:
        return f"{played_san} {refutation.material_swing}."
    if played_san == best_san:
        return f"{played_san} matches the engine's top choice."
    if classification in ("Best", "Excellent"):
        return f"{played_san} is fine; the engine slightly preferred {best_san}."
    return (
        f"No forcing punishment found; the engine simply rates "
        f"{best_san} higher."
    )


def _explain(
    board: chess.Board,
    *,
    played_move: chess.Move,
    best_move: chess.Move,
    played_san: str,
    best_san: str,
    classification: str,
    centipawn_loss: int,
    missed_mate: bool,
    mate_for_opponent: bool,
) -> str:
    if played_move == best_move:
        return f"{played_san} matches the engine's top choice."
    if classification == "Best":
        return (
            f"{played_san} is within 0.10 pawns of the top choice, "
            f"{best_san}."
        )

    best_description = describe_move_purpose(board, best_move)
    if missed_mate:
        return f"The engine preferred {best_san}, which forces mate."
    # Mate scores +-99999, so a pawn figure here would read "1000.34 pawns".
    if mate_for_opponent:
        return (
            f"The engine preferred {best_san}, which {best_description}, "
            f"and avoids the mate."
        )
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
    line_time_limit: float = DEFAULT_LINE_TIME_LIMIT,
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

    # What the opponent does to punish the move, and what the engine expected
    # in reply to its own choice. Both are built after the root search, so the
    # classification can decide how deep to look.
    hanging = hangs_after(board, played_move)
    refutation_plies = REFUTATION_PLIES.get(classification, 0)
    # Split the line budget so the total stays within line_time_limit.
    refutation_budget = line_time_limit * 0.75
    best_line_budget = line_time_limit * 0.25
    board.push(played_move)
    try:
        refutation = build_line(
            board,
            plies=refutation_plies,
            time_limit=refutation_budget,
            evaluator=evaluate,
            mover=turn,
        )
    finally:
        board.pop()

    board.push(best_move)
    try:
        best_line = build_line(
            board,
            plies=BEST_LINE_PLIES,
            time_limit=best_line_budget,
            evaluator=evaluate,
            mover=turn,
        )
        missed_mate = board.is_checkmate() or best_line.ends_in_mate
    finally:
        board.pop()

    reason = _reason(
        board,
        played_move=played_move,
        played_san=played_san,
        best_san=best_san,
        classification=classification,
        refutation=refutation,
        hanging=hanging,
    )
    explanation = _explain(
        board,
        played_move=played_move,
        best_move=best_move,
        played_san=played_san,
        best_san=best_san,
        classification=classification,
        centipawn_loss=centipawn_loss,
        missed_mate=missed_mate,
        mate_for_opponent=refutation.ends_in_mate,
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
        reason=reason,
        refutation_san=refutation.san,
        best_line_san=best_line.san,
        material_swing=refutation.material_swing,
        mate_for_opponent=refutation.ends_in_mate,
        missed_mate=missed_mate,
    )


def format_loss(analysis: MoveAnalysis) -> str:
    """Pawn figure, or a mate verdict when a pawn count would be nonsense.

    A forced mate scores +-99999, which printed as "1000.49 pawn loss".
    """
    if analysis.mate_for_opponent:
        return "allows forced mate"
    if analysis.missed_mate:
        return "misses forced mate"
    return f"{analysis.centipawn_loss / 100:.2f} pawn loss"


def format_move_analysis(analysis: MoveAnalysis) -> str:
    lines = [
        f"  Coach: {analysis.classification} ({format_loss(analysis)})",
        f"  Best move: {analysis.best_san}",
    ]
    if analysis.used_static_fallback:
        # No search depth completed, so the verdict came from a static
        # evaluation that cannot see the opponent's reply. Say so.
        lines.append(
            "  NOTE: static estimate only - raise --coach-time-limit"
        )
    if analysis.reason:
        lines.append(f"  Why: {analysis.reason}")
    if analysis.refutation_san:
        line = format_line(analysis.played_san, analysis.refutation_san)
        suffix = f"  ({analysis.material_swing})" if analysis.material_swing else ""
        lines.append(f"  Line: {line}{suffix}")
    if analysis.best_line_san:
        expected = format_line(analysis.best_san, analysis.best_line_san)
        lines.append(f"  Expected: {expected}")
    lines.append(f"  Engine: {analysis.explanation}")
    return "\n".join(lines)
