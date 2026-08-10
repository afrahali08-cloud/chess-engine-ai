"""Search-free board facts used to explain why a move was good or bad.

Everything here is derived from the position, never guessed. If a fact cannot be
computed it is reported as ``None`` and the caller says nothing rather than
inventing a plausible-sounding chess reason.
"""

from __future__ import annotations

import chess


# Same scale the hand-crafted evaluator uses, so swings read in familiar units.
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}
BACK_RANKS = {chess.WHITE: 0, chess.BLACK: 7}


def material_balance(board: chess.Board) -> int:
    """White-relative material in centipawns."""
    total = 0
    for piece_type, value in PIECE_VALUES.items():
        if not value:
            continue
        total += value * len(board.pieces(piece_type, chess.WHITE))
        total -= value * len(board.pieces(piece_type, chess.BLACK))
    return total


def _name(piece_type: chess.PieceType) -> str:
    return chess.piece_name(piece_type)


def _article(word: str) -> str:
    return "an" if word[0] in "aeiou" else "a"


PIECE_NAME_TOLERANCE = 60  # cp; wider than this and a piece name would mislead


def _swing_noun(magnitude: int) -> str:
    """Name a material amount without overstating what it is.

    A 580cp swing is queen-for-knight, not "a rook". Rather than pick the
    nearest piece and imply something false, fall back to a plain pawn count.
    """
    for piece_type, value in PIECE_VALUES.items():
        if value and abs(magnitude - value) <= PIECE_NAME_TOLERANCE:
            return f"{_article(_name(piece_type))} {_name(piece_type)}"
    pawns = magnitude / PIECE_VALUES[chess.PAWN]
    if abs(pawns - round(pawns)) < 0.15 and round(pawns) > 1:
        return f"{round(pawns)} pawns"
    return f"{pawns:.1f} pawns of material"


def describe_material_swing(
    before: int,
    after: int,
    mover: chess.Color,
) -> str | None:
    """Phrase a material change from the mover's point of view.

    ``None`` when nothing changed, so the caller can stay silent instead of
    padding the explanation.
    """
    delta = after - before
    if mover == chess.BLACK:
        delta = -delta
    if abs(delta) < PIECE_VALUES[chess.PAWN]:
        return None
    return f"{'wins' if delta > 0 else 'loses'} {_swing_noun(abs(delta))}"


def attackers_and_defenders(
    board: chess.Board,
    square: chess.Square,
    owner: chess.Color,
) -> tuple[int, int]:
    """Count of enemy attackers and friendly defenders of ``square``."""
    return (
        len(board.attackers(not owner, square)),
        len(board.attackers(owner, square)),
    )


def cheapest_attacker_value(
    board: chess.Board,
    square: chess.Square,
    owner: chess.Color,
) -> int | None:
    values = [
        PIECE_VALUES[board.piece_at(attacker).piece_type]
        for attacker in board.attackers(not owner, square)
        if board.piece_at(attacker) is not None
    ]
    return min(values) if values else None


def hanging_pieces(board: chess.Board, owner: chess.Color) -> list[chess.Square]:
    """Squares where ``owner`` has a piece that can be profitably taken.

    Deliberately conservative: a piece counts as hanging only when it is
    undefended, or when its cheapest attacker is worth clearly less than it.
    That avoids calling a defended knight "hanging" just because a rook eyes it.
    """
    found: list[chess.Square] = []
    for square, piece in board.piece_map().items():
        if piece.color != owner or piece.piece_type == chess.KING:
            continue
        attackers, defenders = attackers_and_defenders(board, square, owner)
        if not attackers:
            continue
        value = PIECE_VALUES[piece.piece_type]
        cheapest = cheapest_attacker_value(board, square, owner)
        if not defenders:
            found.append(square)
        elif cheapest is not None and cheapest + 50 < value:
            found.append(square)
    found.sort(
        key=lambda sq: PIECE_VALUES[board.piece_at(sq).piece_type], reverse=True
    )
    return found


def most_valuable_hanging(
    board: chess.Board,
    owner: chess.Color,
) -> tuple[chess.Square, chess.PieceType] | None:
    squares = hanging_pieces(board, owner)
    if not squares:
        return None
    square = squares[0]
    return square, board.piece_at(square).piece_type


def hangs_after(
    board: chess.Board,
    move: chess.Move,
) -> tuple[chess.Square, chess.PieceType] | None:
    """The mover's most valuable piece left en prise once ``move`` is played."""
    mover = board.turn
    board.push(move)
    try:
        return most_valuable_hanging(board, mover)
    finally:
        board.pop()


def describe_move_purpose(board: chess.Board, move: chess.Move) -> str:
    """What a move concretely does, in the position it is played.

    Replaces the old catch-all "improves the engine evaluation". Every clause
    here is checked against the board.
    """
    details: list[str] = []
    piece = board.piece_at(move.from_square)
    mover = board.turn

    if board.is_capture(move):
        captured = board.piece_at(move.to_square)
        if captured is None and board.is_en_passant(move):
            details.append("captures a pawn")
        elif captured is not None:
            details.append(f"captures {_article(_name(captured.piece_type))} "
                           f"{_name(captured.piece_type)}")
    if move.promotion:
        details.append(f"promotes to {_article(_name(move.promotion))} "
                       f"{_name(move.promotion)}")
    if board.is_castling(move):
        details.append("castles the king to safety")
    elif (
        piece is not None
        and piece.piece_type in (chess.KNIGHT, chess.BISHOP)
        and chess.square_rank(move.from_square) == BACK_RANKS[mover]
        and chess.square_rank(move.to_square) != BACK_RANKS[mover]
    ):
        details.append(f"develops {_article(_name(piece.piece_type))} "
                       f"{_name(piece.piece_type)}")

    # Only mention a rescue when it is the point of the move; after a capture
    # it reads as redundant padding ("captures a pawn and saves the pawn").
    if not details:
        rescued = _rescues_hanging_piece(board, move)
        if rescued is not None:
            details.append(f"saves {_article(_name(rescued))} {_name(rescued)}")

    board.push(move)
    try:
        if board.is_checkmate():
            details.append("delivers checkmate")
        elif board.is_check():
            details.append("gives check")
    finally:
        board.pop()

    if not details:
        # Everything checkable came back negative, so say only that.
        return "is a quiet move"
    if len(details) == 1:
        return details[0]
    return ", ".join(details[:-1]) + f" and {details[-1]}"


def _rescues_hanging_piece(
    board: chess.Board,
    move: chess.Move,
) -> chess.PieceType | None:
    """Piece type that was hanging before ``move`` and is safe after it."""
    mover = board.turn
    before = most_valuable_hanging(board, mover)
    if before is None:
        return None
    before_square, before_type = before
    board.push(move)
    try:
        after = {square for square in hanging_pieces(board, mover)}
    finally:
        board.pop()
    # Either the piece moved away, or it is no longer profitably takeable.
    if move.from_square == before_square and move.to_square not in after:
        return before_type
    if before_square not in after and move.from_square != before_square:
        return before_type
    return None
