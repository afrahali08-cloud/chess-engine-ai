"""Feature encoding shared by model training and runtime evaluation."""

from __future__ import annotations

import chess


PIECE_SQUARE_FEATURES = 6 * 64
TURN_FEATURE = PIECE_SQUARE_FEATURES
KINGSIDE_CASTLING_FEATURE = TURN_FEATURE + 1
QUEENSIDE_CASTLING_FEATURE = TURN_FEATURE + 2
FEATURE_COUNT = QUEENSIDE_CASTLING_FEATURE + 1


def board_features(board: chess.Board) -> list[tuple[int, float]]:
    """Encode a board using color-symmetric piece-square features."""
    features: list[tuple[int, float]] = []

    for square, piece in board.piece_map().items():
        normalized_square = (
            square if piece.color == chess.WHITE else chess.square_mirror(square)
        )
        feature_index = (piece.piece_type - 1) * 64 + normalized_square
        feature_value = 1.0 if piece.color == chess.WHITE else -1.0
        features.append((feature_index, feature_value))

    features.append((TURN_FEATURE, 1.0 if board.turn == chess.WHITE else -1.0))

    kingside_rights = int(board.has_kingside_castling_rights(chess.WHITE)) - int(
        board.has_kingside_castling_rights(chess.BLACK)
    )
    if kingside_rights:
        features.append((KINGSIDE_CASTLING_FEATURE, float(kingside_rights)))

    queenside_rights = int(board.has_queenside_castling_rights(chess.WHITE)) - int(
        board.has_queenside_castling_rights(chess.BLACK)
    )
    if queenside_rights:
        features.append((QUEENSIDE_CASTLING_FEATURE, float(queenside_rights)))

    return features


def fen_features(fen: str) -> list[tuple[int, float]]:
    """Parse a FEN and return its model features."""
    return board_features(chess.Board(fen))
