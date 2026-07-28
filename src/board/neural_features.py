"""Feature encoding shared by neural evaluator training and inference."""

from __future__ import annotations

import chess
import numpy as np


FEATURE_ENCODING = "neural_piece_square_knowledge_v2"
PIECE_SQUARE_DIM = 6 * 64
MATERIAL_OFFSET = PIECE_SQUARE_DIM
PAWN_STRUCTURE_OFFSET = MATERIAL_OFFSET + 5
KING_SAFETY_OFFSET = PAWN_STRUCTURE_OFFSET + 3
GAME_PHASE_OFFSET = KING_SAFETY_OFFSET + 2
TURN_OFFSET = GAME_PHASE_OFFSET + 1
CASTLING_OFFSET = TURN_OFFSET + 1
TOTAL_FEATURES = CASTLING_OFFSET + 4

FILE_MASKS = [0x0101010101010101 << file_index for file_index in range(8)]
ADJACENT_FILE_MASKS = [
    FILE_MASKS[1]
    if file_index == 0
    else (
        FILE_MASKS[6]
        if file_index == 7
        else FILE_MASKS[file_index - 1] | FILE_MASKS[file_index + 1]
    )
    for file_index in range(8)
]


def _pawn_structure_features(
    board: chess.Board,
) -> tuple[int, int, int]:
    white_pawns = int(board.pawns & board.occupied_co[chess.WHITE])
    black_pawns = int(board.pawns & board.occupied_co[chess.BLACK])
    white_doubled = white_isolated = black_doubled = black_isolated = 0

    for file_index in range(8):
        white_on_file = (white_pawns & FILE_MASKS[file_index]).bit_count()
        black_on_file = (black_pawns & FILE_MASKS[file_index]).bit_count()
        white_doubled += max(0, white_on_file - 1)
        black_doubled += max(0, black_on_file - 1)
        if white_on_file and not (
            white_pawns & ADJACENT_FILE_MASKS[file_index]
        ):
            white_isolated += 1
        if black_on_file and not (
            black_pawns & ADJACENT_FILE_MASKS[file_index]
        ):
            black_isolated += 1

    white_passed = black_passed = 0
    for square in board.pieces(chess.PAWN, chess.WHITE):
        file_index = chess.square_file(square)
        front_mask = 0
        for rank_index in range(chess.square_rank(square) + 1, 8):
            front_mask |= 1 << chess.square(file_index, rank_index)
            if file_index > 0:
                front_mask |= 1 << chess.square(file_index - 1, rank_index)
            if file_index < 7:
                front_mask |= 1 << chess.square(file_index + 1, rank_index)
        if not black_pawns & front_mask:
            white_passed += 1

    for square in board.pieces(chess.PAWN, chess.BLACK):
        file_index = chess.square_file(square)
        front_mask = 0
        for rank_index in range(chess.square_rank(square) - 1, -1, -1):
            front_mask |= 1 << chess.square(file_index, rank_index)
            if file_index > 0:
                front_mask |= 1 << chess.square(file_index - 1, rank_index)
            if file_index < 7:
                front_mask |= 1 << chess.square(file_index + 1, rank_index)
        if not white_pawns & front_mask:
            black_passed += 1

    return (
        white_doubled - black_doubled,
        white_isolated - black_isolated,
        white_passed - black_passed,
    )


def _game_phase(board: chess.Board) -> float:
    phase = 0
    for color in chess.COLORS:
        phase += len(board.pieces(chess.KNIGHT, color))
        phase += len(board.pieces(chess.BISHOP, color))
        phase += 2 * len(board.pieces(chess.ROOK, color))
        phase += 4 * len(board.pieces(chess.QUEEN, color))
    return min(phase, 24) / 24.0


def board_to_features(board: chess.Board) -> np.ndarray:
    """Convert a board to the fixed-width neural feature vector."""
    features = np.zeros(TOTAL_FEATURES, dtype=np.float32)

    for square, piece in board.piece_map().items():
        normalized_square = (
            square if piece.color == chess.WHITE else chess.square_mirror(square)
        )
        feature_index = (piece.piece_type - 1) * 64 + normalized_square
        features[feature_index] += 1.0 if piece.color == chess.WHITE else -1.0

    material_types = (
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
    )
    for index, piece_type in enumerate(material_types):
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        features[MATERIAL_OFFSET + index] = white_count - black_count

    features[
        PAWN_STRUCTURE_OFFSET : PAWN_STRUCTURE_OFFSET + 3
    ] = _pawn_structure_features(board)

    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    features[KING_SAFETY_OFFSET] = float(white_king in (chess.G1, chess.C1))
    features[KING_SAFETY_OFFSET + 1] = float(black_king in (chess.G8, chess.C8))
    features[GAME_PHASE_OFFSET] = _game_phase(board)
    features[TURN_OFFSET] = 1.0 if board.turn == chess.WHITE else -1.0
    features[CASTLING_OFFSET] = float(
        board.has_kingside_castling_rights(chess.WHITE)
    )
    features[CASTLING_OFFSET + 1] = float(
        board.has_queenside_castling_rights(chess.WHITE)
    )
    features[CASTLING_OFFSET + 2] = float(
        board.has_kingside_castling_rights(chess.BLACK)
    )
    features[CASTLING_OFFSET + 3] = float(
        board.has_queenside_castling_rights(chess.BLACK)
    )

    return features


def fen_to_features(fen: str) -> np.ndarray:
    """Parse a FEN string and return its neural feature vector."""
    return board_to_features(chess.Board(fen))
