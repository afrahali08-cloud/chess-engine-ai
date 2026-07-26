from __future__ import annotations
import chess
import numpy as np


# -------------------------------------------------------------------
# Feature dimensions
# -------------------------------------------------------------------
PIECE_SQUARE_DIM = 384   # 6 piece types x 64 squares
KNOWLEDGE_DIM    = 23    # hand-crafted knowledge features
TOTAL_FEATURES   = PIECE_SQUARE_DIM + KNOWLEDGE_DIM


# -------------------------------------------------------------------
# File masks for pawn structure (reused from evaluation.py logic)
# -------------------------------------------------------------------
FILE_MASKS = [0x0101010101010101 << i for i in range(8)]
ADJACENT_FILE_MASKS = [
    FILE_MASKS[1] if i == 0
    else (FILE_MASKS[6] if i == 7
          else FILE_MASKS[i-1] | FILE_MASKS[i+1])
    for i in range(8)
]


def board_to_features(board: chess.Board) -> np.ndarray:
    """
    Convert a chess.Board into a feature vector of length TOTAL_FEATURES.
    
    Returns:
        numpy array of shape (TOTAL_FEATURES,) with dtype float32
        Positive values favor white, negative favor black.
    """
    features = np.zeros(TOTAL_FEATURES, dtype=np.float32)

    # ------------------------------------------------------------------
    # Group 1: Piece-square features (0-383)
    # Same encoding as his model for comparability
    # Each piece gets +1.0 (white) or -1.0 (black) at its square index
    # Square is mirrored for black so tables are always from white's view
    # ------------------------------------------------------------------
    for square, piece in board.piece_map().items():
        mirrored = square if piece.color == chess.WHITE \
                   else chess.square_mirror(square)
        idx = (piece.piece_type - 1) * 64 + mirrored
        features[idx] += 1.0 if piece.color == chess.WHITE else -1.0

    # ------------------------------------------------------------------
    # Group 2: Material count per piece type (384-393)
    # White count minus black count for each piece type
    # Gives the model explicit material balance signal
    # ------------------------------------------------------------------
    base = PIECE_SQUARE_DIM
    for i, piece_type in enumerate([chess.PAWN, chess.KNIGHT, chess.BISHOP,
                                     chess.ROOK, chess.QUEEN]):
        w = len(board.pieces(piece_type, chess.WHITE))
        b = len(board.pieces(piece_type, chess.BLACK))
        features[base + i] = w - b  # positive = white has more

    # ------------------------------------------------------------------
    # Group 3: Pawn structure (394-397)
    # ------------------------------------------------------------------
    base = PIECE_SQUARE_DIM + 5
    w_pawns = int(board.pawns & board.occupied_co[chess.WHITE])
    b_pawns = int(board.pawns & board.occupied_co[chess.BLACK])

    w_doubled = w_isolated = b_doubled = b_isolated = 0

    for f in range(8):
        w_on_file = bin(w_pawns & FILE_MASKS[f]).count('1')
        b_on_file = bin(b_pawns & FILE_MASKS[f]).count('1')
        if w_on_file > 1: w_doubled += w_on_file - 1
        if b_on_file > 1: b_doubled += b_on_file - 1
        if w_on_file > 0 and not (w_pawns & ADJACENT_FILE_MASKS[f]):
            w_isolated += 1
        if b_on_file > 0 and not (b_pawns & ADJACENT_FILE_MASKS[f]):
            b_isolated += 1

    features[base + 0] = w_doubled  - b_doubled   # doubled pawns delta
    features[base + 1] = w_isolated - b_isolated  # isolated pawns delta

    # passed pawns
    w_passed = b_passed = 0
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        front_mask = 0
        for r in range(rank_idx + 1, 8):
            front_mask |= (1 << (r * 8 + file_idx))
            if file_idx > 0: front_mask |= (1 << (r * 8 + file_idx - 1))
            if file_idx < 7: front_mask |= (1 << (r * 8 + file_idx + 1))
        if not (b_pawns & front_mask):
            w_passed += 1

    for sq in board.pieces(chess.PAWN, chess.BLACK):
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)
        front_mask = 0
        for r in range(chess.square_rank(sq) - 1, -1, -1):
            front_mask |= (1 << (r * 8 + file_idx))
            if file_idx > 0: front_mask |= (1 << (r * 8 + file_idx - 1))
            if file_idx < 7: front_mask |= (1 << (r * 8 + file_idx + 1))
        if not (w_pawns & front_mask):
            b_passed += 1

    features[base + 2] = w_passed - b_passed  # passed pawns delta

    # ------------------------------------------------------------------
    # Group 4: King safety (397-398)
    # 1.0 = castled, 0.0 = not castled, for each side
    # ------------------------------------------------------------------
    base = PIECE_SQUARE_DIM + 8
    w_king = board.king(chess.WHITE)
    b_king = board.king(chess.BLACK)
    features[base + 0] = 1.0 if w_king in (chess.G1, chess.C1) else 0.0
    features[base + 1] = 1.0 if b_king in (chess.G8, chess.C8) else 0.0

    # ------------------------------------------------------------------
    # Group 5: Game phase (399)
    # 1.0 = full middlegame, 0.0 = endgame
    # Tells the model what stage of the game it's evaluating
    # ------------------------------------------------------------------
    base = PIECE_SQUARE_DIM + 10
    phase = 0
    phase += len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
    phase += len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))
    phase += (len(board.pieces(chess.ROOK, chess.WHITE)) + len(board.pieces(chess.ROOK, chess.BLACK))) * 2
    phase += (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))) * 4
    features[base] = min(phase, 24) / 24.0  # normalized 0-1

    # ------------------------------------------------------------------
    # Group 6: Turn and castling rights (400-406)
    # ------------------------------------------------------------------
    base = PIECE_SQUARE_DIM + 11
    features[base + 0] = 1.0 if board.turn == chess.WHITE else -1.0
    features[base + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    features[base + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    features[base + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    features[base + 4] = float(board.has_queenside_castling_rights(chess.BLACK))

    return features


def fen_to_features(fen: str) -> np.ndarray:
    """Parse a FEN string and return its feature vector."""
    return board_to_features(chess.Board(fen))