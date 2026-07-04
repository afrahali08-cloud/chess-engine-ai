

import chess

PIECE_VALUES_MG = {
    chess.PAWN:   100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0
}

PIECE_VALUES_EG = {
    chess.PAWN:   130,
    chess.KNIGHT: 310,
    chess.BISHOP: 320,
    chess.ROOK:   500,
    chess.QUEEN:  900,
    chess.KING:   0
}

PST_MG = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
         0,  0,  0,  5,  5,  0,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         5, 10, 10, 10, 10, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}

PST_EG = {
    chess.PAWN: [
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        30, 30, 35, 40, 40, 35, 30, 30,
        20, 20, 25, 30, 30, 25, 20, 20,
        10, 10, 15, 20, 20, 15, 10, 10,
         5,  5,  5, 10, 10,  5,  5,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20
    ],
    chess.ROOK: [
         0,  0,  0,  5,  5,  0,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
         5, 10, 10, 10, 10, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -50,-40,-30,-20,-20,-30,-40,-50,
        -30,-20,-10,  0,  0,-10,-20,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 30, 40, 40, 30,-10,-30,
        -30,-10, 20, 30, 30, 20,-10,-30,
        -30,-30,  0,  0,  0,  0,-30,-30,
        -50,-30,-30,-30,-30,-30,-30,-50
    ]
}

FILE_MASKS = [0x0101010101010101 << i for i in range(8)]
ADJACENT_FILE_MASKS = [
    FILE_MASKS[1] if i == 0
    else (FILE_MASKS[6] if i == 7
          else FILE_MASKS[i - 1] | FILE_MASKS[i + 1])
    for i in range(8)
]

def get_game_phase(board):
    phase = 0
    phase += len(board.pieces(chess.KNIGHT, chess.WHITE)) + len(board.pieces(chess.KNIGHT, chess.BLACK))
    phase += len(board.pieces(chess.BISHOP, chess.WHITE)) + len(board.pieces(chess.BISHOP, chess.BLACK))
    phase += (len(board.pieces(chess.ROOK,  chess.WHITE)) + len(board.pieces(chess.ROOK,  chess.BLACK))) * 2
    phase += (len(board.pieces(chess.QUEEN, chess.WHITE)) + len(board.pieces(chess.QUEEN, chess.BLACK))) * 4
    return min(phase, 24)  # cap at 24

def calculate_material_and_activity(board):
    """ΔM + ΔA — material value + positional bonus, tapered by phase."""
    mg_w, mg_b, eg_w, eg_b = 0, 0, 0, 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue

        pt    = piece.piece_type
        color = piece.color
        # mirror square for black so tables are always from white's perspective
        p_sq  = square if color == chess.WHITE else chess.square_mirror(square)

        if color == chess.WHITE:
            mg_w += PIECE_VALUES_MG[pt] + PST_MG[pt][p_sq]
            eg_w += PIECE_VALUES_EG[pt] + PST_EG[pt][p_sq]
        else:
            mg_b += PIECE_VALUES_MG[pt] + PST_MG[pt][p_sq]
            eg_b += PIECE_VALUES_EG[pt] + PST_EG[pt][p_sq]

    return (mg_w - mg_b), (eg_w - eg_b)


def calculate_pawn_structure_delta(board):
    """ΔS — doubled, isolated, and passed pawn bonuses/penalties."""
    w_pawns = int(board.pawns & board.occupied_co[chess.WHITE])
    b_pawns = int(board.pawns & board.occupied_co[chess.BLACK])

    mg_delta, eg_delta = 0, 0

    # White pawns
    for sq in board.pieces(chess.PAWN, chess.WHITE):
        file_idx = chess.square_file(sq)
        rank_idx = chess.square_rank(sq)

        # doubled pawn penalty
        if bin(w_pawns & FILE_MASKS[file_idx]).count('1') > 1:
            mg_delta -= 15; eg_delta -= 20

        # isolated pawn penalty
        if not (w_pawns & ADJACENT_FILE_MASKS[file_idx]):
            mg_delta -= 10; eg_delta -= 15

        # passed pawn bonus (scales by rank)
        front_mask = 0
        for r in range(rank_idx + 1, 8):
            front_mask |= (1 << (r * 8 + file_idx))
            if file_idx > 0: front_mask |= (1 << (r * 8 + file_idx - 1))
            if file_idx < 7: front_mask |= (1 << (r * 8 + file_idx + 1))
        if not (b_pawns & front_mask):
            bonus = (rank_idx ** 2) * 10
            mg_delta += bonus
            eg_delta += int(bonus * 1.5)

    # Black pawns
    for sq in board.pieces(chess.PAWN, chess.BLACK):
        file_idx = chess.square_file(sq)
        rank_idx = 7 - chess.square_rank(sq)  # invert for black

        if bin(b_pawns & FILE_MASKS[file_idx]).count('1') > 1:
            mg_delta += 15; eg_delta += 20

        if not (b_pawns & ADJACENT_FILE_MASKS[file_idx]):
            mg_delta += 10; eg_delta += 15

        front_mask = 0
        for r in range(chess.square_rank(sq) - 1, -1, -1):
            front_mask |= (1 << (r * 8 + file_idx))
            if file_idx > 0: front_mask |= (1 << (r * 8 + file_idx - 1))
            if file_idx < 7: front_mask |= (1 << (r * 8 + file_idx + 1))
        if not (w_pawns & front_mask):
            bonus = (rank_idx ** 2) * 10
            mg_delta -= bonus
            eg_delta -= int(bonus * 1.5)

    return mg_delta, eg_delta


def calculate_king_safety_delta(board):
    """ΔK — pawn shield and open line penalties near king (middlegame only)."""
    mg_delta = 0

    # White king shield
    w_king_sq = board.king(chess.WHITE)
    if w_king_sq is not None and chess.square_rank(w_king_sq) <= 1:
        w_file = chess.square_file(w_king_sq)
        shield = [w_king_sq + 8]
        if w_file > 0: shield.append(w_king_sq + 7)
        if w_file < 7: shield.append(w_king_sq + 9)
        for sq in shield:
            if 0 <= sq < 64 and board.piece_at(sq) == chess.Piece(chess.PAWN, chess.WHITE):
                mg_delta += 10
            else:
                mg_delta -= 15

    # Black king shield
    b_king_sq = board.king(chess.BLACK)
    if b_king_sq is not None and chess.square_rank(b_king_sq) >= 6:
        b_file = chess.square_file(b_king_sq)
        shield = [b_king_sq - 8]
        if b_file > 0: shield.append(b_king_sq - 9)
        if b_file < 7: shield.append(b_king_sq - 7)
        for sq in shield:
            if 0 <= sq < 64 and board.piece_at(sq) == chess.Piece(chess.PAWN, chess.BLACK):
                mg_delta -= 10
            else:
                mg_delta += 15

    return mg_delta, 0  # king safety only middlegame 


def evaluate_board(board):
    """
    Returns a centipawn score for the position.
    Positive = white is better, negative = black is better.
    Magnitude: ~100 per pawn, checkmate = ±99999.
    """
    # terminal states
    if board.is_checkmate():
        return -99999 if board.turn == chess.WHITE else 99999

    if (board.is_stalemate() or
            board.is_insufficient_material() or
            board.is_seventyfive_moves() or
            board.is_fivefold_repetition()):
        return 0

    # gather deltas
    ma_mg, ma_eg = calculate_material_and_activity(board)
    s_mg,  s_eg  = calculate_pawn_structure_delta(board)
    k_mg,  k_eg  = calculate_king_safety_delta(board)

    # sum middlegame and endgame totals
    total_mg = ma_mg + s_mg + k_mg
    total_eg = ma_eg + s_eg + k_eg

    
    phase      = get_game_phase(board)   
    mg_weight  = phase
    eg_weight  = 24 - phase

    blended = (total_mg * mg_weight + total_eg * eg_weight) / 24.0
    return int(blended)



if __name__ == "__main__":
   
    b = chess.Board()
    print(f"Start:               {evaluate_board(b):+d} cp")

    
    b = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print(f"Black missing queen: {evaluate_board(b):+d} cp")

    
    b = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    print(f"White missing queen: {evaluate_board(b):+d} cp")