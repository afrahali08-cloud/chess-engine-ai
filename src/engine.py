"""
Chess engine — minimax search with alpha-beta pruning.
Optimizations included:
  1. Move ordering (MVV-LVA captures first, then checks, then promotions)
  2. Quiescence search (no more mid-capture-sequence blindness)
  3. Transposition table (memoization — never re-evaluate same position)
  4. Iterative deepening (time-based search instead of fixed depth)
"""

from __future__ import annotations

from math import inf
from time import time
from typing import Any

EVAL_MODE = "neural"  # options: "handcrafted", "ridge", "neural"

if EVAL_MODE == "neural":
    try:
        from .board.neural_evaluation import evaluate_neural_board as evaluate_board
    except ImportError:
        from board.neural_evaluation import evaluate_neural_board as evaluate_board

elif EVAL_MODE == "ridge":
    try:
        from .board.learned_evaluation import evaluate_learned_board as evaluate_board
    except ImportError:
        from board.learned_evaluation import evaluate_learned_board as evaluate_board

else:  # handcrafted
    try:
        from .board.evaluation import evaluate_board
    except ImportError:
        from board.evaluation import evaluate_board


# ==============================================================================
# TRANSPOSITION TABLE
# stores: position_key -> (depth, score, flag)
# flag: 'exact' | 'lower' | 'upper' (for alpha-beta bounds)
# ==============================================================================
_tt: dict = {}

def _tt_key(board: Any) -> str:
    return " ".join(board.fen().split()[:4])

def _tt_store(key: str, depth: int, score: float, flag: str):
    _tt[key] = (depth, score, flag)

def _tt_lookup(key: str, depth: int, alpha: float, beta: float):
    if key not in _tt:
        return None
    stored_depth, score, flag = _tt[key]
    if stored_depth < depth:
        return None  # stored at shallower depth, not reliable enough
    if flag == 'exact':
        return score
    if flag == 'lower' and score >= beta:
        return score
    if flag == 'upper' and score <= alpha:
        return score
    return None

# ==============================================================================
# MOVE ORDERING
# search captures first (most valuable victim, least valuable attacker)
# then checks, then quiet moves
# better ordering = more alpha-beta cutoffs = faster search
# ==============================================================================
PIECE_ORDER_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000}

def _move_score(board: Any, move: Any) -> int:
    score = 0

    # captures — MVV-LVA
    if board.is_capture(move):
        victim   = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim and attacker:
            score += 10 * PIECE_ORDER_VALUES.get(victim.piece_type, 0) \
                       - PIECE_ORDER_VALUES.get(attacker.piece_type, 0)
        else:
            score += 500  # en passant

    # promotions
    if move.promotion:
        score += PIECE_ORDER_VALUES.get(move.promotion, 0)

    # checks — push forward in ordering but don't pay the full cost of
    # board.push/pop for every move; only do it if capture score is low
    if score < 100:
        board.push(move)
        if board.is_check():
            score += 50
        board.pop()

    return score


def _order_moves(board: Any, moves: list) -> list:
    return sorted(moves, key=lambda m: _move_score(board, m), reverse=True)

# ==============================================================================
# QUIESCENCE SEARCH
# called at depth=0 instead of returning evaluate_board() directly
# keeps searching captures until the position is "quiet" (no captures left)
# eliminates the "sees capture but not recapture" class of blunders
# ==============================================================================
def _quiescence(board: Any, alpha: float, beta: float) -> float:
    """Search tactical continuations using White-relative evaluation scores."""
    if board.is_game_over():
        return evaluate_board(board)

    in_check = board.is_check()
    moves = list(board.legal_moves) if in_check else [
        move for move in board.legal_moves if board.is_capture(move)
    ]
    moves = _order_moves(board, moves)

    if board.turn:
        best_score = -inf if in_check else evaluate_board(board)
        if best_score >= beta:
            return best_score
        alpha = max(alpha, best_score)

        for move in moves:
            board.push(move)
            score = _quiescence(board, alpha, beta)
            board.pop()

            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break

        return best_score

    best_score = inf if in_check else evaluate_board(board)
    if best_score <= alpha:
        return best_score
    beta = min(beta, best_score)

    for move in moves:
        board.push(move)
        score = _quiescence(board, alpha, beta)
        board.pop()

        best_score = min(best_score, score)
        beta = min(beta, score)
        if beta <= alpha:
            break

    return best_score

# ==============================================================================
# MINIMAX WITH ALPHA-BETA PRUNING
# ==============================================================================
def minimax(
    board: Any,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
) -> float:

    # transposition table lookup
    key = _tt_key(board)
    cached = _tt_lookup(key, depth, alpha, beta)
    if cached is not None:
        return cached

    original_alpha = alpha
    original_beta = beta

    # base case — use quiescence search instead of raw eval
    if depth <= 0 or board.is_game_over():
        score = _quiescence(board, alpha, beta)
        if score <= original_alpha:
            flag = 'upper'
        elif score >= original_beta:
            flag = 'lower'
        else:
            flag = 'exact'
        _tt_store(key, depth, score, flag)
        return score

    moves = _order_moves(board, list(board.legal_moves))

    if maximizing_player:
        best_score = -inf
        for move in moves:
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break  # beta cutoff

    else:
        best_score = inf
        for move in moves:
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()

            best_score = min(best_score, score)
            beta = min(beta, score)
            if beta <= alpha:
                break  # alpha cutoff

    if best_score <= original_alpha:
        flag = 'upper'
    elif best_score >= original_beta:
        flag = 'lower'
    else:
        flag = 'exact'
    _tt_store(key, depth, best_score, flag)
    return best_score

# ==============================================================================
# ITERATIVE DEEPENING
# searches depth 1, 2, 3... until time runs out
# always returns best move found so far — never wastes thinking time
# ==============================================================================
def choose_best_move(
    board: Any,
    depth: int = 5,
    time_limit: float = 5.0
) -> tuple[Any, float]:

    if not list(board.legal_moves) or board.is_game_over():
        return None, evaluate_board(board)

    # clear transposition table between moves
    _tt.clear()

    best_move  = None
    best_score = -inf if board.turn else inf
    start      = time()

    for current_depth in range(1, depth + 1):
        if time() - start > time_limit:
            break  # out of time — return best found so far

        moves = _order_moves(board, list(board.legal_moves))
        depth_best_move  = None
        depth_best_score = -inf if board.turn else inf

        if board.turn:  # white maximizes
            for move in moves:
                board.push(move)
                score = minimax(board, current_depth - 1, -inf, inf, False)
                board.pop()

                if score > depth_best_score:
                    depth_best_score = score
                    depth_best_move  = move
        else:  # black minimizes
            for move in moves:
                board.push(move)
                score = minimax(board, current_depth - 1, -inf, inf, True)
                board.pop()

                if score < depth_best_score:
                    depth_best_score = score
                    depth_best_move  = move

        if depth_best_move is not None:
            best_move  = depth_best_move
            best_score = depth_best_score

    return best_move, best_score
