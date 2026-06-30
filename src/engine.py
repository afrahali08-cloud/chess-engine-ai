"""Minimax search with alpha-beta pruning for python-chess boards."""

from __future__ import annotations

from math import inf
from typing import Any


try:
    from .evaluation import evaluate_board
except ImportError:
    try:
        from evaluation import evaluate_board
    except ImportError:

        def evaluate_board(board: Any) -> float:
            raise NotImplementedError(
                "evaluate_board(board) is not available yet. "
                "Add it to src/evaluation.py before running the engine."
            )


def minimax(
    board: Any,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
) -> float:
    """Return the minimax evaluation for the current board position."""
    if depth <= 0 or board.is_game_over():
        return evaluate_board(board)

    if maximizing_player:
        best_score = -inf

        for move in list(board.legal_moves):
            board.push(move)
            score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            best_score = max(best_score, score)
            alpha = max(alpha, score)

            if beta <= alpha:
                break

        return best_score

    best_score = inf

    for move in list(board.legal_moves):
        board.push(move)
        score = minimax(board, depth - 1, alpha, beta, True)
        board.pop()

        best_score = min(best_score, score)
        beta = min(beta, score)

        if beta <= alpha:
            break

    return best_score


def choose_best_move(board: Any, depth: int = 2) -> tuple[Any | None, float]:
    """Choose the best legal move and return it with its evaluation score."""
    legal_moves = list(board.legal_moves)

    if not legal_moves or board.is_game_over():
        return None, evaluate_board(board)

    search_depth = max(depth - 1, 0)

    # In python-chess, board.turn is True for White and False for Black.
    if board.turn:
        best_move = None
        best_score = -inf

        for move in legal_moves:
            board.push(move)
            score = minimax(board, search_depth, -inf, inf, False)
            board.pop()

            if score > best_score:
                best_score = score
                best_move = move

        return best_move, best_score

    best_move = None
    best_score = inf

    for move in legal_moves:
        board.push(move)
        score = minimax(board, search_depth, -inf, inf, True)
        board.pop()

        if score < best_score:
            best_score = score
            best_move = move

    return best_move, best_score
