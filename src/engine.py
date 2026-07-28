"""Minimax chess search with alpha-beta pruning and a hard deadline."""

from __future__ import annotations

from math import inf
from time import monotonic
from typing import Any, Callable

try:
    from .board.evaluators import DEFAULT_EVALUATOR, Evaluator, resolve_evaluator
except ImportError:
    from board.evaluators import DEFAULT_EVALUATOR, Evaluator, resolve_evaluator


PIECE_ORDER_VALUES = {1: 100, 2: 320, 3: 330, 4: 500, 5: 900, 6: 20000}
_tt: dict[str, tuple[int, float, str]] = {}


class SearchTimeout(RuntimeError):
    """Internal signal used to stop the current iterative-deepening pass."""


def _resolve_search_evaluator(
    evaluator: Evaluator | str | None,
) -> Evaluator:
    if evaluator is None:
        return resolve_evaluator(DEFAULT_EVALUATOR).evaluate
    if isinstance(evaluator, str):
        return resolve_evaluator(evaluator).evaluate
    if callable(evaluator):
        return evaluator
    raise TypeError("evaluator must be a name or callable")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and monotonic() >= deadline:
        raise SearchTimeout


def _tt_key(board: Any) -> str:
    # The halfmove clock affects automatic draw detection and belongs in the key.
    return " ".join(board.fen().split()[:5])


def _tt_store(key: str, depth: int, score: float, flag: str) -> None:
    _tt[key] = (depth, score, flag)


def _tt_lookup(
    key: str,
    depth: int,
    alpha: float,
    beta: float,
) -> float | None:
    entry = _tt.get(key)
    if entry is None:
        return None
    stored_depth, score, flag = entry
    if stored_depth < depth:
        return None
    if flag == "exact":
        return score
    if flag == "lower" and score >= beta:
        return score
    if flag == "upper" and score <= alpha:
        return score
    return None


def _move_score(
    board: Any,
    move: Any,
    *,
    deadline: float | None = None,
) -> int:
    _check_deadline(deadline)
    score = 0

    if board.is_capture(move):
        victim = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)
        if victim is not None and attacker is not None:
            score += 10 * PIECE_ORDER_VALUES.get(
                victim.piece_type, 0
            ) - PIECE_ORDER_VALUES.get(attacker.piece_type, 0)
        else:
            score += 500

    if move.promotion:
        score += PIECE_ORDER_VALUES.get(move.promotion, 0)

    if score < 100:
        board.push(move)
        try:
            if board.is_check():
                score += 50
        finally:
            board.pop()

    _check_deadline(deadline)
    return score


def _order_moves(
    board: Any,
    moves: list[Any],
    *,
    deadline: float | None = None,
) -> list[Any]:
    scored_moves = [
        (_move_score(board, move, deadline=deadline), move) for move in moves
    ]
    scored_moves.sort(key=lambda item: item[0], reverse=True)
    return [move for _score, move in scored_moves]


def _quiescence(
    board: Any,
    alpha: float,
    beta: float,
    *,
    evaluator: Evaluator | str | None = None,
    deadline: float | None = None,
) -> float:
    """Search captures and check evasions using White-relative scores."""
    evaluate = _resolve_search_evaluator(evaluator)
    _check_deadline(deadline)

    if board.is_game_over():
        return evaluate(board)

    in_check = board.is_check()
    moves = (
        list(board.legal_moves)
        if in_check
        else [move for move in board.legal_moves if board.is_capture(move)]
    )
    moves = _order_moves(board, moves, deadline=deadline)

    if board.turn:
        best_score = -inf if in_check else evaluate(board)
        _check_deadline(deadline)
        if best_score >= beta:
            return best_score
        alpha = max(alpha, best_score)

        for move in moves:
            _check_deadline(deadline)
            board.push(move)
            try:
                score = _quiescence(
                    board,
                    alpha,
                    beta,
                    evaluator=evaluate,
                    deadline=deadline,
                )
            finally:
                board.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if alpha >= beta:
                break
        return best_score

    best_score = inf if in_check else evaluate(board)
    _check_deadline(deadline)
    if best_score <= alpha:
        return best_score
    beta = min(beta, best_score)

    for move in moves:
        _check_deadline(deadline)
        board.push(move)
        try:
            score = _quiescence(
                board,
                alpha,
                beta,
                evaluator=evaluate,
                deadline=deadline,
            )
        finally:
            board.pop()
        best_score = min(best_score, score)
        beta = min(beta, score)
        if beta <= alpha:
            break
    return best_score


def minimax(
    board: Any,
    depth: int,
    alpha: float,
    beta: float,
    maximizing_player: bool,
    *,
    evaluator: Evaluator | str | None = None,
    deadline: float | None = None,
) -> float:
    """Search a position while preserving the caller's board state."""
    evaluate = _resolve_search_evaluator(evaluator)
    _check_deadline(deadline)

    key = _tt_key(board)
    cached = _tt_lookup(key, depth, alpha, beta)
    if cached is not None:
        return cached

    original_alpha = alpha
    original_beta = beta

    if depth <= 0 or board.is_game_over():
        score = _quiescence(
            board,
            alpha,
            beta,
            evaluator=evaluate,
            deadline=deadline,
        )
        if score <= original_alpha:
            flag = "upper"
        elif score >= original_beta:
            flag = "lower"
        else:
            flag = "exact"
        _tt_store(key, depth, score, flag)
        return score

    moves = _order_moves(
        board,
        list(board.legal_moves),
        deadline=deadline,
    )

    if maximizing_player:
        best_score = -inf
        for move in moves:
            _check_deadline(deadline)
            board.push(move)
            try:
                score = minimax(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    False,
                    evaluator=evaluate,
                    deadline=deadline,
                )
            finally:
                board.pop()
            best_score = max(best_score, score)
            alpha = max(alpha, score)
            if beta <= alpha:
                break
    else:
        best_score = inf
        for move in moves:
            _check_deadline(deadline)
            board.push(move)
            try:
                score = minimax(
                    board,
                    depth - 1,
                    alpha,
                    beta,
                    True,
                    evaluator=evaluate,
                    deadline=deadline,
                )
            finally:
                board.pop()
            best_score = min(best_score, score)
            beta = min(beta, score)
            if beta <= alpha:
                break

    if best_score <= original_alpha:
        flag = "upper"
    elif best_score >= original_beta:
        flag = "lower"
    else:
        flag = "exact"
    _tt_store(key, depth, best_score, flag)
    return best_score


def choose_best_move(
    board: Any,
    depth: int = 5,
    time_limit: float = 5.0,
    *,
    evaluator: Evaluator | str | None = None,
) -> tuple[Any, float]:
    """Return the best move from the last fully completed search depth."""
    if depth <= 0:
        raise ValueError("depth must be greater than zero")
    if time_limit <= 0:
        raise ValueError("time_limit must be greater than zero")

    evaluate = _resolve_search_evaluator(evaluator)
    legal_moves = list(board.legal_moves)
    if not legal_moves or board.is_game_over():
        return None, evaluate(board)

    _tt.clear()
    deadline = monotonic() + time_limit
    best_move = legal_moves[0]
    best_score = float(evaluate(board))

    try:
        root_moves = _order_moves(board, legal_moves, deadline=deadline)
        best_move = root_moves[0]

        for current_depth in range(1, depth + 1):
            _check_deadline(deadline)
            depth_best_move = None
            depth_best_score = -inf if board.turn else inf

            for move in root_moves:
                _check_deadline(deadline)
                board.push(move)
                try:
                    score = minimax(
                        board,
                        current_depth - 1,
                        -inf,
                        inf,
                        bool(board.turn),
                        evaluator=evaluate,
                        deadline=deadline,
                    )
                finally:
                    board.pop()

                if board.turn and score > depth_best_score:
                    depth_best_score = score
                    depth_best_move = move
                elif not board.turn and score < depth_best_score:
                    depth_best_score = score
                    depth_best_move = move

            if depth_best_move is not None:
                best_move = depth_best_move
                best_score = depth_best_score
    except SearchTimeout:
        pass

    return best_move, best_score
