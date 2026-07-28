import chess
import pytest
from math import inf
from time import monotonic, sleep

from board.evaluation import evaluate_handcrafted_board
from engine import _quiescence, analyze_position, choose_best_move
from main import main


@pytest.mark.parametrize(
    "board",
    [
        chess.Board(),
        chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"),
    ],
)
def test_engine_returns_a_legal_move(board):
    legal_moves = set(board.legal_moves)

    move, _ = choose_best_move(
        board,
        depth=2,
        time_limit=1.0,
        evaluator=evaluate_handcrafted_board,
    )

    assert move in legal_moves


def test_search_does_not_change_the_board():
    board = chess.Board()
    fen_before_search = board.fen()
    move_stack_before_search = list(board.move_stack)

    choose_best_move(
        board,
        depth=3,
        time_limit=1.0,
        evaluator=evaluate_handcrafted_board,
    )

    assert board.fen() == fen_before_search
    assert board.move_stack == move_stack_before_search


def test_position_analysis_scores_every_root_move():
    board = chess.Board()
    fen_before_search = board.fen()

    result = analyze_position(
        board,
        depth=2,
        time_limit=1.0,
        evaluator=evaluate_handcrafted_board,
    )

    assert result.completed_depth >= 1
    assert set(result.move_scores) == set(board.legal_moves)
    assert result.best_move in result.move_scores
    assert result.best_score == result.move_scores[result.best_move]
    assert board.fen() == fen_before_search


def test_engine_recognizes_checkmate():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    move, score = choose_best_move(
        board,
        depth=2,
        evaluator=evaluate_handcrafted_board,
    )

    assert board.is_checkmate()
    assert move is None
    assert score == 99999


def test_quiescence_uses_white_relative_scores_for_both_sides():
    white_to_move = chess.Board("7k/8/8/8/8/8/q7/R6K w - - 0 1")
    black_to_move = chess.Board("r6k/Q7/8/8/8/8/8/7K b - - 0 1")

    assert _quiescence(
        white_to_move,
        -inf,
        inf,
        evaluator=evaluate_handcrafted_board,
    ) > evaluate_handcrafted_board(white_to_move)
    assert _quiescence(
        black_to_move,
        -inf,
        inf,
        evaluator=evaluate_handcrafted_board,
    ) < evaluate_handcrafted_board(black_to_move)


def test_search_deadline_is_bounded_and_preserves_board():
    board = chess.Board()
    fen_before_search = board.fen()
    move_stack_before_search = list(board.move_stack)

    def slow_evaluator(position):
        sleep(0.005)
        return evaluate_handcrafted_board(position)

    start = monotonic()
    move, _score = choose_best_move(
        board,
        depth=8,
        time_limit=0.05,
        evaluator=slow_evaluator,
    )
    elapsed = monotonic() - start

    assert move in board.legal_moves
    assert elapsed < 0.15
    assert board.fen() == fen_before_search
    assert board.move_stack == move_stack_before_search


def test_main_responds_after_human_move(monkeypatch, capsys):
    moves = iter(["e2e4", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(moves))

    with pytest.raises(SystemExit):
        main(
            [
                "--evaluator",
                "handcrafted",
                "--depth",
                "1",
                "--time-limit",
                "0.1",
            ]
        )

    output = capsys.readouterr().out
    assert "You played: e2e4" in output
    assert "Engine is thinking..." in output
    assert "Engine played:" in output


def test_main_prints_coach_feedback(monkeypatch, capsys):
    moves = iter(["e2e4", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(moves))

    with pytest.raises(SystemExit):
        main(
            [
                "--evaluator",
                "handcrafted",
                "--depth",
                "1",
                "--time-limit",
                "0.1",
                "--coach",
                "--coach-depth",
                "1",
                "--coach-time-limit",
                "0.1",
            ]
        )

    output = capsys.readouterr().out
    assert "Move coach: enabled" in output
    assert "Coach:" in output
    assert "Best move:" in output
