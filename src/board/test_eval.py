import chess

import board.evaluation as evaluation
from board.evaluation import evaluate_board, evaluate_handcrafted_board
from board.learned_evaluation import evaluate_learned_board


def test_evaluate_board_uses_learned_model():
    board = chess.Board()

    assert evaluate_board(board) == evaluate_learned_board(board)
    assert abs(evaluate_board(board)) < 100


def test_missing_black_queen_favors_white():
    board = chess.Board(
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )

    assert evaluate_board(board) > 500


def test_missing_white_queen_favors_black():
    board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
    )

    assert evaluate_board(board) < -500


def test_handcrafted_evaluation_remains_available():
    assert evaluate_handcrafted_board(chess.Board()) == 0


def test_missing_model_falls_back_to_handcrafted(monkeypatch):
    board = chess.Board()

    def missing_model(_board):
        raise FileNotFoundError

    monkeypatch.setattr(evaluation, "evaluate_learned_board", missing_model)

    assert evaluate_board(board) == evaluate_handcrafted_board(board)


def test_terminal_score_takes_priority_over_model():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    assert evaluate_board(board) == 99999
