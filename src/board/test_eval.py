import chess

from board.evaluation import evaluate_board


def test_starting_position_is_even():
    board = chess.Board()

    assert evaluate_board(board) == 0


def test_missing_black_queen_favors_white():
    board = chess.Board(
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    )

    assert evaluate_board(board) > 800


def test_missing_white_queen_favors_black():
    board = chess.Board(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1"
    )

    assert evaluate_board(board) < -800
