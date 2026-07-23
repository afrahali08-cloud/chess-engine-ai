import chess
import pytest

from engine import choose_best_move
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

    move, _ = choose_best_move(board, depth=2)

    assert move in legal_moves


def test_search_does_not_change_the_board():
    board = chess.Board()
    fen_before_search = board.fen()
    move_stack_before_search = list(board.move_stack)

    choose_best_move(board, depth=3)

    assert board.fen() == fen_before_search
    assert board.move_stack == move_stack_before_search


def test_engine_recognizes_checkmate():
    board = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    move, score = choose_best_move(board, depth=2)

    assert board.is_checkmate()
    assert move is None
    assert score == 99999


def test_main_responds_after_human_move(monkeypatch, capsys):
    moves = iter(["e2e4", "quit"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(moves))

    with pytest.raises(SystemExit):
        main()

    output = capsys.readouterr().out
    assert "You played: e2e4" in output
    assert "Engine is thinking..." in output
    assert "Engine played:" in output

