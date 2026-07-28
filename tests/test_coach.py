import chess
import pytest

import coach
from board.evaluation import evaluate_handcrafted_board
from coach import (
    analyze_move,
    calculate_centipawn_loss,
    classify_centipawn_loss,
    format_move_analysis,
)
from engine import SearchResult


@pytest.mark.parametrize(
    ("loss", "expected"),
    [
        (0, "Best"),
        (10, "Best"),
        (11, "Excellent"),
        (30, "Excellent"),
        (80, "Good"),
        (150, "Inaccuracy"),
        (300, "Mistake"),
        (301, "Blunder"),
    ],
)
def test_centipawn_loss_classification(loss, expected):
    assert classify_centipawn_loss(loss) == expected


def test_centipawn_loss_uses_the_moving_side_perspective():
    assert calculate_centipawn_loss(
        chess.WHITE,
        best_score=100,
        played_score=-50,
    ) == 150
    assert calculate_centipawn_loss(
        chess.BLACK,
        best_score=-100,
        played_score=50,
    ) == 150


def test_analyze_move_compares_moves_from_one_root_search(monkeypatch):
    board = chess.Board()
    best_move = chess.Move.from_uci("d2d4")
    played_move = chess.Move.from_uci("e2e4")

    def fake_analysis(*_args, **_kwargs):
        return SearchResult(
            best_move=best_move,
            best_score=80,
            move_scores={best_move: 80, played_move: -90},
            completed_depth=3,
            elapsed_seconds=0.1,
        )

    monkeypatch.setattr(coach, "analyze_position", fake_analysis)
    analysis = analyze_move(
        board,
        played_move,
        evaluator=evaluate_handcrafted_board,
    )

    assert analysis.centipawn_loss == 170
    assert analysis.classification == "Mistake"
    assert analysis.best_san == "d4"
    assert "engine preferred d4" in analysis.explanation
    assert "Coach: Mistake" in format_move_analysis(analysis)
    assert board.fen() == chess.STARTING_FEN


def test_real_coach_analysis_preserves_board():
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    fen_before = board.fen()

    analysis = analyze_move(
        board,
        move,
        depth=1,
        time_limit=0.5,
        evaluator=evaluate_handcrafted_board,
    )

    assert analysis.best_move in board.legal_moves
    assert analysis.played_move == move
    assert analysis.search_depth == 1
    assert board.fen() == fen_before
