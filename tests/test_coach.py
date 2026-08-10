import chess
import pytest

import coach
from board.evaluation import evaluate_handcrafted_board
from coach import (
    REFUTATION_PLIES,
    Line,
    analyze_move,
    build_line,
    calculate_centipawn_loss,
    classify_centipawn_loss,
    format_loss,
    format_move_analysis,
)
from engine import SearchResult

HANGS_QUEEN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3"
ALLOWS_MATE = "rnb1kbnr/pppp1ppp/8/4p3/5Pq1/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1"


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


# ------------------------------------------------------- explaining why


def test_blunder_reason_names_the_hanging_piece():
    """The whole point of the feature: say what is wrong, not just what is best."""
    board = chess.Board(HANGS_QUEEN)
    analysis = analyze_move(
        board,
        board.parse_san("Qg5"),
        depth=3,
        time_limit=0.5,
        evaluator=evaluate_handcrafted_board,
        line_time_limit=0.6,
    )

    assert analysis.classification == "Blunder"
    assert "queen" in analysis.reason
    assert "g5" in analysis.reason
    assert "Nxg5" in analysis.refutation_san
    assert analysis.material_swing == "loses a queen"


def test_refutation_line_is_cut_at_the_move_that_makes_the_point():
    board = chess.Board(HANGS_QUEEN)
    analysis = analyze_move(
        board,
        board.parse_san("Qg5"),
        depth=3,
        time_limit=0.5,
        evaluator=evaluate_handcrafted_board,
        line_time_limit=0.6,
    )
    # Six plies are searched for a blunder, but only the decisive ones are kept.
    assert len(analysis.refutation_san) <= 3
    assert analysis.refutation_san[-1].startswith("Nxg5")


def test_mate_is_reported_as_mate_not_as_a_thousand_pawns():
    """Regression: a forced mate used to print "1000.49 pawn loss"."""
    board = chess.Board(ALLOWS_MATE)
    analysis = analyze_move(
        board,
        board.parse_san("h3"),
        depth=3,
        time_limit=0.5,
        evaluator=evaluate_handcrafted_board,
        line_time_limit=1.0,
    )

    assert analysis.mate_for_opponent is True
    assert format_loss(analysis) == "allows forced mate"
    rendered = format_move_analysis(analysis)
    assert "1000" not in rendered
    assert "pawn loss" not in rendered
    assert analysis.refutation_san[-1].endswith("#")


def test_a_mate_line_does_not_also_claim_a_material_swing():
    board = chess.Board(ALLOWS_MATE)
    analysis = analyze_move(
        board,
        board.parse_san("h3"),
        depth=3,
        time_limit=0.5,
        evaluator=evaluate_handcrafted_board,
        line_time_limit=1.0,
    )
    assert analysis.material_swing is None


def test_the_best_move_gets_no_invented_criticism():
    board = chess.Board(HANGS_QUEEN)
    analysis = analyze_move(
        board,
        board.parse_san("Nf6"),
        depth=3,
        time_limit=0.5,
        evaluator=evaluate_handcrafted_board,
        line_time_limit=0.4,
    )
    assert analysis.classification == "Best"
    assert "en prise" not in analysis.reason
    assert "loses" not in analysis.reason


def test_explanation_never_falls_back_to_the_old_vacuous_phrase():
    board = chess.Board(HANGS_QUEEN)
    analysis = analyze_move(
        board,
        board.parse_san("Qg5"),
        depth=3,
        time_limit=0.5,
        evaluator=evaluate_handcrafted_board,
        line_time_limit=0.6,
    )
    assert "improves the engine evaluation" not in analysis.explanation


def test_analysis_restores_the_board_even_with_lines_enabled():
    board = chess.Board(HANGS_QUEEN)
    before = board.fen()
    analyze_move(
        board,
        board.parse_san("Qg5"),
        depth=3,
        time_limit=0.5,
        evaluator=evaluate_handcrafted_board,
        line_time_limit=0.6,
    )
    assert board.fen() == before


# ------------------------------------------------------------ build_line


def test_build_line_returns_nothing_for_zero_plies():
    board = chess.Board()
    line = build_line(
        board, plies=0, time_limit=0.2,
        evaluator=evaluate_handcrafted_board, mover=chess.WHITE,
    )
    assert line == Line(san=(), ends_in_mate=False, material_swing=None)


def test_build_line_restores_the_board():
    board = chess.Board(HANGS_QUEEN)
    before = board.fen()
    build_line(
        board, plies=4, time_limit=0.3,
        evaluator=evaluate_handcrafted_board, mover=chess.BLACK,
    )
    assert board.fen() == before


def test_build_line_stops_on_an_illegal_suggestion(monkeypatch):
    """A stale or wrong best_move must never be pushed onto the board."""
    board = chess.Board()
    before = board.fen()

    def wrong_move(*_args, **_kwargs):
        return SearchResult(
            best_move=chess.Move.from_uci("a8a1"),  # not legal here
            best_score=0,
            move_scores={},
            completed_depth=1,
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(coach, "analyze_position", wrong_move)
    line = build_line(
        board, plies=3, time_limit=0.2,
        evaluator=evaluate_handcrafted_board, mover=chess.WHITE,
    )
    assert line.san == ()
    assert board.fen() == before


def test_worse_classifications_are_allotted_deeper_lines():
    assert REFUTATION_PLIES["Blunder"] > REFUTATION_PLIES["Inaccuracy"]
    assert REFUTATION_PLIES["Inaccuracy"] > REFUTATION_PLIES["Best"]
