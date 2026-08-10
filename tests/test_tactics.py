"""Pure board facts behind the coach's explanations. No search, no pygame."""

import chess
import pytest

from tactics import (
    PIECE_VALUES,
    describe_material_swing,
    describe_move_purpose,
    hangs_after,
    hanging_pieces,
    material_balance,
    most_valuable_hanging,
)


ITALIAN = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 3"


# ------------------------------------------------------------- material


def test_starting_position_is_materially_level():
    assert material_balance(chess.Board()) == 0


def test_material_balance_is_white_relative():
    # Black is missing a queen.
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    assert material_balance(board) == PIECE_VALUES[chess.QUEEN]


def test_kings_do_not_count_toward_material():
    assert PIECE_VALUES[chess.KING] == 0


# --------------------------------------------------------------- swing


def test_swing_is_silent_when_nothing_changed():
    assert describe_material_swing(0, 0, chess.WHITE) is None


def test_swing_ignores_sub_pawn_noise():
    assert describe_material_swing(0, 40, chess.WHITE) is None


def test_swing_reads_from_the_movers_point_of_view():
    # The same white-relative drop is a loss for White and a gain for Black.
    assert describe_material_swing(0, -900, chess.WHITE) == "loses a queen"
    assert describe_material_swing(0, -900, chess.BLACK) == "wins a queen"


def test_swing_names_a_piece_only_when_the_value_matches():
    assert describe_material_swing(0, 320, chess.WHITE) == "wins a knight"
    assert describe_material_swing(0, 500, chess.WHITE) == "wins a rook"


def test_swing_falls_back_to_pawns_rather_than_naming_the_wrong_piece():
    """580cp is queen-for-knight; calling it "a rook" would be a fabrication."""
    described = describe_material_swing(0, -580, chess.WHITE)
    assert "rook" not in described
    assert described == "loses 5.8 pawns of material"


def test_swing_counts_whole_pawns():
    assert describe_material_swing(0, 200, chess.WHITE) == "wins 2 pawns"


# ------------------------------------------------------------- hanging


def test_undefended_attacked_piece_is_hanging():
    board = chess.Board(ITALIAN)
    board.push_san("Qg5")  # queen walks onto a square the knight covers
    found = most_valuable_hanging(board, chess.BLACK)
    assert found is not None
    square, piece_type = found
    assert chess.square_name(square) == "g5"
    assert piece_type == chess.QUEEN


def test_hangs_after_reports_the_piece_the_move_leaves_en_prise():
    board = chess.Board(ITALIAN)
    found = hangs_after(board, board.parse_san("Qg5"))
    assert found is not None
    assert found[1] == chess.QUEEN


def test_hangs_after_restores_the_board():
    board = chess.Board(ITALIAN)
    before = board.fen()
    hangs_after(board, board.parse_san("Qg5"))
    assert board.fen() == before


def test_a_defended_piece_attacked_by_an_equal_piece_is_not_hanging():
    """Only call it hanging when taking actually wins material."""
    board = chess.Board(ITALIAN)
    assert not any(
        chess.square_name(square) == "c6" for square in hanging_pieces(board, chess.BLACK)
    )


def test_starting_position_has_nothing_hanging():
    board = chess.Board()
    assert hanging_pieces(board, chess.WHITE) == []
    assert hanging_pieces(board, chess.BLACK) == []


# -------------------------------------------------------------- purpose


def test_purpose_names_development():
    board = chess.Board(ITALIAN)
    assert describe_move_purpose(board, board.parse_san("Nf6")) == "develops a knight"


def test_purpose_names_a_capture_with_the_right_piece():
    board = chess.Board(ITALIAN)
    board.push_san("Qg5")
    assert "captures a queen" in describe_move_purpose(board, board.parse_san("Nxg5"))


def test_purpose_names_castling():
    board = chess.Board(
        "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    )
    assert "castles" in describe_move_purpose(board, chess.Move.from_uci("e1g1"))


def test_purpose_names_checkmate():
    # Fool's mate: after 1.f3 e5 2.g4, Qh4 is mate.
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2")
    assert "checkmate" in describe_move_purpose(board, board.parse_san("Qh4"))


def test_purpose_stays_quiet_rather_than_inventing_a_reason():
    """A move that does nothing checkable must not get a flattering story."""
    board = chess.Board()
    assert describe_move_purpose(board, board.parse_san("a3")) == "is a quiet move"


def test_purpose_does_not_pad_a_capture_with_a_redundant_rescue():
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/5Pq1/8/PPPPP1PP/RNBQKBNR w KQkq - 0 1")
    described = describe_move_purpose(board, board.parse_san("fxe5"))
    assert described.count("pawn") == 1


def test_purpose_restores_the_board():
    board = chess.Board(ITALIAN)
    before = board.fen()
    describe_move_purpose(board, board.parse_san("Nf6"))
    assert board.fen() == before
