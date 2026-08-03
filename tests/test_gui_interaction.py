"""Click state machine tests. Pure logic, no pygame."""

import chess
import pytest

from gui.interaction import (
    EMPTY,
    ERROR_FLASH_SECONDS,
    PROMOTION_OPTIONS,
    SelectionState,
    clear,
    click_promotion,
    click_square,
    expire_error,
    legal_targets,
)


def click(state, board, square, **kwargs):
    return click_square(state, board, square, **kwargs)


# ------------------------------------------------------------ selecting


def test_selecting_a_pawn_offers_both_advances():
    board = chess.Board()
    state, move = click(EMPTY, board, chess.E2)

    assert move is None
    assert state.selected == chess.E2
    assert set(state.targets) == {chess.E3, chess.E4}


def test_clicking_an_empty_square_is_a_silent_noop():
    board = chess.Board()
    state, move = click(EMPTY, board, chess.E4)

    assert move is None
    assert state == EMPTY
    assert state.error_square is None


def test_clicking_an_opposing_piece_is_a_silent_noop():
    board = chess.Board()
    state, move = click(EMPTY, board, chess.E7)

    assert move is None
    assert state.selected is None
    assert state.error_square is None


def test_selecting_a_piece_with_no_legal_moves_flashes():
    board = chess.Board()
    state, move = click(EMPTY, board, chess.A1, now=10.0)

    assert move is None
    assert state.selected is None
    assert state.error_square == chess.A1
    assert state.error_expires_at == pytest.approx(10.0 + ERROR_FLASH_SECONDS)


def test_input_can_be_disabled_while_the_engine_moves():
    board = chess.Board()
    state, move = click(EMPTY, board, chess.E2, allow_input=False)

    assert move is None
    assert state is EMPTY


def test_human_color_gate_blocks_moving_the_engine_pieces():
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))  # black to move
    state, move = click(EMPTY, board, chess.E7, human_color=chess.WHITE)

    assert move is None
    assert state.selected is None


# -------------------------------------------------------------- moving


def test_clicking_a_target_returns_the_move_and_clears():
    board = chess.Board()
    selected, _ = click(EMPTY, board, chess.E2)
    state, move = click(selected, board, chess.E4)

    assert move == chess.Move.from_uci("e2e4")
    assert state.selected is None
    assert state.targets == {}


def test_clicking_the_same_square_deselects():
    board = chess.Board()
    selected, _ = click(EMPTY, board, chess.E2)
    state, move = click(selected, board, chess.E2)

    assert move is None
    assert state.selected is None


def test_clicking_another_own_piece_reselects():
    board = chess.Board()
    selected, _ = click(EMPTY, board, chess.E2)
    state, move = click(selected, board, chess.D2)

    assert move is None
    assert state.selected == chess.D2
    assert set(state.targets) == {chess.D3, chess.D4}


def test_clicking_an_illegal_square_deselects_and_flashes():
    board = chess.Board()
    selected, _ = click(EMPTY, board, chess.E2)
    state, move = click(selected, board, chess.H6, now=5.0)

    assert move is None
    assert state.selected is None
    assert state.error_square == chess.H6


def test_capture_targets_are_offered():
    board = chess.Board()
    for uci in ("e2e4", "d7d5"):
        board.push(chess.Move.from_uci(uci))
    state, _ = click(EMPTY, board, chess.E4)

    assert chess.D5 in state.targets
    assert state.targets[chess.D5] == (chess.Move.from_uci("e4d5"),)


# ----------------------------------------------------------- promotion

PROMO_FEN = "8/4P3/8/8/8/8/8/K6k w - - 0 1"


def test_promotion_target_opens_a_prompt_instead_of_moving():
    board = chess.Board(PROMO_FEN)
    selected, _ = click(EMPTY, board, chess.E7)
    state, move = click(selected, board, chess.E8)

    assert move is None
    assert state.promotion is not None
    assert state.promotion.from_square == chess.E7
    assert state.promotion.to_square == chess.E8
    assert set(state.promotion.options) == set(PROMOTION_OPTIONS)


def test_choosing_a_promotion_piece_returns_the_full_move():
    board = chess.Board(PROMO_FEN)
    selected, _ = click(EMPTY, board, chess.E7)
    prompted, _ = click(selected, board, chess.E8)
    state, move = click_promotion(prompted, chess.QUEEN)

    assert move == chess.Move.from_uci("e7e8q")
    assert move in board.legal_moves
    assert state.promotion is None


def test_promotion_to_a_knight_is_supported():
    board = chess.Board(PROMO_FEN)
    selected, _ = click(EMPTY, board, chess.E7)
    prompted, _ = click(selected, board, chess.E8)
    _state, move = click_promotion(prompted, chess.KNIGHT)

    assert move == chess.Move.from_uci("e7e8n")


def test_cancelling_the_prompt_yields_no_move():
    board = chess.Board(PROMO_FEN)
    selected, _ = click(EMPTY, board, chess.E7)
    prompted, _ = click(selected, board, chess.E8)
    state, move = click_promotion(prompted, None)

    assert move is None
    assert state.promotion is None
    assert state.selected is None


def test_a_board_click_cancels_an_open_prompt():
    board = chess.Board(PROMO_FEN)
    selected, _ = click(EMPTY, board, chess.E7)
    prompted, _ = click(selected, board, chess.E8)
    state, move = click(prompted, board, chess.A1)

    assert move is None
    assert state.promotion is None


def test_click_promotion_on_a_closed_prompt_is_a_noop():
    state, move = click_promotion(EMPTY, chess.QUEEN)
    assert move is None
    assert state is EMPTY


def test_an_unlisted_piece_type_cancels_rather_than_moving():
    board = chess.Board(PROMO_FEN)
    selected, _ = click(EMPTY, board, chess.E7)
    prompted, _ = click(selected, board, chess.E8)
    _state, move = click_promotion(prompted, chess.PAWN)

    assert move is None


# ------------------------------------------------------------- helpers


def test_legal_targets_groups_promotions_under_one_square():
    board = chess.Board(PROMO_FEN)
    targets = legal_targets(board, chess.E7)

    assert set(targets) == {chess.E8}
    assert len(targets[chess.E8]) == 4


def test_legal_targets_is_empty_for_a_stuck_piece():
    assert legal_targets(chess.Board(), chess.A1) == {}


def test_expire_error_clears_only_after_the_deadline():
    state = SelectionState(error_square=chess.A1, error_expires_at=5.0)

    assert expire_error(state, 4.9).error_square == chess.A1
    assert expire_error(state, 5.0).error_square is None


def test_clear_keeps_an_active_flash_but_drops_the_selection():
    state = SelectionState(
        selected=chess.E2,
        targets={chess.E4: ()},
        error_square=chess.A1,
        error_expires_at=9.0,
    )
    cleared = clear(state)

    assert cleared.selected is None
    assert cleared.targets == {}
    assert cleared.error_square == chess.A1


def test_error_active_respects_the_clock():
    state = SelectionState(error_square=chess.A1, error_expires_at=3.0)
    assert state.error_active(2.5) is True
    assert state.error_active(3.5) is False
    assert EMPTY.error_active(0.0) is False
