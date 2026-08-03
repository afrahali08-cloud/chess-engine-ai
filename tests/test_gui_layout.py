"""Geometry tests. These import no pygame, so they run anywhere."""

import chess
import pytest

from gui.layout import (
    Rect,
    approach,
    build_layout,
    eval_fraction,
    format_eval,
    promotion_cell_rects,
    square_at,
    square_rect,
)


LAYOUT = build_layout()


def center(rect: Rect) -> tuple[int, int]:
    return rect.center


# --------------------------------------------------------------- regions


def test_regions_stay_inside_the_window():
    for rect in (
        LAYOUT.header,
        LAYOUT.eval_bar,
        LAYOUT.board,
        LAYOUT.controls,
        LAYOUT.coach,
        LAYOUT.moves,
        LAYOUT.status,
    ):
        assert rect.x >= 0 and rect.y >= 0
        assert rect.right <= LAYOUT.width
        assert rect.bottom <= LAYOUT.height


def test_regions_do_not_overlap():
    rects = {
        "header": LAYOUT.header,
        "eval_bar": LAYOUT.eval_bar,
        "board": LAYOUT.board,
        "controls": LAYOUT.controls,
        "coach": LAYOUT.coach,
        "moves": LAYOUT.moves,
        "status": LAYOUT.status,
    }
    names = list(rects)
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            a, b = rects[first], rects[second]
            separated = (
                a.right <= b.x or b.right <= a.x or a.bottom <= b.y or b.bottom <= a.y
            )
            assert separated, f"{first} overlaps {second}"


def test_board_is_square_and_divides_into_eight():
    assert LAYOUT.board.w == LAYOUT.board.h
    assert LAYOUT.board.w == LAYOUT.square * 8


@pytest.mark.parametrize("size", [(1280, 800), (1024, 768), (1600, 1000)])
def test_layout_scales_to_other_window_sizes(size):
    layout = build_layout(*size)
    assert layout.board.w == layout.square * 8
    assert layout.status.bottom <= layout.height
    assert layout.moves.right <= layout.width


# ------------------------------------------------------- square mapping


@pytest.mark.parametrize("flipped", [False, True])
def test_square_at_inverts_square_rect_for_every_square(flipped):
    for square in chess.SQUARES:
        rect = square_rect(square, LAYOUT, flipped)
        assert square_at(center(rect), LAYOUT, flipped) == square


@pytest.mark.parametrize("flipped", [False, True])
def test_square_at_covers_every_corner_pixel(flipped):
    for square in chess.SQUARES:
        rect = square_rect(square, LAYOUT, flipped)
        corners = (
            (rect.x, rect.y),
            (rect.right - 1, rect.y),
            (rect.x, rect.bottom - 1),
            (rect.right - 1, rect.bottom - 1),
        )
        for corner in corners:
            assert square_at(corner, LAYOUT, flipped) == square


def test_a1_is_bottom_left_when_not_flipped():
    a1 = square_rect(chess.A1, LAYOUT, False)
    a8 = square_rect(chess.A8, LAYOUT, False)
    assert a1.x == LAYOUT.board.x
    assert a1.bottom == LAYOUT.board.bottom
    assert a8.y == LAYOUT.board.y


def test_flipping_puts_a1_top_right():
    a1 = square_rect(chess.A1, LAYOUT, True)
    assert a1.right == LAYOUT.board.right
    assert a1.y == LAYOUT.board.y


def test_square_at_returns_none_off_the_board():
    assert square_at((0, 0), LAYOUT) is None
    assert square_at(LAYOUT.controls.center, LAYOUT) is None
    assert square_at((LAYOUT.board.right + 1, LAYOUT.board.y), LAYOUT) is None
    assert square_at((LAYOUT.board.x, LAYOUT.board.bottom + 1), LAYOUT) is None


# ------------------------------------------------------------ promotion


def test_promotion_column_grows_down_from_the_top_rank():
    options = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    cells = promotion_cell_rects(chess.D8, options, LAYOUT, False)

    assert [piece for piece, _ in cells] == list(options)
    assert cells[0][1] == square_rect(chess.D8, LAYOUT, False)
    ys = [rect.y for _, rect in cells]
    assert ys == sorted(ys)
    for _, rect in cells:
        assert LAYOUT.board.y <= rect.y and rect.bottom <= LAYOUT.board.bottom


def test_promotion_column_grows_up_from_the_bottom_rank():
    options = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    cells = promotion_cell_rects(chess.D1, options, LAYOUT, False)

    ys = [rect.y for _, rect in cells]
    assert ys == sorted(ys, reverse=True)
    for _, rect in cells:
        assert LAYOUT.board.y <= rect.y and rect.bottom <= LAYOUT.board.bottom


def test_promotion_cells_stay_on_board_when_flipped():
    options = (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT)
    for square in (chess.A1, chess.H8, chess.D1, chess.E8):
        for flipped in (False, True):
            for _, rect in promotion_cell_rects(square, options, LAYOUT, flipped):
                assert LAYOUT.board.y <= rect.y
                assert rect.bottom <= LAYOUT.board.bottom


# ----------------------------------------------------------- eval scale


def test_eval_fraction_is_even_at_zero():
    assert eval_fraction(0.0) == pytest.approx(0.5)


def test_eval_fraction_saturates_at_mate_scores():
    assert eval_fraction(99999) == 1.0
    assert eval_fraction(-99999) == 0.0


def test_eval_fraction_is_clamped_but_never_fully_empty():
    assert 0.0 < eval_fraction(-5000) <= 0.03
    assert 0.97 <= eval_fraction(5000) < 1.0


def test_eval_fraction_is_monotonic():
    samples = [-3000, -800, -200, -50, 0, 50, 200, 800, 3000]
    values = [eval_fraction(cp) for cp in samples]
    assert values == sorted(values)


def test_eval_fraction_is_symmetric_about_zero():
    for cp in (25, 120, 640):
        assert eval_fraction(cp) + eval_fraction(-cp) == pytest.approx(1.0)


def test_format_eval_covers_none_and_mate():
    assert format_eval(None) == "--"
    assert format_eval(99999) == "M"
    assert format_eval(-99999) == "-M"
    assert format_eval(35.0) == "+0.35"
    assert format_eval(-120.0) == "-1.20"


# -------------------------------------------------------- easing (bar)


def test_approach_moves_toward_the_target():
    moved = approach(0.0, 1.0, 0.05)
    assert 0.0 < moved < 1.0


def test_approach_never_overshoots():
    current = 0.0
    for _ in range(200):
        current = approach(current, 1.0, 0.05)
        assert current <= 1.0
    assert current == 1.0


def test_approach_works_downward_too():
    current = 1.0
    for _ in range(200):
        current = approach(current, 0.2, 0.05)
        assert current >= 0.2
    assert current == 0.2


def test_approach_snaps_once_it_is_close_enough():
    """Without a snap the value creeps forever and the bar never settles."""
    assert approach(0.9999999, 1.0, 0.016) == 1.0


def test_approach_is_a_noop_for_zero_or_negative_dt():
    assert approach(0.3, 1.0, 0.0) == 0.3
    assert approach(0.3, 1.0, -0.1) == 0.3


def test_approach_is_frame_rate_independent():
    """Same wall-clock time must land in the same place at 30fps and 60fps."""
    at_60 = 0.0
    for _ in range(30):  # 30 frames of 1/60s = 0.5s
        at_60 = approach(at_60, 1.0, 1 / 60)
    at_30 = 0.0
    for _ in range(15):  # 15 frames of 1/30s = 0.5s
        at_30 = approach(at_30, 1.0, 1 / 30)

    assert at_60 == pytest.approx(at_30, abs=0.01)


def test_approach_settles_within_about_half_a_second():
    current = 0.0
    for _ in range(30):
        current = approach(current, 1.0, 1 / 60)
    assert current > 0.98


def test_approach_reaches_the_target_exactly_not_asymptotically():
    current = 0.5
    for _ in range(600):
        current = approach(current, 0.0, 1 / 60)
    assert current == 0.0


# ------------------------------------------------------------------ rect


def test_rect_contains_is_half_open():
    rect = Rect(10, 10, 5, 5)
    assert rect.contains((10, 10))
    assert rect.contains((14, 14))
    assert not rect.contains((15, 14))
    assert not rect.contains((9, 10))
