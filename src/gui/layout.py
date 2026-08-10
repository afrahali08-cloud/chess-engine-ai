"""Pure geometry for the window. No pygame, so every function is unit-testable.

Rects are plain ``(x, y, w, h)`` value objects; ``render`` converts them to
``pygame.Rect`` at the last moment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess


WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800

MARGIN = 16
HEADER_TOP = 8
HEADER_HEIGHT = 32
CONTENT_TOP = 48
STATUS_HEIGHT = 44
STATUS_GAP = 12
EVAL_BAR_WIDTH = 32
EVAL_BAR_GAP = 12
BOARD_PANEL_GAP = 16
PANEL_GAP = 12
# Fractions of the content column height given to the three right-hand panels.
CONTROLS_SHARE = 184 / 688
COACH_SHARE = 208 / 688


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def right(self) -> int:
        return self.x + self.w

    @property
    def bottom(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def contains(self, pos: tuple[int, int]) -> bool:
        px, py = pos
        return self.x <= px < self.right and self.y <= py < self.bottom

    def inflate(self, dx: int, dy: int) -> "Rect":
        return Rect(self.x - dx, self.y - dy, self.w + 2 * dx, self.h + 2 * dy)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass(frozen=True)
class Layout:
    width: int
    height: int
    square: int
    header: Rect
    eval_bar: Rect
    board: Rect
    controls: Rect
    coach: Rect
    moves: Rect
    status: Rect


def build_layout(width: int = WINDOW_WIDTH, height: int = WINDOW_HEIGHT) -> Layout:
    """Derive every region from the window size. Board squares stay integral."""
    content_top = CONTENT_TOP
    content_height = height - content_top - STATUS_HEIGHT - STATUS_GAP - MARGIN
    square = max(8, content_height // 8)
    board_size = square * 8

    eval_x = MARGIN
    board_x = eval_x + EVAL_BAR_WIDTH + EVAL_BAR_GAP
    panel_x = board_x + board_size + BOARD_PANEL_GAP
    panel_w = width - panel_x - MARGIN

    controls_h = int(board_size * CONTROLS_SHARE)
    coach_h = int(board_size * COACH_SHARE)
    moves_h = board_size - controls_h - coach_h - 2 * PANEL_GAP
    coach_y = content_top + controls_h + PANEL_GAP
    moves_y = coach_y + coach_h + PANEL_GAP

    return Layout(
        width=width,
        height=height,
        square=square,
        header=Rect(MARGIN, HEADER_TOP, width - 2 * MARGIN, HEADER_HEIGHT),
        eval_bar=Rect(eval_x, content_top, EVAL_BAR_WIDTH, board_size),
        board=Rect(board_x, content_top, board_size, board_size),
        controls=Rect(panel_x, content_top, panel_w, controls_h),
        coach=Rect(panel_x, coach_y, panel_w, coach_h),
        moves=Rect(panel_x, moves_y, panel_w, moves_h),
        status=Rect(
            MARGIN,
            content_top + board_size + STATUS_GAP,
            width - 2 * MARGIN,
            STATUS_HEIGHT,
        ),
    )


def square_to_cell(square: int, flipped: bool) -> tuple[int, int]:
    """Map a python-chess square index to a ``(col, row)`` grid cell."""
    file_index = chess.square_file(square)
    rank_index = chess.square_rank(square)
    col = 7 - file_index if flipped else file_index
    row = rank_index if flipped else 7 - rank_index
    return col, row


def square_rect(square: int, layout: Layout, flipped: bool = False) -> Rect:
    col, row = square_to_cell(square, flipped)
    size = layout.square
    return Rect(
        layout.board.x + col * size,
        layout.board.y + row * size,
        size,
        size,
    )


def square_at(
    pos: tuple[int, int],
    layout: Layout,
    flipped: bool = False,
) -> int | None:
    """Inverse of :func:`square_rect`; ``None`` when the point is off the board."""
    if not layout.board.contains(pos):
        return None
    col = (pos[0] - layout.board.x) // layout.square
    row = (pos[1] - layout.board.y) // layout.square
    file_index = 7 - col if flipped else col
    rank_index = row if flipped else 7 - row
    return chess.square(file_index, rank_index)


def promotion_cell_rects(
    to_square: int,
    options: tuple[int, ...],
    layout: Layout,
    flipped: bool = False,
) -> tuple[tuple[int, Rect], ...]:
    """Stack option cells from the promotion square toward the nearer edge.

    Returns ``(piece_type, rect)`` pairs in draw order.
    """
    base = square_rect(to_square, layout, flipped)
    size = layout.square
    # Grow downward only when the column would otherwise run off the top.
    downward = base.y - (len(options) - 1) * size < layout.board.y
    cells = []
    for index, piece_type in enumerate(options):
        offset = index * size if downward else -index * size
        cells.append((piece_type, Rect(base.x, base.y + offset, size, size)))
    return tuple(cells)


def eval_fraction(centipawns: float) -> float:
    """Map a White-relative score to a 0..1 bar fill.

    Logistic rather than linear: a linear bar makes +3 and +9 look identical.
    The clamp keeps a sliver of both colors visible so the bar never looks broken.
    """
    if centipawns >= 90000:  # engine.py uses +-99999 for mate
        return 1.0
    if centipawns <= -90000:
        return 0.0
    expected = 1.0 / (1.0 + 10 ** (-centipawns / 400.0))
    return min(0.97, max(0.03, expected))


EVAL_EASE_RATE = 9.0  # ~90% of the way in 0.25s
EVAL_SNAP_EPSILON = 0.0005


def approach(
    current: float,
    target: float,
    dt: float,
    *,
    rate: float = EVAL_EASE_RATE,
    epsilon: float = EVAL_SNAP_EPSILON,
) -> float:
    """Ease ``current`` toward ``target`` over ``dt`` seconds.

    Exponential rather than a fixed per-frame step, so the motion takes the same
    wall-clock time whether the loop is running at 60fps or dropping to 30 while
    the search holds the GIL.
    """
    if dt <= 0.0:
        return current
    remaining = (target - current) * math.exp(-rate * dt)
    if abs(remaining) < epsilon:
        return target
    return target - remaining


def format_eval(centipawns: float | None) -> str:
    if centipawns is None:
        return "--"
    if centipawns >= 90000:
        return "M"
    if centipawns <= -90000:
        return "-M"
    return f"{centipawns / 100:+.2f}"
