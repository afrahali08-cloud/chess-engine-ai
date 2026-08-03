"""Draws a :class:`ViewModel` into a caller-supplied surface.

Nothing here touches ``pygame.display``, so a full frame can be rendered in a
test with no window open.
"""

from __future__ import annotations

import chess
import pygame

from . import layout as layout_mod
from .layout import Layout, Rect, format_eval
from .pieces import Assets
from .viewmodel import ViewModel


COORD_INSET = 4


def _rect(rect: Rect) -> pygame.Rect:
    return pygame.Rect(rect.x, rect.y, rect.w, rect.h)


def _fill_alpha(surface: pygame.Surface, rect: Rect, color) -> None:
    overlay = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
    overlay.fill(color)
    surface.blit(overlay, (rect.x, rect.y))


def _panel(surface: pygame.Surface, rect: Rect, assets: Assets, title: str | None):
    theme = assets.theme
    pygame.draw.rect(surface, theme.panel, _rect(rect), border_radius=8)
    pygame.draw.rect(surface, theme.panel_border, _rect(rect), width=1, border_radius=8)
    if title:
        label = assets.fonts.small.render(title.upper(), True, theme.dim_text)
        surface.blit(label, (rect.x + 14, rect.y + 10))
    return rect.y + (34 if title else 12)


def _text(surface, font, text, pos, color):
    surface.blit(font.render(text, True, color), pos)
    return font.size(text)[0]


def wrap_text(font, text: str, max_width: int) -> list[str]:
    """Greedy word wrap against a pygame font."""
    if not text:
        return []
    lines: list[str] = []
    words = text.split()
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if font.size(candidate)[0] <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


# ----------------------------------------------------------------- board


def draw_board(surface: pygame.Surface, vm: ViewModel, assets: Assets, layout: Layout):
    theme = assets.theme
    board = vm.board
    flipped = vm.flipped
    state = vm.selection

    check_square = None
    if board.is_check():
        check_square = board.king(board.turn)

    for square in chess.SQUARES:
        cell = layout_mod.square_rect(square, layout, flipped)
        is_light = (chess.square_file(square) + chess.square_rank(square)) % 2 == 1
        pygame.draw.rect(
            surface,
            theme.light_square if is_light else theme.dark_square,
            _rect(cell),
        )

    if vm.last_move is not None:
        for square in (vm.last_move.from_square, vm.last_move.to_square):
            _fill_alpha(
                surface, layout_mod.square_rect(square, layout, flipped), theme.last_move
            )

    if check_square is not None:
        cell = layout_mod.square_rect(check_square, layout, flipped)
        glow = pygame.Surface((cell.w, cell.h), pygame.SRCALPHA)
        radius = cell.w // 2
        for step in range(radius, 0, -2):
            alpha = int(160 * (step / radius) ** 2)
            pygame.draw.circle(
                glow, (*theme.check_glow, alpha), (cell.w // 2, cell.h // 2), step
            )
        surface.blit(glow, (cell.x, cell.y))

    if state.selected is not None:
        _fill_alpha(
            surface,
            layout_mod.square_rect(state.selected, layout, flipped),
            theme.selected,
        )

    if state.error_active(vm.now):
        _fill_alpha(
            surface,
            layout_mod.square_rect(state.error_square, layout, flipped),
            theme.error_flash,
        )

    _draw_coordinates(surface, assets, layout, flipped)

    for square, piece in board.piece_map().items():
        assets.pieces.blit(
            surface, piece, layout_mod.square_rect(square, layout, flipped)
        )

    # Markers go on top of the pieces so a capture target stays visible.
    for square, moves in state.targets.items():
        cell = layout_mod.square_rect(square, layout, flipped)
        marker = pygame.Surface((cell.w, cell.h), pygame.SRCALPHA)
        occupied = board.piece_at(square) is not None or any(
            board.is_en_passant(move) for move in moves
        )
        if occupied:
            pygame.draw.circle(
                marker,
                theme.legal_marker,
                (cell.w // 2, cell.h // 2),
                cell.w // 2 - 3,
                width=6,
            )
        else:
            pygame.draw.circle(
                marker, theme.legal_marker, (cell.w // 2, cell.h // 2), cell.w // 7
            )
        surface.blit(marker, (cell.x, cell.y))

    if state.promotion is not None:
        _draw_promotion(surface, vm, assets, layout)


def _draw_coordinates(surface, assets: Assets, layout: Layout, flipped: bool):
    theme = assets.theme
    font = assets.fonts.small
    for rank_index in range(8):
        square = chess.square(0 if not flipped else 7, rank_index)
        cell = layout_mod.square_rect(square, layout, flipped)
        is_light = (chess.square_file(square) + rank_index) % 2 == 1
        color = theme.dark_square if is_light else theme.light_square
        surface.blit(
            font.render(str(rank_index + 1), True, color),
            (cell.x + COORD_INSET, cell.y + COORD_INSET),
        )
    for file_index in range(8):
        square = chess.square(file_index, 0 if not flipped else 7)
        cell = layout_mod.square_rect(square, layout, flipped)
        is_light = (file_index + chess.square_rank(square)) % 2 == 1
        color = theme.dark_square if is_light else theme.light_square
        glyph = font.render(chess.FILE_NAMES[file_index], True, color)
        surface.blit(
            glyph,
            (
                cell.right - glyph.get_width() - COORD_INSET,
                cell.bottom - glyph.get_height() - COORD_INSET,
            ),
        )


def _draw_promotion(surface, vm: ViewModel, assets: Assets, layout: Layout):
    theme = assets.theme
    prompt = vm.selection.promotion
    _fill_alpha(surface, layout.board, theme.promotion_scrim)
    cells = layout_mod.promotion_cell_rects(
        prompt.to_square, prompt.options, layout, vm.flipped
    )
    color = vm.board.turn
    for piece_type, cell in cells:
        pygame.draw.rect(surface, theme.light_square, _rect(cell), border_radius=6)
        pygame.draw.rect(
            surface, theme.accent, _rect(cell), width=2, border_radius=6
        )
        assets.pieces.blit(surface, chess.Piece(piece_type, color), cell)


# ---------------------------------------------------------------- panels


def draw_eval_bar(surface, vm: ViewModel, assets: Assets, layout: Layout):
    theme = assets.theme
    rect = layout.eval_bar
    pygame.draw.rect(surface, theme.eval_black, _rect(rect), border_radius=6)

    fraction = (
        vm.eval_fill
        if vm.eval_fill is not None
        else layout_mod.eval_fraction(vm.eval_cp if vm.eval_cp is not None else 0.0)
    )
    white_height = int(rect.h * fraction)
    # "Your" side always fills from the bottom.
    if vm.flipped:
        white_rect = Rect(rect.x, rect.y, rect.w, white_height)
    else:
        white_rect = Rect(rect.x, rect.bottom - white_height, rect.w, white_height)
    pygame.draw.rect(surface, theme.eval_white, _rect(white_rect), border_radius=6)
    pygame.draw.rect(
        surface, theme.panel_border, _rect(rect), width=1, border_radius=6
    )

    # The number sits on its own chip so it stays legible no matter where the
    # fill boundary happens to fall.
    glyph = assets.fonts.small.render(
        format_eval(vm.eval_cp), True, theme.text
    )
    glyph = pygame.transform.rotate(glyph, 90)
    chip = Rect(
        rect.x + 2,
        rect.y + 6,
        rect.w - 4,
        glyph.get_height() + 10,
    )
    _fill_alpha(surface, chip, (*theme.background, 190))
    if vm.eval_is_stale:
        glyph.set_alpha(120)
    surface.blit(
        glyph,
        (rect.x + (rect.w - glyph.get_width()) // 2, chip.y + 5),
    )


def draw_header(surface, vm: ViewModel, assets: Assets, layout: Layout):
    theme = assets.theme
    rect = layout.header
    _text(
        surface,
        assets.fonts.title,
        "Chess Engine AI",
        (rect.x, rect.y),
        theme.text,
    )
    subtitle = (
        f"{vm.evaluator_name}  |  depth {vm.depth}  |  {vm.time_limit:.1f}s"
        f"  |  coach {'on' if vm.coach_enabled else 'off'}"
    )
    glyph = assets.fonts.body.render(subtitle, True, theme.dim_text)
    surface.blit(glyph, (rect.right - glyph.get_width(), rect.y + 6))


def draw_coach_panel(surface, vm: ViewModel, assets: Assets, layout: Layout):
    theme = assets.theme
    rect = layout.coach
    y = _panel(surface, rect, assets, "coach")
    left = rect.x + 14
    width = rect.w - 28

    if vm.coach is None:
        message = (
            "Analyzing your move..."
            if vm.coach_pending
            else (
                "Make a move to see feedback."
                if vm.coach_enabled
                else "Coach is off. Press C to enable."
            )
        )
        _text(surface, assets.fonts.body, message, (left, y + 4), theme.dim_text)
        return

    coach = vm.coach
    color = theme.classification_color(coach.classification)
    pygame.draw.rect(surface, color, pygame.Rect(left, y + 6, 12, 12), border_radius=3)
    _text(
        surface,
        assets.fonts.label,
        f"{coach.classification}",
        (left + 22, y),
        color,
    )
    loss = assets.fonts.body.render(
        f"{coach.centipawn_loss / 100:.2f} pawn loss", True, theme.dim_text
    )
    surface.blit(loss, (rect.right - 14 - loss.get_width(), y + 3))

    y += 30
    detail = f"played {coach.played_san}   best {coach.best_san}   depth {coach.search_depth}"
    _text(surface, assets.fonts.small, detail, (left, y), theme.dim_text)

    y += 20
    if coach.used_static_fallback:
        # The search did not finish depth 1, so these numbers came from a static
        # evaluation. Say so rather than presenting them as a real verdict.
        _text(
            surface,
            assets.fonts.small,
            "static estimate - raise the coach time limit",
            (left, y),
            theme.warn,
        )
        y += 20

    y += 4
    for line in wrap_text(assets.fonts.body, coach.explanation, width):
        if y + 20 > rect.bottom - 10:
            break
        _text(surface, assets.fonts.body, line, (left, y), theme.text)
        y += 22


def draw_status(surface, vm: ViewModel, assets: Assets, layout: Layout):
    theme = assets.theme
    rect = layout.status
    pygame.draw.rect(surface, theme.panel, _rect(rect), border_radius=8)
    pygame.draw.rect(surface, theme.panel_border, _rect(rect), width=1, border_radius=8)
    colors = {
        "dim": theme.dim_text,
        "text": theme.text,
        "accent": theme.accent,
        "warn": theme.warn,
        "error": theme.error,
    }
    text = vm.result_text or vm.status_text
    color = theme.accent if vm.result_text else colors.get(vm.status_color, theme.dim_text)
    _text(surface, assets.fonts.body, text, (rect.x + 14, rect.y + 12), color)

    if vm.thinking:
        # Discrete animation: smooth spinners visibly stutter while the search
        # holds the GIL, a stepped ellipsis does not.
        dots = "." * (1 + int(vm.now * 3) % 3)
        label = f"{vm.thinking_label}{dots}"
        glyph = assets.fonts.body.render(label, True, theme.accent)
        surface.blit(glyph, (rect.right - 14 - glyph.get_width(), rect.y + 12))


def draw_frame(surface: pygame.Surface, vm: ViewModel, assets: Assets) -> None:
    layout = layout_mod.build_layout(*surface.get_size())
    surface.fill(assets.theme.background)
    draw_header(surface, vm, assets, layout)
    draw_eval_bar(surface, vm, assets, layout)
    draw_board(surface, vm, assets, layout)
    _panel(surface, layout.controls, assets, "controls")
    draw_coach_panel(surface, vm, assets, layout)
    _panel(surface, layout.moves, assets, "moves")
    draw_status(surface, vm, assets, layout)
    for widget in vm.widgets:
        widget.draw(surface, assets)
