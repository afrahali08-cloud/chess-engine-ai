"""Cached, outlined piece glyphs rendered from a system font.

Two details matter for legibility at projector size:

* glyphs are sized and centered by their *ink* extent, not the surface box --
  DejaVu's chess glyphs carry uneven padding and side bearings, so centering on
  the surface leaves pieces visibly off-center;
* every glyph is composited with a contrasting outline, which is what keeps a
  black piece readable on a dark square.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import chess
import pygame

from .theme import (
    MONO_FONT_CANDIDATES,
    PIECE_FONT_CANDIDATES,
    PIECE_GLYPHS,
    UI_FONT_CANDIDATES,
    DEFAULT_THEME,
    Theme,
)


OUTLINE_SAMPLES = 16  # 8 compass offsets leave visible corners at 3px


class FontUnavailableError(RuntimeError):
    """Raised when no candidate font could be loaded."""


def resolve_font_path(candidates: tuple[str, ...], *, match: str | None = None) -> str:
    for path in candidates:
        if path and os.path.exists(path):
            return path
    if match:
        found = pygame.font.match_font(match)
        if found:
            return found
    raise FontUnavailableError(f"none of these fonts exist: {candidates}")


def load_font(candidates: tuple[str, ...], size: int, *, match: str | None = None):
    return pygame.font.Font(resolve_font_path(candidates, match=match), size)


@dataclass(frozen=True)
class Glyph:
    surface: pygame.Surface
    ink: pygame.Rect  # ink extent within `surface`


def _stroked_glyph(font, text: str, fill, outline, thickness: int) -> pygame.Surface:
    inner = font.render(text, True, fill)
    ring = font.render(text, True, outline)
    width, height = inner.get_size()
    out = pygame.Surface((width + 2 * thickness, height + 2 * thickness), pygame.SRCALPHA)
    for index in range(OUTLINE_SAMPLES):
        angle = 2 * math.pi * index / OUTLINE_SAMPLES
        dx = round(thickness * math.cos(angle))
        dy = round(thickness * math.sin(angle))
        out.blit(ring, (thickness + dx, thickness + dy))
    out.blit(inner, (thickness, thickness))
    return out


class PieceRenderer:
    """Builds and caches the twelve piece surfaces for one square size."""

    def __init__(
        self,
        square_px: int,
        theme: Theme = DEFAULT_THEME,
        *,
        font_path: str | None = None,
    ) -> None:
        self.square_px = square_px
        self.theme = theme
        self.font_path = font_path or resolve_font_path(
            PIECE_FONT_CANDIDATES, match="dejavusans"
        )
        self.font_size = self._fit_font_size()
        self._font = pygame.font.Font(self.font_path, self.font_size)
        self._cache: dict[tuple[int, bool], Glyph] = {}
        for piece_type in PIECE_GLYPHS:
            for is_white in (True, False):
                self._cache[(piece_type, is_white)] = self._build(piece_type, is_white)

    def _fit_font_size(self) -> int:
        """Scale the font so the tallest glyph's ink fills the target ratio."""
        target = self.square_px * self.theme.piece_square_ratio
        probe_size = max(8, int(self.square_px * 0.80))
        probe = pygame.font.Font(self.font_path, probe_size)
        tallest = 0
        for glyph in PIECE_GLYPHS.values():
            surface = probe.render(glyph, True, (255, 255, 255))
            ink = surface.get_bounding_rect()
            if ink.height == 0:
                raise FontUnavailableError(
                    f"font {self.font_path} renders {glyph!r} as blank"
                )
            tallest = max(tallest, ink.height)
        scaled = round(probe_size * target / tallest)
        # Leave room for the outline on both sides.
        max_size = max(8, int(self.square_px * 1.4))
        return max(8, min(scaled, max_size))

    def _build(self, piece_type: int, is_white: bool) -> Glyph:
        fill, outline = self.theme.piece_colors(is_white)
        surface = _stroked_glyph(
            self._font,
            PIECE_GLYPHS[piece_type],
            fill,
            outline,
            self.theme.piece_stroke_px,
        )
        return Glyph(surface=surface, ink=surface.get_bounding_rect())

    def glyph_for(self, piece: chess.Piece) -> Glyph:
        return self._cache[(piece.piece_type, piece.color == chess.WHITE)]

    def surface_for(self, piece: chess.Piece) -> pygame.Surface:
        return self.glyph_for(piece).surface

    def blit(
        self,
        target: pygame.Surface,
        piece: chess.Piece,
        rect,
        *,
        alpha: int = 255,
    ) -> None:
        """Draw ``piece`` centered by ink extent inside ``rect``."""
        glyph = self.glyph_for(piece)
        cx = rect.x + rect.w // 2
        cy = rect.y + rect.h // 2
        position = (cx - glyph.ink.centerx, cy - glyph.ink.centery)
        if alpha >= 255:
            target.blit(glyph.surface, position)
            return
        faded = glyph.surface.copy()
        faded.set_alpha(alpha)
        target.blit(faded, position)


@dataclass
class Fonts:
    """The UI text fonts, loaded once."""

    small: pygame.font.Font
    body: pygame.font.Font
    label: pygame.font.Font
    title: pygame.font.Font
    mono: pygame.font.Font

    @classmethod
    def load(cls, theme: Theme = DEFAULT_THEME) -> "Fonts":
        return cls(
            small=load_font(UI_FONT_CANDIDATES, theme.font_small, match="dejavusans"),
            body=load_font(UI_FONT_CANDIDATES, theme.font_body, match="dejavusans"),
            label=load_font(UI_FONT_CANDIDATES, theme.font_label, match="dejavusans"),
            title=load_font(UI_FONT_CANDIDATES, theme.font_title, match="dejavusans"),
            mono=load_font(
                MONO_FONT_CANDIDATES, theme.font_mono, match="dejavusansmono"
            ),
        )


@dataclass
class Assets:
    """Everything the renderer needs that must be built against pygame."""

    fonts: Fonts
    pieces: PieceRenderer
    theme: Theme = DEFAULT_THEME

    @classmethod
    def load(cls, square_px: int, theme: Theme = DEFAULT_THEME) -> "Assets":
        return cls(
            fonts=Fonts.load(theme),
            pieces=PieceRenderer(square_px, theme),
            theme=theme,
        )
