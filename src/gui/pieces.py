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
import sys
from dataclasses import dataclass

import chess
import pygame

from .theme import (
    MONO_FONT_CANDIDATES,
    MONO_FONT_FAMILIES,
    NOTDEF_PROBE,
    PIECE_FONT_CANDIDATES,
    PIECE_FONT_FAMILIES,
    PIECE_GLYPHS,
    UI_FONT_CANDIDATES,
    UI_FONT_FAMILIES,
    DEFAULT_THEME,
    Theme,
)


OUTLINE_SAMPLES = 16  # 8 compass offsets leave visible corners at 3px
GLYPH_PROBE_SIZE = 48
BUNDLED_FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")


class FontUnavailableError(RuntimeError):
    """Raised when no font with the required glyphs could be found."""


def _raster(font, text: str) -> bytes:
    return pygame.image.tostring(font.render(text, True, (255, 255, 255)), "RGBA")


def font_supports(font, glyphs) -> bool:
    """True if ``font`` draws real glyphs rather than missing-glyph boxes.

    Font.metrics() returns plausible values for absent characters and a tofu box
    has plenty of ink, so neither is a usable test. Rendering a noncharacter
    gives this font's tofu; any glyph matching it byte-for-byte is missing.
    """
    tofu = _raster(font, NOTDEF_PROBE)
    for glyph in glyphs:
        surface = font.render(glyph, True, (255, 255, 255))
        if surface.get_bounding_rect().height == 0:
            return False
        if _raster(font, glyph) == tofu:
            return False
    return True


def candidate_font_paths(
    explicit: tuple[str, ...],
    families: tuple[str, ...],
    *,
    bundled_names: tuple[str, ...] = (),
) -> list[str]:
    """Ordered, de-duplicated font paths to try."""
    found: list[str] = []

    def add(path: str | None) -> None:
        if path and path not in found and os.path.exists(path):
            found.append(path)

    for name in bundled_names:
        add(os.path.join(BUNDLED_FONT_DIR, name))
    for path in explicit:
        add(path)
    # matplotlib always ships DejaVuSans and is common in course environments.
    try:
        import matplotlib

        add(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"))
        add(
            os.path.join(
                matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSansMono.ttf"
            )
        )
    except Exception:  # noqa: BLE001 - matplotlib is optional
        pass
    for family in families:
        add(pygame.font.match_font(family))
    return found


def resolve_piece_font() -> str:
    """First font that can actually draw the chess pieces."""
    tried = candidate_font_paths(
        PIECE_FONT_CANDIDATES,
        PIECE_FONT_FAMILIES,
        bundled_names=("DejaVuSans.ttf", "FreeSerif.ttf"),
    )
    for path in tried:
        try:
            font = pygame.font.Font(path, GLYPH_PROBE_SIZE)
        except Exception:  # noqa: BLE001 - unreadable font file, try the next
            continue
        if font_supports(font, PIECE_GLYPHS.values()):
            return path
    raise FontUnavailableError(
        "no installed font contains the chess piece glyphs "
        "(U+265A-U+265F).\n"
        f"Checked {len(tried)} font(s): "
        + (", ".join(tried) if tried else "none found")
        + "\nInstall one of: DejaVu Sans (Linux: `sudo apt install fonts-dejavu-core`, "
        "or `pip install matplotlib`), GNU FreeFont, or Noto Sans Symbols 2. "
        "Run `python src/gui_main.py --check-fonts` for details."
    )


def resolve_ui_font(
    explicit: tuple[str, ...],
    families: tuple[str, ...],
) -> str | None:
    """Any readable font will do for UI text; ``None`` means pygame's default."""
    for path in candidate_font_paths(explicit, families):
        try:
            pygame.font.Font(path, GLYPH_PROBE_SIZE)
        except Exception:  # noqa: BLE001
            continue
        return path
    return None


def load_font(candidates: tuple[str, ...], size: int, *, families: tuple[str, ...] = ()):
    """Load a UI font, falling back to pygame's built-in rather than failing."""
    return pygame.font.Font(resolve_ui_font(candidates, families), size)


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
        self.font_path = font_path or resolve_piece_font()
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
        ui = UI_FONT_CANDIDATES
        fam = UI_FONT_FAMILIES
        return cls(
            small=load_font(ui, theme.font_small, families=fam),
            body=load_font(ui, theme.font_body, families=fam),
            label=load_font(ui, theme.font_label, families=fam),
            title=load_font(ui, theme.font_title, families=fam),
            mono=load_font(
                MONO_FONT_CANDIDATES, theme.font_mono, families=MONO_FONT_FAMILIES
            ),
        )


def describe_fonts() -> str:
    """Human-readable report of which fonts were found and whether they work."""
    pygame.font.init()
    lines = [
        f"platform: {sys.platform}",
        f"pygame:   {pygame.version.ver}",
        f"bundled font dir: {BUNDLED_FONT_DIR}"
        + ("" if os.path.isdir(BUNDLED_FONT_DIR) else " (absent)"),
        "",
        "piece font candidates (need U+265A-U+265F):",
    ]
    candidates = candidate_font_paths(
        PIECE_FONT_CANDIDATES,
        PIECE_FONT_FAMILIES,
        bundled_names=("DejaVuSans.ttf", "FreeSerif.ttf"),
    )
    if not candidates:
        lines.append("  (none of the candidate paths or families exist)")
    for path in candidates:
        try:
            font = pygame.font.Font(path, GLYPH_PROBE_SIZE)
        except Exception as error:  # noqa: BLE001
            lines.append(f"  [unreadable] {path}  ({error})")
            continue
        ok = font_supports(font, PIECE_GLYPHS.values())
        lines.append(f"  [{'OK      ' if ok else 'NO GLYPH'}] {path}")

    try:
        chosen = resolve_piece_font()
        lines += ["", f"selected piece font: {chosen}"]
    except FontUnavailableError as error:
        lines += ["", f"NO USABLE PIECE FONT: {error}"]

    ui = resolve_ui_font(UI_FONT_CANDIDATES, UI_FONT_FAMILIES)
    mono = resolve_ui_font(MONO_FONT_CANDIDATES, MONO_FONT_FAMILIES)
    lines.append(f"ui font:             {ui or 'pygame built-in default'}")
    lines.append(f"mono font:           {mono or 'pygame built-in default'}")
    return "\n".join(lines)


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
