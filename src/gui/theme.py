"""Colors, fonts, and sizes. No pygame import, so this stays headless-safe."""

from __future__ import annotations

from dataclasses import dataclass, field


RGB = tuple[int, int, int]
RGBA = tuple[int, int, int, int]

PIECE_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
)
UI_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)
MONO_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)

# Filled glyphs are used for BOTH colors and tinted, so the two sides share a
# silhouette the way a real set does. The hollow U+2654 range is thin-stroke and
# washes out on a projector.
PIECE_GLYPHS = {
    6: "♚",  # king
    5: "♛",  # queen
    4: "♜",  # rook
    3: "♝",  # bishop
    2: "♞",  # knight
    1: "♟",  # pawn
}

CLASSIFICATION_ORDER = (
    "Best",
    "Excellent",
    "Good",
    "Inaccuracy",
    "Mistake",
    "Blunder",
)


def _classification_colors() -> dict[str, RGB]:
    return {
        "Best": (127, 166, 80),
        "Excellent": (147, 181, 99),
        "Good": (184, 176, 74),
        "Inaccuracy": (208, 140, 62),
        "Mistake": (208, 102, 62),
        "Blunder": (199, 75, 75),
    }


@dataclass(frozen=True)
class Theme:
    # board
    light_square: RGB = (235, 236, 208)
    dark_square: RGB = (119, 149, 86)
    selected: RGBA = (246, 246, 105, 150)
    last_move: RGBA = (205, 210, 106, 130)
    legal_marker: RGBA = (0, 0, 0, 45)
    error_flash: RGBA = (199, 75, 75, 110)
    check_glow: RGB = (224, 58, 58)
    promotion_scrim: RGBA = (0, 0, 0, 170)

    # pieces
    white_fill: RGB = (248, 248, 246)
    white_outline: RGB = (32, 34, 30)
    black_fill: RGB = (43, 44, 40)
    black_outline: RGB = (237, 238, 230)
    piece_stroke_px: int = 3
    piece_square_ratio: float = 0.78

    # chrome
    background: RGB = (26, 25, 23)
    panel: RGB = (38, 36, 33)
    panel_border: RGB = (61, 58, 53)
    text: RGB = (232, 230, 225)
    dim_text: RGB = (154, 150, 142)
    accent: RGB = (127, 166, 80)
    warn: RGB = (208, 140, 62)
    error: RGB = (199, 75, 75)

    # eval bar
    eval_white: RGB = (235, 236, 208)
    eval_black: RGB = (56, 54, 50)

    # widgets
    widget_bg: RGB = (52, 49, 45)
    widget_hover: RGB = (66, 62, 57)
    widget_press: RGB = (80, 76, 69)
    widget_border: RGB = (78, 74, 68)

    # type sizes
    font_small: int = 15
    font_body: int = 17
    font_label: int = 20
    font_title: int = 24
    font_mono: int = 17

    classification: dict[str, RGB] = field(default_factory=_classification_colors)

    def classification_color(self, name: str | None) -> RGB:
        if not name:
            return self.dim_text
        return self.classification.get(name, self.text)

    def piece_colors(self, is_white: bool) -> tuple[RGB, RGB]:
        """Return ``(fill, outline)`` for a piece color."""
        if is_white:
            return self.white_fill, self.white_outline
        return self.black_fill, self.black_outline


DEFAULT_THEME = Theme()
