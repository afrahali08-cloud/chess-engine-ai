"""Glyph and full-frame render tests.

The env vars must be set before pygame is imported, and ``importorskip`` keeps
the suite green for anyone who has not installed pygame.
"""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import chess
import pytest

pygame = pytest.importorskip("pygame")

from gui import interaction
from gui.layout import build_layout, square_rect
from gui.pieces import (
    Assets,
    FontUnavailableError,
    PieceRenderer,
    candidate_font_paths,
    describe_fonts,
    font_supports,
    resolve_piece_font,
    resolve_ui_font,
)
from gui.render import coach_panel_lines, draw_frame, wrap_text
from gui.theme import DEFAULT_THEME, NOTDEF_PROBE, PIECE_GLYPHS
from gui.viewmodel import CoachView, MoveRow, ViewModel


SQUARE = 86


@pytest.fixture(scope="module")
def assets():
    pygame.display.init()
    pygame.font.init()
    pygame.display.set_mode((1280, 800))
    made = Assets.load(SQUARE)
    yield made
    pygame.quit()


def surface(size=(1280, 800)):
    return pygame.Surface(size)


# --------------------------------------------------------- font choice


def glyphless_font_path():
    """A font on this machine that lacks the chess glyphs, if there is one."""
    for family in ("ubuntu", "liberationsans", "notosansmono", "freesans"):
        path = pygame.font.match_font(family)
        if not path:
            continue
        try:
            font = pygame.font.Font(path, 48)
        except Exception:  # noqa: BLE001
            continue
        if not font_supports(font, PIECE_GLYPHS.values()):
            return path
    return None


def test_pieces_do_not_render_as_missing_glyph_boxes(assets):
    """Regression: a font without chess glyphs used to draw plain rectangles.

    Font.metrics() reports real values for absent characters and a tofu box has
    plenty of ink, so the only reliable check is against the font's own
    missing-glyph render.
    """
    font = pygame.font.Font(assets.pieces.font_path, 48)
    tofu = pygame.image.tostring(
        font.render(NOTDEF_PROBE, True, (255, 255, 255)), "RGBA"
    )
    for glyph in PIECE_GLYPHS.values():
        rendered = pygame.image.tostring(
            font.render(glyph, True, (255, 255, 255)), "RGBA"
        )
        assert rendered != tofu, f"{glyph!r} renders as a missing-glyph box"


def test_font_supports_accepts_the_resolved_piece_font():
    font = pygame.font.Font(resolve_piece_font(), 48)
    assert font_supports(font, PIECE_GLYPHS.values()) is True


def test_font_supports_rejects_a_font_without_chess_glyphs():
    path = glyphless_font_path()
    if path is None:
        pytest.skip("every font on this machine has the chess glyphs")
    assert font_supports(pygame.font.Font(path, 48), PIECE_GLYPHS.values()) is False


def test_resolve_piece_font_picks_something_that_works():
    path = resolve_piece_font()
    assert os.path.exists(path)


def test_resolve_piece_font_reports_clearly_when_nothing_works(monkeypatch):
    """The teammate-facing failure: say what is wrong, do not draw boxes."""
    from gui import pieces

    glyphless = glyphless_font_path()
    monkeypatch.setattr(
        pieces, "PIECE_FONT_CANDIDATES", (glyphless,) if glyphless else ()
    )
    monkeypatch.setattr(pieces, "PIECE_FONT_FAMILIES", ())
    monkeypatch.setattr(pieces, "BUNDLED_FONT_DIR", "/nonexistent")
    monkeypatch.setitem(__import__("sys").modules, "matplotlib", None)

    with pytest.raises(FontUnavailableError) as caught:
        pieces.resolve_piece_font()
    message = str(caught.value)
    assert "U+265A" in message
    assert "--check-fonts" in message


def test_candidate_paths_are_deduplicated_and_exist():
    paths = candidate_font_paths(
        ("/nope/missing.ttf",) + tuple(), ("dejavusans", "dejavusans")
    )
    assert len(paths) == len(set(paths))
    assert all(os.path.exists(path) for path in paths)


def test_ui_font_falls_back_to_the_pygame_default():
    """UI text must never hard-fail; only the pieces need specific glyphs."""
    assert resolve_ui_font(("/nope/missing.ttf",), ()) is None
    assert pygame.font.Font(None, 16) is not None


def test_describe_fonts_reports_the_selection():
    report = describe_fonts()
    assert "selected piece font:" in report
    assert "platform:" in report


# ------------------------------------------------------------- glyphs


def test_every_piece_renders_visible_ink(assets):
    """A blank bounding rect means a tofu box or a font that failed to load."""
    for piece_type in PIECE_GLYPHS:
        for color in (chess.WHITE, chess.BLACK):
            glyph = assets.pieces.glyph_for(chess.Piece(piece_type, color))
            assert glyph.ink.width > 0
            assert glyph.ink.height > 0


def test_white_and_black_pieces_are_not_identical(assets):
    for piece_type in PIECE_GLYPHS:
        white = assets.pieces.surface_for(chess.Piece(piece_type, chess.WHITE))
        black = assets.pieces.surface_for(chess.Piece(piece_type, chess.BLACK))
        assert pygame.image.tostring(white, "RGBA") != pygame.image.tostring(
            black, "RGBA"
        )


def test_glyphs_fit_inside_a_square(assets):
    for piece_type in PIECE_GLYPHS:
        for color in (chess.WHITE, chess.BLACK):
            glyph = assets.pieces.glyph_for(chess.Piece(piece_type, color))
            assert glyph.ink.width <= SQUARE
            assert glyph.ink.height <= SQUARE


def test_autofit_hits_the_target_height(assets):
    """Ink height, minus the outline it carries, should track the theme ratio."""
    stroke = 2 * DEFAULT_THEME.piece_stroke_px
    tallest = max(
        assets.pieces.glyph_for(chess.Piece(piece_type, chess.WHITE)).ink.height
        for piece_type in PIECE_GLYPHS
    )
    target = SQUARE * DEFAULT_THEME.piece_square_ratio
    assert abs((tallest - stroke) - target) / target < 0.06


def test_pieces_are_centered_by_ink_not_by_surface(assets):
    """Off-center pieces are the classic giveaway of naive glyph blitting."""
    target = surface((SQUARE, SQUARE))
    from gui.layout import Rect

    assets.pieces.blit(target, chess.Piece(chess.PAWN, chess.WHITE), Rect(0, 0, SQUARE, SQUARE))
    ink = target.get_bounding_rect()
    assert abs(ink.centerx - SQUARE // 2) <= 2
    assert abs(ink.centery - SQUARE // 2) <= 2


def test_a_renderer_can_be_built_for_another_square_size():
    renderer = PieceRenderer(48)
    glyph = renderer.glyph_for(chess.Piece(chess.KING, chess.WHITE))
    assert 0 < glyph.ink.height <= 48


# -------------------------------------------------------- full frames


def test_draw_frame_renders_the_start_position(assets):
    target = surface()
    draw_frame(target, ViewModel(board=chess.Board()), assets)
    assert target.get_at((0, 0))[:3] == DEFAULT_THEME.background


def test_draw_frame_handles_a_midgame_with_every_panel_populated(assets):
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push(chess.Move.from_uci(uci))
    view = ViewModel(
        board=board,
        selection=interaction.SelectionState(
            selected=chess.F3, targets=interaction.legal_targets(board, chess.F3)
        ),
        last_move=chess.Move.from_uci("b8c6"),
        eval_cp=42.0,
        thinking=True,
        thinking_label="Engine thinking",
        status_text="Engine thinking",
        coach_enabled=True,
        coach=CoachView(
            classification="Blunder",
            centipawn_loss=430,
            played_san="Nc6",
            best_san="Nf6",
            explanation="A long explanation " * 12,
            search_depth=4,
        ),
        moves=(MoveRow(1, "e4", "e5"), MoveRow(2, "Nf3", "Nc6", None, "Blunder")),
    )
    draw_frame(surface(), view, assets)


def test_draw_frame_handles_an_open_promotion_prompt(assets):
    board = chess.Board("8/4P3/8/8/8/8/8/K6k w - - 0 1")
    state = interaction.SelectionState(
        selected=chess.E7,
        targets=interaction.legal_targets(board, chess.E7),
        promotion=interaction.PromotionPrompt(chess.E7, chess.E8),
    )
    draw_frame(surface(), ViewModel(board=board, selection=state), assets)


def test_draw_frame_handles_game_over_and_check(assets):
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push(chess.Move.from_uci(uci))
    draw_frame(
        surface(),
        ViewModel(board=board, result_text="Checkmate - Black wins", eval_cp=-99999),
        assets,
    )


def test_draw_frame_handles_a_flipped_board_and_static_coach_note(assets):
    view = ViewModel(
        board=chess.Board(),
        flipped=True,
        eval_cp=None,
        coach_enabled=True,
        coach=CoachView(
            classification="Inaccuracy",
            centipawn_loss=120,
            played_san="e4",
            best_san="d4",
            explanation="short",
            search_depth=0,
            used_static_fallback=True,
        ),
    )
    draw_frame(surface(), view, assets)


def test_draw_frame_handles_an_error_flash(assets):
    state = interaction.SelectionState(error_square=chess.A1, error_expires_at=10.0)
    draw_frame(
        surface(),
        ViewModel(board=chess.Board(), selection=state, now=9.0),
        assets,
    )


@pytest.mark.parametrize("size", [(1280, 800), (1024, 768), (1600, 1000)])
def test_draw_frame_adapts_to_other_window_sizes(assets, size):
    layout = build_layout(*size)
    local = Assets.load(layout.square)
    draw_frame(surface(size), ViewModel(board=chess.Board()), local)


def test_board_is_drawn_where_the_layout_says_it_is(assets):
    target = surface()
    draw_frame(target, ViewModel(board=chess.Board(None)), assets)
    layout = build_layout()
    a1 = square_rect(chess.A1, layout, False)
    b1 = square_rect(chess.B1, layout, False)

    assert target.get_at(a1.center)[:3] == DEFAULT_THEME.dark_square
    assert target.get_at(b1.center)[:3] == DEFAULT_THEME.light_square


# ------------------------------------------------------------ helpers


def test_wrap_text_breaks_on_width(assets):
    lines = wrap_text(assets.fonts.body, "word " * 40, 200)
    assert len(lines) > 1
    for line in lines:
        assert assets.fonts.body.size(line)[0] <= 200


def test_wrap_text_handles_empty_and_long_words(assets):
    assert wrap_text(assets.fonts.body, "", 200) == []
    assert len(wrap_text(assets.fonts.body, "x" * 200, 50)) == 1


# -------------------------------------------------------- coach panel


def coach_view(**overrides):
    base = dict(
        classification="Blunder",
        centipawn_loss=885,
        played_san="Qg5",
        best_san="Nf6",
        explanation="The engine preferred Nf6, which develops a knight.",
        search_depth=3,
        reason="Qg5 leaves your queen on g5 en prise.",
        refutation_san=("Nxg5",),
        best_line_san=("Nc3",),
        material_swing="loses a queen",
    )
    base.update(overrides)
    return CoachView(**base)


def test_coach_panel_puts_the_reason_first():
    """Only ~4 lines fit, so the "why" must not be the thing that gets cut."""
    lines = coach_panel_lines(coach_view())
    assert lines[0][0].startswith("Qg5 leaves your queen")


def test_coach_panel_shows_the_line_with_its_material_swing():
    lines = [text for text, _tone in coach_panel_lines(coach_view())]
    assert "Qg5 Nxg5  (loses a queen)" in lines


def test_coach_panel_labels_the_expected_continuation():
    lines = [text for text, _tone in coach_panel_lines(coach_view())]
    assert "expected Nf6 Nc3" in lines


def test_coach_panel_drops_the_engine_sentence_last():
    lines = [text for text, _tone in coach_panel_lines(coach_view())]
    assert lines[-1].startswith("The engine preferred")


def test_coach_panel_omits_empty_sections():
    lines = coach_panel_lines(
        coach_view(reason="", refutation_san=(), best_line_san=(), material_swing=None)
    )
    assert len(lines) == 1


def test_coach_panel_does_not_repeat_the_reason_as_the_explanation():
    view = coach_view(explanation="Qg5 leaves your queen on g5 en prise.")
    texts = [text for text, _tone in coach_panel_lines(view)]
    assert texts.count("Qg5 leaves your queen on g5 en prise.") == 1


def test_mate_loss_text_replaces_the_pawn_figure():
    assert coach_view(mate_for_opponent=True).loss_text == "allows forced mate"
    assert coach_view(missed_mate=True).loss_text == "misses forced mate"
    assert coach_view().loss_text == "8.85 pawn loss"


def test_full_frame_renders_with_a_populated_coach_panel(assets):
    surface = pygame.Surface((1280, 800))
    vm = ViewModel(board=chess.Board(), coach=coach_view(), coach_enabled=True)
    draw_frame(surface, vm, assets)
    assert surface.get_bounding_rect().width > 0
