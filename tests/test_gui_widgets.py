"""Widget behavior tests. Needs pygame for its event types, but never a window."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pytest

pygame = pytest.importorskip("pygame")

from gui.layout import Rect
from gui.viewmodel import MoveRow
from gui.widgets import Action, Button, Cycler, MoveListView, Stepper, dispatch


RECT = Rect(100, 100, 200, 30)


def press(widget, pos, button=1):
    event = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {"pos": pos, "button": button}
    )
    return widget.handle(event)


def move_mouse(widget, pos):
    return widget.handle(pygame.event.Event(pygame.MOUSEMOTION, {"pos": pos}))


def wheel(widget, dy):
    return widget.handle(pygame.event.Event(pygame.MOUSEWHEEL, {"y": dy, "x": 0}))


# -------------------------------------------------------------- button


def test_button_emits_its_action_when_clicked():
    button = Button(RECT, "New Game", "new_game")
    assert press(button, RECT.center) == Action("new_game")


def test_button_ignores_clicks_outside_itself():
    button = Button(RECT, "New Game", "new_game")
    assert press(button, (0, 0)) is None


def test_button_ignores_right_clicks():
    button = Button(RECT, "New Game", "new_game")
    assert press(button, RECT.center, button=3) is None


def test_disabled_widgets_emit_nothing():
    button = Button(RECT, "New Game", "new_game", enabled=False)
    assert press(button, RECT.center) is None


def test_hover_tracks_the_pointer():
    button = Button(RECT, "New Game", "new_game")
    move_mouse(button, RECT.center)
    assert button.hovered is True
    move_mouse(button, (0, 0))
    assert button.hovered is False


# -------------------------------------------------------------- cycler


def test_cycler_advances_and_wraps():
    cycler = Cycler(RECT, "Evaluator", ["a", "b", "c"], "evaluator")
    assert press(cycler, RECT.center) == Action("evaluator", "b")
    assert press(cycler, RECT.center) == Action("evaluator", "c")
    assert press(cycler, RECT.center) == Action("evaluator", "a")


def test_right_click_cycles_backwards():
    cycler = Cycler(RECT, "Evaluator", ["a", "b", "c"], "evaluator")
    assert press(cycler, RECT.center, button=3) == Action("evaluator", "c")


def test_cycler_set_value_syncs_the_index():
    cycler = Cycler(RECT, "Evaluator", ["a", "b", "c"], "evaluator")
    cycler.set_value("c")
    assert cycler.value == "c"


def test_cycler_set_value_ignores_unknown_values():
    cycler = Cycler(RECT, "Evaluator", ["a", "b"], "evaluator", index=1)
    cycler.set_value("nope")
    assert cycler.value == "b"


# ------------------------------------------------------------- stepper


def test_stepper_emits_minus_one_on_the_left_button():
    stepper = Stepper(RECT, "Depth", "depth", value=4)
    assert press(stepper, stepper.minus_rect.center) == Action("depth", -1)


def test_stepper_emits_plus_one_on_the_right_button():
    stepper = Stepper(RECT, "Depth", "depth", value=4)
    assert press(stepper, stepper.plus_rect.center) == Action("depth", +1)


def test_stepper_center_click_does_nothing():
    stepper = Stepper(RECT, "Depth", "depth", value=4)
    assert press(stepper, RECT.center) is None


def test_stepper_wheel_steps_the_value():
    stepper = Stepper(RECT, "Depth", "depth", value=4)
    move_mouse(stepper, RECT.center)
    assert wheel(stepper, 1) == Action("depth", 1)
    assert wheel(stepper, -1) == Action("depth", -1)


def test_stepper_button_zones_do_not_overlap():
    stepper = Stepper(Rect(0, 0, 200, 30), "Depth", "depth")
    assert stepper.minus_rect.right <= stepper.plus_rect.x


# ------------------------------------------------------------ movelist


def rows(count):
    return [MoveRow(number=i + 1, white=f"w{i}", black=f"b{i}") for i in range(count)]


def test_movelist_follows_the_tail_as_moves_are_appended():
    view = MoveListView(Rect(0, 0, 300, 240))  # 10 visible rows
    view.set_rows(rows(30))

    assert view.pinned_to_bottom is True
    assert view.scroll == view.max_scroll


def test_scrolling_up_unpins_and_new_rows_do_not_yank_the_view():
    view = MoveListView(Rect(0, 0, 300, 240))
    view.set_rows(rows(30))
    view.hovered = True
    wheel(view, 1)  # scroll up

    assert view.pinned_to_bottom is False
    held = view.scroll
    view.set_rows(rows(31))
    assert view.scroll == held


def test_scrolling_back_to_the_bottom_repins():
    view = MoveListView(Rect(0, 0, 300, 240))
    view.set_rows(rows(30))
    view.hovered = True
    wheel(view, 1)
    assert view.pinned_to_bottom is False
    wheel(view, -5)
    assert view.pinned_to_bottom is True


def test_scroll_is_clamped_at_both_ends():
    view = MoveListView(Rect(0, 0, 300, 240))
    view.set_rows(rows(30))
    view.hovered = True
    for _ in range(40):
        wheel(view, 1)
    assert view.scroll == 0
    for _ in range(40):
        wheel(view, -1)
    assert view.scroll == view.max_scroll


def test_short_lists_do_not_scroll():
    view = MoveListView(Rect(0, 0, 300, 240))
    view.set_rows(rows(3))
    assert view.max_scroll == 0
    view.hovered = True
    wheel(view, -3)
    assert view.scroll == 0


def test_shrinking_the_list_clamps_the_scroll():
    view = MoveListView(Rect(0, 0, 300, 240))
    view.set_rows(rows(30))
    view.set_rows(rows(4))
    assert view.scroll <= view.max_scroll


def test_wheel_is_ignored_when_the_pointer_is_elsewhere():
    view = MoveListView(Rect(0, 0, 300, 240))
    view.set_rows(rows(30))
    view.hovered = False
    before = view.scroll
    wheel(view, 1)
    assert view.scroll == before


# ------------------------------------------------------------ dispatch


def test_dispatch_collects_actions_from_every_widget():
    first = Button(Rect(0, 0, 50, 20), "A", "a")
    second = Button(Rect(0, 0, 50, 20), "B", "b")  # same spot, both fire
    event = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {"pos": (10, 10), "button": 1}
    )

    assert dispatch([first, second], event) == [Action("a"), Action("b")]


def test_dispatch_returns_nothing_for_untouched_widgets():
    button = Button(Rect(500, 500, 50, 20), "A", "a")
    event = pygame.event.Event(
        pygame.MOUSEBUTTONDOWN, {"pos": (10, 10), "button": 1}
    )
    assert dispatch([button], event) == []
