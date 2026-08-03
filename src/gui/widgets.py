"""Minimal hand-rolled widgets.

pygame ships no widget toolkit. Rather than build dropdowns and sliders -- which
need popup surfaces, z-ordering, click-outside dismissal, and drag capture --
this uses cyclers and steppers: the current value is always visible as text,
which reads better on a projector and needs no event capture.

Widgets return *action tokens* from :meth:`Widget.handle` instead of invoking
callbacks, so every state change happens in one place in ``app``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import pygame

from .layout import Rect
from .theme import DEFAULT_THEME, Theme


@dataclass(frozen=True)
class Action:
    name: str
    value: Any = None


def _rect(rect: Rect) -> pygame.Rect:
    return pygame.Rect(rect.x, rect.y, rect.w, rect.h)


def draw_chip(
    surface,
    rect: Rect,
    theme: Theme,
    *,
    hovered: bool = False,
    pressed: bool = False,
    enabled: bool = True,
    border: bool = True,
):
    if not enabled:
        color = theme.panel
    elif pressed:
        color = theme.widget_press
    elif hovered:
        color = theme.widget_hover
    else:
        color = theme.widget_bg
    pygame.draw.rect(surface, color, _rect(rect), border_radius=6)
    if border:
        pygame.draw.rect(
            surface, theme.widget_border, _rect(rect), width=1, border_radius=6
        )


def draw_centered(surface, font, text, rect: Rect, color, *, dy: int = 0):
    glyph = font.render(text, True, color)
    surface.blit(
        glyph,
        (
            rect.x + (rect.w - glyph.get_width()) // 2,
            rect.y + (rect.h - glyph.get_height()) // 2 + dy,
        ),
    )


class Widget:
    def __init__(self, rect: Rect, *, name: str = "", enabled: bool = True) -> None:
        self.rect = rect
        self.name = name
        self.enabled = enabled
        self.hovered = False
        self.pressed = False

    # -- events -------------------------------------------------------
    def handle(self, event, now: float = 0.0) -> Action | None:
        if not self.enabled:
            return None
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.contains(event.pos)
            return None
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.contains(event.pos):
            if event.button in (1, 3):
                self.pressed = True
                return self.on_click(event.pos, event.button)
            return self.on_scroll_button(event.button)
        if event.type == pygame.MOUSEBUTTONUP:
            self.pressed = False
        if event.type == pygame.MOUSEWHEEL and self.hovered:
            return self.on_wheel(event.y)
        return None

    def on_click(self, pos, button: int) -> Action | None:
        return None

    def on_scroll_button(self, button: int) -> Action | None:
        return None

    def on_wheel(self, dy: int) -> Action | None:
        return None

    # -- drawing ------------------------------------------------------
    def draw(self, surface, assets) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class Button(Widget):
    def __init__(self, rect: Rect, label: str, action: str, **kwargs) -> None:
        super().__init__(rect, name=action, **kwargs)
        self.label = label
        self.action = action

    def on_click(self, pos, button: int) -> Action | None:
        if button == 1:
            return Action(self.action)
        return None

    def draw(self, surface, assets) -> None:
        theme = assets.theme
        draw_chip(
            surface,
            self.rect,
            theme,
            hovered=self.hovered,
            pressed=self.pressed,
            enabled=self.enabled,
        )
        draw_centered(
            surface,
            assets.fonts.body,
            self.label,
            self.rect,
            theme.text if self.enabled else theme.dim_text,
        )


class Cycler(Widget):
    """Click advances to the next option; right-click goes back."""

    def __init__(
        self,
        rect: Rect,
        label: str,
        options: Sequence[Any],
        action: str,
        *,
        index: int = 0,
        formatter=str,
        **kwargs,
    ) -> None:
        super().__init__(rect, name=action, **kwargs)
        self.label = label
        self.options = list(options)
        self.action = action
        self.index = index
        self.formatter = formatter

    @property
    def value(self):
        return self.options[self.index % len(self.options)]

    def set_value(self, value) -> None:
        if value in self.options:
            self.index = self.options.index(value)

    def on_click(self, pos, button: int) -> Action | None:
        step = 1 if button == 1 else -1
        self.index = (self.index + step) % len(self.options)
        return Action(self.action, self.value)

    def draw(self, surface, assets) -> None:
        theme = assets.theme
        draw_chip(
            surface,
            self.rect,
            theme,
            hovered=self.hovered,
            pressed=self.pressed,
            enabled=self.enabled,
        )
        font = assets.fonts.body
        label = font.render(f"{self.label}", True, theme.dim_text)
        surface.blit(
            label,
            (self.rect.x + 10, self.rect.y + (self.rect.h - label.get_height()) // 2),
        )
        value = font.render(self.formatter(self.value), True, theme.text)
        chevron = font.render("›", True, theme.dim_text)
        surface.blit(
            chevron,
            (
                self.rect.right - 12 - chevron.get_width(),
                self.rect.y + (self.rect.h - chevron.get_height()) // 2,
            ),
        )
        surface.blit(
            value,
            (
                self.rect.right - 24 - chevron.get_width() - value.get_width(),
                self.rect.y + (self.rect.h - value.get_height()) // 2,
            ),
        )


class Stepper(Widget):
    """``[-] label value [+]`` -- exact values, no drag capture needed."""

    BUTTON_W = 30

    def __init__(
        self,
        rect: Rect,
        label: str,
        action: str,
        *,
        value: float = 0,
        formatter=str,
        **kwargs,
    ) -> None:
        super().__init__(rect, name=action, **kwargs)
        self.label = label
        self.action = action
        self.value = value
        self.formatter = formatter

    @property
    def minus_rect(self) -> Rect:
        return Rect(self.rect.x, self.rect.y, self.BUTTON_W, self.rect.h)

    @property
    def plus_rect(self) -> Rect:
        return Rect(
            self.rect.right - self.BUTTON_W, self.rect.y, self.BUTTON_W, self.rect.h
        )

    def on_click(self, pos, button: int) -> Action | None:
        if button != 1:
            return None
        if self.minus_rect.contains(pos):
            return Action(self.action, -1)
        if self.plus_rect.contains(pos):
            return Action(self.action, +1)
        return None

    def on_wheel(self, dy: int) -> Action | None:
        return Action(self.action, 1 if dy > 0 else -1)

    def draw(self, surface, assets) -> None:
        theme = assets.theme
        draw_chip(surface, self.rect, theme, hovered=self.hovered, enabled=self.enabled)
        font = assets.fonts.body
        for rect, glyph in ((self.minus_rect, "−"), (self.plus_rect, "+")):
            hovered = self.hovered and rect.contains(pygame.mouse.get_pos())
            draw_chip(
                surface,
                rect.inflate(-3, -3),
                theme,
                hovered=hovered,
                border=False,
            )
            draw_centered(surface, font, glyph, rect, theme.text)
        inner = Rect(
            self.rect.x + self.BUTTON_W,
            self.rect.y,
            self.rect.w - 2 * self.BUTTON_W,
            self.rect.h,
        )
        text = f"{self.label} {self.formatter(self.value)}"
        draw_centered(surface, font, text, inner, theme.text)


class MoveListView(Widget):
    """Scrollable numbered move list with follow-tail behavior."""

    ROW_HEIGHT = 24
    SCROLL_ROWS = 3
    SCROLLBAR_W = 6

    def __init__(self, rect: Rect, **kwargs) -> None:
        super().__init__(rect, name="moves", **kwargs)
        self.rows: tuple = ()
        self.scroll = 0
        self.pinned_to_bottom = True

    @property
    def visible_rows(self) -> int:
        return max(1, self.rect.h // self.ROW_HEIGHT)

    @property
    def max_scroll(self) -> int:
        return max(0, len(self.rows) - self.visible_rows)

    def set_rows(self, rows) -> None:
        grew = len(rows) != len(self.rows)
        self.rows = tuple(rows)
        if grew and self.pinned_to_bottom:
            self.scroll = self.max_scroll
        else:
            self.scroll = min(self.scroll, self.max_scroll)

    def on_wheel(self, dy: int) -> Action | None:
        self.scroll = max(0, min(self.max_scroll, self.scroll - dy * self.SCROLL_ROWS))
        self.pinned_to_bottom = self.scroll >= self.max_scroll
        return None

    def on_scroll_button(self, button: int) -> Action | None:
        if button == 4:
            return self.on_wheel(1)
        if button == 5:
            return self.on_wheel(-1)
        return None

    def draw(self, surface, assets) -> None:
        theme = assets.theme
        font = assets.fonts.mono
        previous_clip = surface.get_clip()
        surface.set_clip(_rect(self.rect))  # rows must never bleed past the panel
        try:
            y = self.rect.y
            for row in self.rows[self.scroll : self.scroll + self.visible_rows]:
                number = font.render(f"{row.number:>3}.", True, theme.dim_text)
                surface.blit(number, (self.rect.x, y + 3))
                x = self.rect.x + number.get_width() + 10
                for san, classification in (
                    (row.white, row.white_classification),
                    (row.black, row.black_classification),
                ):
                    if san:
                        color = (
                            theme.classification_color(classification)
                            if classification
                            else theme.text
                        )
                        surface.blit(font.render(san, True, color), (x, y + 3))
                    x += 92
                y += self.ROW_HEIGHT
        finally:
            surface.set_clip(previous_clip)

        if self.max_scroll:
            track_h = self.rect.h
            thumb_h = max(24, int(track_h * self.visible_rows / len(self.rows)))
            span = track_h - thumb_h
            offset = int(span * self.scroll / self.max_scroll)
            pygame.draw.rect(
                surface,
                theme.widget_border,
                pygame.Rect(
                    self.rect.right - self.SCROLLBAR_W,
                    self.rect.y + offset,
                    self.SCROLLBAR_W,
                    thumb_h,
                ),
                border_radius=3,
            )


def dispatch(widgets: Sequence[Widget], event, now: float = 0.0) -> list[Action]:
    """Feed one event to every widget and collect the actions they emit."""
    actions = []
    for widget in widgets:
        action = widget.handle(event, now)
        if action is not None:
            actions.append(action)
    return actions
