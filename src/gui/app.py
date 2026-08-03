"""Window, event loop, and the app state machine.

All session mutation happens on this thread. The worker only ever runs the
zero-argument callables handed out by :class:`GameSession`, which close over a
board *copy*, so a search can never observe a half-applied move.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")  # no sound; skip PulseAudio init

import chess
import pygame

from . import interaction, layout as layout_mod
from .engine_worker import BackgroundRunner, JobResult
from .layout import Rect
from .pieces import Assets
from .render import draw_frame
from .viewmodel import CoachView, MoveRow, ViewModel
from .widgets import Button, Cycler, MoveListView, Stepper, dispatch

try:
    from ..board.evaluators import EVALUATOR_CHOICES
    from ..game_session import GameSession, SessionConfig
except ImportError:
    from board.evaluators import EVALUATOR_CHOICES
    from game_session import GameSession, SessionConfig


TITLE = "Chess Engine AI - CMPT 310"
FPS_IDLE = 60
FPS_BUSY = 30  # the pure-Python search holds the GIL; asking for 60 just adds contention

DEPTH_RANGE = (1, 8)
TIME_STEPS = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0)


class ChessApp:
    def __init__(
        self,
        session: GameSession,
        *,
        width: int = layout_mod.WINDOW_WIDTH,
        height: int = layout_mod.WINDOW_HEIGHT,
        fullscreen: bool = False,
        runner: BackgroundRunner | None = None,
        pending_evaluator: str | None = None,
    ) -> None:
        self.session = session
        self.width = width
        self.height = height
        self.fullscreen = fullscreen
        self.runner = runner or BackgroundRunner()
        self.selection = interaction.EMPTY
        self.manual_flip = False
        self.running = False
        self.eval_cp: float | None = None
        self.eval_is_stale = False
        self.eval_fill = layout_mod.eval_fraction(0.0)  # animated bar fill
        self._last_now: float | None = None
        self.status_text = "Your move."
        self.status_color = "dim"
        self.screen = None
        self.assets: Assets | None = None
        self.layout = layout_mod.build_layout(width, height)
        self.pending_evaluator = pending_evaluator
        self.coach_pending = False
        self.widgets: list = []
        self.move_list: MoveListView | None = None
        self._build_widgets()

    # ------------------------------------------------------------------
    # widgets
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        panel = self.layout.controls
        pad = 14
        left = panel.x + pad
        width = panel.w - 2 * pad
        row_h = 30
        gap = 6
        y = panel.y + 34
        half = (width - gap) // 2
        third = (width - 2 * gap) // 3

        evaluator = Cycler(
            Rect(left, y, width, row_h),
            "Evaluator",
            list(EVALUATOR_CHOICES),
            "evaluator",
            index=max(0, EVALUATOR_CHOICES.index(self.session.config.evaluator))
            if self.session.config.evaluator in EVALUATOR_CHOICES
            else 0,
        )
        y += row_h + gap
        depth = Stepper(
            Rect(left, y, half, row_h),
            "Depth",
            "depth",
            value=self.session.config.depth,
        )
        time_limit = Stepper(
            Rect(left + half + gap, y, half, row_h),
            "Time",
            "time_limit",
            value=self.session.config.time_limit,
            formatter=lambda v: f"{v:g}s",
        )
        y += row_h + gap
        play_as = Cycler(
            Rect(left, y, half, row_h),
            "Play as",
            ["White", "Black"],
            "play_as",
            index=0 if self.session.config.human_color == chess.WHITE else 1,
        )
        coach = Cycler(
            Rect(left + half + gap, y, half, row_h),
            "Coach",
            ["Off", "On"],
            "coach",
            index=1 if self.session.config.coach else 0,
        )
        y += row_h + gap
        new_game = Button(Rect(left, y, third, row_h), "New Game", "new_game")
        undo = Button(Rect(left + third + gap, y, third, row_h), "Undo", "undo")
        flip = Button(
            Rect(left + 2 * (third + gap), y, third, row_h), "Flip", "flip"
        )

        moves_panel = self.layout.moves
        self.move_list = MoveListView(
            Rect(
                moves_panel.x + pad,
                moves_panel.y + 34,
                moves_panel.w - 2 * pad,
                moves_panel.h - 34 - 10,
            )
        )

        self.widgets = [
            evaluator,
            depth,
            time_limit,
            play_as,
            coach,
            new_game,
            undo,
            flip,
            self.move_list,
        ]
        self._by_name = {widget.name: widget for widget in self.widgets}

    def sync_widgets(self, rows) -> None:
        """Push current session values into the widgets before drawing."""
        config = self.session.config
        self._by_name["evaluator"].set_value(
            self.pending_evaluator or config.evaluator
        )
        self._by_name["depth"].value = config.depth
        self._by_name["time_limit"].value = config.time_limit
        self._by_name["play_as"].set_value(
            "White" if config.human_color == chess.WHITE else "Black"
        )
        self._by_name["coach"].set_value("On" if config.coach else "Off")
        self.move_list.set_rows(rows)

    def apply_action(self, action) -> None:
        config = self.session.config
        if action.name == "evaluator":
            self.start_evaluator_load(action.value)
        elif action.name == "depth":
            low, high = DEPTH_RANGE
            self.session.set_config(
                config.replace(
                    depth=max(low, min(high, config.depth + action.value))
                )
            )
        elif action.name == "time_limit":
            steps = TIME_STEPS
            try:
                index = steps.index(config.time_limit)
            except ValueError:
                index = min(
                    range(len(steps)),
                    key=lambda i: abs(steps[i] - config.time_limit),
                )
            index = max(0, min(len(steps) - 1, index + action.value))
            self.session.set_config(config.replace(time_limit=steps[index]))
        elif action.name == "play_as":
            wanted = chess.WHITE if action.value == "White" else chess.BLACK
            if wanted != config.human_color:
                self.new_game(config=config.replace(human_color=wanted))
        elif action.name == "coach":
            self.session.set_config(config.replace(coach=action.value == "On"))
        elif action.name == "new_game":
            self.new_game()
        elif action.name == "undo":
            self.undo()
        elif action.name == "flip":
            self.manual_flip = not self.manual_flip

    # ------------------------------------------------------------------
    @property
    def flipped(self) -> bool:
        human_is_black = self.session.config.human_color == chess.BLACK
        return human_is_black != self.manual_flip

    def _init_display(self) -> None:
        pygame.display.init()
        pygame.font.init()
        flags = pygame.SCALED | (pygame.FULLSCREEN if self.fullscreen else 0)
        try:
            self.screen = pygame.display.set_mode(
                (self.width, self.height), flags, vsync=1
            )
        except pygame.error:
            # vsync is unavailable on some drivers, notably under WSLg.
            self.screen = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption(TITLE)
        self.assets = Assets.load(self.layout.square)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        pygame.display.toggle_fullscreen()

    # ------------------------------------------------------------------
    # turn sequencing
    # ------------------------------------------------------------------
    def start_evaluator_load(self, name: str) -> None:
        self.pending_evaluator = name
        self.status_text = f"Loading {name} evaluator..."
        self.status_color = "warn"
        self.runner.submit("evaluator", self.session.evaluator_job(name))

    def request_engine_move(self) -> None:
        if self.session.engine_to_move():
            self.runner.submit("engine", self.session.engine_job())

    def play_human_move(self, move: chess.Move) -> None:
        """Apply immediately for feedback, then queue the coach and the reply."""
        before = self.session.board.copy()
        record = self.session.apply_move(move, by_engine=False)
        self.selection = interaction.EMPTY

        if self.session.config.coach:
            self.coach_pending = True
            self.runner.submit(
                "coach",
                self.session.coach_job(before, move),
                payload=record.ply,
            )
        self.request_engine_move()
        self.refresh_status()

    def new_game(self, *, config: SessionConfig | None = None) -> None:
        self.runner.bump_generation()
        self.session.reset(config=config)
        self.selection = interaction.EMPTY
        self.eval_cp = None
        self.eval_is_stale = False
        self.coach_pending = False
        self.request_engine_move()
        self.refresh_status()

    def undo(self) -> None:
        self.runner.bump_generation()
        self.coach_pending = False
        if self.session.undo_to_human_turn():
            self.eval_is_stale = True
        self.selection = interaction.EMPTY
        self.refresh_status()

    @property
    def eval_target_fill(self) -> float:
        return layout_mod.eval_fraction(self.eval_cp if self.eval_cp is not None else 0.0)

    def advance_animation(self, dt: float) -> None:
        """Ease the eval bar toward the current score."""
        self.eval_fill = layout_mod.approach(self.eval_fill, self.eval_target_fill, dt)

    def refresh_status(self) -> None:
        if self.session.is_game_over():
            self.status_text = self.session.result_text() or "Game over."
            self.status_color = "accent"
        elif self.pending_evaluator:
            self.status_text = f"Loading {self.pending_evaluator} evaluator..."
            self.status_color = "warn"
        elif self.runner.outstanding("engine"):
            self.status_text = "Engine is thinking"
            self.status_color = "text"
        elif self.session.human_to_move():
            self.status_text = "Your move."
            self.status_color = "dim"
        else:
            self.status_text = ""
            self.status_color = "dim"

    # ------------------------------------------------------------------
    # worker results
    # ------------------------------------------------------------------
    def handle_result(self, result: JobResult) -> None:
        if result.kind == "evaluator":
            self._handle_evaluator_result(result)
            return
        if result.stale:
            return  # the board moved on; this answer is for a dead position
        if not result.ok:
            self.status_text = f"{result.kind} failed: {result.error}"
            self.status_color = "error"
            if result.kind == "coach":
                self.coach_pending = False
            return
        if result.kind == "engine":
            self._handle_engine_result(result)
        elif result.kind == "coach":
            self.coach_pending = False
            if result.value is not None:
                self.session.attach_analysis(result.payload, result.value)

    def _handle_engine_result(self, result: JobResult) -> None:
        search = result.value
        self.eval_cp = search.best_score
        self.eval_is_stale = False
        if search.best_move is None:
            self.refresh_status()
            return
        # Belt and braces: even a fresh-generation result must still be legal.
        if search.best_move not in self.session.board.legal_moves:
            self.refresh_status()
            return
        self.session.apply_move(
            search.best_move, by_engine=True, score=search.best_score
        )
        self.refresh_status()

    def _handle_evaluator_result(self, result: JobResult) -> None:
        requested = self.pending_evaluator
        if not result.ok:
            self.pending_evaluator = None
            self.status_text = f"Could not load evaluator: {result.error}"
            self.status_color = "error"
            return
        selection = result.value
        if requested is not None and selection.requested != requested:
            return  # the user moved on to a different evaluator
        self.pending_evaluator = None
        self.session.set_selection(selection)
        self.session.set_config(
            self.session.config.replace(evaluator=selection.selected)
        )
        if selection.fallback_reasons:
            self.status_text = (
                f"{selection.requested} unavailable, using {selection.selected}: "
                f"{selection.fallback_reasons[0]}"
            )
            self.status_color = "warn"
        else:
            self.refresh_status()

    # ------------------------------------------------------------------
    # input
    # ------------------------------------------------------------------
    def handle_board_click(self, pos: tuple[int, int], now: float) -> None:
        if self.selection.promotion is not None:
            cells = layout_mod.promotion_cell_rects(
                self.selection.promotion.to_square,
                self.selection.promotion.options,
                self.layout,
                self.flipped,
            )
            for piece_type, cell in cells:
                if cell.contains(pos):
                    self.selection, move = interaction.click_promotion(
                        self.selection, piece_type
                    )
                    if move is not None:
                        self.play_human_move(move)
                    return
            self.selection, _ = interaction.click_promotion(self.selection, None)
            return

        square = layout_mod.square_at(pos, self.layout, self.flipped)
        if square is None:
            return
        self.selection, move = interaction.click_square(
            self.selection,
            self.session.board,
            square,
            human_color=self.session.config.human_color,
            now=now,
            allow_input=self.session.human_to_move(),
        )
        if move is not None:
            self.play_human_move(move)

    def cycle_evaluator(self, step: int = 1) -> None:
        current = self.pending_evaluator or self.session.config.evaluator
        try:
            index = EVALUATOR_CHOICES.index(current)
        except ValueError:
            index = 0
        self.start_evaluator_load(
            EVALUATOR_CHOICES[(index + step) % len(EVALUATOR_CHOICES)]
        )

    def toggle_coach(self) -> None:
        config = self.session.config
        self.session.set_config(config.replace(coach=not config.coach))

    def swap_sides(self) -> None:
        config = self.session.config
        human = chess.BLACK if config.human_color == chess.WHITE else chess.WHITE
        self.new_game(config=config.replace(human_color=human))

    def handle_key(self, event) -> None:
        if event.key == pygame.K_ESCAPE:
            if self.selection.promotion is not None:
                self.selection, _ = interaction.click_promotion(self.selection, None)
            else:
                self.selection = interaction.clear(self.selection)
        elif event.key == pygame.K_f:
            self.manual_flip = not self.manual_flip
        elif event.key == pygame.K_n:
            self.new_game()
        elif event.key == pygame.K_u:
            self.undo()
        elif event.key == pygame.K_c:
            self.toggle_coach()
        elif event.key == pygame.K_e:
            self.cycle_evaluator()
        elif event.key == pygame.K_b:
            self.swap_sides()
        elif event.key == pygame.K_F11:
            self.toggle_fullscreen()
        elif event.key == pygame.K_q or (
            event.key == pygame.K_w and event.mod & pygame.KMOD_CTRL
        ):
            self.running = False

    # ------------------------------------------------------------------
    def build_view(self, now: float) -> ViewModel:
        session = self.session
        config = session.config

        coach_view = None
        for record in reversed(session.history):
            if record.analysis is not None:
                analysis = record.analysis
                coach_view = CoachView(
                    classification=analysis.classification,
                    centipawn_loss=analysis.centipawn_loss,
                    played_san=analysis.played_san,
                    best_san=analysis.best_san,
                    explanation=analysis.explanation,
                    search_depth=analysis.search_depth,
                    used_static_fallback=analysis.used_static_fallback,
                )
                break

        rows = tuple(
            MoveRow(
                number=pair.number,
                white=pair.white,
                black=pair.black,
                white_classification=(
                    pair.white_analysis.classification if pair.white_analysis else None
                ),
                black_classification=(
                    pair.black_analysis.classification if pair.black_analysis else None
                ),
            )
            for pair in session.move_pairs()
        )

        thinking_kind = ""
        if self.runner.outstanding("engine"):
            thinking_kind = "Engine thinking"
        elif self.runner.outstanding("coach"):
            thinking_kind = "Coach analyzing"
        elif self.pending_evaluator:
            thinking_kind = f"Loading {self.pending_evaluator}"

        return ViewModel(
            board=session.board,
            flipped=self.flipped,
            selection=self.selection,
            last_move=session.last_move(),
            now=now,
            evaluator_name=session.selection.selected,
            requested_evaluator=session.selection.requested,
            depth=config.depth,
            time_limit=config.time_limit,
            coach_enabled=config.coach,
            human_color=config.human_color,
            thinking=bool(thinking_kind),
            thinking_label=thinking_kind,
            status_text=self.status_text,
            status_color=self.status_color,
            eval_cp=self.eval_cp,
            eval_is_stale=self.eval_is_stale,
            eval_fill=self.eval_fill,
            coach=coach_view,
            coach_pending=self.coach_pending,
            moves=rows,
            result_text=session.result_text(),
            widgets=tuple(self.widgets),
        )

    # ------------------------------------------------------------------
    def run(self) -> int:
        self._init_display()
        clock = pygame.time.Clock()
        self.running = True

        if self.pending_evaluator:
            self.start_evaluator_load(self.pending_evaluator)
        self.request_engine_move()
        self.refresh_status()

        while self.running:
            now = pygame.time.get_ticks() / 1000.0
            dt = 0.0 if self._last_now is None else max(0.0, now - self._last_now)
            self._last_now = now
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    continue
                if event.type == pygame.KEYDOWN:
                    self.handle_key(event)
                    continue

                for action in dispatch(self.widgets, event, now):
                    self.apply_action(action)

                if (
                    event.type == pygame.MOUSEBUTTONDOWN
                    and event.button == 1
                    and self.layout.board.contains(event.pos)
                ):
                    self.handle_board_click(event.pos, now)

            for result in self.runner.poll():
                self.handle_result(result)

            self.selection = interaction.expire_error(self.selection, now)
            self.advance_animation(dt)
            view = self.build_view(now)
            self.sync_widgets(view.moves)
            draw_frame(self.screen, view, self.assets)
            pygame.display.flip()
            clock.tick(FPS_BUSY if self.runner.busy() else FPS_IDLE)

        self.runner.shutdown()
        pygame.quit()
        return 0


def run_app(
    config: SessionConfig,
    *,
    fullscreen: bool = False,
    width: int = layout_mod.WINDOW_WIDTH,
    height: int = layout_mod.WINDOW_HEIGHT,
) -> int:
    """Open the window first, then load the real evaluator in the background.

    Resolving ``neural`` imports torch and reads a checkpoint, which takes long
    enough that doing it before the first frame looks like a hang.
    """
    from board.evaluators import resolve_evaluator

    boot = resolve_evaluator("handcrafted")
    session = GameSession(config.replace(evaluator="handcrafted"), selection=boot)
    pending = config.evaluator if config.evaluator != "handcrafted" else None

    return ChessApp(
        session,
        width=width,
        height=height,
        fullscreen=fullscreen,
        pending_evaluator=pending,
    ).run()
