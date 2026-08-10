"""App state-machine tests.

These drive the real handlers with the real background runner, but never open a
window. The focus is the async hazard: a search result arriving after the board
has already moved on.
"""

import time

import chess
import pytest

pygame = pytest.importorskip("pygame")

from board.evaluation import evaluate_handcrafted_board
from board.evaluators import EvaluatorSelection
from coach import MoveAnalysis
from engine import SearchResult
from game_session import GameSession, SessionConfig
from gui import interaction
from gui.app import DEPTH_RANGE, TIME_STEPS, ChessApp, _step_time
from gui.engine_worker import JobResult
from gui.widgets import Action

HANDCRAFTED = EvaluatorSelection(
    requested="handcrafted",
    selected="handcrafted",
    evaluate=evaluate_handcrafted_board,
)


def make_app(**config_changes) -> ChessApp:
    config = SessionConfig(
        evaluator="handcrafted",
        depth=config_changes.pop("depth", 2),
        time_limit=config_changes.pop("time_limit", 0.1),
        **config_changes,
    )
    session = GameSession(config, selection=HANDCRAFTED)
    return ChessApp(session)


@pytest.fixture
def app():
    made = make_app()
    yield made
    made.runner.shutdown()


def pump(app, *, timeout=8.0, until=None):
    """Poll the runner like the event loop does, until quiet or ``until``."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for result in app.runner.poll():
            app.handle_result(result)
        if until is not None and until():
            return True
        if until is None and not app.runner.busy():
            # one last drain so the final result is handled
            for result in app.runner.poll():
                app.handle_result(result)
            return True
        time.sleep(0.005)
    return False


# --------------------------------------------------------- normal play


def test_human_move_is_applied_immediately(app):
    app.play_human_move(chess.Move.from_uci("e2e4"))

    assert app.session.board.piece_at(chess.E4) is not None
    assert app.session.history[0].san == "e4"


def test_engine_replies_after_a_human_move(app):
    app.play_human_move(chess.Move.from_uci("e2e4"))
    assert pump(app)

    assert len(app.session.history) == 2
    assert app.session.history[1].by_engine is True
    assert app.session.human_to_move() is True
    assert app.eval_cp is not None


def test_engine_opens_when_the_human_plays_black():
    app = make_app(human_color=chess.BLACK)
    try:
        app.request_engine_move()
        assert pump(app)

        assert len(app.session.history) == 1
        assert app.session.history[0].color == chess.WHITE
        assert app.session.history[0].by_engine is True
        assert app.session.human_to_move() is True
        assert app.flipped is True
    finally:
        app.runner.shutdown()


def test_board_input_is_refused_while_the_engine_is_to_move(app):
    app.play_human_move(chess.Move.from_uci("e2e4"))
    # Engine has not replied yet, so a board click must not select anything.
    state, move = interaction.click_square(
        app.selection,
        app.session.board,
        chess.E7,
        human_color=app.session.config.human_color,
        allow_input=app.session.human_to_move(),
    )
    assert move is None
    assert state.selected is None
    assert pump(app)


# ------------------------------------------------ the stale-result hazard


def test_undo_during_a_search_discards_the_result():
    """The classic async bug: a reply for a position that no longer exists."""
    app = make_app(depth=6, time_limit=0.6)
    try:
        app.play_human_move(chess.Move.from_uci("e2e4"))
        assert app.runner.outstanding("engine") == 1

        app.undo()  # takes the human move back mid-search
        fen_after_undo = app.session.board.fen()
        assert pump(app)

        assert app.session.board.fen() == fen_after_undo
        assert app.session.history == ()
        assert app.session.board.move_stack == []
    finally:
        app.runner.shutdown()


def test_new_game_during_a_search_discards_the_result():
    app = make_app(depth=6, time_limit=0.6)
    try:
        app.play_human_move(chess.Move.from_uci("e2e4"))
        app.new_game()
        assert pump(app)

        assert app.session.board.fen() == chess.Board().fen()
        assert app.session.history == ()
    finally:
        app.runner.shutdown()


def test_repeated_undo_spam_during_searches_never_corrupts_the_board():
    """Acceptance check for hammering Undo mid-search, as a demo audience would."""
    app = make_app(depth=6, time_limit=0.3)
    try:
        for _ in range(6):
            if app.session.human_to_move():
                legal = next(iter(app.session.board.legal_moves))
                app.play_human_move(legal)
            app.undo()
            app.session.board.fen()  # must always be parseable / consistent
        assert pump(app, timeout=10.0)

        # Whatever survived must be a legal, self-consistent position.
        assert app.session.board.is_valid()
        assert len(app.session.history) == len(app.session.board.move_stack)
    finally:
        app.runner.shutdown()


def test_a_fresh_but_illegal_engine_move_is_refused(app):
    """Belt and braces behind the generation counter."""
    app.session.apply_move(chess.Move.from_uci("e2e4"), by_engine=False)
    before = app.session.board.fen()
    bogus = SearchResult(
        best_move=chess.Move.from_uci("a1a8"),  # not legal here
        best_score=0.0,
        move_scores={},
        completed_depth=1,
        elapsed_seconds=0.0,
    )

    app.handle_result(
        JobResult(job_id=1, generation=app.runner.generation, kind="engine", value=bogus)
    )

    assert app.session.board.fen() == before


def test_an_engine_result_with_no_move_is_survivable(app):
    empty = SearchResult(None, 0.0, {}, 0, 0.0)
    app.handle_result(
        JobResult(job_id=1, generation=app.runner.generation, kind="engine", value=empty)
    )
    assert app.session.board.fen() == chess.Board().fen()


def test_a_failed_job_surfaces_in_the_status_line(app):
    app.handle_result(
        JobResult(
            job_id=1,
            generation=app.runner.generation,
            kind="engine",
            error=RuntimeError("boom"),
        )
    )
    assert "boom" in app.status_text
    assert app.status_color == "error"


# ------------------------------------------------------- eval bar anim


def test_eval_bar_starts_level(app):
    assert app.eval_fill == pytest.approx(0.5)


def test_eval_bar_eases_toward_a_new_score_rather_than_jumping(app):
    app.eval_cp = 800.0
    target = app.eval_target_fill
    app.advance_animation(1 / 60)

    assert app.eval_fill > 0.5  # started moving
    assert app.eval_fill < target  # but did not snap there in one frame


def test_eval_bar_reaches_the_target_after_enough_frames(app):
    app.eval_cp = 800.0
    for _ in range(120):
        app.advance_animation(1 / 60)

    assert app.eval_fill == pytest.approx(app.eval_target_fill)


def test_eval_bar_animates_back_toward_level_on_a_new_game(app):
    app.eval_cp = -900.0
    for _ in range(120):
        app.advance_animation(1 / 60)
    assert app.eval_fill < 0.2

    app.new_game()
    assert app.eval_cp is None
    for _ in range(120):
        app.advance_animation(1 / 60)
    assert app.eval_fill == pytest.approx(0.5)


def test_eval_bar_fill_is_published_to_the_view(app):
    app.eval_cp = 400.0
    app.advance_animation(1 / 60)
    view = app.build_view(0.0)

    assert view.eval_fill == app.eval_fill
    assert view.eval_cp == 400.0  # the number itself does not lag


def test_mate_scores_drive_the_bar_fully(app):
    app.eval_cp = 99999
    for _ in range(180):
        app.advance_animation(1 / 60)
    assert app.eval_fill == pytest.approx(1.0)


# ---------------------------------------------------------------- coach


def test_coach_result_attaches_to_the_move_it_analyzed(app):
    app.session.set_config(app.session.config.replace(coach=True))
    app.session.apply_move(chess.Move.from_uci("e2e4"), by_engine=False)
    app.session.apply_move(chess.Move.from_uci("e7e5"), by_engine=True)
    analysis = MoveAnalysis(
        played_move=chess.Move.from_uci("e2e4"),
        best_move=chess.Move.from_uci("d2d4"),
        played_san="e4",
        best_san="d4",
        played_score=0.0,
        best_score=50.0,
        centipawn_loss=50,
        classification="Good",
        explanation="stub",
        search_depth=2,
        used_static_fallback=False,
    )
    app.coach_pending = True

    app.handle_result(
        JobResult(
            job_id=1,
            generation=app.runner.generation,
            kind="coach",
            value=analysis,
            payload=0,  # ply of the human's move
        )
    )

    assert app.coach_pending is False
    assert app.session.history[0].analysis is analysis
    assert app.session.history[1].analysis is None


def test_coach_job_is_submitted_only_when_enabled(app):
    app.play_human_move(chess.Move.from_uci("e2e4"))
    assert app.runner.outstanding("coach") == 0
    assert pump(app)

    app.session.set_config(app.session.config.replace(coach=True))
    app.play_human_move(next(iter(app.session.board.legal_moves)))
    assert app.coach_pending is True
    assert pump(app)


# ----------------------------------------------------------- evaluator


def test_evaluator_result_is_applied_and_updates_config(app):
    app.pending_evaluator = "handcrafted"
    app.handle_result(
        JobResult(
            job_id=1,
            generation=app.runner.generation,
            kind="evaluator",
            value=HANDCRAFTED,
        )
    )

    assert app.pending_evaluator is None
    assert app.session.selection is HANDCRAFTED
    assert app.session.config.evaluator == "handcrafted"


def test_a_superseded_evaluator_result_is_ignored(app):
    app.pending_evaluator = "neural"  # user has since asked for something else
    app.handle_result(
        JobResult(
            job_id=1,
            generation=app.runner.generation,
            kind="evaluator",
            value=HANDCRAFTED,
        )
    )

    assert app.pending_evaluator == "neural"


def test_evaluator_fallback_reasons_reach_the_status_line(app):
    fallen_back = EvaluatorSelection(
        requested="neural",
        selected="handcrafted",
        evaluate=evaluate_handcrafted_board,
        fallback_reasons=("neural: FileNotFoundError: missing checkpoint",),
    )
    app.pending_evaluator = "neural"
    app.handle_result(
        JobResult(
            job_id=1,
            generation=app.runner.generation,
            kind="evaluator",
            value=fallen_back,
        )
    )

    assert "neural" in app.status_text
    assert "missing checkpoint" in app.status_text
    assert app.status_color == "warn"


def test_a_failed_evaluator_load_does_not_wedge_the_pending_flag(app):
    app.pending_evaluator = "neural"
    app.handle_result(
        JobResult(
            job_id=1,
            generation=app.runner.generation,
            kind="evaluator",
            error=ImportError("no torch"),
        )
    )

    assert app.pending_evaluator is None
    assert app.status_color == "error"


def test_cycle_evaluator_advances_through_the_choices(app):
    app.cycle_evaluator()
    assert app.pending_evaluator == "ridge"
    app.cycle_evaluator()
    assert app.pending_evaluator == "neural"


# ------------------------------------------------------------- session


def test_swap_sides_starts_a_new_game_as_black(app):
    app.session.apply_move(chess.Move.from_uci("e2e4"), by_engine=False)
    app.swap_sides()

    assert app.session.config.human_color == chess.BLACK
    assert app.session.history == ()
    assert app.flipped is True
    assert pump(app)


def test_toggle_coach_flips_the_config(app):
    assert app.session.config.coach is False
    app.toggle_coach()
    assert app.session.config.coach is True


def test_manual_flip_is_independent_of_side(app):
    assert app.flipped is False
    app.manual_flip = True
    assert app.flipped is True


def test_status_reports_the_result_when_the_game_ends(app):
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        app.session.apply_move(chess.Move.from_uci(uci), by_engine=False)
    app.refresh_status()

    assert "Checkmate" in app.status_text
    assert app.status_color == "accent"


def test_no_engine_job_is_queued_once_the_game_is_over(app):
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        app.session.apply_move(chess.Move.from_uci(uci), by_engine=False)
    app.request_engine_move()

    assert app.runner.outstanding("engine") == 0


# ------------------------------------------------- coach vs engine budget


def test_coach_and_engine_budgets_are_independent(app):
    """The Depth/Time steppers drive the search; the coach has its own pair."""
    before = (app.session.config.coach_depth, app.session.config.coach_time_limit)

    app.apply_action(Action("depth", 1))
    app.apply_action(Action("time_limit", 1))

    assert (
        app.session.config.coach_depth,
        app.session.config.coach_time_limit,
    ) == before


def test_coach_time_stepper_changes_only_the_coach(app):
    engine_before = (app.session.config.depth, app.session.config.time_limit)

    app.apply_action(Action("coach_time_limit", 1))
    app.apply_action(Action("coach_depth", 1))

    assert (app.session.config.depth, app.session.config.time_limit) == engine_before
    assert app.session.config.coach_time_limit > 1.0
    assert app.session.config.coach_depth == 5


def test_coach_depth_is_clamped_to_the_same_range_as_the_engine(app):
    for _ in range(40):
        app.apply_action(Action("coach_depth", 1))
    assert app.session.config.coach_depth == DEPTH_RANGE[1]
    for _ in range(40):
        app.apply_action(Action("coach_depth", -1))
    assert app.session.config.coach_depth == DEPTH_RANGE[0]


def test_coach_time_walks_the_shared_step_ladder(app):
    for _ in range(40):
        app.apply_action(Action("coach_time_limit", 1))
    assert app.session.config.coach_time_limit == TIME_STEPS[-1]
    for _ in range(40):
        app.apply_action(Action("coach_time_limit", -1))
    assert app.session.config.coach_time_limit == TIME_STEPS[0]


def test_step_time_snaps_a_value_that_is_off_the_ladder():
    assert _step_time(0.9, 1) == 2.0   # nearest notch is 1.0, then step up
    assert _step_time(0.9, -1) == 0.5


def test_coach_widgets_reflect_the_session(app):
    app.apply_action(Action("coach_depth", 1))
    app.apply_action(Action("coach_time_limit", 1))
    app.sync_widgets(())

    assert app._by_name["coach_depth"].value == app.session.config.coach_depth
    assert (
        app._by_name["coach_time_limit"].value
        == app.session.config.coach_time_limit
    )
