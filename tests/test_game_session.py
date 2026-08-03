"""Tests for the headless session shared by the terminal and GUI frontends."""

import chess
import pytest

import game_session
from board.evaluation import evaluate_handcrafted_board
from board.evaluators import EvaluatorSelection
from coach import MoveAnalysis
from engine import SearchResult
from game_session import GameSession, SessionConfig


HANDCRAFTED = EvaluatorSelection(
    requested="handcrafted",
    selected="handcrafted",
    evaluate=evaluate_handcrafted_board,
)


def make_session(**config_changes) -> GameSession:
    """Build a session without loading torch or touching the model files."""
    return GameSession(SessionConfig(**config_changes), selection=HANDCRAFTED)


def play(session: GameSession, *ucis: str) -> None:
    for uci in ucis:
        session.apply_move(chess.Move.from_uci(uci), by_engine=False)


def fake_search(best_uci: str, score: float = 42.0) -> SearchResult:
    move = chess.Move.from_uci(best_uci)
    return SearchResult(
        best_move=move,
        best_score=score,
        move_scores={move: score},
        completed_depth=3,
        elapsed_seconds=0.01,
    )


# ---------------------------------------------------------------- apply_move


def test_apply_move_records_san_color_and_source():
    session = make_session()
    record = session.apply_move(chess.Move.from_uci("e2e4"), by_engine=False)

    assert record.ply == 0
    assert record.san == "e4"
    assert record.color == chess.WHITE
    assert record.by_engine is False
    assert record.analysis is None
    assert session.board.move_stack == [chess.Move.from_uci("e2e4")]


def test_apply_move_computes_san_before_pushing():
    """SAN needs the pre-move position; a capture proves it was not computed late."""
    session = make_session()
    play(session, "e2e4", "d7d5")
    record = session.apply_move(chess.Move.from_uci("e4d5"), by_engine=True)

    assert record.san == "exd5"
    assert record.by_engine is True


def test_apply_move_rejects_illegal_move():
    session = make_session()
    with pytest.raises(ValueError, match="illegal move"):
        session.apply_move(chess.Move.from_uci("e2e5"), by_engine=False)


# --------------------------------------------------------------- move_pairs


def test_move_pairs_leaves_black_none_on_odd_ply_count():
    session = make_session()
    play(session, "e2e4", "e7e5", "g1f3")
    pairs = session.move_pairs()

    assert [(p.number, p.white, p.black) for p in pairs] == [
        (1, "e4", "e5"),
        (2, "Nf3", None),
    ]


def test_move_pairs_and_san_history_empty_at_start():
    session = make_session()
    assert session.move_pairs() == ()
    assert session.san_history() == ()


def test_move_pairs_carries_coach_analysis():
    session = make_session()
    play(session, "e2e4")
    analysis = MoveAnalysis(
        played_move=chess.Move.from_uci("e2e4"),
        best_move=chess.Move.from_uci("d2d4"),
        played_san="e4",
        best_san="d4",
        played_score=10.0,
        best_score=60.0,
        centipawn_loss=50,
        classification="Good",
        explanation="stub",
        search_depth=2,
        used_static_fallback=False,
    )
    assert session.attach_analysis(0, analysis) is True

    pair = session.move_pairs()[0]
    assert pair.white_analysis is analysis
    assert pair.black_analysis is None
    assert session.history[0].analysis is analysis


def test_attach_analysis_returns_false_for_unknown_ply():
    session = make_session()
    assert session.attach_analysis(7, None) is False


# -------------------------------------------------------------------- turns


def test_turn_helpers_track_human_color():
    session = make_session(human_color=chess.WHITE)
    assert session.human_to_move() is True
    assert session.engine_to_move() is False

    play(session, "e2e4")
    assert session.human_to_move() is False
    assert session.engine_to_move() is True


def test_turn_helpers_when_human_plays_black():
    session = make_session(human_color=chess.BLACK)
    assert session.human_to_move() is False
    assert session.engine_to_move() is True


def test_turn_helpers_are_false_once_the_game_is_over():
    session = make_session()
    play(session, "f2f3", "e7e5", "g2g4", "d8h4")

    assert session.is_game_over() is True
    assert session.human_to_move() is False
    assert session.engine_to_move() is False


# --------------------------------------------------------------------- undo


def test_undo_to_human_turn_pops_the_pair():
    session = make_session()
    play(session, "e2e4", "e7e5")

    assert session.undo_to_human_turn() == 2
    assert session.board.move_stack == []
    assert session.history == ()


def test_undo_to_human_turn_pops_one_when_engine_has_not_replied():
    session = make_session()
    play(session, "e2e4")

    assert session.undo_to_human_turn() == 1
    assert session.board.fen() == chess.Board().fen()


def test_undo_to_human_turn_is_a_noop_on_an_empty_board():
    session = make_session()
    assert session.undo_to_human_turn() == 0


def test_undo_clamps_to_available_plies():
    session = make_session()
    play(session, "e2e4")
    assert session.undo(5) == 1
    assert session.history == ()


# -------------------------------------------------------------- result_text


def test_result_text_is_none_while_in_progress():
    assert make_session().result_text() is None


def test_result_text_reports_checkmate_winner():
    session = make_session()
    play(session, "f2f3", "e7e5", "g2g4", "d8h4")
    assert session.result_text() == "Checkmate - Black wins"


def test_result_text_reports_stalemate():
    session = make_session()
    session.board = chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    assert session.result_text() == "Draw - STALEMATE"


# --------------------------------------------------------------------- jobs


def test_engine_job_does_not_mutate_the_session_board():
    """The repo-wide invariant: a search must leave the caller's board alone."""
    session = make_session(depth=2, time_limit=0.05)
    play(session, "e2e4", "e7e5")
    fen_before = session.board.fen()
    stack_before = list(session.board.move_stack)

    result = session.engine_job()()

    assert session.board.fen() == fen_before
    assert session.board.move_stack == stack_before
    assert result.best_move in session.board.legal_moves


def test_engine_job_snapshots_the_position_at_call_time(monkeypatch):
    seen = []
    monkeypatch.setattr(
        game_session,
        "analyze_position",
        lambda board, **kwargs: seen.append(board.fen()) or fake_search("e2e4"),
    )
    session = make_session()
    job = session.engine_job()
    fen_at_submit = session.board.fen()

    play(session, "d2d4")  # board moves on after the job was created
    job()

    assert seen == [fen_at_submit]


def test_coach_job_receives_the_pre_push_board(monkeypatch):
    seen = []
    monkeypatch.setattr(
        game_session,
        "analyze_move",
        lambda board, move, **kwargs: seen.append((board.fen(), move)) or None,
    )
    session = make_session()
    before = session.board.copy()
    move = chess.Move.from_uci("e2e4")
    job = session.coach_job(before, move)

    session.apply_move(move, by_engine=False)  # applied before the job runs
    job()

    assert seen == [(chess.Board().fen(), move)]


def test_play_engine_move_records_the_score(monkeypatch):
    monkeypatch.setattr(
        game_session,
        "analyze_position",
        lambda board, **kwargs: fake_search("e2e4", score=125.0),
    )
    session = make_session()
    record = session.play_engine_move()

    assert record.move == chess.Move.from_uci("e2e4")
    assert record.score == 125.0
    assert record.by_engine is True


def test_play_engine_move_returns_none_when_there_is_no_move(monkeypatch):
    monkeypatch.setattr(
        game_session,
        "analyze_position",
        lambda board, **kwargs: SearchResult(None, 0.0, {}, 0, 0.0),
    )
    assert make_session().play_engine_move() is None


def test_play_human_move_skips_the_coach_when_disabled(monkeypatch):
    monkeypatch.setattr(
        game_session,
        "analyze_move",
        lambda *a, **k: pytest.fail("coach must not run when disabled"),
    )
    session = make_session(coach=False)
    record, analysis = session.play_human_move(chess.Move.from_uci("e2e4"))

    assert analysis is None
    assert record.san == "e4"


def test_play_human_move_attaches_the_coach_analysis(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(game_session, "analyze_move", lambda *a, **k: sentinel)
    session = make_session(coach=True)
    record, analysis = session.play_human_move(chess.Move.from_uci("e2e4"))

    assert analysis is sentinel
    assert record.analysis is sentinel
    assert session.history[0].analysis is sentinel


# -------------------------------------------------------------------- reset


def test_reset_clears_board_and_history():
    session = make_session()
    play(session, "e2e4", "e7e5")
    session.reset()

    assert session.board.fen() == chess.Board().fen()
    assert session.history == ()
    assert session.move_pairs() == ()


def test_reset_can_swap_the_config():
    session = make_session(human_color=chess.WHITE)
    session.reset(config=session.config.replace(human_color=chess.BLACK))

    assert session.config.human_color == chess.BLACK
    assert session.engine_to_move() is True
