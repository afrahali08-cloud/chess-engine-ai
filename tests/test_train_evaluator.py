import chess
import numpy as np

from scripts.train_evaluator import FEATURE_COUNT, fen_features, split_for_game


def dense_features(board: chess.Board) -> np.ndarray:
    features = np.zeros(FEATURE_COUNT, dtype=np.float32)
    for index, value in fen_features(board.fen()):
        features[index] += value
    return features


def test_color_mirrored_position_has_opposite_features():
    board = chess.Board()

    original = dense_features(board)
    mirrored = dense_features(board.mirror())

    assert np.array_equal(original, -mirrored)


def test_game_split_is_deterministic_and_has_known_values():
    game_ids = [f"https://lichess.org/game-{index}" for index in range(100)]
    first_assignments = [split_for_game(game_id) for game_id in game_ids]
    second_assignments = [split_for_game(game_id) for game_id in game_ids]

    assert first_assignments == second_assignments
    assert set(first_assignments) == {"train", "validation", "test"}


def test_every_position_from_a_game_uses_the_same_split():
    game_id = "https://lichess.org/one-complete-game"

    assignments = [split_for_game(game_id) for _ in range(20)]

    assert len(set(assignments)) == 1
