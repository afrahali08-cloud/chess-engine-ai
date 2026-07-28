import csv

import chess

from scripts.train_evaluator import train as train_ridge
from scripts.train_neural_eval import train as train_neural
from scripts.training_common import SPLIT_NAMES, split_for_game


def write_balanced_training_csv(path) -> None:
    game_ids = {name: [] for name in SPLIT_NAMES}
    candidate = 0
    while any(len(values) < 3 for values in game_ids.values()):
        game_id = f"game-{candidate}"
        split_name = split_for_game(game_id)
        if len(game_ids[split_name]) < 3:
            game_ids[split_name].append(game_id)
        candidate += 1

    positions = []
    board = chess.Board()
    for move in ("e2e4", "e7e5", "g1f3"):
        board.push_uci(move)
        positions.append(board.fen())

    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("game_id", "fen", "cp", "ply"),
        )
        writer.writeheader()
        score = -120
        for split_name in SPLIT_NAMES:
            for game_id in game_ids[split_name]:
                for ply, fen in enumerate(positions, start=1):
                    writer.writerow(
                        {
                            "game_id": game_id,
                            "fen": fen,
                            "cp": score,
                            "ply": ply,
                        }
                    )
                    score += 10


def test_ridge_and_neural_training_share_splits_and_metadata(tmp_path):
    input_path = tmp_path / "positions.csv"
    ridge_path = tmp_path / "ridge.json"
    neural_path = tmp_path / "neural.pt"
    write_balanced_training_csv(input_path)

    ridge = train_ridge(
        input_path,
        ridge_path,
        alpha=10.0,
        clip_cp=1500,
    )
    neural = train_neural(
        input_path,
        neural_path,
        epochs=1,
        batch_size=8,
        learning_rate=1e-3,
        clip_cp=1500,
    )

    assert ridge["dataset"] == neural["dataset"]
    assert ridge["split_counts"] == neural["split_counts"]
    assert set(ridge["metrics"]) == set(SPLIT_NAMES)
    assert set(neural["metrics"]) == set(SPLIT_NAMES)
    assert neural["training"]["epochs_requested"] == 1
    assert neural["training"]["best_epoch"] == 1
