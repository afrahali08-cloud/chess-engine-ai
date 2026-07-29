import chess
import torch

from board.neural_evaluation import evaluate_neural_board
from board.neural_features import FEATURE_ENCODING, TOTAL_FEATURES, board_to_features
from board.neural_model import EvalNet


def write_constant_model(path, score: float) -> None:
    model = EvalNet()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.fill_(score)

    torch.save(
        {
            "format_version": 2,
            "model_type": "mlp",
            "model_state": model.state_dict(),
            "input_size": TOTAL_FEATURES,
            "feature_encoding": FEATURE_ENCODING,
            "clip_cp": 1500,
        },
        path,
    )


def test_neural_feature_vector_has_no_unused_tail():
    features = board_to_features(chess.Board())

    assert features.shape == (400,)
    assert TOTAL_FEATURES == 400
    assert features[-4:].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_neural_evaluation_preserves_trained_centipawn_scale(tmp_path):
    model_path = tmp_path / "constant.pt"
    write_constant_model(model_path, 100.0)

    assert evaluate_neural_board(chess.Board(), model_path) == 100


def test_neural_terminal_score_does_not_require_model(tmp_path):
    checkmate = chess.Board("7k/6Q1/6K1/8/8/8/8/8 b - - 0 1")

    assert evaluate_neural_board(checkmate, tmp_path / "missing.pt") == 99999
