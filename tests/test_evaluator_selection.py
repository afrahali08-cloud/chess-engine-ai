from pathlib import Path

from board.evaluators import resolve_evaluator


def test_missing_neural_model_falls_back_to_ridge(tmp_path):
    selection = resolve_evaluator(
        "neural",
        neural_model_path=tmp_path / "missing.pt",
    )

    assert selection.requested == "neural"
    assert selection.selected == "ridge"
    assert selection.fallback_reasons


def test_missing_learned_models_fall_back_to_handcrafted(tmp_path):
    selection = resolve_evaluator(
        "neural",
        neural_model_path=tmp_path / "missing.pt",
        ridge_model_path=tmp_path / "missing.json",
    )

    assert selection.selected == "handcrafted"
    assert len(selection.fallback_reasons) == 2


def test_handcrafted_selection_never_needs_model_files(tmp_path):
    selection = resolve_evaluator(
        "handcrafted",
        neural_model_path=Path(tmp_path / "missing.pt"),
        ridge_model_path=Path(tmp_path / "missing.json"),
    )

    assert selection.selected == "handcrafted"
    assert selection.fallback_reasons == ()
