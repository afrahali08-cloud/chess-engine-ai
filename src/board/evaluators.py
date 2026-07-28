"""Evaluator selection with lazy imports and deterministic fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import chess

from .evaluation import evaluate_handcrafted_board
from .learned_evaluation import (
    DEFAULT_MODEL_PATH as DEFAULT_RIDGE_MODEL_PATH,
    evaluate_learned_board,
    load_model as load_ridge_model,
)


Evaluator = Callable[[chess.Board], int]
EVALUATOR_CHOICES = ("handcrafted", "ridge", "neural")
DEFAULT_EVALUATOR = "neural"


@dataclass(frozen=True)
class EvaluatorSelection:
    requested: str
    selected: str
    evaluate: Evaluator
    fallback_reasons: tuple[str, ...] = ()


def _load_ridge_evaluator(model_path: Path) -> Evaluator:
    load_ridge_model(model_path)
    return partial(evaluate_learned_board, model_path=model_path)


def _load_neural_evaluator(model_path: Path) -> Evaluator:
    from .neural_evaluation import evaluate_neural_board, load_neural_model

    load_neural_model(model_path)
    return partial(evaluate_neural_board, model_path=model_path)


def resolve_evaluator(
    requested: str,
    *,
    ridge_model_path: Path = DEFAULT_RIDGE_MODEL_PATH,
    neural_model_path: Path | None = None,
) -> EvaluatorSelection:
    """Resolve an evaluator and fall back from neural to Ridge to handcrafted."""
    normalized = requested.lower()
    if normalized not in EVALUATOR_CHOICES:
        choices = ", ".join(EVALUATOR_CHOICES)
        raise ValueError(f"unknown evaluator {requested!r}; choose from {choices}")

    if neural_model_path is None:
        neural_model_path = (
            Path(__file__).resolve().parents[2] / "models" / "neural_evaluator.pt"
        )

    candidates = {
        "handcrafted": lambda: evaluate_handcrafted_board,
        "ridge": lambda: _load_ridge_evaluator(Path(ridge_model_path)),
        "neural": lambda: _load_neural_evaluator(Path(neural_model_path)),
    }
    fallback_order = {
        "handcrafted": ("handcrafted",),
        "ridge": ("ridge", "handcrafted"),
        "neural": ("neural", "ridge", "handcrafted"),
    }
    reasons: list[str] = []

    for candidate in fallback_order[normalized]:
        try:
            evaluator = candidates[candidate]()
        except (ImportError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
            reasons.append(f"{candidate}: {type(error).__name__}: {error}")
            continue
        return EvaluatorSelection(
            requested=normalized,
            selected=candidate,
            evaluate=evaluator,
            fallback_reasons=tuple(reasons),
        )

    raise RuntimeError("handcrafted evaluator could not be loaded")
