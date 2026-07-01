from __future__ import annotations

import sys
from pathlib import Path


def _import_python_chess():
    """Import python-chess even though this repo also has src/chess/."""
    src_dir = Path(__file__).resolve().parent
    removed_paths = []

    for path in list(sys.path):
        if path and Path(path).resolve() == src_dir:
            sys.path.remove(path)
            removed_paths.append(path)

    try:
        import chess
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The python-chess package is required. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc
    finally:
        for path in reversed(removed_paths):
            sys.path.insert(0, path)

    if not hasattr(chess, "Board"):
        raise ImportError(
            "The python-chess package is required. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    return chess


def _import_choose_best_move():
    try:
        from .engine import choose_best_move
    except ImportError:
        from engine import choose_best_move

    return choose_best_move


def main():
    chess = _import_python_chess()
    choose_best_move = _import_choose_best_move()
    board = chess.Board()
    best_move, score = choose_best_move(board, depth=2)

    print("Best move:", best_move)
    print("Evaluation score:", score)


if __name__ == '__main__':
    main()
