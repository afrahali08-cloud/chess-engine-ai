

"""
Main entry point — human vs AI chess game.
Human plays white, engine plays black.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

# fix import path so python finds python-chess not our local chess folder
src_dir = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != src_dir]

# re-add src so engine/evaluation imports still work
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import chess

try:
    from .board.evaluators import (
        DEFAULT_EVALUATOR,
        EVALUATOR_CHOICES,
        resolve_evaluator,
    )
    from .engine import choose_best_move
except ImportError:
    from board.evaluators import (
        DEFAULT_EVALUATOR,
        EVALUATOR_CHOICES,
        resolve_evaluator,
    )
    from engine import choose_best_move


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play chess against the engine.")
    parser.add_argument(
        "--evaluator",
        choices=EVALUATOR_CHOICES,
        default=DEFAULT_EVALUATOR,
        help=f"position evaluator (default: {DEFAULT_EVALUATOR})",
    )
    parser.add_argument(
        "--depth",
        type=_positive_int,
        default=4,
        help="maximum iterative-deepening depth (default: 4)",
    )
    parser.add_argument(
        "--time-limit",
        type=_positive_float,
        default=5.0,
        help="maximum search time per engine move in seconds (default: 5)",
    )
    return parser


def print_board(board: chess.Board):
    """Print board as unicode with rank/file labels."""
    print()
    print("    a b c d e f g h")
    print("  +" + "-" * 17 + "+")
    for rank in range(7, -1, -1):
        row = f"{rank + 1} | "
        for file in range(8):
            square = chess.square(file, rank)
            piece  = board.piece_at(square)
            row   += (piece.unicode_symbol() if piece else ".") + " "
        row += f"| {rank + 1}"
        print(row)
    print("  +" + "-" * 17 + "+")
    print("    a b c d e f g h")
    print()


def get_human_move(board: chess.Board):
    while True:
        try:
            raw = input("Your move (e.g. e2e4, or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if raw.lower() in ("quit", "exit", "q"):
            return None

        try:
            move = chess.Move.from_uci(raw)
            if move in board.legal_moves:
                return move
            else:
                print("  Illegal move — try again.")
        except ValueError:
            print("  Bad format — use e2e4 format.")


def print_game_summary(board: chess.Board):
    """Print move history, final FEN, and UCI move list after game ends."""
    print()
    print("=" * 40)
    print("GAME SUMMARY")
    print("=" * 40)

    # reconstruct SAN move list from move stack
    board_copy = chess.Board()
    moves_san = []
    for move in board.move_stack:
        moves_san.append(board_copy.san(move))
        board_copy.push(move)

    # print moves in standard chess notation
    print("Moves played:")
    move_text = ""
    for i, san in enumerate(moves_san):
        if i % 2 == 0:
            move_text += f"{i//2 + 1}. {san} "
        else:
            move_text += f"{san}  "
    print(move_text.strip())

    # print final FEN
    print()
    print(f"Final FEN: {board.fen()}")

    # print UCI move list
    print()
    print("UCI moves:", " ".join(str(m) for m in board.move_stack))
    print("=" * 40)


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    evaluator_selection = resolve_evaluator(args.evaluator)
    board = chess.Board()

    print()
    print("Chess Engine AI — CMPT 310")
    print("You play White. Engine plays Black.")
    print(f"Evaluator: {evaluator_selection.selected}")
    if evaluator_selection.selected != evaluator_selection.requested:
        print(
            f"Requested {evaluator_selection.requested}, "
            f"using {evaluator_selection.selected} fallback."
        )
    print("Enter moves in UCI format: e2e4, g1f3, etc.")
    print("Type 'quit' to exit.")

    while not board.is_game_over():
        print_board(board)

        if board.turn == chess.WHITE:
            move = get_human_move(board)
            if move is None:
                print("Game ended.")
                print_game_summary(board)
                sys.exit(0)
            board.push(move)
            print(f"  You played: {move}")

        else:
            print("  Engine is thinking...")
            best_move, score = choose_best_move(
                board,
                depth=args.depth,
                time_limit=args.time_limit,
                evaluator=evaluator_selection.evaluate,
            )

            if best_move is None:
                break

            board.push(best_move)
            print(f"  Engine played: {best_move}  (eval: {score / 100:+.2f})")

    # game over
    print_board(board)
    outcome = board.outcome()

    if outcome is None:
        print("Game over.")
    elif outcome.winner == chess.WHITE:
        print("Checkmate — You win!")
    elif outcome.winner == chess.BLACK:
        print("Checkmate — Engine wins!")
    else:
        print(f"Draw — {outcome.termination.name}")

    print_game_summary(board)


if __name__ == "__main__":
    main()
