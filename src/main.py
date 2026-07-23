

"""
Main entry point — human vs AI chess game.
Human plays white, engine plays black.
"""

from __future__ import annotations
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
    from .engine import choose_best_move
except ImportError:
    from engine import choose_best_move


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


def main():
    board = chess.Board()
    depth = 4  # increase for stronger play, decrease if too slow

    print()
    print("Chess Engine AI — CMPT 310")
    print("You play White. Engine plays Black.")
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
            best_move, score = choose_best_move(board, depth=depth, time_limit= 5.0)

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