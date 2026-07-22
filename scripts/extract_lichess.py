#!/usr/bin/env python3
"""Stream evaluated chess positions from a Lichess PGN Zstandard archive."""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
from typing import Iterable

import chess.pgn
import zstandard as zstd


CSV_FIELDS = ("game_id", "fen", "cp", "ply")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _sample_evenly(items: list[tuple[str, int, int]], count: int) -> Iterable[tuple[str, int, int]]:
    if len(items) <= count:
        return items
    if count == 1:
        return [items[len(items) // 2]]

    last_index = len(items) - 1
    return [items[round(index * last_index / (count - 1))] for index in range(count)]


def _evaluated_positions(game: chess.pgn.Game) -> list[tuple[str, int, int]]:
    board = game.board()
    positions = []

    for node in game.mainline():
        board.push(node.move)
        evaluation = node.eval()
        if evaluation is None:
            continue

        centipawns = evaluation.white().score()
        if centipawns is None:
            continue

        positions.append((board.fen(), int(centipawns), board.ply()))

    return positions


def extract_positions(
    input_path: Path,
    output_path: Path,
    *,
    limit: int,
    positions_per_game: int,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Extract evaluated positions and return games read and rows written."""
    if not input_path.is_file():
        raise FileNotFoundError(f"input archive not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"output already exists: {output_path} (use --overwrite to replace it)"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    games_read = 0
    rows_written = 0
    next_progress_report = 10_000
    seen_positions: set[str] = set()

    with input_path.open("rb") as compressed_file, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
        decompressor = zstd.ZstdDecompressor()
        with decompressor.stream_reader(compressed_file) as reader:
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while rows_written < limit:
                game = chess.pgn.read_game(text_stream)
                if game is None:
                    break

                games_read += 1
                game_id = game.headers.get("Site") or f"game-{games_read}"
                positions = _sample_evenly(
                    _evaluated_positions(game), positions_per_game
                )

                for fen, centipawns, ply in positions:
                    position_key = " ".join(fen.split()[:4])
                    if position_key in seen_positions:
                        continue

                    writer.writerow(
                        {
                            "game_id": game_id,
                            "fen": fen,
                            "cp": centipawns,
                            "ply": ply,
                        }
                    )
                    seen_positions.add(position_key)
                    rows_written += 1

                    if rows_written >= next_progress_report:
                        print(
                            f"Collected {rows_written:,} positions "
                            f"from {games_read:,} games"
                        )
                        next_progress_report += 10_000

                    if rows_written >= limit:
                        break

    return games_read, rows_written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract Stockfish-evaluated positions from a Lichess PGN archive."
    )
    parser.add_argument("--input", required=True, type=Path, help="input .pgn.zst file")
    parser.add_argument("--output", required=True, type=Path, help="output CSV file")
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=100_000,
        help="maximum number of positions to write (default: 100000)",
    )
    parser.add_argument(
        "--positions-per-game",
        type=_positive_int,
        default=20,
        help="maximum evenly sampled positions per game (default: 20)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace the output file if it already exists",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        games_read, rows_written = extract_positions(
            args.input,
            args.output,
            limit=args.limit,
            positions_per_game=args.positions_per_game,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Finished: wrote {rows_written:,} positions from {games_read:,} games "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
