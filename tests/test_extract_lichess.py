import csv
from pathlib import Path

import pytest
import zstandard as zstd

from scripts.extract_lichess import extract_positions


TEST_PGN = """[Event "Extraction test"]
[Site "https://lichess.org/testgame"]
[Date "2026.07.21"]
[Round "-"]
[White "White"]
[Black "Black"]
[Result "*"]

1. e4 { [%eval 0.20] } e5 { [%eval 0.10] }
2. Nf3 { [%eval #-3] } *
"""


def write_test_archive(path: Path) -> None:
    path.write_bytes(zstd.ZstdCompressor().compress(TEST_PGN.encode("utf-8")))


def test_extracts_centipawn_positions_and_skips_mate(tmp_path):
    input_path = tmp_path / "games.pgn.zst"
    output_path = tmp_path / "positions.csv"
    write_test_archive(input_path)

    games_read, rows_written = extract_positions(
        input_path,
        output_path,
        limit=10,
        positions_per_game=10,
    )

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert games_read == 1
    assert rows_written == 2
    assert [row["cp"] for row in rows] == ["20", "10"]
    assert [row["ply"] for row in rows] == ["1", "2"]
    assert all(row["game_id"] == "https://lichess.org/testgame" for row in rows)


def test_refuses_to_overwrite_existing_output(tmp_path):
    input_path = tmp_path / "games.pgn.zst"
    output_path = tmp_path / "positions.csv"
    write_test_archive(input_path)
    output_path.write_text("existing data", encoding="utf-8")

    with pytest.raises(FileExistsError):
        extract_positions(
            input_path,
            output_path,
            limit=10,
            positions_per_game=10,
        )
