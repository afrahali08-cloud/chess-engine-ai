"""Entry point for the windowed frontend.

    python src/gui_main.py --evaluator neural --coach
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Same guard as main.py: keep src/ from shadowing python-chess with a local
# `chess` directory before anything imports it.
src_dir = Path(__file__).resolve().parent
sys.path = [p for p in sys.path if Path(p).resolve() != src_dir]
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import chess

try:
    from .game_session import SessionConfig
    from .main import build_parser
except ImportError:
    from game_session import SessionConfig
    from main import build_parser


def build_gui_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description = "Play chess against the engine in a window."
    parser.add_argument(
        "--play-as",
        choices=("white", "black"),
        default="white",
        help="which color you play (default: white)",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="start in fullscreen",
    )
    parser.add_argument(
        "--window",
        default="1280x800",
        help="window size as WIDTHxHEIGHT (default: 1280x800)",
    )
    parser.add_argument(
        "--check-fonts",
        action="store_true",
        help="report which fonts were found for the pieces, then exit",
    )
    return parser


def parse_window(value: str) -> tuple[int, int]:
    try:
        width, height = value.lower().split("x")
        return int(width), int(height)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid window size {value!r}; use WIDTHxHEIGHT"
        ) from None


def main(argv: list[str] | None = None) -> int:
    args = build_gui_parser().parse_args(argv)

    if args.check_fonts:
        from gui.pieces import describe_fonts

        print(describe_fonts())
        return 0

    width, height = parse_window(args.window)
    config = SessionConfig(
        evaluator=args.evaluator,
        depth=args.depth,
        time_limit=args.time_limit,
        coach=args.coach,
        coach_depth=args.coach_depth,
        coach_time_limit=args.coach_time_limit,
        human_color=chess.WHITE if args.play_as == "white" else chess.BLACK,
    )

    from gui.app import run_app

    return run_app(config, fullscreen=args.fullscreen, width=width, height=height)


if __name__ == "__main__":
    raise SystemExit(main())
