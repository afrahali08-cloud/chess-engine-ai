# Chess-Engine-AI
AI chess engine with search, learned position evaluation, and a move-quality coach — CMPT 310 project


## What this project does

We're building a chess engine that:
- **Plays chess** using classic AI search (minimax with alpha-beta pruning)
- **Evaluates positions** — scoring who's ahead and by how much (e.g. +2.3, -1.6), 
  starting from a hand-crafted evaluation function and aiming to replace it with 
  one learned from real game data
- **Coaches** — analyzes a move you made and explains whether it was good or bad, 
  and why

## Current architecture

The playable engine path is based on `python-chess`:
- `src/main.py` runs the command-line game.
- `src/engine.py` searches legal moves with minimax and alpha-beta pruning.
- `src/board/evaluation.py` evaluates `python-chess` board positions.

The custom board implementation in `src/board/board.py`, `src/board/piece.py`,
and `src/board/game.py` is kept as an experimental/manual rules module, but it is
not used by the main engine loop.

## Preparing training data

Lichess standard-game archives can be read directly without writing the much
larger uncompressed PGN to disk. The extractor keeps positions with a numeric
`%eval`, converts each score to centipawns from White's point of view, and skips
mate-only evaluations.

```bash
python scripts/extract_lichess.py \
  --input ~/Downloads/lichess_db_standard_rated_2026-06.pgn.zst \
  --output data/lichess_evaluations.csv \
  --limit 100000
```

The output columns are `game_id`, `fen`, `cp`, and `ply`. Local archives and the
generated `data/` directory are ignored by Git.

Train the first learned evaluation model from the extracted positions:

```bash
python scripts/train_evaluator.py \
  --input data/lichess_evaluations.csv \
  --output models/learned_evaluator.json
```

Positions are split by game into training, validation, and test sets. The model
learns symmetric piece-square weights plus side-to-move and castling features.
Its JSON output contains the learned weights, data counts, and evaluation
metrics, so training is reproducible and the result can be inspected in Git.

At runtime, `evaluate_board()` loads `models/learned_evaluator.json` once and
uses it for all non-terminal positions searched by minimax. Checkmate and draw
scores remain rule-based. If the model is unavailable or invalid, evaluation
falls back to the hand-crafted implementation.
