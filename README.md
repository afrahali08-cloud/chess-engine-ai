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

## Install and play

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The engine supports hand-crafted, Ridge, and neural evaluation:

```bash
python src/main.py --evaluator neural --time-limit 3 --depth 4
python src/main.py --evaluator ridge --time-limit 3 --depth 4
python src/main.py --evaluator handcrafted --time-limit 3 --depth 4
```

Neural evaluation is the default because the 1,000,000-position model has both
the lowest held-out test MAE and the best fixed-time playing-strength result.
If PyTorch or the neural checkpoint is unavailable, neural mode falls back to
Ridge, then to the hand-crafted evaluator.

Enable move coaching to classify each human move by centipawn loss and explain
the engine's preferred alternative:

```bash
python src/main.py \
  --evaluator neural \
  --coach \
  --coach-time-limit 1 \
  --time-limit 3
```

## Preparing training data

Lichess standard-game archives can be read directly without writing the much
larger uncompressed PGN to disk. The extractor keeps positions with a numeric
`%eval`, converts each score to centipawns from White's point of view, and skips
mate-only evaluations.

```bash
python scripts/extract_lichess.py \
  --input ~/Downloads/lichess_db_standard_rated_2026-06.pgn.zst \
  --output data/lichess_evaluations.csv \
  --limit 1000000
```

The output columns are `game_id`, `fen`, `cp`, and `ply`. Local archives and the
generated `data/` directory are ignored by Git.

Train both learned evaluators from the same extracted positions:

```bash
python scripts/train_evaluator.py \
  --input data/lichess_evaluations.csv \
  --output models/learned_evaluator.json \
  --seed 42 \
  --overwrite

python scripts/train_neural_eval.py \
  --input data/lichess_evaluations.csv \
  --output models/neural_evaluator.pt \
  --epochs 20 \
  --seed 42 \
  --overwrite
```

Both trainers assign complete games to deterministic 80/10/10 training,
validation, and test splits. This prevents neighboring positions from one game
appearing in different splits. Model artifacts record the source-data hash,
split counts, random seed, feature version, training parameters, and MAE/RMSE/R2
metrics.

Compare all evaluators on the shared test split:

```bash
python scripts/compare_evaluators.py \
  --input data/lichess_evaluations.csv \
  --overwrite
```

Current results from 1,000,000 positions:

| Evaluator | Test MAE | Test RMSE | R2 |
| --- | ---: | ---: | ---: |
| Hand-crafted | 200.9 cp | 286.1 cp | 0.463 |
| Ridge | 167.6 cp | 236.4 cp | 0.634 |
| Neural MLP | 139.5 cp | 209.2 cp | 0.713 |

## Playing-strength benchmark

Run color-swapped games from six fixed openings with equal search time:

```bash
python scripts/benchmark_evaluators.py \
  --first ridge \
  --second neural \
  --time-limit 0.1 \
  --depth 5 \
  --openings 6 \
  --overwrite
```

The current 12-game result is:

| Evaluator | Wins | Draws | Losses | Unresolved | Points | Average depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 1 | 6 | 2 | 3 | 4.0 | 1.73 |
| Neural MLP | 2 | 6 | 1 | 3 | 5.0 | 1.40 |

The benchmark uses both colors for every opening. Nine games ended naturally;
three reached the 100-ply limit and are recorded as unresolved. Results and
per-game move data are stored in `models/strength_benchmark.json`.

Run the complete test suite with:

```bash
python -m pytest -q
```
