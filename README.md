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
