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
