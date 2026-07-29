"""
extract_hf.py
-------------
Streams chess positions from Lichess/chess-evaluations on Hugging Face.

Why this instead of downloading the 19GB file?
    Hugging Face lets us stream the dataset — we download only
    what we need, one batch at a time, without ever storing the
    full file. We stop after N positions and have our CSV ready.

Input:  Hugging Face dataset (streamed, no local file needed)
Output: data/eval_positions.csv
"""

from datasets import load_dataset
import csv
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("data/eval_positions.csv")
LIMIT       = 4_000_000   # how many positions to collect
EVAL_CAP    = 1500      # clip evals beyond ±1500cp

def main():
    print(f"Streaming Lichess eval dataset from Hugging Face...")
    print(f"Collecting {LIMIT:,} positions — no full download needed.")

    dataset = load_dataset(
        "Lichess/chess-position-evaluations",
        split="train",
        streaming=True   # key — streams instead of downloading everything
    )

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    count   = 0
    skipped = 0

    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["game_id", "fen", "cp", "ply"])
        writer.writeheader()

        for row in dataset:
            if count >= LIMIT:
                break

            try:
                fen = row.get("fen", "")
                cp  = row.get("cp")
                
                if not fen or cp is None:
                    skipped += 1
                    continue

                cp = int(np.clip(int(cp), -EVAL_CAP, EVAL_CAP))

                writer.writerow({
                    "game_id": f"hf_{count}",
                    "fen":     fen,
                    "cp":      cp,
                    "ply":     0,
                })
                count += 1

                if count % 50_000 == 0:
                    print(f"  {count:,} positions collected...")

            except (ValueError, TypeError):
                skipped += 1
                continue

    print(f"Done — {count:,} positions saved, {skipped:,} skipped")
    print(f"Output: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
