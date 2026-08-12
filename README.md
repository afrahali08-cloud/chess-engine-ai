# Chess-Engine-AI

AI chess engine with alpha-beta search, learned position evaluation, a Pygame
interface, and a move-quality coach. This is a CMPT 310 course project.

## Features

- Playable command-line and Pygame interfaces built on `python-chess`.
- Iterative-deepening minimax with alpha-beta pruning, quiescence search,
  transposition caching, and a per-move deadline.
- Three interchangeable evaluators: hand-crafted, Ridge regression, and a
  PyTorch multilayer perceptron (MLP).
- Automatic evaluator fallback: neural to Ridge to hand-crafted.
- Move coaching based on centipawn loss, short refutation lines, and tactical
  board facts.
- Reproducible Lichess extraction, game-level data splitting, training,
  evaluator comparison, and fixed-time self-play benchmarking.

## Fresh installation tutorial

The trained Ridge and neural model files are included in the repository. A new
user can play immediately after installing the dependencies; downloading the
Lichess archive or retraining the models is **not** required. The original
`.pgn.zst` archive and generated training CSV are too large for this repository
and are not included on GitHub.

### Step 1: install the prerequisites

Install these tools before cloning the project:

- Git
- Python 3.11 or Python 3.12

Check that both commands are available:

```bash
git --version
python3 --version
```

On Windows, use `py --version` if `python3` is not recognized.

### Step 2: clone the repository

Open Terminal, PowerShell, or Command Prompt and run:

```bash
git clone https://github.com/afrahali08-cloud/chess-engine-ai.git
cd chess-engine-ai
```

All remaining commands in this README must be run from the repository root,
the directory that contains `README.md`, `requirements.txt`, `src/`, and
`models/`.

### Step 3: create a clean virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
```

On Windows:

```powershell
py -3.12 -m venv .venv
```

The `.venv` directory is local and ignored by Git.

### Step 4: activate the virtual environment

On macOS or Linux:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

After activation, the terminal prompt should begin with `(.venv)`. Check that
the environment's Python is being used:

```bash
python --version
```

### Step 5: install every dependency

Upgrade pip and install the complete runtime, GUI, training, extraction, and
test dependency set:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The last command should print `No broken requirements found.`

### Step 6: verify the included model and GUI assets

Confirm that the GUI can find its bundled chess font:

```bash
python src/gui_main.py --check-fonts
```

The output should show `[OK]` for
`src/gui/assets/DejaVuSans.ttf` and display it as the selected piece font.
The trained models used by the application are already present at:

```text
models/neural_evaluator.pt
models/learned_evaluator.json
```

### Step 7: run the automated tests

```bash
python -m pytest -q
```

The current repository should report `258 passed` with no skipped or failed
tests. A different passing count is acceptable if new tests were added later.

### Step 8: start the GUI

The recommended way to run and demonstrate the project is:

```bash
python src/gui_main.py
```

The GUI uses the trained neural evaluator by default. For an explicit
presentation command with the move coach enabled, run:

```bash
python src/gui_main.py \
  --evaluator neural \
  --time-limit 3 \
  --coach
```

On Windows PowerShell, the same command can be entered on one line:

```powershell
python src/gui_main.py --evaluator neural --time-limit 3 --coach
```

### Step 9: play and exit

Click one of your pieces and then click its destination square. The evaluator,
search depth, time limit, side, and coach can be changed in the window. Press
`q` or close the window to exit. When finished, leave the virtual environment
with:

```bash
deactivate
```

The terminal interface described below is retained as a debugging and fallback
entry point.

## Run the application

### Windowed interface (recommended)

```bash
python src/gui_main.py
python src/gui_main.py --evaluator neural --time-limit 3 --coach
python src/gui_main.py --evaluator ridge --play-as black
```

Click a piece and then its destination. The evaluator, depth, time limit,
player color, and coach can also be changed in the window. Useful keys are `n`
for a new game, `u` to undo, `f` to flip, `F11` for fullscreen, and `q` to quit.

Check GUI font availability without starting a game:

```bash
python src/gui_main.py --check-fonts
```

### Terminal interface

```bash
python src/main.py --evaluator neural --time-limit 3 --depth 4
python src/main.py --evaluator ridge --time-limit 3 --depth 4
python src/main.py --evaluator handcrafted --time-limit 3 --depth 4
```

Enable move coaching with:

```bash
python src/main.py \
  --evaluator neural \
  --coach \
  --coach-time-limit 1 \
  --time-limit 3
```

Neural evaluation is the default because it has the lowest held-out test MAE.
If PyTorch or the neural checkpoint is unavailable, the application falls back
to Ridge and then to the hand-crafted evaluator.

## Common setup problems

### `No module named ...`

The virtual environment is probably inactive or the dependencies were not
installed. Return to the repository root and run:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows users should replace the activation command with the PowerShell or
Command Prompt command from Step 4.

### PowerShell blocks `Activate.ps1`

Allow local activation for the current PowerShell process, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### `python: can't open file 'src/gui_main.py'`

The command is being run from the wrong directory. Run `cd chess-engine-ai`
and confirm that `README.md` and `src/` are visible before trying again.

### The neural model falls back to Ridge

Check that `models/neural_evaluator.pt` exists and that PyTorch imports:

```bash
python -c "import torch; print(torch.__version__)"
```

Reinstall `requirements.txt` if the import fails. The fallback keeps the game
playable, but the normal fresh installation should select `neural`.

## Evaluator models and training

The search engine does not predict a move directly. Minimax generates and
searches legal moves, while the selected evaluator predicts a single
White-relative position score in centipawns. Positive scores favor White and
negative scores favor Black.

### Hand-crafted evaluator

The baseline evaluator uses manually selected chess knowledge such as material,
piece-square tables, pawn structure, king safety, and game phase. It requires no
training and remains available as the final fallback evaluator.

### Ridge regression evaluator

The Ridge model is a linear supervised regression model with 387 inputs:

- 384 color-symmetric piece-square features (`6 piece types x 64 squares`)
- 1 side-to-move feature
- 2 castling-right difference features

It was trained with `alpha=10.0`, the `lsqr` solver, and random seed 42. Ridge
is fast and interpretable, but a linear model cannot represent all nonlinear
interactions between pieces.

### Neural MLP evaluator

The default predictor is a feed-forward multilayer perceptron regression model,
not KNN and not a move-classification model. Its architecture is:

```text
400 input features
  -> Linear(400, 256)
  -> ReLU
  -> Dropout(0.1)
  -> Linear(256, 128)
  -> ReLU
  -> Dropout(0.1)
  -> Linear(128, 1)
  -> centipawn evaluation
```

The network has two hidden layers, three trainable linear layers, and 135,681
trainable parameters. The final layer has no activation because the regression
output must support both positive and negative centipawn values.

Its 400 inputs contain 384 piece-square values plus material balance, pawn
structure, king safety, game phase, side to move, and castling rights. Training
uses Adam, mean squared error, learning rate `0.001`, batch size 2048, 20 epochs,
and random seed 42. The trainer retains the epoch with the lowest validation
MAE; the committed 4M checkpoint selected epoch 20.

### Stored checkpoint metrics

Both learned models were trained and evaluated using the same 4M dataset and
game-level split. Their saved floating-point checkpoint metrics are:

| Model | Split | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| Ridge | Train | 171.2 cp | 241.7 cp | 0.634 |
| Ridge | Validation | 170.8 cp | 241.3 cp | 0.636 |
| Ridge | Test | 172.1 cp | 242.6 cp | 0.633 |
| Neural MLP | Train | 131.9 cp | 198.6 cp | 0.753 |
| Neural MLP | Validation | 135.7 cp | 205.0 cp | 0.737 |
| Neural MLP | Test | **136.7 cp** | **206.2 cp** | **0.734** |

MAE is the average absolute centipawn error. RMSE penalizes large prediction
errors more strongly. R2 measures the fraction of label variance explained by
the model, where values closer to 1 are better.

## Current models and data

The committed Ridge and neural models were trained on the same **4,000,000
positions** extracted from the June 2026 Lichess standard rated-game archive.[^1]
Numeric Stockfish `%eval` annotations are converted to centipawns from White's
point of view and clipped to `[-1500, 1500]`.

Complete games are assigned to deterministic splits using a SHA-256 hash of
`seed:game_id`, with seed 42. Keeping every position from one game in one split
prevents leakage between training and evaluation.

| Split | Positions | Percentage |
| --- | ---: | ---: |
| Train | 3,194,752 | 79.87% |
| Validation | 403,685 | 10.09% |
| Test | 401,563 | 10.04% |
| Total | 4,000,000 | 100% |

The CSV archives are intentionally ignored by Git. The committed model files
contain the dataset hash, split counts, seed, feature version, training
parameters, and regression metrics needed to identify the training run.

## Dataset source and citation

The original data comes from the
[Lichess Open Database standard-games collection](https://database.lichess.org/#standard_games).
This project used the **2026 - June** standard rated-games archive:

- Archive: [`lichess_db_standard_rated_2026-06.pgn.zst`](https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst)
- Official listed size: 28.2 GB
- Games in the monthly archive: 86,483,328
- Format: compressed PGN using Zstandard

Lichess monthly archives are independent rather than cumulative. The extractor
scans the June archive and keeps positions that contain a numeric Stockfish
`%eval`; it does not treat every game or every position as a labeled example.

For reports or presentations, cite the dataset as:

> Lichess. (2026). *Lichess Open Database: Standard rated games, June 2026*
> [Data set]. https://database.lichess.org/

## Optional: reproduce the training data

This section is only for users who want to retrain the evaluators. It is not
part of the normal installation or GUI startup process.

The source `.pgn.zst` and the generated `data/lichess_evaluations_4m.csv` are
not committed to GitHub. To reproduce the project locally, first download the
exact June 2026 archive from the direct link above and place it somewhere on
your own disk, such as `~/Downloads/`.

The extractor streams a `.pgn.zst` archive directly, so an uncompressed PGN is
not required:

```bash
python scripts/extract_lichess.py \
  --input ~/Downloads/lichess_db_standard_rated_2026-06.pgn.zst \
  --output data/lichess_evaluations_4m.csv \
  --limit 4000000 \
  --positions-per-game 20 \
  --overwrite
```

The generated CSV columns are `game_id`, `fen`, `cp`, and `ply`.

The command above matches the current project's extraction settings, but an
independent retraining run is not guaranteed to reproduce byte-for-byte model
weights or identical metrics. Results can differ when a user selects another
month, changes extraction limits or sampling settings, uses a different source
archive, or trains with different PyTorch, NumPy, CPU, CUDA, or operating-system
versions. The game-level split and random seed are deterministic, so using the
same June 2026 archive, repository version, command-line parameters, and seed 42
provides the closest reproduction of the committed run.

## Run the complete model pipeline

`scripts/run_pipeline.py` is the main reproducible workflow entry point. It
runs Ridge training, neural training, held-out comparison, and self-play
benchmarking in separate processes:

```bash
python scripts/run_pipeline.py \
  --input data/lichess_evaluations_4m.csv \
  --expected-positions 4000000 \
  --overwrite
```

The neural stage uses 20 epochs, batch size 2048, Adam/MSE, learning rate
`0.001`, and saves the checkpoint with the lowest validation MAE. A complete 4M
run requires substantial RAM and can take several minutes. Preview all commands
without running them using `--dry-run`.

Run selected stages when trained models already exist:

```bash
python scripts/run_pipeline.py --stages compare --overwrite
python scripts/run_pipeline.py --stages benchmark --overwrite
```

The individual scripts remain available for experiments:

```bash
python scripts/train_evaluator.py --help
python scripts/train_neural_eval.py --help
python scripts/compare_evaluators.py --help
python scripts/benchmark_evaluators.py --help
```

## Held-out evaluation

The current runtime comparison uses the shared 401,563-position test split:

| Evaluator | Test MAE | Test RMSE | R2 | Positions/second |
| --- | ---: | ---: | ---: | ---: |
| Hand-crafted | 206.4 cp | 292.4 cp | 0.466 | 24,460 |
| Ridge | 171.6 cp | 241.3 cp | 0.636 | 76,677 |
| Neural MLP | **136.3 cp** | **205.4 cp** | **0.736** | 12,060 |

The checkpoint table above measures the trainers' raw floating-point outputs.
This comparison passes positions through the production evaluator, which rounds
predictions to integer centipawns; that is why the figures differ slightly.

These metrics measure agreement with held-out Stockfish labels. They do not by
themselves establish Elo or playing strength.

## 4M playing-strength benchmark

The committed benchmark was regenerated from the 4M Ridge and neural model
artifacts. It uses six fixed openings, swaps colors for every opening, allows
0.1 seconds per move, searches to a maximum depth of 5, and stops unresolved
games after 100 searched plies.

| Evaluator | Wins | Draws | Losses | Unresolved | Points | Average depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 1 | 8 | 1 | 2 | 5.0 | 1.64 |
| Neural MLP | 1 | 8 | 1 | 2 | 5.0 | 1.28 |

The result is **inconclusive**. Ten games ended naturally and two reached the
ply limit. The JSON artifact records model hashes and verifies that both learned
evaluators use the same 4,000,000-position dataset. A larger match is required
for a meaningful Elo estimate.

Reproduce it with:

```bash
python scripts/benchmark_evaluators.py \
  --first ridge \
  --second neural \
  --time-limit 0.1 \
  --depth 5 \
  --max-plies 100 \
  --openings 6 \
  --expected-positions 4000000 \
  --overwrite
```

## Tests

Run the complete test suite after installing all requirements:

```bash
python -m pytest -q
```

## Project layout

```text
src/main.py                     terminal game entry point
src/gui_main.py                 Pygame entry point
src/engine.py                   minimax and search control
src/coach.py                    move-quality analysis
src/board/evaluators.py         evaluator selection and fallback
src/board/neural_model.py       MLP architecture
scripts/extract_lichess.py      streaming archive extractor
scripts/run_pipeline.py         complete model workflow
models/                         trained models and evaluation artifacts
tests/                          automated tests
```

The older custom board implementation in `src/board/board.py`, `piece.py`, and
`game.py` is retained as an experimental module. The playable engine path uses
`python-chess` exclusively.

[^1]: Lichess. (2026). *Lichess Open Database: Standard rated games, June
    2026* [Data set]. [Database page](https://database.lichess.org/#standard_games),
    [June 2026 PGN.ZST archive](https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst).
