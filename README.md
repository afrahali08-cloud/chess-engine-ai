# Chess-Engine-AI

Chess-Engine-AI is a local chess application that combines classical game-tree
search, learned position evaluation, and move-quality coaching. It provides a
Pygame interface for playing against the engine and a reproducible pipeline for
extracting evaluated positions, training models, comparing evaluators, and
benchmarking playing strength.

## Features

- Pygame interface with configurable player color, evaluator, search depth,
  time limit, undo, board flipping, and move history.
- Iterative-deepening Minimax with Alpha-Beta Pruning.
- Quiescence Search, move ordering, and a Transposition Table.
- Deadline-aware search with complete board-state restoration.
- Handcrafted, Ridge, and Neural MLP position evaluators.
- Automatic evaluator fallback from Neural to Ridge to Handcrafted.
- Move-quality Coach with centipawn loss, move classification, recommended
  alternatives, and short refutation lines.
- Lichess data extraction, game-level dataset splitting, model training,
  held-out evaluation, and color-swapped self-play benchmarking.

## Installation and Quick Start

The trained Ridge and Neural model files are included in the repository. Playing
the game does not require downloading the training archive or retraining the
models.

### 1. Install prerequisites

Install Git and Python 3.11 or 3.12, then verify them:

```bash
git --version
python3 --version
```

On Windows, use `py --version` if `python3` is unavailable.

### 2. Clone the repository

```bash
git clone https://github.com/afrahali08-cloud/chess-engine-ai.git
cd chess-engine-ai
```

Run all remaining commands from the repository root, which contains
`README.md`, `requirements.txt`, `src/`, and `models/`.

### 3. Create a virtual environment

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
```

After activation, the terminal prompt should begin with `(.venv)`.

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

The final command should report `No broken requirements found.`

### 5. Verify the installation

Check the bundled GUI font and run the automated tests:

```bash
python src/gui_main.py --check-fonts
python -m pytest -q
```

The current version has 258 passing tests with no failures or skipped tests.

### 6. Start the GUI

The recommended way to run the project is:

```bash
python src/gui_main.py
```

For a demonstration with explicit Neural evaluation and coaching:

```bash
python src/gui_main.py --evaluator neural --time-limit 3 --coach
```

Click a piece and then its destination square. The evaluator, depth, time
limit, player color, and Coach can also be changed inside the application.
Press `n` for a new game, `u` to undo, `f` to flip the board, `F11` for
fullscreen, and `q` to quit.

### Terminal interface

The terminal interface is available as a lightweight alternative:

```bash
python src/main.py --evaluator neural --time-limit 3 --depth 4
python src/main.py --evaluator ridge --time-limit 3 --depth 4
python src/main.py --evaluator handcrafted --time-limit 3 --depth 4
```

## System Architecture

The application uses `python-chess` for chess rules and legal move generation.
The selected evaluator assigns White-relative centipawn scores to positions,
while Minimax searches legal continuations and selects the move. The evaluator
does not select moves directly.

```text
Human move
    -> python-chess legality checking
    -> game state
    -> Minimax search
       -> Iterative Deepening
       -> Alpha-Beta Pruning
       -> Quiescence Search
       -> Transposition Table
       -> position evaluator
          -> Handcrafted / Ridge / Neural MLP
    -> best legal move
    -> engine move and GUI update

Human move + search result
    -> Move-quality Coach
    -> centipawn loss
    -> move classification
    -> recommended alternative and refutation line
    -> GUI feedback
```

The model-development pipeline is separate from normal gameplay:

```text
Lichess June 2026 PGN.ZST
    -> stream Zstandard decompression
    -> parse standard rated games
    -> extract positions with numeric %eval
    -> convert labels to White-relative centipawns
    -> sample positions and remove duplicates
    -> clip labels to [-1500, 1500]
    -> split complete games by game_id
       -> training set
       -> validation set
       -> test set
    -> train Ridge and Neural MLP on the same data
    -> select the best Neural checkpoint using validation MAE
    -> compare Handcrafted, Ridge, and Neural on the test set
    -> save models, metadata, metrics, and gameplay benchmark
```

## Position Evaluators

All evaluators return a centipawn score from White's perspective. Positive
values favor White, negative values favor Black, and approximately 100
centipawns represent one pawn of evaluation.

### Handcrafted evaluator

The Handcrafted evaluator combines material values, piece-square tables, pawn
structure, king safety, and game phase. It requires no training and serves as
the final fallback when a learned model cannot be loaded.

### Ridge evaluator

The Ridge evaluator is a supervised linear regression model with 387 inputs:

- 384 color-symmetric piece-square features (`6 piece types x 64 squares`)
- 1 side-to-move feature
- 2 castling-right difference features

It was trained with regularization strength `alpha=10.0`, the `lsqr` solver,
and random seed 42. Its sparse linear representation makes inference fast, but
limits the nonlinear relationships it can learn between pieces.

### Neural MLP evaluator

The default position evaluator is a feed-forward multilayer perceptron trained
as a supervised regression model. It converts 400 chess-position features into
a single White-relative centipawn evaluation.

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

The model has two hidden layers, three trainable Linear layers, and 135,681
trainable parameters. The output layer has no activation because the predicted
score must support both positive and negative values.

The 400 inputs contain 384 piece-square features together with material
balance, pawn structure, king safety, game phase, side to move, and castling
rights. Training uses Adam, mean squared error, learning rate `0.001`, batch
size 2048, 20 epochs, and random seed 42. The trainer saves the checkpoint with
the lowest validation MAE; the committed model selected epoch 20.

## Dataset and Preprocessing

The models were trained on positions extracted from the **June 2026 Lichess
standard rated-games archive**.[^1] The official monthly archive is a 28.2 GB
Zstandard-compressed PGN containing 86,483,328 games.

The extraction process:

1. Streams the `.pgn.zst` archive without creating an uncompressed PGN.
2. Reads positions with a numeric Stockfish `%eval` annotation.
3. Converts evaluations to centipawns from White's perspective.
4. Samples at most 20 evaluated positions per game.
5. Removes duplicate positions.
6. Skips annotations without a numeric centipawn value.
7. Clips training labels to `[-1500, 1500]` centipawns.

The final dataset contains 4,000,000 positions. Complete games are assigned to
one deterministic split using a SHA-256 hash of `seed:game_id`, with seed 42.
This prevents neighboring positions from the same game from appearing in both
training and evaluation data.

| Split | Positions | Percentage |
| --- | ---: | ---: |
| Train | 3,194,752 | 79.87% |
| Validation | 403,685 | 10.09% |
| Test | 401,563 | 10.04% |
| **Total** | **4,000,000** | **100%** |

## Training Results

### Saved model metrics

The saved Ridge and Neural checkpoints record metrics from the same dataset and
game-level split:

| Model | Split | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: |
| Ridge | Train | 171.2 cp | 241.7 cp | 0.634 |
| Ridge | Validation | 170.8 cp | 241.3 cp | 0.636 |
| Ridge | Test | 172.1 cp | 242.6 cp | 0.633 |
| Neural MLP | Train | 131.9 cp | 198.6 cp | 0.753 |
| Neural MLP | Validation | 135.7 cp | 205.0 cp | 0.737 |
| Neural MLP | Test | **136.7 cp** | **206.2 cp** | **0.734** |

MAE is the average absolute centipawn error. RMSE penalizes large errors more
strongly. R2 measures the fraction of label variance explained by the model,
where values closer to 1 are better.

### Runtime evaluator comparison

The production evaluators were compared on the shared 401,563-position test
split:

| Evaluator | Test MAE | Test RMSE | R2 | Positions/second |
| --- | ---: | ---: | ---: | ---: |
| Handcrafted | 206.4 cp | 292.4 cp | 0.466 | 24,460 |
| Ridge | 171.6 cp | 241.3 cp | 0.636 | 76,677 |
| Neural MLP | **136.3 cp** | **205.4 cp** | **0.736** | 12,060 |

The checkpoint metrics use raw floating-point predictions. The runtime
comparison uses production evaluators that round predictions to integer
centipawns, which causes the small numerical difference between the two tables.
These metrics measure agreement with held-out Stockfish labels rather than Elo
or direct playing strength.

## 4M Gameplay Benchmark

The committed benchmark compares the 4M Ridge and Neural model artifacts using
six fixed openings. Each opening is played twice with colors swapped. Both
evaluators receive 0.1 seconds per move, a maximum search depth of 5, and a
100-ply limit after the opening.

| Evaluator | Wins | Draws | Losses | Unresolved | Points | Average depth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Ridge | 1 | 8 | 1 | 2 | 5.0 | 1.64 |
| Neural MLP | 1 | 8 | 1 | 2 | 5.0 | 1.28 |

The result is **inconclusive**. Ten games ended naturally and two reached the
ply limit. The benchmark artifact records each model's SHA-256 hash and verifies
that both learned evaluators use the same 4,000,000-position dataset. A larger
match is required for a meaningful Elo estimate.

Reproduce the committed benchmark with:

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

## Optional Retraining

Retraining is optional and is not required to run the GUI. The original
`.pgn.zst` archive and generated 4M CSV are not included in the GitHub
repository. To reproduce the training process, download the June 2026 archive
from the Lichess link in the citation and place it on local storage.

Extract the training positions:

```bash
python scripts/extract_lichess.py \
  --input ~/Downloads/lichess_db_standard_rated_2026-06.pgn.zst \
  --output data/lichess_evaluations_4m.csv \
  --limit 4000000 \
  --positions-per-game 20 \
  --overwrite
```

Run Ridge training, Neural training, held-out comparison, and benchmarking:

```bash
python scripts/run_pipeline.py \
  --input data/lichess_evaluations_4m.csv \
  --expected-positions 4000000 \
  --overwrite
```

Preview the commands without running them:

```bash
python scripts/run_pipeline.py --dry-run
```

An independent run is not guaranteed to reproduce byte-identical model weights
or exactly identical metrics. Results can differ if the source month,
extraction settings, repository version, dependency versions, operating system,
or hardware changes. Using the same June 2026 archive, command-line settings,
code version, and seed 42 provides the closest reproduction.

## Limitations

- The training set contains only positions with numeric `%eval` annotations and
  may not represent all game phases, openings, ratings, or rare positions
  equally.
- Lower evaluation error does not directly establish higher Elo. The current
  self-play benchmark contains only 12 games.
- Neural inference is slower than Ridge inference, so the Neural evaluator may
  reach a lower search depth under an equal time limit.
- Coach feedback depends on the configured evaluator, search depth, and time
  budget. It can miss combinations beyond its search horizon.
- Reproducing the 4M training run requires the external 28.2 GB archive,
  substantial local storage, memory, and training time.

The project reduces these risks by using game-level splits, deterministic
seeds, label clipping, position deduplication, held-out metrics, color-swapped
benchmark games, explicit unresolved results, evaluator fallback, and model and
dataset hashes in generated artifacts.

## Project Layout

```text
src/gui_main.py                 Pygame application entry point
src/main.py                     terminal application entry point
src/game_session.py             shared board state and move history
src/engine.py                   Minimax and search control
src/coach.py                    move-quality analysis
src/tactics.py                  tactical board facts
src/board/evaluators.py         evaluator selection and fallback
src/board/evaluation.py         Handcrafted evaluator
src/board/learned_evaluation.py Ridge runtime inference
src/board/neural_model.py       Neural MLP architecture
src/board/neural_evaluation.py  Neural runtime inference
scripts/extract_lichess.py      Lichess archive extraction
scripts/run_pipeline.py         complete training and evaluation workflow
models/                         trained models and result artifacts
tests/                          automated tests
```

## External Libraries

- `python-chess` for chess rules, legal moves, board state, and PGN parsing.
- PyTorch for Neural MLP training and inference.
- Scikit-learn and SciPy for Ridge regression and sparse matrices.
- NumPy for numerical features and regression metrics.
- Pygame for the desktop interface.
- Zstandard for streaming `.pgn.zst` decompression.
- pytest for automated testing.

Install the complete dependency set from `requirements.txt`.

## Data Citation

> Lichess. (2026). *Lichess Open Database: Standard rated games, June 2026*
> [Data set]. https://database.lichess.org/

- [Lichess standard-games database](https://database.lichess.org/#standard_games)
- [June 2026 PGN.ZST archive](https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst)

## License

This project is distributed under the [MIT License](LICENSE). The bundled
DejaVu Sans font is distributed under its included license in
`src/gui/assets/DejaVuSans-LICENSE.txt`.

[^1]: Lichess. (2026). *Lichess Open Database: Standard rated games, June
    2026* [Data set]. [Database page](https://database.lichess.org/#standard_games),
    [June 2026 archive](https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst).
