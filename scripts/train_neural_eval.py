#!/usr/bin/env python3
"""Train the neural evaluator with deterministic game-level data splits."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from board.neural_features import (  # noqa: E402
    FEATURE_ENCODING,
    TOTAL_FEATURES,
    fen_to_features,
)
from board.neural_model import EvalNet  # noqa: E402
from scripts.training_common import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    SPLIT_NAMES,
    LoadedDataset,
    dataset_metadata,
    load_split_rows,
    regression_metrics,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def encode_splits(
    dataset: LoadedDataset,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for split_name in SPLIT_NAMES:
        rows = dataset.splits[split_name]
        print(f"Encoding {split_name}: {len(rows):,} positions")
        matrices[split_name] = np.asarray(
            [fen_to_features(row.fen) for row in rows],
            dtype=np.float32,
        )
        labels[split_name] = np.asarray(
            [row.score for row in rows],
            dtype=np.float32,
        )
    return matrices, labels


def make_loader(
    matrix: np.ndarray,
    labels: np.ndarray,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(matrix), torch.from_numpy(labels))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def predict(
    model: EvalNet,
    loader: DataLoader,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    predictions: list[np.ndarray] = []
    expected_values: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for features, labels in loader:
            values = model(features.to(device))
            predictions.append(values.cpu().numpy())
            expected_values.append(labels.numpy())
    return np.concatenate(predictions), np.concatenate(expected_values)


def train(
    input_path: Path,
    output_path: Path,
    *,
    epochs: int = 20,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    clip_cp: int = 1500,
    seed: int = DEFAULT_RANDOM_SEED,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"model exists: {output_path} (use --overwrite)")

    set_random_seed(seed)
    dataset = load_split_rows(
        input_path,
        clip_cp=clip_cp,
        seed=seed,
        limit=limit,
    )
    matrices, labels = encode_splits(dataset)
    loaders = {
        split_name: make_loader(
            matrices[split_name],
            labels[split_name],
            batch_size=batch_size,
            shuffle=split_name == "train",
            seed=seed,
        )
        for split_name in SPLIT_NAMES
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")
    model = EvalNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=2,
        factor=0.5,
    )

    best_validation_mae = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []

    print(f"Training for {epochs} epochs")
    print(f"{'Epoch':>6} {'Train MAE':>10} {'Val MAE':>10} {'LR':>10}")
    print("-" * 42)

    for epoch in range(1, epochs + 1):
        model.train()
        train_predictions: list[np.ndarray] = []
        train_labels: list[np.ndarray] = []

        for features, expected in loaders["train"]:
            features = features.to(device)
            expected = expected.to(device)
            optimizer.zero_grad()
            values = model(features)
            loss = criterion(values, expected)
            loss.backward()
            optimizer.step()
            train_predictions.append(values.detach().cpu().numpy())
            train_labels.append(expected.cpu().numpy())

        train_metrics = regression_metrics(
            np.concatenate(train_labels),
            np.concatenate(train_predictions),
        )
        validation_predictions, validation_labels = predict(
            model,
            loaders["validation"],
            device=device,
        )
        validation_metrics = regression_metrics(
            validation_labels,
            validation_predictions,
        )
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        validation_mae = validation_metrics["mae_cp"]
        scheduler.step(validation_mae)

        history.append(
            {
                "epoch": epoch,
                "train_mae_cp": train_metrics["mae_cp"],
                "validation_mae_cp": validation_mae,
                "learning_rate": current_learning_rate,
            }
        )
        print(
            f"{epoch:>6} {train_metrics['mae_cp']:>10.1f} "
            f"{validation_mae:>10.1f} {current_learning_rate:>10.2e}"
        )

        if validation_mae < best_validation_mae:
            best_validation_mae = validation_mae
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a model checkpoint")

    model.load_state_dict(best_state)
    model.to(device)
    metrics = {}
    for split_name in SPLIT_NAMES:
        predictions, expected = predict(
            model,
            loaders[split_name],
            device=device,
        )
        metrics[split_name] = regression_metrics(expected, predictions)

    checkpoint: dict[str, object] = {
        "format_version": 2,
        "model_type": "mlp",
        "model_state": best_state,
        "architecture": {
            "hidden_layers": [256, 128],
            "activation": "relu",
            "dropout": 0.1,
        },
        "input_size": TOTAL_FEATURES,
        "feature_encoding": FEATURE_ENCODING,
        "label": "white_centipawns",
        "clip_cp": clip_cp,
        "dataset": dataset_metadata(
            dataset,
            input_path=input_path,
            clip_cp=clip_cp,
            seed=seed,
        ),
        "training": {
            "algorithm": "adam_mse",
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
            "batch_size": batch_size,
            "initial_learning_rate": learning_rate,
            "random_seed": seed,
        },
        "split_counts": dataset.split_counts,
        "metrics": metrics,
        "history": history,
        "val_mae": metrics["validation"]["mae_cp"],
        "test_mae": metrics["test"]["mae_cp"],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    print(f"Best epoch: {best_epoch}")
    for split_name in SPLIT_NAMES:
        split_metrics = metrics[split_name]
        print(
            f"{split_name}: MAE={split_metrics['mae_cp']:.1f} cp, "
            f"RMSE={split_metrics['rmse_cp']:.1f} cp, "
            f"R2={split_metrics['r2']:.3f}"
        )
    print(f"Model saved to {output_path}")
    return checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the neural evaluator with game-level data splits."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=_positive_int, default=20)
    parser.add_argument("--batch-size", type=_positive_int, default=2048)
    parser.add_argument("--lr", type=_positive_float, default=1e-3)
    parser.add_argument("--clip-cp", type=_positive_int, default=1500)
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--limit", type=_positive_int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        train(
            args.input,
            args.output,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            clip_cp=args.clip_cp,
            seed=args.seed,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
