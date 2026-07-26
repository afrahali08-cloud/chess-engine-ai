from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from board.neural_features import TOTAL_FEATURES, fen_to_features


# ------------------------------------------------------------------
# Model architecture
# ------------------------------------------------------------------
class EvalNet(nn.Module):
    """
    Small MLP for chess position evaluation.
    
    Three layers with ReLU activations — enough capacity to learn
    non-linear positional patterns without overfitting on 500k positions.
    """
    def __init__(self, input_size: int = TOTAL_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),        # prevents overfitting
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)       # single output: centipawn score
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
def train(
    input_path: Path,
    output_path: Path,
    *,
    epochs: int = 20,
    batch_size: int = 2048,
    lr: float = 1e-3,
    clip_cp: int = 1500,
    overwrite: bool = False,
) -> None:

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"model exists: {output_path} (use --overwrite)")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("Loading positions...")
    fens, labels = [], []

    with open(input_path, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 200_000:
                break
            cp = float(np.clip(int(row["cp"]), -clip_cp, clip_cp))
            fens.append(row["fen"])
            labels.append(cp)

    print(f"  Loaded {len(fens):,} positions")

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    print("Extracting features...")
    X = np.array([fen_to_features(fen) for fen in fens], dtype=np.float32)
    y = np.array(labels, dtype=np.float32)
    print(f"  Feature matrix: {X.shape}")

    # ------------------------------------------------------------------
    # Train/val/test split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42
    )
    print(f"  Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    def make_loader(X, y, shuffle=False):
        dataset = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32)
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    # ------------------------------------------------------------------
    # Model, loss, optimizer
    # ------------------------------------------------------------------
    model     = EvalNet().to(device)
    criterion = nn.MSELoss()             # mean squared error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )

    print(f"\nTraining for {epochs} epochs...")
    print(f"{'Epoch':>6} {'Train MAE':>10} {'Val MAE':>10} {'LR':>10}")
    print("-" * 40)

    best_val_mae = float('inf')
    best_state   = None

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        train_preds, train_true = [], []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_preds.extend(preds.detach().cpu().numpy())
            train_true.extend(y_batch.cpu().numpy())

        # validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                preds   = model(X_batch)
                val_preds.extend(preds.cpu().numpy())
                val_true.extend(y_batch.numpy())

        train_mae = mean_absolute_error(train_true, train_preds)
        val_mae   = mean_absolute_error(val_true, val_preds)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"{epoch:>6} {train_mae:>10.1f} {val_mae:>10.1f} {current_lr:>10.2e}")

        scheduler.step(val_mae)

        # save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch.to(device))
            test_preds.extend(preds.cpu().numpy())
            test_true.extend(y_batch.numpy())

    test_mae = mean_absolute_error(test_true, test_preds)

    print(f"\nFinal test MAE: {test_mae:.1f} cp")
    print(f"Best val MAE:   {best_val_mae:.1f} cp")

    # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": best_state,
        "input_size":  TOTAL_FEATURES,
        "clip_cp":     clip_cp,
        "val_mae":     best_val_mae,
        "test_mae":    test_mae,
    }, output_path)

    print(f"Model saved to {output_path}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input",     required=True, type=Path)
    p.add_argument("--output",    required=True, type=Path)
    p.add_argument("--epochs",    type=int,   default=20)
    p.add_argument("--batch-size",type=int,   default=2048)
    p.add_argument("--lr",        type=float, default=1e-3)
    p.add_argument("--clip-cp",   type=int,   default=1500)
    p.add_argument("--overwrite", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    train(
        Path(args.input),
        Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        clip_cp=args.clip_cp,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()