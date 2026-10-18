#!/usr/bin/env python3
"""Train a small local service-verification MLP from labelled feature rows.

Input is JSON/JSONL.  Each row must contain ``label`` (boolean or 0/1) and a
``features`` object with every name from ``LOCAL_FEATURE_NAMES``.  The output is
the strict JSON checkpoint consumed by ``LocalServiceVerifier``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from flowmetric.infrastructure.astra import _atomic_write_json
from flowmetric.service.local import LOCAL_FEATURE_NAMES, LOCAL_MLP_FORMAT_VERSION


def load_labelled_rows(path: Path) -> tuple[np.ndarray, np.ndarray]:
    path = Path(path)
    try:
        if path.suffix.lower() == ".jsonl":
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("samples") if isinstance(payload, dict) else payload
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read labelled local-service features: {exc}") from exc
    if not isinstance(rows, list) or len(rows) < 4:
        raise ValueError("training data must contain at least four labelled rows")

    features: list[list[float]] = []
    labels: list[float] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or "label" not in row or "features" not in row:
            raise ValueError(f"training row {index} must contain label and features")
        label = row["label"]
        if type(label) is bool:
            label_value = float(label)
        elif type(label) in (int, float) and float(label) in (0.0, 1.0):
            label_value = float(label)
        else:
            raise ValueError(f"training row {index} label must be boolean or 0/1")
        raw_features = row["features"]
        if not isinstance(raw_features, dict) or set(raw_features) != set(LOCAL_FEATURE_NAMES):
            raise ValueError(f"training row {index} has an incompatible feature set")
        vector = []
        for name in LOCAL_FEATURE_NAMES:
            value = raw_features[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"training row {index} feature {name} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"training row {index} feature {name} must be finite")
            vector.append(number)
        features.append(vector)
        labels.append(label_value)
    if len(set(labels)) != 2:
        raise ValueError("training data must contain both positive and negative rows")
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32)


def train_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    hidden: int = 8,
    epochs: int = 1000,
    learning_rate: float = 0.02,
    seed: int = 17,
    threshold: float = 0.5,
) -> dict[str, Any]:
    if features.ndim != 2 or features.shape[1] != len(LOCAL_FEATURE_NAMES):
        raise ValueError("features have an incompatible shape")
    if labels.shape != (features.shape[0],):
        raise ValueError("labels have an incompatible shape")
    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden <= 0:
        raise ValueError("hidden must be a positive integer")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    if not np.isfinite(features).all() or not np.isfinite(labels).all():
        raise ValueError("training data must be finite")

    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale < 1e-6, 1.0, scale)
    normalized = (features - mean) / scale

    torch.manual_seed(seed)
    inputs = torch.as_tensor(normalized, dtype=torch.float32)
    targets = torch.as_tensor(labels[:, None], dtype=torch.float32)
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 1),
    )
    positives = float(labels.sum())
    negatives = float(len(labels) - positives)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / max(positives, 1.0)], dtype=torch.float32)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

    first = model[0]
    second = model[2]
    return {
        "format_version": LOCAL_MLP_FORMAT_VERSION,
        "feature_names": list(LOCAL_FEATURE_NAMES),
        "mean": [float(value) for value in mean],
        "scale": [float(value) for value in scale],
        "layers": [
            {
                "weights": first.weight.detach().cpu().numpy().astype(float).tolist(),
                "bias": first.bias.detach().cpu().numpy().astype(float).tolist(),
                "activation": "relu",
            },
            {
                "weights": second.weight.detach().cpu().numpy().astype(float).tolist(),
                "bias": second.bias.detach().cpu().numpy().astype(float).tolist(),
                "activation": "sigmoid",
            },
        ],
        "threshold": float(threshold),
        "metadata": {
            "name": "flowmetric-local-service-mlp",
            "kind": "trained MLP",
            "sample_count": int(len(labels)),
            "positive_count": int(labels.sum()),
            "hidden": hidden,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "seed": seed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True, help="labelled JSON/JSONL rows")
    parser.add_argument("--out", type=Path, required=True, help="output MLP JSON checkpoint")
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    features, labels = load_labelled_rows(args.features)
    payload = train_mlp(
        features,
        labels,
        hidden=args.hidden,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        threshold=args.threshold,
    )
    _atomic_write_json(payload, args.out)
    print(
        f"saved {args.out}: {len(labels)} samples, "
        f"{int(labels.sum())} positive, {len(labels) - int(labels.sum())} negative"
    )


if __name__ == "__main__":
    main()
