#!/usr/bin/env python3
"""Train a small scene-specific relighting proxy from recorded Krust paths.

This stays intentionally outside the Rust renderer. It uses PyTorch when
available, reads JSONL path vertices from `path_recording.rs`, synthesizes
targets with the same point-light gather used by the Rust prototype, and saves
a compact checkpoint for `optimize_lighting.py`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", required=True, help="Path JSONL from record_paths")
    parser.add_argument("--output", required=True, help="Output .pt checkpoint")
    parser.add_argument("--virtual-lights", default="[]", help="JSON list of point lights")
    parser.add_argument("--max-records", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_paths(path: Path, limit: int) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if len(records) >= limit:
                break
    return records


def default_lights() -> list[dict]:
    return [{"position": [0.0, 3.0, 0.0], "color": [1.0, 1.0, 1.0], "intensity": 8.0}]


def contribution(record: dict, light: dict) -> list[float]:
    position = record["position"]
    throughput = record["throughput"]
    light_position = light["position"]
    light_color = light["color"]
    intensity = float(light.get("intensity", 1.0))
    dx = light_position[0] - position[0]
    dy = light_position[1] - position[1]
    dz = light_position[2] - position[2]
    distance_squared = max(dx * dx + dy * dy + dz * dz, 1e-4)
    scale = intensity / distance_squared
    return [throughput[i] * light_color[i] * scale for i in range(3)]


def make_sample(record: dict, light: dict) -> tuple[list[float], list[float]]:
    position = record["position"]
    throughput = record["throughput"]
    outgoing = record["outgoing"]
    light_position = light["position"]
    light_color = light["color"]
    intensity = float(light.get("intensity", 1.0))
    features = [
        position[0],
        position[1],
        position[2],
        throughput[0],
        throughput[1],
        throughput[2],
        outgoing[0],
        outgoing[1],
        outgoing[2],
        light_position[0],
        light_position[1],
        light_position[2],
        light_color[0],
        light_color[1],
        light_color[2],
        math.log(max(intensity, 1e-4)),
    ]
    return features, contribution(record, light)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for training this external proxy script.") from exc

    records = load_paths(Path(args.paths), args.max_records)
    lights = json.loads(args.virtual_lights)
    if not lights:
        lights = default_lights()

    samples = [make_sample(record, light) for record in records for light in lights]
    random.shuffle(samples)
    features = torch.tensor([sample[0] for sample in samples], dtype=torch.float32)
    targets = torch.tensor([sample[1] for sample in samples], dtype=torch.float32)

    model = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
        nn.Softplus(),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(args.epochs):
        permutation = torch.randperm(features.shape[0])
        total_loss = 0.0
        for start in range(0, features.shape[0], args.batch_size):
            batch = permutation[start : start + args.batch_size]
            prediction = model(features[batch])
            loss = loss_fn(prediction, targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.numel()
        print(f"epoch {epoch + 1}: loss={total_loss / features.shape[0]:.8f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "input_dim": 16,
            "hidden_dim": 64,
            "lights": lights,
        },
        output,
    )


if __name__ == "__main__":
    main()
