#!/usr/bin/env python3
"""Optimize a virtual point light through a trained scene proxy.

This does not make the Rust path tracer differentiable. It loads the external
proxy checkpoint created by `train_scene_proxy.py`, optimizes one virtual light
against an image-space mean RGB target, and writes renderer-friendly JSON.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--paths", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-color", nargs=3, type=float, required=True)
    parser.add_argument("--initial-position", nargs=3, type=float, default=[0.0, 3.0, 0.0])
    parser.add_argument("--initial-color", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--initial-intensity", type=float, default=8.0)
    parser.add_argument("--max-records", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-2)
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


def main() -> None:
    args = parse_args()

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for proxy-based lighting optimization.") from exc

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = nn.Sequential(
        nn.Linear(checkpoint["input_dim"], checkpoint["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_dim"], checkpoint["hidden_dim"]),
        nn.ReLU(),
        nn.Linear(checkpoint["hidden_dim"], 3),
        nn.Softplus(),
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    records = load_paths(Path(args.paths), args.max_records)
    path_features = torch.tensor(
        [
            record["position"]
            + record["throughput"]
            + record["outgoing"]
            for record in records
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(args.target_color, dtype=torch.float32)

    position = torch.tensor(args.initial_position, dtype=torch.float32, requires_grad=True)
    raw_color = torch.tensor(args.initial_color, dtype=torch.float32).clamp(1e-3, 1.0).logit()
    raw_color.requires_grad_(True)
    log_intensity = torch.tensor(math.log(max(args.initial_intensity, 1e-4)), requires_grad=True)

    optimizer = torch.optim.Adam([position, raw_color, log_intensity], lr=args.lr)
    for step in range(args.steps):
        color = raw_color.sigmoid()
        light_features = torch.cat(
            [
                position.expand(path_features.shape[0], 3),
                color.expand(path_features.shape[0], 3),
                log_intensity.expand(path_features.shape[0], 1),
            ],
            dim=1,
        )
        prediction = model(torch.cat([path_features, light_features], dim=1)).mean(dim=0)
        loss = torch.mean((prediction - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == args.steps - 1:
            print(f"step {step}: loss={float(loss.detach()):.8f} rgb={prediction.detach().tolist()}")

    output = Path(args.output)
    optimized = {
        "virtual_lights": [
            {
                "position": [float(value) for value in position.detach()],
                "color": [float(value) for value in raw_color.sigmoid().detach()],
                "intensity": float(log_intensity.exp().detach()),
            }
        ],
        "renderer_settings": {
            "gather_relighting": 1,
            "virtual_lights_file": str(output),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(optimized, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
