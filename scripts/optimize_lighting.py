#!/usr/bin/env python3
"""Optimize virtual light parameters through a trained NRP (differentiable inverse).

Paper reference: Eq. 6 — Reinhard tonemapped MSE, reparameterized light params,
mini-batch pixel SGD (Table 3).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import struct
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="NRP .bin weights")
    parser.add_argument("--checkpoint-meta", help="Optional .json metadata")
    parser.add_argument("--paths", required=True, help="Path JSONL (for pixel list)")
    parser.add_argument("--output", required=True, help="Output optimized lights JSON")
    parser.add_argument("--target-image", help="Target EXR/PNG for image loss")
    parser.add_argument("--target-color", nargs=3, type=float, help="Mean RGB target")
    parser.add_argument("--scribble", help="JSON mask [{x,y,r,g,b,weight}, ...]")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--initial-position", nargs=3, type=float, default=[0.0, 5.0, 0.0])
    parser.add_argument("--initial-color", nargs=3, type=float, default=[1.0, 1.0, 1.0])
    parser.add_argument("--initial-intensity", type=float, default=4.0)
    parser.add_argument("--initial-radius", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--pixel-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def reinhard(x: "torch.Tensor") -> "torch.Tensor":
    return x / (1.0 + x)


def load_weights(path: Path) -> tuple[dict, list[float]]:
    raw = path.read_bytes()
    count = len(raw) // 4
    values = struct.unpack(f"{count}f", raw)
    header = {
        "input_dim": int(values[0]),
        "hidden_dim": int(values[1]),
        "output_dim": int(values[2]),
        "num_layers": int(values[3]),
    }
    return header, list(values[8:])


def build_mlp(header: dict, weights: list[float]) -> "torch.nn.Module":
    import torch
    from torch import nn

    input_dim = header["input_dim"]
    hidden_dim = header["hidden_dim"]
    num_layers = header["num_layers"]
    offset = 0

    def take(shape: tuple[int, ...]) -> torch.Tensor:
        nonlocal offset
        size = int(np.prod(shape))
        tensor = torch.tensor(weights[offset : offset + size], dtype=torch.float32).reshape(shape)
        offset += size
        return tensor

    layers: list[nn.Module] = []
    w = take((hidden_dim, input_dim))
    b = take((hidden_dim,))
    layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()])
    layers[0].weight.data = w
    layers[0].bias.data = b

    for _ in range(num_layers - 2):
        lin = nn.Linear(hidden_dim, hidden_dim)
        lin.weight.data = take((hidden_dim, hidden_dim))
        lin.bias.data = take((hidden_dim,))
        layers.extend([lin, nn.ReLU()])
    out = nn.Linear(hidden_dim, 3)
    out.weight.data = take((3, hidden_dim))
    out.bias.data = take((3,))
    layers.extend([out, nn.Softplus()])
    return nn.Sequential(*layers)


def hash_grid(px: float, py: float, level: int) -> np.ndarray:
    scale = 2**level
    fx = px * scale
    fy = py * scale
    h = np.sin(np.array([fx * 12.9898, fy * 78.233])) * 43758.5453
    return np.mod(h, 1.0)


def load_target(args: argparse.Namespace, width: int, height: int) -> np.ndarray:
    if args.target_color:
        return np.array(args.target_color, dtype=np.float32)
    if args.target_image:
        try:
            import imageio.v2 as imageio
        except ImportError as exc:
            raise SystemExit("imageio required for --target-image") from exc
        img = imageio.imread(args.target_image)[..., :3].astype(np.float32)
        if img.max() > 1.5:
            img /= 255.0
        return img
    return np.array([0.6, 0.45, 0.35], dtype=np.float32)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for inverse optimization.") from exc

    header, weights = load_weights(Path(args.checkpoint))
    model = build_mlp(header, weights)
    model.eval()

    width, height = args.width, args.height
    if args.checkpoint_meta:
        meta = json.loads(Path(args.checkpoint_meta).read_text(encoding="utf-8"))
        width = int(meta.get("width", width))
        height = int(meta.get("height", height))

    target = load_target(args, width, height)
    scribble_mask = None
    if args.scribble:
        scribble_mask = json.loads(Path(args.scribble).read_text(encoding="utf-8"))

    pos_min = torch.tensor([-20.0, -2.0, -20.0])
    pos_max = torch.tensor([20.0, 20.0, 20.0])
    position = torch.tensor(args.initial_position, dtype=torch.float32, requires_grad=True)
    raw_color = torch.tensor(args.initial_color, dtype=torch.float32).clamp(1e-3, 1.0).logit()
    raw_color.requires_grad_(True)
    log_intensity = torch.tensor(math.log(max(args.initial_intensity, 1e-4)), requires_grad=True)
    log_radius = torch.tensor(math.log(max(args.initial_radius, 1e-4)), requires_grad=True)

    optimizer = torch.optim.Adam([position, raw_color, log_intensity, log_radius], lr=args.lr)
    pixels = [(x, y) for y in range(height) for x in range(width)]
    batch_size = max(int(len(pixels) * args.pixel_fraction), 64)

    for step in range(args.steps):
        batch = random.sample(pixels, min(batch_size, len(pixels)))
        color = raw_color.sigmoid()
        intensity = log_intensity.exp()
        radius = log_radius.exp()
        preds = []
        targets = []
        for px, py in batch:
            u = px / max(width - 1, 1)
            v = py / max(height - 1, 1)
            h = hash_grid(u, v, 0)
            features = torch.tensor(
                [
                    u,
                    v,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    float(position[0]),
                    float(position[1]),
                    float(position[2]),
                    float(radius),
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                    float(log_intensity),
                    float(h[0]),
                    float(h[1]),
                    float(h[2]),
                ],
                dtype=torch.float32,
            )
            pred = model(features)
            if isinstance(target, np.ndarray) and target.ndim == 3:
                tgt = torch.tensor(target[py, px], dtype=torch.float32)
            else:
                tgt = torch.tensor(target, dtype=torch.float32)
            if scribble_mask:
                weight = 0.0
                for stroke in scribble_mask:
                    if abs(stroke["x"] - px) <= stroke.get("radius", 8) and abs(stroke["y"] - py) <= stroke.get("radius", 8):
                        weight = stroke.get("weight", 1.0)
                        tgt = torch.tensor(stroke.get("color", [1.0, 1.0, 1.0]), dtype=torch.float32)
                if weight == 0.0:
                    continue
            preds.append(reinhard(pred))
            targets.append(reinhard(tgt))

        if not preds:
            continue
        pred_tensor = torch.stack(preds)
        target_tensor = torch.stack(targets)
        loss = torch.mean((pred_tensor - target_tensor) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            position.clamp_(min=pos_min, max=pos_max)
        if step % 50 == 0 or step == args.steps - 1:
            print(f"step {step}: loss={float(loss.detach()):.8f}")

    output = Path(args.output)
    optimized = {
        "virtual_lights": [
            {
                "position": [float(v) for v in position.detach()],
                "color": [float(v) for v in raw_color.sigmoid().detach()],
                "intensity": float(log_intensity.exp().detach()),
                "radius": float(log_radius.exp().detach()),
            }
        ],
        "renderer_settings": {
            "gather_relighting": 1,
            "virtual_lights_file": str(output),
            "nrp_weights_file": str(args.checkpoint),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(optimized, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
