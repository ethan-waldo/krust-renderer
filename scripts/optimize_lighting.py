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
    parser.add_argument("--aux", help="Optional NRP auxiliary feature JSON")
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
        "hash_table_size": int(values[4]),
        "hash_levels": int(values[5]),
        "hash_features": int(values[6]),
        "hash_offset": int(values[7]),
    }
    return header, list(values)


def build_mlp(header: dict, weights: list[float]) -> "torch.nn.Module":
    import torch
    from torch import nn

    input_dim = header["input_dim"]
    hidden_dim = header["hidden_dim"]
    num_layers = header["num_layers"]
    offset = 8

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
    layers.append(out)
    return nn.Sequential(*layers)


def build_hash_table(header: dict, weights: list[float]) -> "torch.nn.Embedding":
    import torch
    from torch import nn

    table_size = header["hash_table_size"]
    levels = header["hash_levels"]
    features = header["hash_features"]
    offset = header["hash_offset"]
    values = torch.tensor(
        weights[offset : offset + table_size * levels * features],
        dtype=torch.float32,
    ).reshape(levels * table_size, features)
    table = nn.Embedding(levels * table_size, features)
    table.weight.data = values
    table.weight.requires_grad_(False)
    return table


def hash_resolution(level: int) -> int:
    return max(1, int(math.floor(16.0 * (1.3**level))))


def hash_cell(ix: int, iy: int, level: int, table_size: int) -> int:
    hashed = ((ix * 73_856_093) ^ (iy * 19_349_663) ^ (level * 83_492_791)) & 0xFFFFFFFF
    return int(hashed % table_size)


def hash_corners(px: int, py: int, width: int, height: int, level: int, table_size: int) -> tuple[list[int], list[float]]:
    u = px / max(width - 1, 1)
    v = py / max(height - 1, 1)
    res = hash_resolution(level)
    fx = u * max(res - 1, 1)
    fy = v * max(res - 1, 1)
    ix0 = min(int(math.floor(fx)), res - 1)
    iy0 = min(int(math.floor(fy)), res - 1)
    ix1 = min(ix0 + 1, res - 1)
    iy1 = min(iy0 + 1, res - 1)
    tx = fx - ix0
    ty = fy - iy0
    return (
        [
            hash_cell(ix0, iy0, level, table_size),
            hash_cell(ix1, iy0, level, table_size),
            hash_cell(ix0, iy1, level, table_size),
            hash_cell(ix1, iy1, level, table_size),
        ],
        [
            (1.0 - tx) * (1.0 - ty),
            tx * (1.0 - ty),
            (1.0 - tx) * ty,
            tx * ty,
        ],
    )


def pixel_hash_encoding(
    px: int,
    py: int,
    width: int,
    height: int,
    header: dict,
    hash_table: "torch.nn.Embedding",
) -> "torch.Tensor":
    import torch

    indices = []
    weights = []
    for level in range(header["hash_levels"]):
        corners, corner_weights = hash_corners(px, py, width, height, level, header["hash_table_size"])
        indices.extend([level * header["hash_table_size"] + corner for corner in corners])
        weights.extend(corner_weights)
    index_tensor = torch.tensor(indices, dtype=torch.long)
    weight_tensor = torch.tensor(weights, dtype=torch.float32).reshape(header["hash_levels"], 4, 1)
    encoded = hash_table(index_tensor).reshape(header["hash_levels"], 4, header["hash_features"])
    return (encoded * weight_tensor).sum(dim=1).flatten()


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


def load_aux(path: str | None, width: int, height: int) -> np.ndarray | None:
    if not path or not Path(path).exists():
        return None
    aux_data = json.loads(Path(path).read_text(encoding="utf-8"))
    feature_count = int(aux_data.get("feature_count", 4))
    aux = np.array(aux_data["features"], dtype=np.float32).reshape(
        int(aux_data.get("height", height)),
        int(aux_data.get("width", width)),
        feature_count,
    )
    if feature_count < 10:
        padded = np.zeros((aux.shape[0], aux.shape[1], 10), dtype=np.float32)
        padded[:, :, :feature_count] = aux
        aux = padded
    return aux


def aux_features(px: int, py: int, aux: np.ndarray | None) -> "torch.Tensor":
    import torch

    if aux is None or py >= aux.shape[0] or px >= aux.shape[1]:
        return torch.zeros(10, dtype=torch.float32)
    return torch.tensor(aux[py, px, :10], dtype=torch.float32)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for inverse optimization.") from exc

    header, weights = load_weights(Path(args.checkpoint))
    model = build_mlp(header, weights)
    hash_table = build_hash_table(header, weights)
    model.eval()

    width, height = args.width, args.height
    if args.checkpoint_meta:
        meta = json.loads(Path(args.checkpoint_meta).read_text(encoding="utf-8"))
        width = int(meta.get("width", width))
        height = int(meta.get("height", height))

    target = load_target(args, width, height)
    aux = load_aux(args.aux, width, height)
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
            hash_features = pixel_hash_encoding(px, py, width, height, header, hash_table)
            light_features = torch.stack(
                [
                    torch.tensor(u, dtype=torch.float32),
                    torch.tensor(v, dtype=torch.float32),
                    position[0],
                    position[1],
                    position[2],
                    radius,
                    color[0],
                    color[1],
                    color[2],
                    log_intensity,
                ]
            )
            features = torch.cat(
                [
                    hash_features,
                    light_features,
                    aux_features(px, py, aux),
                ]
            )
            pred = torch.expm1(model(features)).clamp_min(0.0)
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
