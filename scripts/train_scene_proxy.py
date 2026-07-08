#!/usr/bin/env python3
"""Train a Neural Render Proxy (NRP) from recorded Krust path vertices.

Paper reference: Sancho et al. 2026 — path-based training with on-the-fly
segment gather targets, auxiliary features, 2D hashgrid pixel encoding, and
relative HDR MSE loss. Exports a compact .bin for WGSL inference.
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
    parser.add_argument("--paths", required=True, help="Path JSONL from record_paths")
    parser.add_argument("--aux", help="Optional JSON with width/height and aux arrays")
    parser.add_argument("--output", required=True, help="Output .bin weights")
    parser.add_argument("--virtual-lights-file", help="JSON file with virtual_lights")
    parser.add_argument("--max-records", type=int, default=200_000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
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
    return [{"position": [0.0, 5.0, 0.0], "color": [1.0, 1.0, 1.0], "intensity": 4.0, "radius": 0.5}]


def load_lights(args: argparse.Namespace) -> list[dict]:
    if args.virtual_lights_file:
        value = json.loads(Path(args.virtual_lights_file).read_text(encoding="utf-8"))
        lights = value.get("virtual_lights", [])
        if lights:
            return lights
    return default_lights()


def hash_grid(px: float, py: float, level: int) -> np.ndarray:
    scale = 2**level
    fx = px * scale
    fy = py * scale
    h = np.sin(np.array([fx * 12.9898, fy * 78.233])) * 43758.5453
    return np.mod(h, 1.0)


def segment_gather_target(
    records: list[dict],
    light: dict,
    pixel: tuple[int, int],
    width: int,
    height: int,
) -> np.ndarray:
    """Paper GATHERLIGHT: segment intersection with virtual sphere light."""
    px, py = pixel
    contrib = np.zeros(3, dtype=np.float32)
    position = np.array(light["position"], dtype=np.float32)
    color = np.array(light.get("color", [1.0, 1.0, 1.0]), dtype=np.float32)
    intensity = float(light.get("intensity", 1.0))
    radius = float(light.get("radius", 0.5))

    pixel_records = [r for r in records if r["x"] == px and r["y"] == py]
    if not pixel_records:
        return contrib

    by_sample: dict[int, list[dict]] = {}
    for record in pixel_records:
        by_sample.setdefault(int(record["sample"]), []).append(record)
    sample_count = max(len(by_sample), 1)

    for sample_records in by_sample.values():
        sample_records.sort(key=lambda r: int(r["depth"]))
        for i, record in enumerate(sample_records):
            p0 = np.array(record["position"], dtype=np.float32)
            throughput = np.array(record["throughput"], dtype=np.float32)
            if i + 1 < len(sample_records):
                p1 = np.array(sample_records[i + 1]["position"], dtype=np.float32)
            else:
                outgoing = np.array(record.get("outgoing", [0.0, 1.0, 0.0]), dtype=np.float32)
                norm = np.linalg.norm(outgoing)
                if norm < 1e-6:
                    continue
                p1 = p0 + outgoing / norm * 1000.0

            d = p1 - p0
            f = p0 - position
            a = np.dot(d, d)
            b = 2.0 * np.dot(f, d)
            c = np.dot(f, f) - radius * radius
            disc = b * b - 4.0 * a * c
            if disc >= 0.0 and a > 1e-8:
                s = math.sqrt(disc)
                t0 = (-b - s) / (2.0 * a)
                t1 = (-b + s) / (2.0 * a)
                if (0.0 <= t0 <= 1.0) or (0.0 <= t1 <= 1.0) or (t0 < 0.0 and t1 > 1.0):
                    contrib += throughput * color * intensity

    return contrib / sample_count


def rel_mse_loss(pred: "torch.Tensor", target: "torch.Tensor", eps: float = 0.01) -> "torch.Tensor":
    import torch

    num = (pred - target) ** 2
    den = target.detach() ** 2 + eps
    return torch.mean(num / den)


def export_bin(
    path: Path,
    model: "torch.nn.Module",
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
) -> None:
    import torch

    weights: list[float] = [
        float(input_dim),
        float(hidden_dim),
        float(output_dim),
        float(num_layers),
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    for param in model.parameters():
        weights.extend(param.detach().cpu().flatten().tolist())

    payload = struct.pack(f"{len(weights)}f", *weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_features(
    px: int,
    py: int,
    width: int,
    height: int,
    aux: np.ndarray | None,
    light: dict,
) -> np.ndarray:
    u = px / max(width - 1, 1)
    v = py / max(height - 1, 1)
    if aux is not None and py < aux.shape[0] and px < aux.shape[1]:
        albedo = aux[py, px, :3]
        depth = aux[py, px, 3]
    else:
        albedo = np.zeros(3, dtype=np.float32)
        depth = 0.0
    hash_feat = hash_grid(u, v, 0)
    return np.array(
        [
            u,
            v,
            albedo[0],
            albedo[1],
            albedo[2],
            depth,
            0.0,
            0.0,
            0.0,
            light["position"][0],
            light["position"][1],
            light["position"][2],
            float(light.get("radius", 0.5)),
            light["color"][0],
            light["color"][1],
            light["color"][2],
            math.log(max(float(light.get("intensity", 1.0)), 1e-4)),
            hash_feat[0],
            hash_feat[1],
            hash_feat[2],
        ],
        dtype=np.float32,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for NRP training.") from exc

    records = load_paths(Path(args.paths), args.max_records)
    lights = load_lights(args)
    width, height = args.width, args.height

    aux = None
    if args.aux and Path(args.aux).exists():
        aux_data = json.loads(Path(args.aux).read_text(encoding="utf-8"))
        width = int(aux_data.get("width", width))
        height = int(aux_data.get("height", height))
        aux = np.array(aux_data["features"], dtype=np.float32).reshape(height, width, 4)

    pixels = [(x, y) for y in range(height) for x in range(width)]
    random.shuffle(pixels)
    pixels = pixels[: min(len(pixels), args.max_records // max(len(lights), 1))]

    features = []
    targets = []
    for px, py in pixels:
        light = random.choice(lights)
        features.append(build_features(px, py, width, height, aux, light))
        targets.append(segment_gather_target(records, light, (px, py), width, height))

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    input_dim = x.shape[1]
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers

    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.extend([nn.Linear(hidden_dim, 3), nn.Softplus()])
    model = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        permutation = torch.randperm(x.shape[0])
        total_loss = 0.0
        for start in range(0, x.shape[0], args.batch_size):
            batch = permutation[start : start + args.batch_size]
            pred = model(x[batch])
            loss = rel_mse_loss(pred, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.numel()
        print(f"epoch {epoch + 1}: rel_mse={total_loss / x.shape[0]:.8f}")

    export_bin(
        Path(args.output),
        model,
        input_dim,
        hidden_dim,
        3,
        num_layers,
    )
    meta_path = Path(args.output).with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": 3,
                "num_layers": num_layers,
                "width": width,
                "height": height,
                "lights": lights,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output} and {meta_path}")


if __name__ == "__main__":
    main()
