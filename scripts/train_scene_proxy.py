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
import time
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
    parser.add_argument("--hash-levels", type=int, default=16)
    parser.add_argument("--hash-features", type=int, default=2)
    parser.add_argument("--hash-table-size", type=int, default=131_072)
    parser.add_argument("--random-lights", type=int, default=16)
    parser.add_argument("--lights-per-pixel", type=int, default=2)
    parser.add_argument("--max-training-samples", type=int, default=300_000)
    parser.add_argument("--target-denoise-radius", type=int, default=3)
    parser.add_argument("--target-denoise-passes", type=int, default=2)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def load_paths(path: Path, limit: int) -> list[dict]:
    records: list[dict] = []
    seen = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            seen += 1
            if limit <= 0 or len(records) < limit:
                records.append(record)
            else:
                slot = random.randrange(seen)
                if slot < limit:
                    records[slot] = record
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


def renderer_gather_target(
    pixel_records: list[dict],
    light: dict,
) -> np.ndarray:
    """Paper-style GATHERLIGHT target: add emission when recorded path segments hit virtual lights."""
    contrib = np.zeros(3, dtype=np.float32)
    position = np.array(light["position"], dtype=np.float32)
    color = np.array(light.get("color", [1.0, 1.0, 1.0]), dtype=np.float32)
    intensity = float(light.get("intensity", 1.0))
    radius = float(light.get("radius", 0.5))

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


def expand_training_lights(records: list[dict], lights: list[dict], random_count: int) -> list[dict]:
    training_lights = list(lights)
    if random_count <= 0 or not records:
        return training_lights

    first_hit_positions = np.array(
        [record["position"] for record in records if int(record.get("depth", 0)) == 0],
        dtype=np.float32,
    )
    if first_hit_positions.size == 0:
        first_hit_positions = np.array([record["position"] for record in records], dtype=np.float32)
    base_lights = lights or default_lights()

    for _ in range(random_count):
        base = random.choice(base_lights)
        color = np.array(base.get("color", [1.0, 1.0, 1.0]), dtype=np.float32)
        jitter = np.random.uniform(0.75, 1.25, size=3).astype(np.float32)
        sampled_color = np.clip(color * jitter, 0.05, 1.0)
        intensity = float(base.get("intensity", 1.0)) * float(np.exp(np.random.uniform(-0.75, 0.75)))
        radius = float(base.get("radius", 0.5)) * float(np.exp(np.random.uniform(-0.35, 0.35)))
        anchor = first_hit_positions[np.random.randint(0, len(first_hit_positions))]
        offset = np.array(
            [
                np.random.uniform(-2.5, 2.5),
                np.random.uniform(1.5, 5.0),
                np.random.uniform(-2.5, 2.5),
            ],
            dtype=np.float32,
        )
        position = [
            float(anchor[0] + offset[0]),
            float(anchor[1] + offset[1]),
            float(anchor[2] + offset[2]),
        ]
        training_lights.append(
            {
                "position": position,
                "color": sampled_color.tolist(),
                "intensity": max(intensity, 0.1),
                "radius": max(radius, 0.15),
            }
        )

    return training_lights


def records_by_pixel(records: list[dict]) -> dict[tuple[int, int], list[dict]]:
    by_pixel: dict[tuple[int, int], list[dict]] = {}
    for record in records:
        by_pixel.setdefault((int(record["x"]), int(record["y"])), []).append(record)
    return by_pixel


def box_blur_image(image: np.ndarray, radius: int, passes: int) -> np.ndarray:
    if radius <= 0 or passes <= 0:
        return image
    result = image.astype(np.float32, copy=True)
    kernel = 2 * radius + 1
    for _ in range(passes):
        padded = np.pad(result, ((0, 0), (radius, radius), (0, 0)), mode="edge")
        cumsum = np.cumsum(padded, axis=1, dtype=np.float32)
        cumsum = np.pad(cumsum, ((0, 0), (1, 0), (0, 0)), mode="constant")
        result = (cumsum[:, kernel:, :] - cumsum[:, :-kernel, :]) / kernel

        padded = np.pad(result, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        cumsum = np.cumsum(padded, axis=0, dtype=np.float32)
        cumsum = np.pad(cumsum, ((1, 0), (0, 0), (0, 0)), mode="constant")
        result = (cumsum[kernel:, :, :] - cumsum[:-kernel, :, :]) / kernel
    return result


def build_denoised_target_images(
    records_index: dict[tuple[int, int], list[dict]],
    lights: list[dict],
    width: int,
    height: int,
    radius: int,
    passes: int,
    aux: np.ndarray | None,
) -> np.ndarray:
    targets = np.zeros((len(lights), height, width, 3), dtype=np.float32)
    if not lights:
        return targets

    use_direct_target = aux is not None and aux.shape[2] >= 10
    target_name = "direct-light" if use_direct_target else "segment-gather"
    print(
        f"[NRP train] reconstructing {len(lights)} {target_name} target image(s)",
        flush=True,
    )
    for light_index, light in enumerate(lights):
        image = targets[light_index]
        if use_direct_target:
            for py in range(height):
                for px in range(width):
                    image[py, px] = direct_light_target(aux[py, px], light)
        else:
            for (px, py), pixel_records in records_index.items():
                if px < width and py < height:
                    image[py, px] = renderer_gather_target(pixel_records, light)
        nonzero = int(np.count_nonzero(np.sum(image, axis=2) > 0.0))
        if not use_direct_target and radius > 0 and passes > 0:
            targets[light_index] = box_blur_image(image, radius, passes)
        print(
            f"[NRP train] target {light_index + 1:02d}/{len(lights):02d} "
            f"nonzero_pixels={nonzero:,} denoise_radius={0 if use_direct_target else radius} "
            f"passes={0 if use_direct_target else passes}",
            flush=True,
        )
    return targets


def direct_light_target(aux_pixel: np.ndarray, light: dict) -> np.ndarray:
    albedo = aux_pixel[:3].astype(np.float32)
    depth_log = float(aux_pixel[3])
    normal = aux_pixel[4:7].astype(np.float32)
    position = aux_pixel[7:10].astype(np.float32)
    roughness = 0.5
    metallic = 0.0
    specular_weight = 0.0
    diffuse_weight = 1.0
    specular_color = np.ones(3, dtype=np.float32)
    if aux_pixel.shape[0] >= 14:
        roughness = float(np.clip(aux_pixel[10], 0.02, 1.0))
        metallic = float(np.clip(aux_pixel[11], 0.0, 1.0))
        specular_weight = float(np.clip(aux_pixel[12], 0.0, 1.0))
        diffuse_weight = float(np.clip(aux_pixel[13] - metallic, 0.0, 1.0))
    if aux_pixel.shape[0] >= 20:
        specular_color = aux_pixel[17:20].astype(np.float32)
    if depth_log <= 0.0 or not np.isfinite(position).all():
        return np.zeros(3, dtype=np.float32)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-6:
        return np.zeros(3, dtype=np.float32)
    normal /= normal_len

    light_position = np.array(light["position"], dtype=np.float32)
    light_vec = light_position - position
    dist2 = max(float(np.dot(light_vec, light_vec)), 1e-4)
    light_dir = light_vec / math.sqrt(dist2)
    ndotl = max(float(np.dot(normal, light_dir)), 0.0)

    if aux_pixel.shape[0] >= 17:
        view_dir = aux_pixel[14:17].astype(np.float32)
        view_len = float(np.linalg.norm(view_dir))
        if view_len > 1e-4:
            view_dir = view_dir / view_len
        else:
            view_len = max(float(np.linalg.norm(position)), 1e-4)
            view_dir = -position / view_len
    else:
        view_len = max(float(np.linalg.norm(position)), 1e-4)
        view_dir = -position / view_len
    if ndotl <= 0.0:
        return np.zeros(3, dtype=np.float32)

    half_dir = light_dir + view_dir
    half_len = max(float(np.linalg.norm(half_dir)), 1e-4)
    half_dir = half_dir / half_len
    spec_power = ((1.0 - roughness) ** 4.0) * 1000.0 + 3.5
    dielectric_f0 = np.clip(specular_color, 0.0, 1.0) * (0.04 * specular_weight)
    f0 = (1.0 - metallic) * dielectric_f0 + metallic * albedo
    hv = max(float(np.dot(half_dir, view_dir)), 0.0)
    fresnel = f0 + (1.0 - f0) * ((1.0 - min(hv, 1.0)) ** 5.0)
    specular = fresnel * (max(float(np.dot(normal, half_dir)), 0.0) ** spec_power) * ndotl
    diffuse = albedo * diffuse_weight * ndotl

    color = np.array(light.get("color", [1.0, 1.0, 1.0]), dtype=np.float32)
    intensity = float(light.get("intensity", 1.0))
    radius = max(float(light.get("radius", 0.5)), 0.01)
    radiance = (diffuse + specular) * color * intensity * (radius * radius / dist2)
    return np.minimum(radiance, 32.0).astype(np.float32)


def log_l1_mse_loss(pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    import torch

    return torch.mean((pred - torch.log1p(target)) ** 2)


def export_bin(
    path: Path,
    model: "torch.nn.Module",
    hash_table: "torch.nn.Embedding",
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    hash_levels: int,
    hash_features: int,
    hash_table_size: int,
) -> None:
    import torch

    mlp_weights: list[float] = []
    for param in model.parameters():
        mlp_weights.extend(param.detach().cpu().flatten().tolist())
    hash_offset = 8 + len(mlp_weights)

    weights: list[float] = [
        float(input_dim),
        float(hidden_dim),
        float(output_dim),
        float(num_layers),
        float(hash_table_size),
        float(hash_levels),
        float(hash_features),
        float(hash_offset),
    ]
    weights.extend(mlp_weights)
    weights.extend(hash_table.weight.detach().cpu().flatten().tolist())

    payload = struct.pack(f"{len(weights)}f", *weights)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def build_hash_indices(
    px: int,
    py: int,
    width: int,
    height: int,
    hash_levels: int,
    hash_table_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices: list[int] = []
    weights: list[float] = []
    for level in range(hash_levels):
        corners, corner_weights = hash_corners(px, py, width, height, level, hash_table_size)
        indices.extend([level * hash_table_size + corner for corner in corners])
        weights.extend(corner_weights)
    return np.array(indices, dtype=np.int64), np.array(weights, dtype=np.float32)


def build_light_features(
    px: int,
    py: int,
    width: int,
    height: int,
    light: dict,
) -> np.ndarray:
    u = px / max(width - 1, 1)
    v = py / max(height - 1, 1)
    return np.array(
        [
            u,
            v,
            light["position"][0],
            light["position"][1],
            light["position"][2],
            float(light.get("radius", 0.5)),
            light["color"][0],
            light["color"][1],
            light["color"][2],
            math.log(max(float(light.get("intensity", 1.0)), 1e-4)),
        ],
        dtype=np.float32,
    )


def build_aux_features(px: int, py: int, aux: np.ndarray | None) -> np.ndarray:
    if aux is None or py >= aux.shape[0] or px >= aux.shape[1]:
        features = np.zeros(20, dtype=np.float32)
        features[13] = 1.0
        features[17:20] = 1.0
        return features
    features = aux[py, px, : min(aux.shape[2], 20)].astype(np.float32)
    if features.shape[0] < 20:
        padded = np.zeros(20, dtype=np.float32)
        padded[: features.shape[0]] = features
        padded[13] = 1.0
        padded[17:20] = 1.0
        return padded
    return features


def main() -> None:
    args = parse_args()
    started = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise SystemExit("PyTorch is required for NRP training.") from exc

    records = load_paths(Path(args.paths), args.max_records)
    records_index = records_by_pixel(records)
    lights = load_lights(args)
    training_lights = expand_training_lights(records, lights, args.random_lights)
    width, height = args.width, args.height
    print(
        f"[NRP train] loaded {len(records):,} path vertices, "
        f"{len(lights)} scene light(s), {len(training_lights)} training light(s), target={args.output}",
        flush=True,
    )

    aux = None
    if args.aux and Path(args.aux).exists():
        aux_data = json.loads(Path(args.aux).read_text(encoding="utf-8"))
        width = int(aux_data.get("width", width))
        height = int(aux_data.get("height", height))
        feature_count = int(aux_data.get("feature_count", 4))
        aux = np.array(aux_data["features"], dtype=np.float32).reshape(height, width, feature_count)
        if feature_count < 20:
            padded = np.zeros((height, width, 20), dtype=np.float32)
            padded[:, :, :feature_count] = aux
            padded[:, :, 13] = 1.0
            padded[:, :, 17:20] = 1.0
            aux = padded

    pixels = list(records_index.keys()) or [(x, y) for y in range(height) for x in range(width)]
    random.shuffle(pixels)
    max_pixels = max(args.max_training_samples // max(args.lights_per_pixel, 1), 1)
    pixels = pixels[: min(len(pixels), max_pixels)]
    target_images = build_denoised_target_images(
        records_index,
        training_lights,
        width,
        height,
        args.target_denoise_radius,
        args.target_denoise_passes,
        aux,
    )

    hash_indices = []
    hash_weights = []
    dense_features = []
    targets = []
    for px, py in pixels:
        indices, weights = build_hash_indices(
            px,
            py,
            width,
            height,
            args.hash_levels,
            args.hash_table_size,
        )
        for _ in range(max(args.lights_per_pixel, 1)):
            light_index = random.randrange(len(training_lights))
            light = training_lights[light_index]
            hash_indices.append(indices)
            hash_weights.append(weights)
            dense_features.append(
                np.concatenate(
                    [
                        build_light_features(px, py, width, height, light),
                        build_aux_features(px, py, aux),
                    ]
                )
            )
            targets.append(target_images[light_index, py, px])
    if not dense_features:
        raise SystemExit("No NRP training samples were generated. Increase path_spp/max_records or verify the path file.")
    input_dim = args.hash_levels * args.hash_features + len(dense_features[0])
    print(
        f"[NRP train] training pixels={len(dense_features):,}, feature_dim={input_dim}, "
        f"hash={args.hash_levels}x{args.hash_features}@{args.hash_table_size}",
        flush=True,
    )

    hash_x = torch.from_numpy(np.asarray(hash_indices, dtype=np.int64))
    hash_w = torch.from_numpy(np.asarray(hash_weights, dtype=np.float32))
    dense_x = torch.from_numpy(np.asarray(dense_features, dtype=np.float32))
    y = torch.from_numpy(np.asarray(targets, dtype=np.float32))
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers

    hash_table = nn.Embedding(args.hash_levels * args.hash_table_size, args.hash_features)
    nn.init.uniform_(hash_table.weight, -0.001, 0.001)
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(num_layers - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    layers.append(nn.Linear(hidden_dim, 3))
    model = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(hash_table.parameters()),
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        permutation = torch.randperm(dense_x.shape[0])
        total_loss = 0.0
        for start in range(0, dense_x.shape[0], args.batch_size):
            batch = permutation[start : start + args.batch_size]
            encoded = (
                hash_table(hash_x[batch])
                .reshape(batch.numel(), args.hash_levels, 4, args.hash_features)
                * hash_w[batch].reshape(batch.numel(), args.hash_levels, 4, 1)
            ).sum(dim=2).reshape(batch.numel(), -1)
            pred = model(torch.cat([encoded, dense_x[batch]], dim=1))
            loss = log_l1_mse_loss(pred, y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch.numel()
        progress = (epoch + 1) / args.epochs
        filled = int(progress * 24)
        bar = "#" * filled + "-" * (24 - filled)
        elapsed = time.time() - started
        print(
            f"[NRP train] epoch {epoch + 1:03d}/{args.epochs:03d} [{bar}] {progress * 100:5.1f}% "
            f"log_mse={total_loss / dense_x.shape[0]:.8f} elapsed={elapsed:0.1f}s",
            flush=True,
        )

    export_bin(
        Path(args.output),
        model,
        hash_table,
        input_dim,
        hidden_dim,
        3,
        num_layers,
        args.hash_levels,
        args.hash_features,
        args.hash_table_size,
    )
    with torch.no_grad():
        preview_count = min(1024, dense_x.shape[0])
        encoded = (
            hash_table(hash_x[:preview_count])
            .reshape(preview_count, args.hash_levels, 4, args.hash_features)
            * hash_w[:preview_count].reshape(preview_count, args.hash_levels, 4, 1)
        ).sum(dim=2).reshape(preview_count, -1)
        preview = torch.expm1(model(torch.cat([encoded, dense_x[:preview_count]], dim=1))).clamp_min(0.0)
        print(
            "[NRP train] preview radiance "
            f"mean={preview.mean().item():.6f} max={preview.max().item():.6f}",
            flush=True,
        )
    meta_path = Path(args.output).with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "output_dim": 3,
                "num_layers": num_layers,
                "hash_levels": args.hash_levels,
                "hash_features": args.hash_features,
                "hash_table_size": args.hash_table_size,
                "width": width,
                "height": height,
                "lights": lights,
                "training_light_count": len(training_lights),
                "random_lights": args.random_lights,
                "lights_per_pixel": args.lights_per_pixel,
                "max_training_samples": args.max_training_samples,
                "aux_feature_count": 20 if aux is not None else 0,
                "target_space": "log1p_radiance",
                "target_model": "direct_material_specular_v8" if aux is not None and aux.shape[2] >= 20 else "segment_gather_denoised_v3",
                "hash_interpolation": "bilinear",
                "target_denoise_radius": args.target_denoise_radius,
                "target_denoise_passes": args.target_denoise_passes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {args.output} and {meta_path}")


if __name__ == "__main__":
    main()
