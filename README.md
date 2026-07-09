# Krust Renderer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
![Example diffuse render](img/crocodiles_example.png)
![Example render showcasing normal maps and ggx](img/bust_example.png)
![Simple sphere](img/simple_sphere.png)


## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Acknowledgements](#acknowledgements)
- [License](#license)


## Overview <a name="overview"></a>
This project showcases a simple raytracer written in Rust. Though relatively naive, the renderer is capable of producing very appealing results in a reasonable timeframe. Multiple importance sampling has been utilized to help converge more efficiently, and the BVH implementation allows for relatively quick scene traversals. GGX sampling is used for the specular response, while the principled material allows for blending of different shading techniques to create varied and realistic surfaces with ease. 

Future improvements currently in development:
- Subdivision (catclark and adaptive)
- Subsurface scattering (diffusion and randomwalk)
- Volumes
- Radiance caching
- More robust integration with Maya


## Installation <a name="installation"></a>
To run this project, you will need Rust installed and the following dependencies:

- rand = "0.8.3"
- image = "0.24.5"
- indicatif = "0.17.1"
- serde_json = "1.0"
- show-image = "0.13.1"
- rayon = "1.5.1"
- num_cpus = "1.14.0"

## Usage <a name="usage"></a>
Scenes can be generated within maya using the provided plugin and scripts in the src/maya directory. A few simple example scenes are available to test as well. To render an example scene simply input the scene file, and output directory into the main function as follows:

```rust
render_scene(
    Some("path to scene file"),
    "path to output directory"
);
 ```

 Provided examples scenes:
 - examples/spheres.json 
 - examples/dog.json 

The binary also accepts a scene file and output directory:

```bash
cargo run -- examples/spheres.json /tmp/krust-output/
```

## Neural Relighting Workflow

Relighting uses the Neural Render Proxy pipeline by default. Exact path gather is kept as
training/reference infrastructure. If NRP weights are missing, the renderer records paths,
launches the Python trainer, writes `scene_proxy.bin`, then runs neural relighting.

1. Run the full automatic path-record, train, and relight flow:

```json
{
  "preview_window": 0,
  "export_exr": 1,
  "gather_relighting": 1,
  "record_paths": 1,
  "path_spp": 4,
  "path_depth": 8,
  "path_output_file": "/tmp/krust-output/paths.jsonl",
  "nrp_aux_file": "/tmp/krust-output/nrp_aux_features.json",
  "nrp_weights_file": "/tmp/krust-output/scene_proxy.bin",
  "relight_output_file": "/tmp/krust-output/nrp_relight.exr",
  "open_relight_editor_after_render": 1,
  "virtual_lights": [
    {
      "position": [0.0, 3.0, 0.0],
      "color": [1.0, 0.8, 0.6],
      "intensity": 12.0
    }
  ]
}
```

The Rust renderer does not link against ML libraries. It shells out to
`scripts/train_scene_proxy.py` and streams the trainer's progress lines. Useful settings:

```json
{
  "auto_train_nrp": 1,
  "retrain_nrp": 0,
  "nrp_train_epochs": 20,
  "nrp_train_max_records": 200000,
  "nrp_train_batch_size": 4096,
  "nrp_hash_levels": 16,
  "nrp_hash_features": 2,
  "nrp_hash_table_size": 131072,
  "nrp_random_lights": 16,
  "nrp_lights_per_pixel": 2,
  "nrp_max_training_samples": 300000,
  "nrp_target_denoise_radius": 3,
  "nrp_target_denoise_passes": 2,
  "nrp_python": "python3",
  "open_relight_editor_after_render": 1
}
```

2. Optimize lights through the trained proxy:

```bash
python3 scripts/optimize_lighting.py \
  --checkpoint /tmp/krust-output/scene_proxy.bin \
  --checkpoint-meta /tmp/krust-output/scene_proxy.json \
  --paths /tmp/krust-output/paths.jsonl \
  --aux /tmp/krust-output/nrp_aux_features.json \
  --target-color 0.6 0.45 0.35 \
  --output /tmp/krust-output/optimized_light.json
```

The renderer accepts the optimizer output through `virtual_lights_file`.

3. Reuse existing weights and optimized lights:

```json
{
  "preview_window": 0,
  "relight_only": 1,
  "gather_relighting": 1,
  "record_paths": 0,
  "auto_train_nrp": 0,
  "relight_output_file": "/tmp/krust-output/nrp_relight.exr",
  "nrp_weights_file": "/tmp/krust-output/scene_proxy.bin",
  "virtual_lights_file": "/tmp/krust-output/optimized_light.json"
}
```

`gather_relighting` is retained as the compatibility switch for "run relighting", but the output
path goes through NRP inference. To emit an exact gather reference for validation, set
`reference_relight_output_file`.

## Interactive Relight Editor

Launch the realtime GPU relighting editor with:

```json
{
  "render_backend": "relight_editor",
  "path_spp": 2,
  "path_depth": 8,
  "record_paths": 0,
  "nrp_weights_file": "/tmp/krust-output/scene_proxy.bin"
}
```

```bash
KRUST_WGPU_BACKEND=metal cargo run -- examples/spheres.json /tmp/krust-output/
```

The editor records paths once into a GPU-resident fp16/rgb9e5 packed cache, renders first-hit
albedo/normal/depth auxiliary features, and starts in NRP mode when weights are provided. Press
`N` to toggle the exact gather reference for validation. Press `I` to run the Python inverse
optimizer and reload optimized lights.

Keyboard shortcuts:
- `1/2/3`: select light
- drag: move selected light
- `+/-`: intensity
- `Tab`: sphere/quad type
- `A`: add light
- `S`: export paths JSONL
- `O`: export lights JSON
- `N`: toggle NRP
- `I`: solve lighting via NRP optimizer

## GPU Backend

The GPU backend is opt-in:

```json
{
  "preview_window": 0,
  "export_exr": 1,
  "render_backend": "gpu"
}
```

`"gpu"` runs an iterative `wgpu` compute path tracer that writes beauty, diffuse, specular,
emission, normal, albedo, roughness, depth, position, and variance data back into the existing
`FrameBuffers` and EXR exporter. `"gpu_aov"` runs only the deterministic first-hit AOV shader,
which is useful when validating camera rays, geometry upload, and material packing.

On Apple Silicon, the backend tries Metal first. You can force that path explicitly:

```bash
KRUST_WGPU_BACKEND=metal cargo run -- examples/spheres.json /tmp/krust-output/
```

If a compatible GPU adapter is unavailable, explicit GPU mode fails fast instead of silently
using the CPU path. The GPU path currently supports flattened sphere/triangle/quad-light geometry,
directional lights,
constant material properties, material texture maps, bump maps, normal maps, metallic/specular
branching, refraction branching, direct quad-light sampling, direct directional-light sampling,
and GPU-side JSONL path-record emission for relighting. Scene parsing, EXR export, KPCN metadata,
path-record JSONL, and relighting remain the same renderer workflows.

The GPU path is intended to match the CPU architecture and exported data contract, but it is not
expected to be bit-for-bit identical because stochastic sampling and shader math differ from the
recursive CPU implementation.

## Acknowledgements <a name="acknowledgements"></a>
This project was inspired by the work of [Shirley et al.](https://raytracing.github.io/)


## License <a name="license"></a>
This project is licensed under the MIT License - see the LICENSE.md file for details.
