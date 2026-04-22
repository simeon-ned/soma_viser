# Soma Viser

`soma_viser` is a web-based SOMA motion viewer built on [Viser](https://github.com/nerfstudio-project/viser).
It is designed as a practical visualization companion: load BVH motions, inspect skeleton/mesh behavior, debug joint frames, and validate trajectories before or after retargeting. It can also be reused as a building block in your own Viser-based applications.

## What It Provides

- Interactive BVH playback with scrubber, play/pause, loop, and stepped speed controls
- SOMA skeleton rendering
- Optional SOMA skinned mesh rendering
- Grouped joint-frame inspector (`Body / Head / Arms / Hands / Legs / Other`)
- In-scene frame labels with compact/full text modes
- Root/scale/alignment controls for quick pose inspection

## Installation

Recommended setup uses `mamba` + `pip`:

```bash

mamba create -n soma-viser python=3.12 -y
mamba activate soma-viser
pip install -e .
```

If your shell does not auto-enable `mamba activate`, initialize it once:

```bash
mamba init
```

## Run

Default (uses `soma_viser/motions`):

```bash
soma-viser
```

Custom motions directory and port:

```bash
soma-viser --motions-dir /path/to/motions --port 8090
```

---

## UI Overview

<!-- Place your screen capture under assets/, for example:
     assets/ui_overview.gif or assets/ui_overview.mp4 -->
![Soma Viser UI Overview](assets/ui_overview.gif)

### Motion tab

- select trajectory (`.bvh`)
- frame timeline + playback controls
- playback state information (clip/frame/speed/FPS/mesh status)

### Visualization tab

- skeleton visibility, color, line width
- mesh visibility, color, opacity
- joint frame inspector with grouped categories
- scene text toggles for frame labels

### Controls tab

- root offset (`x`, `y`, `z`)
- body scale
- alignment rotation (`x`, `y`, `z` in degrees)

---

## Motion Data

- This repository includes reference SOMA BVH motions in `motions/` for immediate testing.
- For large-scale motion data, use the [SEED dataset](https://huggingface.co/datasets/bones-studio/seed) (Skeletal Everyday Embodiment Dataset) by [Bones Studio](https://huggingface.co/bones-studio).
- SEED provides large-scale motions on the SOMA uniform-proportion skeleton, and included G1 robot motion data was retargeted using SOMA Retargeter.

## References
`soma_viser` borrows playback/UI ideas from `mjviser` and adapts them for SOMA motion inspection.

- [Viser](https://github.com/viser-project/viser) - Core web visualization framework used by this project
- [mujocolab/mjviser](https://github.com/mujocolab/mjviser/tree/main) - Reference project for tab/UI and playback interaction patterns
- [NVlabs/SOMA-X](https://github.com/NVlabs/SOMA-X) - SOMA body model ecosystem and source skeleton context
- [NVIDIA/soma-retargeter](https://github.com/NVIDIA/soma-retargeter) - SOMA BVH retargeting pipeline and mesh/skeleton tooling references
