# Soma Viser

`soma_viser` is a web-based SOMA motion viewer built on [Viser](https://github.com/nerfstudio-project/viser).
The primary goal of this project is integration into Viser-based pipelines, where it acts as an interactive inspection/debug stage for SOMA trajectories before and after retargeting. It can also run as a standalone viewer for day-to-day motion exploration. The repository includes reference SOMA BVH motions in `motions/` for immediate testing, and for large-scale workflows it is designed to pair with the [SEED dataset](https://huggingface.co/datasets/bones-studio/seed) (Skeletal Everyday Embodiment Dataset) by [Bones Studio](https://huggingface.co/bones-studio).

## What It Provides

- Interactive BVH playback with scrubber, play/pause, loop, and stepped speed controls
- Trajectory search and folder-based filtering for large motion libraries
- SOMA skeleton rendering
- Optional SOMA skinned mesh rendering
- Grouped joint-frame inspector (`Body / Head / Arms / Hands / Legs / Other`)
- In-scene frame labels with compact/full text modes
- Root/scale/alignment controls for quick pose inspection
- Joint controls (main joints, non-fingers) with gizmo-based adjustments and BVH export

## Installation

You may install `soma_viser` as regular python package, following setup uses `mamba` + `pip`, but you may use `uv` as well:

```bash

mamba create -n soma-viser python=3.12 -y
mamba activate soma-viser
pip install -e .
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
<p align="center">
  <video src="https://github.com/user-attachments/assets/a38334bd-b772-4696-bb48-14fecac844ef.mp4" width="100%" autoplay loop muted playsinline></video>
</p>

### Motion tab

- select trajectory (`.bvh`)
- search/filter trajectories
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
- joint controls and joint gizmos (main joints, no fingers)

---

## Architecture Map

Code is organized around three main module groups:

- `soma_viser/panes/` - UI/presentation layer (tab construction, callbacks, inspector widgets)
- `soma_viser/core/` - non-UI logic (motion loading/parsing/export, playback/state, math helpers, joint override logic)
- `soma_viser/render/` - rendering layer (frame pipeline plus skeleton/mesh drawing)

Guideline used in this project: 

- If code manipulates GUI widgets or scene labels directly, it belongs in `panes`.
- If code is reusable logic without UI handles, it belongs in `core`.
- If code computes or applies renderable geometry/transforms, it belongs in `render`.

## References
`soma_viser` borrows playback/UI ideas from `mjviser` and adapts them for SOMA motion inspection.

- [Viser](https://github.com/viser-project/viser) - Core web visualization framework used by this project
- [mujocolab/mjviser](https://github.com/mujocolab/mjviser/tree/main) - Reference project for tab/UI and playback interaction patterns
- [NVlabs/SOMA-X](https://github.com/NVlabs/SOMA-X) - SOMA body model ecosystem and source skeleton context
- [NVIDIA/soma-retargeter](https://github.com/NVIDIA/soma-retargeter) - SOMA BVH retargeting pipeline and mesh/skeleton tooling references
