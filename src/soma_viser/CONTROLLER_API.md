## Controller Reuse API

This document defines the stable API surface intended for downstream integrations (for example `humo_target`).

### Controllers

- `MotionController` in `soma_viser.controllers.motion_controller`
- `VisualizationController` in `soma_viser.controllers.visualization_controller`
- `ControlsController` in `soma_viser.controllers.controls_controller`

Each controller exposes the same lifecycle contract:

- `attach(gui_tab, server, session_state)`  
  Build controller-owned GUI controls and register callbacks.
- `tick(dt)`  
  Advance controller-local runtime work (debounce/poll/update hooks).
- `dispose()`  
  Release controller resources if needed.

### Current Session Model

- `PlaybackState` in `soma_viser.state.session`
  - `playing`, `loop`, `speed_idx`, `accumulator`, `frame_idx`, `needs_redraw`

### Render Pipeline

- `RenderPipeline` in `soma_viser.render.pipeline`
  - `redraw(frame_idx)`
  - Runs the full draw pass:
    - apply joint overrides
    - compute global transforms
    - update skeleton lines
    - update inspector frames/labels
    - update skinned mesh

### BVH I/O Helpers

- `parse_bvh_channel_map`
- `extract_bvh_motion_rows`
- `estimate_bvh_units_per_meter`

from `soma_viser.io_bvh`.

These remain utility-level and can be reused by export and future retarget flows.
