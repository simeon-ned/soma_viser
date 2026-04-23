## Pane Reuse API

This document defines the stable API surface intended for downstream integrations (for example `humo_target`).

### Panes

- `MotionPane` in `soma_viser.panes.motion_pane`
- `VisualizationPane` in `soma_viser.panes.visualization_pane`
- `ControlsPane` in `soma_viser.panes.controls_pane`

Each pane exposes the same lifecycle contract:

- `attach(gui_tab, server, session_state)`  
  Build pane-owned GUI controls and register callbacks.
- `tick(dt)`  
  Advance pane-local runtime work (debounce/poll/update hooks).
- `dispose()`  
  Release pane resources if needed.

### Current Session Model

- `SessionState` in `soma_viser.state.session`
  - `playback: PlaybackState`
  - `visual: VisualizationState`
  - `motion: MotionLibraryState`
- `PaneContext` in `soma_viser.panes.context`
  - shared dependencies passed to panes (`viewer`, `session`)

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
