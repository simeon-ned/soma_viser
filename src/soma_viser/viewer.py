"""Interactive SOMA BVH viewer with Motion/Visualization/Controls tabs."""

from __future__ import annotations

import time
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import viser

from .controllers.controls_controller import ControlsController
from .controllers.motion_controller import MotionController
from .controllers.visualization_controller import VisualizationController
from .mesh_skinner import WarpMeshSkinner, try_load_soma_skeletal_mesh
from .io_bvh import parse_bvh_channel_map
from .motion import MotionClip, MotionEntry, index_bvh_files, load_motion_clip
from .render.pipeline import RenderPipeline
from .state.session import PlaybackState
from .skeleton import (
  decode_joint_name,
)
from .viz.joint_inspector import JointInspector
from .viz.joints import resolve_hips_joint_name
from .viz.playback import DEFAULT_SPEEDS, advance_playback, update_speed_index
from .viz.status import build_status_html

_SPEEDS = DEFAULT_SPEEDS
_MOTION_MAX_OPTIONS = 300
_MOTION_FOLDER_SHOW_THRESHOLD = 120
_MOTION_SEARCH_DEBOUNCE_SEC = 0.2

POSE_KEYFRAMES: dict[str, Path] = {
  "Calibration (frame0)": Path(__file__).resolve().parent / "assets" / "poses" / "soma_zero.bvh",
  "T pose": Path(__file__).resolve().parent / "assets" / "poses" / "soma_tpose.bvh",
}


class SomaViewer:
  """SOMA skeleton viewer for BVH clips with realtime playback."""

  def __init__(
    self,
    motions_dir: Path,
    port: int = 8080,
    up_axis: str = "+z",
  ) -> None:
    self.motions_dir = motions_dir
    self.server = viser.ViserServer(port=port)
    self.server.scene.set_up_direction(up_axis)
    self.server.scene.add_grid("/grid", plane="xy" if up_axis == "+z" else "xz")
    self._lock = RLock()

    self._clip: MotionClip | None = None
    self._session = PlaybackState(
      playing=True,
      loop=True,
      speed_idx=_SPEEDS.index(1.0),
      accumulator=0.0,
      frame_idx=0,
      needs_redraw=True,
    )
    self._playing = self._session.playing
    self._loop = self._session.loop
    self._speed_idx = self._session.speed_idx
    self._accumulator = self._session.accumulator
    self._frame_idx = self._session.frame_idx
    self._needs_redraw = self._session.needs_redraw

    self._skeleton_color = (160, 220, 100)
    self._line_width = 2.8
    self._mesh_color = (195, 170, 215)
    self._mesh_opacity = 0.45
    self._root_offset = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    self._align_euler = np.array([90.0, 0.0, 90.0], dtype=np.float64)
    self._body_scale = 1.0
    self._show_skeleton = True
    self._show_mesh = True
    self._show_frame_text = True
    self._show_frame_text_full_info = True
    self._joint_names: list[str] = []
    self._joint_name_to_idx: dict[str, int] = {}
    self._hips_joint_name: str | None = None
    self._hips_joint_idx: int = 0
    self._suppress_gui_updates = False
    self._suppress_root_ui = False
    self._root_gizmo: Any | None = None
    self._root_number_controls: tuple[Any, Any, Any] | None = None

    self._line_handle = self.server.scene.add_line_segments(
      "/soma/skeleton",
      np.zeros((1, 2, 3), dtype=np.float32),
      colors=(160, 220, 100),
      line_width=self._line_width,
    )
    self._last_skeleton_segments: np.ndarray | None = None
    self._mesh_handle: Any = None
    self._mesh_skinner: WarpMeshSkinner | None = None
    self._mesh_warn: str | None = None
    self._skeleton_instance: Any = None

    self._status_html: Any = None
    self._frame_slider: Any = None
    self._motion_dropdown: Any = None
    self._motion_search_text: Any = None
    self._motion_folder_dropdown: Any = None
    self._motion_page_text: Any = None
    self._motion_entries: list[MotionEntry] = []
    self._motion_filtered_entries: list[MotionEntry] = []
    self._motion_entry_by_relpath: dict[str, MotionEntry] = {}
    self._motion_folder_counts: dict[str, int] = {}
    self._motion_search_pending = False
    self._motion_search_deadline = 0.0
    self._suppress_motion_ui = False
    self._joint_inspector: JointInspector | None = None
    self._main_joint_controls: dict[str, Any] = {}
    self._main_joint_overrides_deg: dict[str, np.ndarray] = {}
    self._main_joint_display_deg: dict[str, np.ndarray] = {}
    self._main_joint_knobs: dict[str, viser.TransformControlsHandle] = {}
    self._show_joint_knobs = True
    self._suppress_joint_ui = False
    self._bvh_channel_map: dict[str, tuple[int, list[str]]] = {}
    self._last_rendered_row: Any | None = None

    self._motion_controller = MotionController(self, POSE_KEYFRAMES)
    self._visualization_controller = VisualizationController(self)
    self._controls_controller = ControlsController(self)
    self._render_pipeline = RenderPipeline(self)

    self._build_gui()
    self._load_initial_clip()

  @property
  def _speed(self) -> float:
    return _SPEEDS[self._speed_idx]

  def _build_gui(self) -> None:
    tabs = self.server.gui.add_tab_group()
    with tabs.add_tab("Motion", icon=viser.Icon.PLAYER_PLAY):
      self._motion_controller.attach(gui_tab="Motion", server=self.server, session_state=self)
    with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
      self._visualization_controller.attach(gui_tab="Visualization", server=self.server, session_state=self)
    with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
      self._controls_controller.attach(gui_tab="Controls", server=self.server, session_state=self)

  def _build_motion_tab(self) -> None:
    self._motion_controller.build_tab()

  def _refresh_motion_library(self) -> None:
    self._motion_controller.refresh_motion_library()

  def _apply_motion_filters(self) -> None:
    self._motion_controller.apply_motion_filters()

  def _build_visualization_tab(self) -> None:
    self._visualization_controller.build_tab()

  def _build_controls_tab(self) -> None:
    self._controls_controller.build_tab()

  def _load_initial_clip(self) -> None:
    if not self._motion_entries:
      self._refresh_motion_library()
    if not self._motion_entries:
      self._update_status_text("No BVH clips found.")
      return
    self._load_clip_by_path(self._motion_entries[0].path)

  def _load_clip_by_relpath(self, rel_path: str) -> None:
    entry = self._motion_entry_by_relpath.get(rel_path)
    if entry is None:
      candidate = self.motions_dir / rel_path
      if candidate.is_file():
        self._load_clip_by_path(candidate)
      else:
        self._update_status_text(f"Clip not found: {rel_path}")
      return
    self._load_clip_by_path(entry.path)

  def _load_clip_by_path(self, path: Path) -> None:
    clip = load_motion_clip(path)
    try:
      clip.name = path.relative_to(self.motions_dir).as_posix()
    except ValueError:
      clip.name = path.name
    self._clip = clip
    self._frame_idx = 0
    self._accumulator = 0.0
    self._needs_redraw = True
    self._last_skeleton_segments = None
    self._joint_names = [decode_joint_name(clip.skeleton.joint_names[i]) for i in range(clip.skeleton.num_joints)]
    self._joint_name_to_idx = {name: i for i, name in enumerate(self._joint_names)}
    self._hips_joint_name = resolve_hips_joint_name(self._joint_names)
    self._hips_joint_idx = int(self._joint_name_to_idx.get(self._hips_joint_name or "", 0))
    self._skeleton_instance = None
    self._bvh_channel_map = {}
    if path.suffix.lower() == ".bvh" and path.is_file():
      try:
        self._bvh_channel_map = parse_bvh_channel_map(path)
      except Exception:
        self._bvh_channel_map = {}
    self._init_mesh_skinner()
    if self._joint_inspector is not None:
      self._joint_inspector.rebuild(self._joint_names)
    self._rebuild_main_joint_controls()

    self._suppress_gui_updates = True
    try:
      self._frame_slider.max = max(0, clip.num_frames - 1)
      self._frame_slider.value = 0
    finally:
      self._suppress_gui_updates = False
    self._update_status_text()

  def _rebuild_main_joint_controls(self) -> None:
    self._controls_controller.rebuild_main_joint_controls()

  def _init_mesh_skinner(self) -> None:
    self._mesh_skinner = None
    self._mesh_warn = None
    if self._clip is None:
      return
    mesh, warn = try_load_soma_skeletal_mesh(self._clip.skeleton)
    if mesh is None:
      self._mesh_warn = warn
      return
    try:
      self._mesh_skinner = WarpMeshSkinner(mesh)
    except Exception as exc:
      self._mesh_warn = str(exc)

  def _update_status_text(self, extra: str | None = None) -> None:
    if self._status_html is None:
      return
    clip_name = None if self._clip is None else self._clip.name
    num_frames = None if self._clip is None else self._clip.num_frames
    sample_rate = None if self._clip is None else self._clip.sample_rate
    self._status_html.content = build_status_html(
      clip_name=clip_name,
      num_frames=num_frames,
      frame_idx=self._frame_idx,
      speed=self._speed,
      sample_rate=sample_rate,
      playing=self._playing,
      has_mesh_skinner=self._mesh_skinner is not None,
      mesh_warn=self._mesh_warn,
      extra=extra,
    )

  def _scale_about_pivot(self, xyz: np.ndarray, pivot: np.ndarray, s: float) -> np.ndarray:
    return pivot + float(s) * (xyz - pivot)

  def _redraw(self, frame_idx: int) -> None:
    self._render_pipeline.redraw(frame_idx)

  def _tick(self, dt: float) -> None:
    self._motion_controller.tick(dt)
    self._visualization_controller.tick(dt)
    self._controls_controller.tick(dt)

    if self._clip is None:
      return
    clip = self._clip

    if self._joint_inspector is not None and self._joint_inspector.needs_redraw:
      self._joint_inspector.needs_redraw = False
      self._needs_redraw = True

    if self._playing:
      self._frame_idx, self._accumulator, self._playing, advanced = advance_playback(
        self._frame_idx,
        self._accumulator,
        dt,
        sample_rate=clip.sample_rate,
        speed=self._speed,
        num_frames=clip.num_frames,
        loop=self._loop,
        playing=self._playing,
      )
      if advanced:
        self._needs_redraw = True

    if self._needs_redraw:
      with self._lock:
        self._redraw(self._frame_idx)
      self._needs_redraw = False

  def run(self) -> None:
    try:
      import warp as wp
    except ImportError as exc:
      raise ImportError(
        "warp-lang is required by soma_viser runtime. Install with: pip install warp-lang"
      ) from exc
    wp.init()

    self._update_status_text()
    last = time.perf_counter()
    try:
      while True:
        now = time.perf_counter()
        dt = min(now - last, 0.2)
        last = now
        self._tick(dt)
        time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
      pass
    finally:
      self.server.stop()
