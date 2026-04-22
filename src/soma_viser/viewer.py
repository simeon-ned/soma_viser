"""Interactive SOMA BVH viewer with Motion/Visualization/Controls tabs."""

from __future__ import annotations

import time
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import viser

from .mesh_skinner import WarpMeshSkinner, try_load_soma_skeletal_mesh
from .motion import MotionClip, list_bvh_files, load_motion_clip
from .skeleton import (
  build_bone_segments,
  decode_joint_name,
  euler_xyz_extrinsic_deg_to_wp_quat,
  joint_xyz_quat_from_globals,
)
from .viz.joint_inspector import JointInspector
from .viz.joints import resolve_hips_joint_name
from .viz.playback import DEFAULT_SPEEDS, advance_playback, update_speed_index
from .viz.status import build_status_html

_SPEEDS = DEFAULT_SPEEDS


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
    self._playing = True
    self._loop = True
    self._speed_idx = _SPEEDS.index(1.0)
    self._accumulator = 0.0
    self._frame_idx = 0
    self._needs_redraw = True

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
    self._joint_inspector: JointInspector | None = None

    self._build_gui()
    self._load_initial_clip()

  @property
  def _speed(self) -> float:
    return _SPEEDS[self._speed_idx]

  def _build_gui(self) -> None:
    tabs = self.server.gui.add_tab_group()
    with tabs.add_tab("Motion", icon=viser.Icon.PLAYER_PLAY):
      self._build_motion_tab()
    with tabs.add_tab("Visualization", icon=viser.Icon.EYE):
      self._build_visualization_tab()
    with tabs.add_tab("Controls", icon=viser.Icon.SETTINGS):
      self._build_controls_tab()

  def _build_motion_tab(self) -> None:
    motion_files = list_bvh_files(self.motions_dir)
    motion_names = [p.name for p in motion_files] or ["(no .bvh files found)"]

    self._motion_dropdown = self.server.gui.add_dropdown(
      "Motion clip",
      options=motion_names,
      initial_value=motion_names[0],
    )
    self._frame_slider = self.server.gui.add_slider(
      "Frame",
      min=0,
      max=1,
      step=1,
      initial_value=0,
    )
    self._status_html = self.server.gui.add_html("")
    play_btn = self.server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    speed_btn = self.server.gui.add_button_group(
      "Speed", options=["Slower", "1x", "Faster"]
    )
    loop_cb = self.server.gui.add_checkbox("Loop", initial_value=True)

    @self._motion_dropdown.on_update
    def _(_) -> None:
      if self._motion_dropdown.value.startswith("("):
        return
      self._load_clip_by_name(self._motion_dropdown.value)

    @self._frame_slider.on_update
    def _(_) -> None:
      if self._suppress_gui_updates:
        return
      self._frame_idx = int(self._frame_slider.value)
      self._needs_redraw = True

    @play_btn.on_click
    def _(_) -> None:
      self._playing = not self._playing
      play_btn.label = "Pause" if self._playing else "Play"
      play_btn.icon = viser.Icon.PLAYER_PAUSE if self._playing else viser.Icon.PLAYER_PLAY
      self._update_status_text()

    @speed_btn.on_click
    def _(event) -> None:
      self._speed_idx = update_speed_index(self._speed_idx, event.target.value)
      self._update_status_text()

    @loop_cb.on_update
    def _(_) -> None:
      self._loop = bool(loop_cb.value)
      self._update_status_text()

  def _build_visualization_tab(self) -> None:
    show_skeleton_cb = self.server.gui.add_checkbox("Show skeleton", initial_value=True)
    show_mesh_cb = self.server.gui.add_checkbox("Show skinned mesh", initial_value=True)
    color_picker = self.server.gui.add_rgb("Skeleton color", initial_value=self._skeleton_color)
    mesh_color_picker = self.server.gui.add_rgb("Mesh color", initial_value=self._mesh_color)
    mesh_opacity_slider = self.server.gui.add_slider(
      "Mesh opacity",
      min=0.0,
      max=1.0,
      step=0.01,
      initial_value=self._mesh_opacity,
    )
    line_width_slider = self.server.gui.add_slider(
      "Line width", min=1.0, max=8.0, step=0.1, initial_value=self._line_width
    )

    @show_skeleton_cb.on_update
    def _(_) -> None:
      self._show_skeleton = bool(show_skeleton_cb.value)
      self._line_handle.visible = self._show_skeleton

    @show_mesh_cb.on_update
    def _(_) -> None:
      self._show_mesh = bool(show_mesh_cb.value)
      self._needs_redraw = True

    @color_picker.on_update
    def _(_) -> None:
      self._skeleton_color = tuple(int(c) for c in color_picker.value)
      self._needs_redraw = True

    @mesh_color_picker.on_update
    def _(_) -> None:
      self._mesh_color = tuple(int(c) for c in mesh_color_picker.value)
      self._needs_redraw = True

    @mesh_opacity_slider.on_update
    def _(_) -> None:
      self._mesh_opacity = float(mesh_opacity_slider.value)
      if self._mesh_handle is not None:
        self._mesh_handle.opacity = self._mesh_opacity
      self._needs_redraw = True

    @line_width_slider.on_update
    def _(_) -> None:
      self._line_width = float(line_width_slider.value)
      self._line_handle.line_width = self._line_width

    self._joint_inspector = JointInspector(
      self.server,
      show_frame_text=self._show_frame_text,
      show_frame_text_full_info=self._show_frame_text_full_info,
    )
    with self._joint_inspector.folder:
      frame_text_cb = self.server.gui.add_checkbox(
        "Show frame text in scene",
        initial_value=self._show_frame_text,
      )
      frame_text_full_info_cb = self.server.gui.add_checkbox(
        "Show full text info (xyz + quat)",
        initial_value=self._show_frame_text_full_info,
      )
      # self._frame_inspector_header = self.server.gui.add_markdown(
      #   "Load a motion clip, then enable joint checkboxes to display frame axes."
      # )

    @frame_text_cb.on_update
    def _(_) -> None:
      self._show_frame_text = bool(frame_text_cb.value)
      if self._joint_inspector is not None:
        self._joint_inspector.show_frame_text = self._show_frame_text
        self._joint_inspector.update_text_visibility()

    @frame_text_full_info_cb.on_update
    def _(_) -> None:
      self._show_frame_text_full_info = bool(frame_text_full_info_cb.value)
      if self._joint_inspector is not None:
        self._joint_inspector.show_frame_text_full_info = self._show_frame_text_full_info
      self._needs_redraw = True

  def _build_controls_tab(self) -> None:
    root_x = self.server.gui.add_number("Root offset X", initial_value=0.0, step=0.01)
    root_y = self.server.gui.add_number("Root offset Y", initial_value=0.0, step=0.01)
    root_z = self.server.gui.add_number("Root offset Z", initial_value=0.0, step=0.01)
    align_rx = self.server.gui.add_slider("Align rot X (deg)", -180.0, 180.0, 1.0, 90.0)
    align_ry = self.server.gui.add_slider("Align rot Y (deg)", -180.0, 180.0, 1.0, 0.0)
    align_rz = self.server.gui.add_slider("Align rot Z (deg)", -180.0, 180.0, 1.0, 90.0)
    scale_slider = self.server.gui.add_slider(
      "Body scale", min=0.1, max=3.0, step=0.02, initial_value=1.0
    )

    def _mark_redraw() -> None:
      self._needs_redraw = True

    @root_x.on_update
    def _(_) -> None:
      self._root_offset[0] = float(root_x.value)
      _mark_redraw()

    @root_y.on_update
    def _(_) -> None:
      self._root_offset[1] = float(root_y.value)
      _mark_redraw()

    @root_z.on_update
    def _(_) -> None:
      self._root_offset[2] = float(root_z.value)
      _mark_redraw()

    @align_rx.on_update
    def _(_) -> None:
      self._align_euler[0] = float(align_rx.value)
      _mark_redraw()

    @align_ry.on_update
    def _(_) -> None:
      self._align_euler[1] = float(align_ry.value)
      _mark_redraw()

    @align_rz.on_update
    def _(_) -> None:
      self._align_euler[2] = float(align_rz.value)
      _mark_redraw()

    @scale_slider.on_update
    def _(_) -> None:
      self._body_scale = max(0.01, float(scale_slider.value))
      _mark_redraw()

  def _load_initial_clip(self) -> None:
    files = list_bvh_files(self.motions_dir)
    if not files:
      self._update_status_text("No BVH clips found.")
      return
    self._load_clip_by_name(files[0].name)

  def _load_clip_by_name(self, name: str) -> None:
    path = self.motions_dir / name
    clip = load_motion_clip(path)
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
    self._init_mesh_skinner()
    if self._joint_inspector is not None:
      self._joint_inspector.rebuild(self._joint_names)

    self._suppress_gui_updates = True
    try:
      self._frame_slider.max = max(0, clip.num_frames - 1)
      self._frame_slider.value = 0
    finally:
      self._suppress_gui_updates = False
    self._update_status_text()

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
    if self._clip is None:
      return
    import warp as wp

    clip = self._clip
    frame = int(np.clip(frame_idx, 0, clip.num_frames - 1))
    row = np.copy(clip.animation.local_transforms[frame])
    local_transforms = [row[i] for i in range(clip.skeleton.num_joints)]
    q_align = euler_xyz_extrinsic_deg_to_wp_quat(*self._align_euler.tolist())
    root_tx = wp.transform(wp.vec3(*self._root_offset.tolist()), q_align)
    global_tx = clip.skeleton.compute_global_transforms(local_transforms, root_tx)
    xyz, quat_xyzw = joint_xyz_quat_from_globals(global_tx, clip.skeleton.num_joints)
    pivot = self._root_offset.copy()
    xyz = self._scale_about_pivot(xyz, pivot, self._body_scale)

    segments = build_bone_segments(
      clip.skeleton.parent_indices,
      xyz,
      [decode_joint_name(clip.skeleton.joint_names[i]) for i in range(clip.skeleton.num_joints)],
    )
    self._line_handle.visible = self._show_skeleton
    if len(segments) > 0:
      base_rgb = np.array(self._skeleton_color, dtype=np.uint8).reshape(1, 1, 3)
      self._line_handle.points = segments
      self._line_handle.colors = np.broadcast_to(base_rgb, (segments.shape[0], 2, 3))
      self._last_skeleton_segments = segments
    elif self._last_skeleton_segments is not None:
      base_rgb = np.array(self._skeleton_color, dtype=np.uint8).reshape(1, 1, 3)
      self._line_handle.points = self._last_skeleton_segments
      self._line_handle.colors = np.broadcast_to(
        base_rgb, (self._last_skeleton_segments.shape[0], 2, 3)
      )

    inspector_rows = [] if self._joint_inspector is None else self._joint_inspector.rows
    show_full_info = self._show_frame_text_full_info
    show_text = self._show_frame_text
    if self._joint_inspector is not None:
      show_full_info = self._joint_inspector.show_frame_text_full_info
      show_text = self._joint_inspector.show_frame_text

    for inspector_row in inspector_rows:
      if not inspector_row.checkbox.value:
        inspector_row.frame_handle.visible = False
        inspector_row.scene_label.visible = False
        continue
      idx = self._joint_name_to_idx.get(inspector_row.joint_name)
      if idx is None:
        inspector_row.frame_handle.visible = False
        inspector_row.scene_label.visible = False
        continue
      p = xyz[idx]
      q = quat_xyzw[idx]
      inspector_row.frame_handle.position = (float(p[0]), float(p[1]), float(p[2]))
      inspector_row.frame_handle.wxyz = (float(q[3]), float(q[0]), float(q[1]), float(q[2]))
      inspector_row.frame_handle.visible = True
      if show_full_info:
        inspector_row.scene_label.text = (
          f"{inspector_row.joint_name}\n"
          f"xyz: ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})\n"
          f"quat: ({q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}, {q[3]:.3f})"
        )
      else:
        inspector_row.scene_label.text = inspector_row.joint_name
      inspector_row.scene_label.position = (
        float(p[0] + 0.02),
        float(p[1] + 0.02),
        float(p[2] + 0.02),
      )
      inspector_row.scene_label.visible = show_text

    if self._show_mesh and self._mesh_skinner is not None:
      from soma_retargeter.animation.skeleton import SkeletonInstance

      if self._skeleton_instance is None or self._skeleton_instance.skeleton is not clip.skeleton:
        self._skeleton_instance = SkeletonInstance(
          clip.skeleton,
          wp.vec3(0.9, 0.85, 1.0),
          root_tx,
        )
      else:
        self._skeleton_instance.xform = root_tx
      self._skeleton_instance.set_local_transforms(row)
      verts, faces = self._mesh_skinner.skin_to_numpy(self._skeleton_instance)
      verts = self._scale_about_pivot(verts, pivot, self._body_scale)
      if self._mesh_handle is None:
        self._mesh_handle = self.server.scene.add_mesh_simple(
          "/soma/body",
          verts,
          faces,
          color=self._mesh_color,
          opacity=self._mesh_opacity,
          wireframe=False,
        )
      else:
        self._mesh_handle.vertices = verts
        self._mesh_handle.color = self._mesh_color
        self._mesh_handle.opacity = self._mesh_opacity
        self._mesh_handle.visible = True
    elif self._mesh_handle is not None:
      self._mesh_handle.visible = False

    self._frame_idx = frame
    self._suppress_gui_updates = True
    try:
      self._frame_slider.value = frame
    finally:
      self._suppress_gui_updates = False
    self._update_status_text()

  def _tick(self, dt: float) -> None:
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
