"""Interactive SOMA BVH viewer with Motion/Visualization/Controls tabs."""

from __future__ import annotations

import time
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
import viser
import viser.transforms as tf

from .mesh_skinner import WarpMeshSkinner, try_load_soma_skeletal_mesh
from .motion import MotionClip, MotionEntry, index_bvh_files, load_motion_clip
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
_MOTION_MAX_OPTIONS = 300

_POSE_KEYFRAMES: dict[str, Path] = {
  "Calibration (frame0)": Path(__file__).resolve().parent / "assets" / "poses" / "soma_zero.bvh",
  "T pose": Path(__file__).resolve().parent / "assets" / "poses" / "soma_tpose.bvh",
}
_FINGER_TOKENS = (
  "finger",
  "thumb",
  "index",
  "middle",
  "ring",
  "pinky",
  "pinkie",
  "metacarp",
  "knuckle",
)


def _is_main_joint(name: str) -> bool:
  lname = name.strip().lower()
  if lname.endswith("end"):
    return False
  return not any(tok in lname for tok in _FINGER_TOKENS)


def _is_hips_like_joint(name: str) -> bool:
  lname = name.strip().lower()
  return lname in ("hips", "hip", "pelvis", "root", "rootjoint")


def _parse_bvh_channel_map(path: Path) -> dict[str, tuple[int, list[str]]]:
  """Map joint name -> (channel start index, channel labels)."""
  lines = path.read_text(encoding="utf-8").splitlines()
  channel_cursor = 0
  active_name: str | None = None
  out: dict[str, tuple[int, list[str]]] = {}
  for line in lines:
    s = line.strip()
    if s.startswith("MOTION"):
      break
    if s.startswith("ROOT "):
      active_name = s.split(maxsplit=1)[1].strip()
      continue
    if s.startswith("JOINT "):
      active_name = s.split(maxsplit=1)[1].strip()
      continue
    if s.startswith("CHANNELS "):
      parts = s.split()
      if len(parts) >= 2:
        n = int(parts[1])
        labels = parts[2 : 2 + n]
        if active_name is not None:
          out[active_name] = (channel_cursor, labels)
        channel_cursor += n
  return out


def _euler_zyx_deg_to_wxyz(euler_deg: np.ndarray) -> tuple[float, float, float, float]:
  rz = float(np.radians(euler_deg[0]))
  ry = float(np.radians(euler_deg[1]))
  rx = float(np.radians(euler_deg[2]))
  so3 = (
    tf.SO3.from_z_radians(rz)
    @ tf.SO3.from_y_radians(ry)
    @ tf.SO3.from_x_radians(rx)
  )
  w, x, y, z = so3.wxyz
  return (float(w), float(x), float(y), float(z))


def _wxyz_to_euler_zyx_deg(wxyz: tuple[float, float, float, float]) -> np.ndarray:
  r = tf.SO3(wxyz=np.asarray(wxyz, dtype=np.float64)).as_matrix()
  sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
  singular = sy < 1e-8
  if not singular:
    rz = np.arctan2(r[1, 0], r[0, 0])
    ry = np.arctan2(-r[2, 0], sy)
    rx = np.arctan2(r[2, 1], r[2, 2])
  else:
    rz = np.arctan2(-r[0, 1], r[1, 1])
    ry = np.arctan2(-r[2, 0], sy)
    rx = 0.0
  return np.array([np.degrees(rz), np.degrees(ry), np.degrees(rx)], dtype=np.float64)


def _extract_bvh_motion_rows(path: Path) -> tuple[list[str], int]:
  """Return (all lines, motion row start index)."""
  lines = path.read_text(encoding="utf-8").splitlines()
  start_idx = -1
  for i, line in enumerate(lines):
    if line.strip().startswith("Frame Time:"):
      start_idx = i + 1
      break
  if start_idx < 0:
    raise ValueError(f"BVH missing Frame Time: {path}")
  return lines, start_idx


def _estimate_bvh_units_per_meter(
  source_values: list[float],
  channel_map: dict[str, tuple[int, list[str]]],
) -> float:
  """Heuristic scale from viewer meters -> BVH translation units."""
  for jname, (start, labels) in channel_map.items():
    lname = jname.strip().lower()
    if lname not in ("hips", "hip", "pelvis", "root", "rootjoint"):
      continue
    idx_y = None
    for i, lbl in enumerate(labels):
      if lbl == "Yposition":
        idx_y = start + i
        break
    if idx_y is None or idx_y >= len(source_values):
      continue
    y_abs = abs(float(source_values[idx_y]))
    # Most BVH assets here are centimeters (hips around ~100). Otherwise meters.
    return 100.0 if y_abs > 10.0 else 1.0
  return 1.0


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
    self._motion_search_text = self.server.gui.add_text("Search clips", initial_value="")
    self._motion_folder_dropdown = self.server.gui.add_dropdown(
      "Folder",
      options=["(all folders)"],
      initial_value="(all folders)",
    )
    self._motion_dropdown = self.server.gui.add_dropdown(
      "Motion clip",
      options=["(no .bvh files found)"],
      initial_value="(no .bvh files found)",
    )
    refresh_btn = self.server.gui.add_button("Refresh library")
    self._motion_page_text = self.server.gui.add_text(
      "Results",
      initial_value="No clips indexed.",
      disabled=True,
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
    pose_dd = self.server.gui.add_dropdown(
      "Pose keyframe",
      options=list(_POSE_KEYFRAMES.keys()),
      initial_value="Calibration (frame0)",
    )

    @self._motion_search_text.on_update
    def _(_) -> None:
      self._apply_motion_filters()

    @self._motion_folder_dropdown.on_update
    def _(_) -> None:
      if self._suppress_motion_ui:
        return
      self._apply_motion_filters()

    @self._motion_dropdown.on_update
    def _(_) -> None:
      if self._suppress_motion_ui:
        return
      selected = str(self._motion_dropdown.value)
      if selected.startswith("("):
        return
      self._load_clip_by_relpath(selected)

    @refresh_btn.on_click
    def _(_) -> None:
      self._refresh_motion_library()

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

    @pose_dd.on_update
    def _(_) -> None:
      pose_path = _POSE_KEYFRAMES.get(str(pose_dd.value))
      if pose_path is None or not pose_path.is_file():
        self._update_status_text("Pose keyframe file not found.")
        return
      self._load_clip_by_path(pose_path)

    self._refresh_motion_library()

  def _refresh_motion_library(self) -> None:
    self._motion_entries = index_bvh_files(self.motions_dir, recursive=True)
    self._motion_entry_by_relpath = {e.rel_path: e for e in self._motion_entries}

    folder_names = {"(all folders)"}
    for entry in self._motion_entries:
      parent = Path(entry.rel_path).parent.as_posix()
      folder_names.add(parent if parent not in ("", ".") else ".")
    folder_options = sorted(folder_names, key=lambda v: (v != "(all folders)", v.lower()))

    if self._motion_folder_dropdown is not None:
      old_value = str(self._motion_folder_dropdown.value)
      self._suppress_motion_ui = True
      try:
        self._motion_folder_dropdown.options = folder_options
        self._motion_folder_dropdown.value = (
          old_value if old_value in folder_options else "(all folders)"
        )
      finally:
        self._suppress_motion_ui = False
    self._apply_motion_filters()

  def _apply_motion_filters(self) -> None:
    query = ""
    folder = "(all folders)"
    if self._motion_search_text is not None:
      query = str(self._motion_search_text.value).strip().lower()
    if self._motion_folder_dropdown is not None:
      folder = str(self._motion_folder_dropdown.value)

    terms = [t for t in query.split() if t]
    filtered: list[MotionEntry] = []
    for entry in self._motion_entries:
      rel = entry.rel_path.lower()
      parent = Path(entry.rel_path).parent.as_posix()
      parent = parent if parent not in ("", ".") else "."
      if folder != "(all folders)" and parent != folder:
        continue
      if terms and not all(term in rel for term in terms):
        continue
      filtered.append(entry)
    self._motion_filtered_entries = filtered
    visible_entries = filtered[:_MOTION_MAX_OPTIONS]
    options = [e.rel_path for e in visible_entries] or ["(no matches)"]
    selected = options[0]
    if self._motion_dropdown is not None:
      old_selected = str(self._motion_dropdown.value)
      if old_selected in options:
        selected = old_selected
      self._suppress_motion_ui = True
      try:
        self._motion_dropdown.options = options
        self._motion_dropdown.value = selected
      finally:
        self._suppress_motion_ui = False

    if self._motion_page_text is not None:
      if not filtered:
        self._motion_page_text.value = "No clips match current filters."
      else:
        shown = len(visible_entries)
        if shown < len(filtered):
          self._motion_page_text.value = (
            f"Showing first {shown} of {len(filtered)} clips. "
            "Type more in Search clips to narrow results."
          )
        else:
          self._motion_page_text.value = f"Showing {shown} clip(s)."

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
    # Force default-on each run (avoid stale/persisted UI state surprises).
    self._show_joint_knobs = False
    show_gizmo_cb = self.server.gui.add_checkbox("Show root gizmo", initial_value=True)
    root_x = self.server.gui.add_number("Root offset X", initial_value=0.0, step=0.01)
    root_y = self.server.gui.add_number("Root offset Y", initial_value=0.0, step=0.01)
    root_z = self.server.gui.add_number("Root offset Z", initial_value=0.0, step=0.01)
    align_rx = self.server.gui.add_slider("Align rot X (deg)", -180.0, 180.0, 1.0, 90.0)
    align_ry = self.server.gui.add_slider("Align rot Y (deg)", -180.0, 180.0, 1.0, 0.0)
    align_rz = self.server.gui.add_slider("Align rot Z (deg)", -180.0, 180.0, 1.0, 90.0)
    scale_slider = self.server.gui.add_slider(
      "Body scale", min=0.1, max=3.0, step=0.02, initial_value=1.0
    )
    export_name = self.server.gui.add_text("Export file", initial_value="pose_export.bvh")
    export_btn = self.server.gui.add_button("Export BVH (current config)")
    export_status = self.server.gui.add_text("Export status", initial_value="-", disabled=True)
    show_knobs_cb = self.server.gui.add_checkbox(
      "Show joint gizmos",
      initial_value=self._show_joint_knobs,
    )

    def _mark_redraw() -> None:
      self._needs_redraw = True

    self._root_number_controls = (root_x, root_y, root_z)
    self._root_gizmo = self.server.scene.add_transform_controls(
      "/soma/root_gizmo",
      scale=0.14,
      line_width=1.4,
      position=tuple(float(v) for v in self._root_offset),
      wxyz=(1.0, 0.0, 0.0, 0.0),
      depth_test=False,
      disable_rotations=True,
      opacity=0.92,
      visible=True,
    )

    @show_gizmo_cb.on_update
    def _(_) -> None:
      if self._root_gizmo is not None:
        self._root_gizmo.visible = bool(show_gizmo_cb.value)

    @self._root_gizmo.on_update
    def _(event) -> None:
      p = np.asarray(event.target.position, dtype=np.float64)
      self._root_offset[:] = p
      if self._root_number_controls is not None:
        self._suppress_root_ui = True
        try:
          rx, ry, rz = self._root_number_controls
          rx.value = float(p[0])
          ry.value = float(p[1])
          rz.value = float(p[2])
        finally:
          self._suppress_root_ui = False
      _mark_redraw()

    @root_x.on_update
    def _(_) -> None:
      if self._suppress_root_ui:
        return
      self._root_offset[0] = float(root_x.value)
      if self._root_gizmo is not None:
        self._root_gizmo.position = tuple(float(v) for v in self._root_offset)
      _mark_redraw()

    @root_y.on_update
    def _(_) -> None:
      if self._suppress_root_ui:
        return
      self._root_offset[1] = float(root_y.value)
      if self._root_gizmo is not None:
        self._root_gizmo.position = tuple(float(v) for v in self._root_offset)
      _mark_redraw()

    @root_z.on_update
    def _(_) -> None:
      if self._suppress_root_ui:
        return
      self._root_offset[2] = float(root_z.value)
      if self._root_gizmo is not None:
        self._root_gizmo.position = tuple(float(v) for v in self._root_offset)
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

    @export_btn.on_click
    def _(_) -> None:
      if self._clip is None:
        export_status.value = "No clip loaded."
        return
      if not self._clip.path.is_file():
        export_status.value = "Source BVH not found."
        return
      try:
        lines, row_start = _extract_bvh_motion_rows(self._clip.path)
        if self._frame_idx >= max(1, self._clip.num_frames):
          frame_idx = 0
        else:
          frame_idx = int(self._frame_idx)
        src_row_idx = row_start + frame_idx
        if src_row_idx >= len(lines):
          src_row_idx = row_start
        src_vals = [float(x) for x in lines[src_row_idx].strip().split()]
        out_vals = list(src_vals)
        bvh_units_per_meter = _estimate_bvh_units_per_meter(
          src_vals,
          self._bvh_channel_map,
        )
        row_live = self._last_rendered_row
        if row_live is None:
          export_status.value = "No rendered pose cached yet."
          return
        row_arr = np.asarray(row_live)
        packed_transform_array = (
          row_arr.ndim == 2
          and row_arr.shape[0] == len(self._joint_names)
          and row_arr.shape[1] >= 7
          and row_arr.dtype.kind in "fc"
        )
        import warp as wp
        for jname, jidx in self._joint_name_to_idx.items():
          info = self._bvh_channel_map.get(jname)
          if info is None:
            continue
          start, labels = info
          ch_idx = {lbl: start + i for i, lbl in enumerate(labels)}
          if packed_transform_array:
            qx, qy, qz, qw = (
              float(row_live[jidx, 3]),
              float(row_live[jidx, 4]),
              float(row_live[jidx, 5]),
              float(row_live[jidx, 6]),
            )
          else:
            ti = row_live[jidx]
            qr = wp.transform_get_rotation(ti)  # xyzw
            qx, qy, qz, qw = float(qr[0]), float(qr[1]), float(qr[2]), float(qr[3])
          eul = _wxyz_to_euler_zyx_deg((qw, qx, qy, qz))
          # Keep BVH position channels from source row to preserve bind offsets.
          # Apply viewer root-offset only on hips/root-like joints.
          if _is_hips_like_joint(jname):
            if "Xposition" in ch_idx:
              out_vals[ch_idx["Xposition"]] = (
                float(src_vals[ch_idx["Xposition"]])
                + float(self._root_offset[0]) * bvh_units_per_meter
              )
            if "Yposition" in ch_idx:
              out_vals[ch_idx["Yposition"]] = (
                float(src_vals[ch_idx["Yposition"]])
                + float(self._root_offset[1]) * bvh_units_per_meter
              )
            if "Zposition" in ch_idx:
              out_vals[ch_idx["Zposition"]] = (
                float(src_vals[ch_idx["Zposition"]])
                + float(self._root_offset[2]) * bvh_units_per_meter
              )
          if "Zrotation" in ch_idx:
            out_vals[ch_idx["Zrotation"]] = float(eul[0])
          if "Yrotation" in ch_idx:
            out_vals[ch_idx["Yrotation"]] = float(eul[1])
          if "Xrotation" in ch_idx:
            out_vals[ch_idx["Xrotation"]] = float(eul[2])
        header = lines[:row_start]
        frame_time = 1.0 / max(1e-6, float(self._clip.sample_rate))
        for i, line in enumerate(header):
          if line.strip().startswith("Frames:"):
            header[i] = "Frames: 1"
          elif line.strip().startswith("Frame Time:"):
            header[i] = f"Frame Time: {frame_time:.6f}"
        out_name = export_name.value.strip() or "pose_export.bvh"
        if not out_name.endswith(".bvh"):
          out_name += ".bvh"
        out_dir = self.motions_dir / "exports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / out_name
        out_text = "\n".join(header + [" ".join(f"{v:.9g}" for v in out_vals)]) + "\n"
        out_path.write_text(out_text, encoding="utf-8")
        export_status.value = f"Saved: {out_path}"
      except Exception as exc:
        export_status.value = f"Export failed: {exc}"

    with self.server.gui.add_folder("Joint controls (main, no fingers)"):
      self._joint_controls_folder = self.server.gui.add_folder("Joint list")

    @show_knobs_cb.on_update
    def _(_) -> None:
      self._show_joint_knobs = bool(show_knobs_cb.value)
      for h in self._main_joint_knobs.values():
        h.visible = self._show_joint_knobs

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
        self._bvh_channel_map = _parse_bvh_channel_map(path)
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
    folder = getattr(self, "_joint_controls_folder", None)
    if folder is None:
      return
    # Clear old controls.
    for c in self._main_joint_controls.values():
      c.remove()
    self._main_joint_controls.clear()
    for h in self._main_joint_knobs.values():
      h.remove()
    self._main_joint_knobs.clear()
    self._main_joint_overrides_deg.clear()
    self._main_joint_display_deg.clear()
    if self._clip is None:
      return
    with folder:
      for jname in self._joint_names:
        if not _is_main_joint(jname):
          continue
        ctrl = self.server.gui.add_vector3(
          f"{jname} (Z,Y,X deg)",
          initial_value=(0.0, 0.0, 0.0),
          step=1.0,
        )
        self._main_joint_controls[jname] = ctrl
        # Add small rotational knob in scene (rotation only, no translation axes).
        knob = self.server.scene.add_transform_controls(
          f"/soma/joint_knobs/{jname}",
          scale=0.12,
          line_width=1.8,
          disable_axes=True,
          disable_sliders=True,
          visible=self._show_joint_knobs,
          depth_test=False,
          opacity=1.0,
        )
        self._main_joint_knobs[jname] = knob
        self._main_joint_overrides_deg[jname] = np.zeros(3, dtype=np.float64)
        self._main_joint_display_deg[jname] = np.zeros(3, dtype=np.float64)

        def _bind(name: str, h: Any, k: viser.TransformControlsHandle) -> None:
          @h.on_update
          def _(_event) -> None:
            if self._suppress_joint_ui:
              return
            v = np.asarray(h.value, dtype=np.float64)
            prev = self._main_joint_display_deg.get(name, np.zeros(3, dtype=np.float64))
            delta = v - prev
            self._main_joint_overrides_deg[name] = (
              self._main_joint_overrides_deg.get(name, np.zeros(3, dtype=np.float64))
              + delta
            )
            self._main_joint_display_deg[name] = v
            k.wxyz = _euler_zyx_deg_to_wxyz(v)
            self._needs_redraw = True

        _bind(jname, ctrl, knob)

        def _bind_knob(name: str, h: Any, k: viser.TransformControlsHandle) -> None:
          @k.on_update
          def _(_event) -> None:
            if self._suppress_joint_ui:
              return
            v = _wxyz_to_euler_zyx_deg(k.wxyz)
            prev = self._main_joint_display_deg.get(name, np.zeros(3, dtype=np.float64))
            delta = v - prev
            self._main_joint_overrides_deg[name] = (
              self._main_joint_overrides_deg.get(name, np.zeros(3, dtype=np.float64))
              + delta
            )
            self._main_joint_display_deg[name] = v
            self._suppress_joint_ui = True
            try:
              h.value = (float(v[0]), float(v[1]), float(v[2]))
            finally:
              self._suppress_joint_ui = False
            self._needs_redraw = True

        _bind_knob(jname, ctrl, knob)

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
    # Apply user joint overrides as offsets on top of incoming motion rotations.
    row_arr = np.asarray(row)
    packed_transform_array = (
      row_arr.ndim == 2
      and row_arr.shape[0] == clip.skeleton.num_joints
      and row_arr.shape[1] >= 7
      and row_arr.dtype.kind in "fc"
    )
    for jname, deg_zyx in self._main_joint_overrides_deg.items():
      jidx = self._joint_name_to_idx.get(jname)
      if jidx is None:
        continue
      rz = float(np.radians(deg_zyx[0]))
      ry = float(np.radians(deg_zyx[1]))
      rx = float(np.radians(deg_zyx[2]))
      so3 = (
        tf.SO3.from_z_radians(rz)
        @ tf.SO3.from_y_radians(ry)
        @ tf.SO3.from_x_radians(rx)
      )
      w, x, y, z = so3.wxyz
      if packed_transform_array:
        # Packed local transform layout: [tx, ty, tz, qx, qy, qz, qw].
        qx = float(row[jidx, 3])
        qy = float(row[jidx, 4])
        qz = float(row[jidx, 5])
        qw = float(row[jidx, 6])
        base_so3 = tf.SO3(wxyz=(qw, qx, qy, qz))
        out_so3 = base_so3 @ tf.SO3(wxyz=(float(w), float(x), float(y), float(z)))
        ow, ox, oy, oz = out_so3.wxyz
        row[jidx, 3] = float(ox)
        row[jidx, 4] = float(oy)
        row[jidx, 5] = float(oz)
        row[jidx, 6] = float(ow)
      else:
        ti = row[jidx]
        tr = wp.transform_get_translation(ti)
        qr = wp.transform_get_rotation(ti)  # xyzw
        base_so3 = tf.SO3(wxyz=(float(qr[3]), float(qr[0]), float(qr[1]), float(qr[2])))
        out_so3 = base_so3 @ tf.SO3(wxyz=(float(w), float(x), float(y), float(z)))
        ow, ox, oy, oz = out_so3.wxyz
        row[jidx] = wp.transform(tr, wp.quat(float(ox), float(oy), float(oz), float(ow)))
    self._last_rendered_row = np.copy(row)

    # Keep joint control values in sync with the currently rendered local pose.
    # This makes controls "live" while playback/calibration is running.
    self._suppress_joint_ui = True
    try:
      for jname, ctrl in list(self._main_joint_controls.items()):
        jidx = self._joint_name_to_idx.get(jname)
        if jidx is None:
          continue
        if packed_transform_array:
          qx = float(row[jidx, 3])
          qy = float(row[jidx, 4])
          qz = float(row[jidx, 5])
          qw = float(row[jidx, 6])
          eul = _wxyz_to_euler_zyx_deg((qw, qx, qy, qz))
        else:
          qr = wp.transform_get_rotation(row[jidx])  # xyzw
          eul = _wxyz_to_euler_zyx_deg(
            (float(qr[3]), float(qr[0]), float(qr[1]), float(qr[2]))
          )
        self._main_joint_display_deg[jname] = np.asarray(eul, dtype=np.float64)
        ctrl.value = (float(eul[0]), float(eul[1]), float(eul[2]))
    finally:
      self._suppress_joint_ui = False
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

    # Keep scene knobs attached to joints and synced to current override values.
    self._suppress_joint_ui = True
    try:
      for jname, knob in list(self._main_joint_knobs.items()):
        jidx = self._joint_name_to_idx.get(jname)
        if jidx is None:
          continue
        p = xyz[jidx]
        knob.position = (float(p[0]), float(p[1]), float(p[2]))
        # Keep gizmo orientation in the same local-euler space as controls.
        # This avoids world/local jumps while still evolving with playback.
        disp = self._main_joint_display_deg.get(jname, np.zeros(3, dtype=np.float64))
        knob.wxyz = _euler_zyx_deg_to_wxyz(disp)
        knob.visible = self._show_joint_knobs
    finally:
      self._suppress_joint_ui = False

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
