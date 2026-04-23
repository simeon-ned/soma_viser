"""Skeleton and joint knob rendering helpers."""

from __future__ import annotations

from typing import Any

import numpy as np

from .joint_overrides import euler_zyx_deg_to_wxyz, wxyz_to_euler_zyx_deg


def sync_joint_controls_from_row(viewer: Any, row: np.ndarray, packed_transform_array: bool) -> None:
  """Refresh joint control widgets from the rendered row pose."""
  import warp as wp

  viewer._suppress_joint_ui = True
  try:
    for jname, ctrl in list(viewer._main_joint_controls.items()):
      jidx = viewer._joint_name_to_idx.get(jname)
      if jidx is None:
        continue
      if packed_transform_array:
        qx = float(row[jidx, 3])
        qy = float(row[jidx, 4])
        qz = float(row[jidx, 5])
        qw = float(row[jidx, 6])
        eul = wxyz_to_euler_zyx_deg((qw, qx, qy, qz))
      else:
        qr = wp.transform_get_rotation(row[jidx])  # xyzw
        eul = wxyz_to_euler_zyx_deg((float(qr[3]), float(qr[0]), float(qr[1]), float(qr[2])))
      viewer._main_joint_display_deg[jname] = np.asarray(eul, dtype=np.float64)
      ctrl.value = (float(eul[0]), float(eul[1]), float(eul[2]))
  finally:
    viewer._suppress_joint_ui = False


def render_skeleton(viewer: Any, segments: np.ndarray) -> None:
  """Render skeleton lines and preserve last known valid segment set."""
  viewer._line_handle.visible = viewer._show_skeleton
  if len(segments) > 0:
    base_rgb = np.array(viewer._skeleton_color, dtype=np.uint8).reshape(1, 1, 3)
    viewer._line_handle.points = segments
    viewer._line_handle.colors = np.broadcast_to(base_rgb, (segments.shape[0], 2, 3))
    viewer._last_skeleton_segments = segments
    return
  if viewer._last_skeleton_segments is not None:
    base_rgb = np.array(viewer._skeleton_color, dtype=np.uint8).reshape(1, 1, 3)
    viewer._line_handle.points = viewer._last_skeleton_segments
    viewer._line_handle.colors = np.broadcast_to(base_rgb, (viewer._last_skeleton_segments.shape[0], 2, 3))


def sync_joint_knobs(viewer: Any, xyz: np.ndarray) -> None:
  """Update joint knob transforms from latest global joint positions."""
  viewer._suppress_joint_ui = True
  try:
    for jname, knob in list(viewer._main_joint_knobs.items()):
      jidx = viewer._joint_name_to_idx.get(jname)
      if jidx is None:
        continue
      p = xyz[jidx]
      knob.position = (float(p[0]), float(p[1]), float(p[2]))
      disp = viewer._main_joint_display_deg.get(jname, np.zeros(3, dtype=np.float64))
      knob.wxyz = euler_zyx_deg_to_wxyz(disp)
      knob.visible = viewer._show_joint_knobs
  finally:
    viewer._suppress_joint_ui = False
