"""Joint inspector rendering helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def render_joint_inspector(viewer: Any, xyz: np.ndarray, quat_xyzw: np.ndarray) -> None:
  """Update all joint frame handles and labels in the inspector."""
  inspector_rows = [] if viewer._joint_inspector is None else viewer._joint_inspector.rows
  show_full_info = viewer._show_frame_text_full_info
  show_text = viewer._show_frame_text
  if viewer._joint_inspector is not None:
    show_full_info = viewer._joint_inspector.show_frame_text_full_info
    show_text = viewer._joint_inspector.show_frame_text
  for inspector_row in inspector_rows:
    if not inspector_row.checkbox.value:
      inspector_row.frame_handle.visible = False
      inspector_row.scene_label.visible = False
      continue
    idx = viewer._joint_name_to_idx.get(inspector_row.joint_name)
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
    inspector_row.scene_label.position = (float(p[0] + 0.02), float(p[1] + 0.02), float(p[2] + 0.02))
    inspector_row.scene_label.visible = show_text
