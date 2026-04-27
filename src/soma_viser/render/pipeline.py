"""Render pipeline extracted from SomaViewer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.joint_overrides import apply_joint_overrides_to_row
from ..render.mesh import render_body_mesh
from ..render.skeleton import (
  build_bone_segments,
  decode_joint_name,
  euler_xyz_extrinsic_deg_to_wp_quat,
  joint_xyz_quat_from_globals,
  render_skeleton,
)


@dataclass
class RenderFrameResult:
  """Render output data needed by non-render UI flows."""

  frame_idx: int
  row: np.ndarray
  packed_transform_array: bool
  xyz: np.ndarray
  quat_xyzw: np.ndarray

class RenderPipeline:
  """Single-frame rendering pipeline for SomaViewer."""

  def __init__(self, viewer: Any) -> None:
    self.viewer = viewer

  def redraw(self, frame_idx: int) -> RenderFrameResult | None:
    v = self.viewer
    if v._clip is None:
      return None
    import warp as wp

    clip = v._clip
    frame = int(np.clip(frame_idx, 0, clip.num_frames - 1))
    row = np.copy(clip.animation.local_transforms[frame])
    row, packed_transform_array = apply_joint_overrides_to_row(
      row,
      num_joints=clip.skeleton.num_joints,
      main_joint_overrides_deg=v._main_joint_overrides_deg,
      joint_name_to_idx=v._joint_name_to_idx,
    )

    local_transforms = [row[i] for i in range(clip.skeleton.num_joints)]
    q_align = euler_xyz_extrinsic_deg_to_wp_quat(*v._align_euler.tolist())
    root_tx = wp.transform(wp.vec3(*v._root_offset.tolist()), q_align)
    global_tx = clip.skeleton.compute_global_transforms(local_transforms, root_tx)
    xyz, quat_xyzw = joint_xyz_quat_from_globals(global_tx, clip.skeleton.num_joints)
    pivot = v._root_offset.copy()
    xyz = v._scale_about_pivot(xyz, pivot, v._body_scale)

    segments = build_bone_segments(
      clip.skeleton.parent_indices,
      xyz,
      [decode_joint_name(clip.skeleton.joint_names[i]) for i in range(clip.skeleton.num_joints)],
    )
    render_skeleton(v, segments)
    render_body_mesh(v, clip, row, root_tx, pivot)
    return RenderFrameResult(
      frame_idx=frame,
      row=np.copy(row),
      packed_transform_array=packed_transform_array,
      xyz=xyz,
      quat_xyzw=quat_xyzw,
    )
