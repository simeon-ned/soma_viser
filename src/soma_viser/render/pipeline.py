"""Render pipeline extracted from SomaViewer."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..render.joint_overrides import (
  apply_joint_overrides_to_row,
  euler_zyx_deg_to_wxyz,
  wxyz_to_euler_zyx_deg,
)
from ..skeleton import (
  build_bone_segments,
  decode_joint_name,
  euler_xyz_extrinsic_deg_to_wp_quat,
  joint_xyz_quat_from_globals,
)


class RenderPipeline:
  """Single-frame rendering pipeline for SomaViewer."""

  def __init__(self, viewer: Any) -> None:
    self.viewer = viewer

  def redraw(self, frame_idx: int) -> None:
    v = self.viewer
    if v._clip is None:
      return
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
    v._last_rendered_row = np.copy(row)

    v._suppress_joint_ui = True
    try:
      for jname, ctrl in list(v._main_joint_controls.items()):
        jidx = v._joint_name_to_idx.get(jname)
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
        v._main_joint_display_deg[jname] = np.asarray(eul, dtype=np.float64)
        ctrl.value = (float(eul[0]), float(eul[1]), float(eul[2]))
    finally:
      v._suppress_joint_ui = False

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
    v._line_handle.visible = v._show_skeleton
    if len(segments) > 0:
      base_rgb = np.array(v._skeleton_color, dtype=np.uint8).reshape(1, 1, 3)
      v._line_handle.points = segments
      v._line_handle.colors = np.broadcast_to(base_rgb, (segments.shape[0], 2, 3))
      v._last_skeleton_segments = segments
    elif v._last_skeleton_segments is not None:
      base_rgb = np.array(v._skeleton_color, dtype=np.uint8).reshape(1, 1, 3)
      v._line_handle.points = v._last_skeleton_segments
      v._line_handle.colors = np.broadcast_to(base_rgb, (v._last_skeleton_segments.shape[0], 2, 3))

    inspector_rows = [] if v._joint_inspector is None else v._joint_inspector.rows
    show_full_info = v._show_frame_text_full_info
    show_text = v._show_frame_text
    if v._joint_inspector is not None:
      show_full_info = v._joint_inspector.show_frame_text_full_info
      show_text = v._joint_inspector.show_frame_text
    for inspector_row in inspector_rows:
      if not inspector_row.checkbox.value:
        inspector_row.frame_handle.visible = False
        inspector_row.scene_label.visible = False
        continue
      idx = v._joint_name_to_idx.get(inspector_row.joint_name)
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

    v._suppress_joint_ui = True
    try:
      for jname, knob in list(v._main_joint_knobs.items()):
        jidx = v._joint_name_to_idx.get(jname)
        if jidx is None:
          continue
        p = xyz[jidx]
        knob.position = (float(p[0]), float(p[1]), float(p[2]))
        disp = v._main_joint_display_deg.get(jname, np.zeros(3, dtype=np.float64))
        knob.wxyz = euler_zyx_deg_to_wxyz(disp)
        knob.visible = v._show_joint_knobs
    finally:
      v._suppress_joint_ui = False

    if v._show_mesh and v._mesh_skinner is not None:
      from soma_retargeter.animation.skeleton import SkeletonInstance

      if v._skeleton_instance is None or v._skeleton_instance.skeleton is not clip.skeleton:
        v._skeleton_instance = SkeletonInstance(clip.skeleton, wp.vec3(0.9, 0.85, 1.0), root_tx)
      else:
        v._skeleton_instance.xform = root_tx
      v._skeleton_instance.set_local_transforms(row)
      verts, faces = v._mesh_skinner.skin_to_numpy(v._skeleton_instance)
      verts = v._scale_about_pivot(verts, pivot, v._body_scale)
      if v._mesh_handle is None:
        v._mesh_handle = v.server.scene.add_mesh_simple(
          "/soma/body",
          verts,
          faces,
          color=v._mesh_color,
          opacity=v._mesh_opacity,
          wireframe=False,
        )
      else:
        v._mesh_handle.vertices = verts
        v._mesh_handle.color = v._mesh_color
        v._mesh_handle.opacity = v._mesh_opacity
        v._mesh_handle.visible = True
    elif v._mesh_handle is not None:
      v._mesh_handle.visible = False

    v._frame_idx = frame
    v._suppress_gui_updates = True
    try:
      v._frame_slider.value = frame
    finally:
      v._suppress_gui_updates = False
    v._update_status_text()
