"""SOMA mesh rendering helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def render_body_mesh(
  viewer: Any,
  clip: Any,
  row: np.ndarray,
  root_tx: Any,
  pivot: np.ndarray,
) -> None:
  """Render skinned mesh and keep the handle updated."""
  import warp as wp

  if viewer._show_mesh and viewer._mesh_skinner is not None:
    from soma_retargeter.animation.skeleton import SkeletonInstance

    if viewer._skeleton_instance is None or viewer._skeleton_instance.skeleton is not clip.skeleton:
      viewer._skeleton_instance = SkeletonInstance(clip.skeleton, wp.vec3(0.9, 0.85, 1.0), root_tx)
    else:
      viewer._skeleton_instance.xform = root_tx
    viewer._skeleton_instance.set_local_transforms(row)
    verts, faces = viewer._mesh_skinner.skin_to_numpy(viewer._skeleton_instance)
    verts = viewer._scale_about_pivot(verts, pivot, viewer._body_scale)
    if viewer._mesh_handle is None:
      viewer._mesh_handle = viewer.server.scene.add_mesh_simple(
        "/soma/body",
        verts,
        faces,
        color=viewer._mesh_color,
        opacity=viewer._mesh_opacity,
        wireframe=False,
      )
    else:
      viewer._mesh_handle.vertices = verts
      viewer._mesh_handle.color = viewer._mesh_color
      viewer._mesh_handle.opacity = viewer._mesh_opacity
      viewer._mesh_handle.visible = True
    return
  if viewer._mesh_handle is not None:
    viewer._mesh_handle.visible = False
