"""SOMA mesh skinning, loading, and rendering helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


class WarpMeshSkinner:
  """Warp LBS wrapper matching soma-retargeter mesh renderer kernels."""

  def __init__(self, skeletal_mesh: Any) -> None:
    import warp as wp
    from soma_retargeter.renderers.mesh_renderer import (
      skinning_kernel,
      update_skinned_transform_kernel,
    )

    self.skeletal_mesh = skeletal_mesh
    self._skinning_kernel = skinning_kernel
    self._update_kernel = update_skinned_transform_kernel
    self.skinned_points: list[Any] = []
    for i in range(skeletal_mesh.num_skinned_meshes):
      n = skeletal_mesh.skinned_meshes[i].num_points
      self.skinned_points.append(wp.zeros(n, dtype=wp.vec3))
    n_joints = skeletal_mesh.skeleton.num_joints
    self.skinned_transforms = wp.zeros((1, n_joints), dtype=wp.transform)

  def skin_to_numpy(self, skeleton_instance: Any) -> tuple[np.ndarray, np.ndarray]:
    import warp as wp

    sm = self.skeletal_mesh
    animation_transforms = skeleton_instance.get_local_transforms()
    wp.launch(
      self._update_kernel,
      dim=(1),
      inputs=[
        skeleton_instance.num_joints,
        wp.array(animation_transforms, dtype=wp.transform),
        wp.array(skeleton_instance.parent_indices, dtype=wp.int32),
        sm.bind_transforms,
        skeleton_instance.xform,
      ],
      outputs=[self.skinned_transforms],
    )
    for i in range(sm.num_skinned_meshes):
      part = sm.skinned_meshes[i]
      dim = part.num_points
      if dim == 0:
        continue
      wp.launch(
        self._skinning_kernel,
        dim=dim,
        inputs=[
          part.points,
          part.joint_indices,
          part.joint_weights,
          int(part.num_influences),
          wp.array(self.skinned_transforms[0], dtype=wp.transform),
        ],
        outputs=[self.skinned_points[i]],
      )
    verts = np.asarray(self.skinned_points[0].numpy(), dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(sm.skinned_meshes[0].indices.numpy(), dtype=np.int32).reshape(-1, 3)
    return verts, faces


def try_load_soma_skeletal_mesh(skeleton: Any) -> tuple[Any | None, str | None]:
  """Load SOMA USD skinned mesh for a specific skeleton."""
  try:
    from soma_retargeter.pipelines.utils import SourceType, get_source_model_mesh
  except ImportError:
    return None, "soma_retargeter is missing (needed for SOMA USD skin mesh)"
  try:
    mesh = get_source_model_mesh(SourceType.SOMA, skeleton)
  except Exception as exc:
    return None, str(exc)
  if mesh is None:
    return None, "get_source_model_mesh returned None"
  return mesh, None


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
