"""Skeleton extraction and segment generation helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as vtf


def decode_joint_name(name: Any) -> str:
  """Decode byte or object joint names to string."""
  if isinstance(name, bytes):
    return name.decode()
  return str(name)


def euler_xyz_extrinsic_deg_to_wp_quat(rx_deg: float, ry_deg: float, rz_deg: float) -> Any:
  """Extrinsic world X->Y->Z Euler rotation (degrees) as Warp quaternion."""
  import warp as wp

  rx = float(np.radians(rx_deg))
  ry = float(np.radians(ry_deg))
  rz = float(np.radians(rz_deg))
  so3 = (
    vtf.SO3.from_z_radians(rz)
    @ vtf.SO3.from_y_radians(ry)
    @ vtf.SO3.from_x_radians(rx)
  )
  w, x, y, z = so3.wxyz
  return wp.quat(float(x), float(y), float(z), float(w))


def joint_xyz_quat_from_globals(globals_tx: Any, num_joints: int) -> tuple[np.ndarray, np.ndarray]:
  """Extract world-space xyz and xyzw quaternion from Warp transforms."""
  import warp as wp

  xyz = np.zeros((num_joints, 3), dtype=np.float64)
  quat_xyzw = np.zeros((num_joints, 4), dtype=np.float64)
  arr = np.asarray(globals_tx)

  # Fast path for packed float arrays, commonly shaped (J, 7):
  # [tx, ty, tz, qx, qy, qz, qw].
  if arr.ndim == 2 and arr.shape[0] == num_joints and arr.dtype.kind in "fc":
    if arr.shape[1] >= 3:
      xyz[:, :] = arr[:, :3].astype(np.float64)
    if arr.shape[1] >= 7:
      quat_xyzw[:, :] = arr[:, 3:7].astype(np.float64)
    else:
      quat_xyzw[:, 3] = 1.0
    return xyz, quat_xyzw

  for i in range(num_joints):
    ti = arr[i] if arr.ndim == 1 else arr.reshape(-1)[i]
    tr = wp.transform_get_translation(ti)
    qr = wp.transform_get_rotation(ti)
    xyz[i, 0] = float(tr[0])
    xyz[i, 1] = float(tr[1])
    xyz[i, 2] = float(tr[2])
    quat_xyzw[i, 0] = float(qr[0])
    quat_xyzw[i, 1] = float(qr[1])
    quat_xyzw[i, 2] = float(qr[2])
    quat_xyzw[i, 3] = float(qr[3])
  return xyz, quat_xyzw


def build_bone_segments(
  parent_indices: np.ndarray,
  joint_xyz: np.ndarray,
  joint_names: list[str] | None = None,
) -> np.ndarray:
  """Build line segments with shape (N, 2, 3)."""
  segs: list[np.ndarray] = []
  num_joints = int(joint_xyz.shape[0])

  def _append_if_valid(a: np.ndarray, b: np.ndarray) -> None:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
      return
    if np.linalg.norm(b - a) < 1e-9:
      return
    segs.append(np.stack([a, b], axis=0))

  def _collect(filter_root_hips: bool) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    for j in range(1, num_joints):
      parent = int(parent_indices[j])
      if parent < 0 or parent >= num_joints:
        continue
      if filter_root_hips and joint_names is not None:
        pn = joint_names[parent].strip().lower()
        jn = joint_names[j].strip().lower()
        if (pn == "root" and jn == "hips") or (pn == "hips" and jn == "root"):
          continue
      a = joint_xyz[parent]
      b = joint_xyz[j]
      if not np.isfinite(a).all() or not np.isfinite(b).all():
        continue
      if np.linalg.norm(b - a) < 1e-9:
        continue
      out.append(np.stack([a, b], axis=0))
    return out

  # Match humo_target behavior.
  segs = _collect(filter_root_hips=True)
  if not segs:
    segs = _collect(filter_root_hips=False)

  # Fallback: if hierarchy links failed (bad parents or naming/layout mismatch),
  # still draw a continuous polyline over consecutive joints so skeleton remains visible.
  if not segs:
    for j in range(1, num_joints):
      _append_if_valid(joint_xyz[j - 1], joint_xyz[j])

  if not segs:
    return np.zeros((0, 2, 3), dtype=np.float32)
  out = np.stack(segs, axis=0).astype(np.float32)
  # Final finite filter for extra safety.
  finite_mask = np.isfinite(out).all(axis=(1, 2))
  if not finite_mask.any():
    return np.zeros((0, 2, 3), dtype=np.float32)
  return out[finite_mask]
