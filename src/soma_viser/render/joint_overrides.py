"""Joint override math utilities shared by controllers/renderers."""

from __future__ import annotations

from typing import Any

import numpy as np
import viser.transforms as tf

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


def is_main_joint(name: str) -> bool:
  lname = name.strip().lower()
  if lname.endswith("end"):
    return False
  return not any(tok in lname for tok in _FINGER_TOKENS)


def is_hips_like_joint(name: str) -> bool:
  lname = name.strip().lower()
  return lname in ("hips", "hip", "pelvis", "root", "rootjoint")


def euler_zyx_deg_to_wxyz(euler_deg: np.ndarray) -> tuple[float, float, float, float]:
  rz = float(np.radians(euler_deg[0]))
  ry = float(np.radians(euler_deg[1]))
  rx = float(np.radians(euler_deg[2]))
  so3 = tf.SO3.from_z_radians(rz) @ tf.SO3.from_y_radians(ry) @ tf.SO3.from_x_radians(rx)
  w, x, y, z = so3.wxyz
  return (float(w), float(x), float(y), float(z))


def wxyz_to_euler_zyx_deg(wxyz: tuple[float, float, float, float]) -> np.ndarray:
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


def apply_joint_overrides_to_row(
  row: Any,
  *,
  num_joints: int,
  main_joint_overrides_deg: dict[str, np.ndarray],
  joint_name_to_idx: dict[str, int],
) -> tuple[Any, bool]:
  """Apply additive joint euler overrides to one local-transform row."""
  import warp as wp

  row_arr = np.asarray(row)
  packed_transform_array = (
    row_arr.ndim == 2
    and row_arr.shape[0] == num_joints
    and row_arr.shape[1] >= 7
    and row_arr.dtype.kind in "fc"
  )
  for jname, deg_zyx in main_joint_overrides_deg.items():
    jidx = joint_name_to_idx.get(jname)
    if jidx is None:
      continue
    rz = float(np.radians(deg_zyx[0]))
    ry = float(np.radians(deg_zyx[1]))
    rx = float(np.radians(deg_zyx[2]))
    so3 = tf.SO3.from_z_radians(rz) @ tf.SO3.from_y_radians(ry) @ tf.SO3.from_x_radians(rx)
    w, x, y, z = so3.wxyz
    if packed_transform_array:
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
      qr = wp.transform_get_rotation(ti)
      base_so3 = tf.SO3(wxyz=(float(qr[3]), float(qr[0]), float(qr[1]), float(qr[2])))
      out_so3 = base_so3 @ tf.SO3(wxyz=(float(w), float(x), float(y), float(z)))
      ow, ox, oy, oz = out_so3.wxyz
      row[jidx] = wp.transform(tr, wp.quat(float(ox), float(oy), float(oz), float(ow)))
  return row, packed_transform_array
