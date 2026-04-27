"""Math helpers shared across core/render modules."""

from __future__ import annotations

import numpy as np
import viser.transforms as tf


def euler_zyx_deg_to_wxyz(euler_deg: np.ndarray) -> tuple[float, float, float, float]:
  """Convert ZYX Euler degrees to quaternion in wxyz order."""
  rz = float(np.radians(euler_deg[0]))
  ry = float(np.radians(euler_deg[1]))
  rx = float(np.radians(euler_deg[2]))
  so3 = tf.SO3.from_z_radians(rz) @ tf.SO3.from_y_radians(ry) @ tf.SO3.from_x_radians(rx)
  w, x, y, z = so3.wxyz
  return (float(w), float(x), float(y), float(z))


def wxyz_to_euler_zyx_deg(wxyz: tuple[float, float, float, float]) -> np.ndarray:
  """Convert quaternion in wxyz order to ZYX Euler degrees."""
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
