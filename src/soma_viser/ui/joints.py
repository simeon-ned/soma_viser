"""Joint-name related utilities for viewer logic."""

from __future__ import annotations


def resolve_hips_joint_name(joint_names: list[str]) -> str | None:
  """Resolve the best hips/root joint candidate from joint names."""
  if not joint_names:
    return None

  lower_to_name = {n.lower(): n for n in joint_names}
  exact_keys = ("hips", "hip", "pelvis", "root", "rootjoint")
  for key in exact_keys:
    if key in lower_to_name:
      return lower_to_name[key]

  lower_names = [n.lower() for n in joint_names]
  for key in exact_keys:
    for i, lname in enumerate(lower_names):
      if key in lname:
        return joint_names[i]
  return joint_names[0]
