"""BVH parsing/export utility helpers."""

from __future__ import annotations

from pathlib import Path


def parse_bvh_channel_map(path: Path) -> dict[str, tuple[int, list[str]]]:
  """Map joint name -> (channel start index, channel labels)."""
  lines = path.read_text(encoding="utf-8").splitlines()
  channel_cursor = 0
  active_name: str | None = None
  out: dict[str, tuple[int, list[str]]] = {}
  for line in lines:
    s = line.strip()
    if s.startswith("MOTION"):
      break
    if s.startswith("ROOT "):
      active_name = s.split(maxsplit=1)[1].strip()
      continue
    if s.startswith("JOINT "):
      active_name = s.split(maxsplit=1)[1].strip()
      continue
    if s.startswith("CHANNELS "):
      parts = s.split()
      if len(parts) >= 2:
        n = int(parts[1])
        labels = parts[2 : 2 + n]
        if active_name is not None:
          out[active_name] = (channel_cursor, labels)
        channel_cursor += n
  return out


def extract_bvh_motion_rows(path: Path) -> tuple[list[str], int]:
  """Return (all lines, motion row start index)."""
  lines = path.read_text(encoding="utf-8").splitlines()
  start_idx = -1
  for i, line in enumerate(lines):
    if line.strip().startswith("Frame Time:"):
      start_idx = i + 1
      break
  if start_idx < 0:
    raise ValueError(f"BVH missing Frame Time: {path}")
  return lines, start_idx


def estimate_bvh_units_per_meter(
  source_values: list[float],
  channel_map: dict[str, tuple[int, list[str]]],
) -> float:
  """Heuristic scale from viewer meters -> BVH translation units."""
  for jname, (start, labels) in channel_map.items():
    lname = jname.strip().lower()
    if lname not in ("hips", "hip", "pelvis", "root", "rootjoint"):
      continue
    idx_y = None
    for i, lbl in enumerate(labels):
      if lbl == "Yposition":
        idx_y = start + i
        break
    if idx_y is None or idx_y >= len(source_values):
      continue
    y_abs = abs(float(source_values[idx_y]))
    return 100.0 if y_abs > 10.0 else 1.0
  return 1.0
