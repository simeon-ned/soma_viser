"""Motion loading and BVH parsing utilities for SOMA clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .joint_overrides import is_hips_like_joint
from .math import wxyz_to_euler_zyx_deg


@dataclass(frozen=True)
class MotionEntry:
  """Indexed metadata for one BVH file in the clip library."""

  rel_path: str
  name: str
  path: Path


@dataclass
class MotionClip:
  """A loaded BVH clip and its skeleton metadata."""

  name: str
  path: Path
  skeleton: Any
  animation: Any
  sample_rate: float
  num_frames: int


def list_bvh_files(motions_dir: Path) -> list[Path]:
  """Return sorted .bvh files in a motion directory (non-recursive)."""
  if not motions_dir.is_dir():
    return []
  return sorted(p for p in motions_dir.glob("*.bvh") if p.is_file())


def index_bvh_files(motions_dir: Path, recursive: bool = True) -> list[MotionEntry]:
  """Return sorted BVH library entries, optionally including subfolders."""
  if not motions_dir.is_dir():
    return []
  iter_paths = motions_dir.rglob("*.bvh") if recursive else motions_dir.glob("*.bvh")
  entries: list[MotionEntry] = []
  for path in iter_paths:
    if not path.is_file():
      continue
    rel_path = path.relative_to(motions_dir).as_posix()
    entries.append(MotionEntry(rel_path=rel_path, name=path.name, path=path))
  return sorted(entries, key=lambda e: e.rel_path.lower())


def load_motion_clip(path: Path) -> MotionClip:
  """Load a BVH clip using soma-retargeter BVH tools."""
  try:
    from soma_retargeter.assets.bvh import load_bvh
  except ImportError as exc:
    raise ImportError(
      "soma_retargeter is required to load BVH clips. "
      "Install with: pip install -e /path/to/soma-retargeter"
    ) from exc

  skeleton, animation = load_bvh(str(path))
  sample_rate = float(animation.sample_rate) if animation.sample_rate > 0 else 60.0
  return MotionClip(
    name=path.name,
    path=path.resolve(),
    skeleton=skeleton,
    animation=animation,
    sample_rate=sample_rate,
    num_frames=int(animation.num_frames),
  )


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


def export_current_pose_bvh(viewer: Any, export_name: str) -> str:
  """Export the current rendered pose to a single-frame BVH file."""
  if viewer._clip is None:
    return "No clip loaded."
  if not viewer._clip.path.is_file():
    return "Source BVH not found."
  try:
    lines, row_start = extract_bvh_motion_rows(viewer._clip.path)
    frame_idx = 0 if viewer._frame_idx >= max(1, viewer._clip.num_frames) else int(viewer._frame_idx)
    src_row_idx = row_start + frame_idx
    if src_row_idx >= len(lines):
      src_row_idx = row_start
    src_vals = [float(x) for x in lines[src_row_idx].strip().split()]
    out_vals = list(src_vals)
    bvh_units_per_meter = estimate_bvh_units_per_meter(src_vals, viewer._bvh_channel_map)
    row_live = viewer._last_rendered_row
    if row_live is None:
      return "No rendered pose cached yet."
    row_arr = np.asarray(row_live)
    packed_transform_array = (
      row_arr.ndim == 2
      and row_arr.shape[0] == len(viewer._joint_names)
      and row_arr.shape[1] >= 7
      and row_arr.dtype.kind in "fc"
    )
    import warp as wp

    for jname, jidx in viewer._joint_name_to_idx.items():
      info = viewer._bvh_channel_map.get(jname)
      if info is None:
        continue
      start, labels = info
      ch_idx = {lbl: start + i for i, lbl in enumerate(labels)}
      if packed_transform_array:
        qx, qy, qz, qw = (
          float(row_live[jidx, 3]),
          float(row_live[jidx, 4]),
          float(row_live[jidx, 5]),
          float(row_live[jidx, 6]),
        )
      else:
        ti = row_live[jidx]
        qr = wp.transform_get_rotation(ti)
        qx, qy, qz, qw = float(qr[0]), float(qr[1]), float(qr[2]), float(qr[3])
      eul = wxyz_to_euler_zyx_deg((qw, qx, qy, qz))
      if is_hips_like_joint(jname):
        if "Xposition" in ch_idx:
          out_vals[ch_idx["Xposition"]] = float(src_vals[ch_idx["Xposition"]]) + float(viewer._root_offset[0]) * bvh_units_per_meter
        if "Yposition" in ch_idx:
          out_vals[ch_idx["Yposition"]] = float(src_vals[ch_idx["Yposition"]]) + float(viewer._root_offset[1]) * bvh_units_per_meter
        if "Zposition" in ch_idx:
          out_vals[ch_idx["Zposition"]] = float(src_vals[ch_idx["Zposition"]]) + float(viewer._root_offset[2]) * bvh_units_per_meter
      if "Zrotation" in ch_idx:
        out_vals[ch_idx["Zrotation"]] = float(eul[0])
      if "Yrotation" in ch_idx:
        out_vals[ch_idx["Yrotation"]] = float(eul[1])
      if "Xrotation" in ch_idx:
        out_vals[ch_idx["Xrotation"]] = float(eul[2])

    header = lines[:row_start]
    frame_time = 1.0 / max(1e-6, float(viewer._clip.sample_rate))
    for i, line in enumerate(header):
      if line.strip().startswith("Frames:"):
        header[i] = "Frames: 1"
      elif line.strip().startswith("Frame Time:"):
        header[i] = f"Frame Time: {frame_time:.6f}"
    out_name = export_name.strip() or "pose_export.bvh"
    if not out_name.endswith(".bvh"):
      out_name += ".bvh"
    out_dir = viewer.motions_dir / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    out_text = "\n".join(header + [" ".join(f"{value:.9g}" for value in out_vals)]) + "\n"
    out_path.write_text(out_text, encoding="utf-8")
    return f"Saved: {out_path}"
  except Exception as exc:  # pragma: no cover - UI-facing error path
    return f"Export failed: {exc}"
