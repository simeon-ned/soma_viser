"""Motion loading utilities for SOMA BVH clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
