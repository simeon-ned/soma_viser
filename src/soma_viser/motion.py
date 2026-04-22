"""Motion loading utilities for SOMA BVH clips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
  """Return sorted .bvh files in a motion directory."""
  if not motions_dir.is_dir():
    return []
  return sorted(p for p in motions_dir.glob("*.bvh") if p.is_file())


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
