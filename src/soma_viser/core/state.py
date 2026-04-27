"""Viewer session state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PlaybackState:
  """Playback fields used by the viewer tick loop."""

  playing: bool = True
  loop: bool = True
  speed_idx: int = 0
  accumulator: float = 0.0
  frame_idx: int = 0
  needs_redraw: bool = True


@dataclass
class VisualizationState:
  """Visualization and transform values used by panes/renderers."""

  skeleton_color: tuple[int, int, int] = (160, 220, 100)
  line_width: float = 2.8
  mesh_color: tuple[int, int, int] = (195, 170, 215)
  mesh_opacity: float = 0.45
  root_offset: np.ndarray = field(
    default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
  )
  align_euler: np.ndarray = field(
    default_factory=lambda: np.array([90.0, 0.0, 90.0], dtype=np.float64)
  )
  body_scale: float = 1.0
  show_skeleton: bool = True
  show_mesh: bool = True
  show_frame_text: bool = True
  show_frame_text_full_info: bool = True
  show_joint_knobs: bool = True


@dataclass
class MotionLibraryState:
  """Indexed motion list and filtering UI state."""

  entries: list[Any] = field(default_factory=list)
  filtered_entries: list[Any] = field(default_factory=list)
  entry_by_relpath: dict[str, Any] = field(default_factory=dict)
  folder_counts: dict[str, int] = field(default_factory=dict)
  search_pending: bool = False
  search_deadline: float = 0.0
  suppress_motion_ui: bool = False


@dataclass
class SessionState:
  """Top-level viewer session aggregate."""

  playback: PlaybackState = field(default_factory=PlaybackState)
  visual: VisualizationState = field(default_factory=VisualizationState)
  motion: MotionLibraryState = field(default_factory=MotionLibraryState)
