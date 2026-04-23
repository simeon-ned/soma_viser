"""Viewer session state containers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlaybackState:
  """Playback fields used by the viewer tick loop."""

  playing: bool = True
  loop: bool = True
  speed_idx: int = 0
  accumulator: float = 0.0
  frame_idx: int = 0
  needs_redraw: bool = True
