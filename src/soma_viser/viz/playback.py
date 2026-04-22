"""Playback helpers for frame stepping and speed control."""

from __future__ import annotations

DEFAULT_SPEEDS: tuple[float, ...] = (1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0)


def update_speed_index(current_index: int, action: str) -> int:
  """Map speed button action to a new speed index."""
  if action == "Slower":
    return max(0, current_index - 1)
  if action == "Faster":
    return min(len(DEFAULT_SPEEDS) - 1, current_index + 1)
  return DEFAULT_SPEEDS.index(1.0)


def advance_playback(
  frame_idx: int,
  accumulator: float,
  dt: float,
  *,
  sample_rate: float,
  speed: float,
  num_frames: int,
  loop: bool,
  playing: bool,
) -> tuple[int, float, bool, bool]:
  """Advance playback state and report whether frame changed."""
  if not playing:
    return frame_idx, accumulator, playing, False

  fps = max(1e-6, sample_rate * speed)
  step = 1.0 / fps
  accumulator += dt
  advanced = False
  while accumulator >= step:
    accumulator -= step
    frame_idx += 1
    if frame_idx >= num_frames:
      if loop:
        frame_idx = 0
      else:
        frame_idx = num_frames - 1
        playing = False
        break
    advanced = True
  return frame_idx, accumulator, playing, advanced
