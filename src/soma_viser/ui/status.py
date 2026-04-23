"""Status text formatting helpers for the viewer HUD."""

from __future__ import annotations


def build_status_html(
  *,
  clip_name: str | None,
  num_frames: int | None,
  frame_idx: int,
  speed: float,
  sample_rate: float | None,
  playing: bool,
  has_mesh_skinner: bool,
  mesh_warn: str | None,
  extra: str | None = None,
) -> str:
  """Build status panel HTML for the current viewer state."""
  if clip_name is None or num_frames is None or sample_rate is None:
    msg = extra or "No clip loaded."
    return f"<div>{msg}</div>"

  state = "Playing" if playing else "Paused"
  mesh_status = "available" if has_mesh_skinner else "unavailable"
  if mesh_warn:
    mesh_status = f"unavailable ({mesh_warn})"
  msg = extra or ""

  return (
    "<div style='font-size:0.85em;line-height:1.3'>"
    f"<strong>{state}</strong><br/>"
    f"Clip: {clip_name}<br/>"
    f"Frame: {frame_idx}/{num_frames - 1}<br/>"
    f"Speed: {speed:.2f}x<br/>"
    f"Base FPS: {sample_rate:.1f}<br/>"
    f"Mesh: {mesh_status}<br/>"
    f"{msg}"
    "</div>"
  )
