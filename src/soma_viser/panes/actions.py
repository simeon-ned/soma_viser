"""Small action/service layer used by panes."""

from __future__ import annotations

from typing import Any

import viser


class PaneActions:
  """Typed mutations and side-effect helpers for pane callbacks."""

  def __init__(self, viewer: Any) -> None:
    self.viewer = viewer

  def request_redraw(self) -> None:
    self.viewer._needs_redraw = True

  def update_status(self, extra: str | None = None) -> None:
    self.viewer._update_status_text(extra)

  def load_clip_by_relpath(self, relpath: str) -> None:
    self.viewer._load_clip_by_relpath(relpath)

  def load_clip_by_path(self, path: Any) -> None:
    self.viewer._load_clip_by_path(path)

  def toggle_play_pause(self) -> bool:
    self.viewer._playing = not self.viewer._playing
    self.viewer._update_status_text()
    return bool(self.viewer._playing)

  def set_speed_idx(self, speed_idx: int) -> None:
    self.viewer._speed_idx = int(speed_idx)
    self.viewer._update_status_text()

  def set_loop(self, loop: bool) -> None:
    self.viewer._loop = bool(loop)
    self.viewer._update_status_text()

  def set_frame_idx(self, frame_idx: int) -> None:
    self.viewer._frame_idx = int(frame_idx)
    self.viewer._needs_redraw = True

  def set_show_skeleton(self, show: bool) -> None:
    self.viewer._show_skeleton = bool(show)
    self.viewer._line_handle.visible = self.viewer._show_skeleton

  def set_show_mesh(self, show: bool) -> None:
    self.viewer._show_mesh = bool(show)
    self.viewer._needs_redraw = True

  def set_skeleton_color(self, color: tuple[int, int, int]) -> None:
    self.viewer._skeleton_color = tuple(int(c) for c in color)
    self.viewer._needs_redraw = True

  def set_mesh_color(self, color: tuple[int, int, int]) -> None:
    self.viewer._mesh_color = tuple(int(c) for c in color)
    self.viewer._needs_redraw = True

  def set_mesh_opacity(self, opacity: float) -> None:
    self.viewer._mesh_opacity = float(opacity)
    if self.viewer._mesh_handle is not None:
      self.viewer._mesh_handle.opacity = self.viewer._mesh_opacity
    self.viewer._needs_redraw = True

  def set_line_width(self, line_width: float) -> None:
    self.viewer._line_width = float(line_width)
    self.viewer._line_handle.line_width = self.viewer._line_width

  def set_show_frame_text(self, show: bool) -> None:
    self.viewer._show_frame_text = bool(show)
    if self.viewer._joint_inspector is not None:
      self.viewer._joint_inspector.show_frame_text = self.viewer._show_frame_text
      self.viewer._joint_inspector.update_text_visibility()

  def set_show_frame_text_full_info(self, show: bool) -> None:
    self.viewer._show_frame_text_full_info = bool(show)
    if self.viewer._joint_inspector is not None:
      self.viewer._joint_inspector.show_frame_text_full_info = self.viewer._show_frame_text_full_info
    self.viewer._needs_redraw = True

  def set_show_joint_knobs(self, show: bool) -> None:
    self.viewer._show_joint_knobs = bool(show)
    for handle in self.viewer._main_joint_knobs.values():
      handle.visible = self.viewer._show_joint_knobs

  def set_play_button_state(self, button: viser.GuiButtonHandle, playing: bool) -> None:
    button.label = "Pause" if playing else "Play"
    button.icon = viser.Icon.PLAYER_PAUSE if playing else viser.Icon.PLAYER_PLAY
