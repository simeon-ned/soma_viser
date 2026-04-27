"""Visualization tab pane for SomaViewer."""

from __future__ import annotations

from typing import Any

from .common import PaneContext
from .joint_inspector import JointInspector


class VisualizationPane:
  """Owns Visualization tab UI and callbacks."""

  def __init__(self, context: PaneContext) -> None:
    self.context = context
    self.viewer = context.viewer
    self.actions = context.actions

  def build_tab(self) -> None:
    v = self.viewer
    show_skeleton_cb = v.server.gui.add_checkbox("Show skeleton", initial_value=True)
    show_mesh_cb = v.server.gui.add_checkbox("Show skinned mesh", initial_value=True)
    color_picker = v.server.gui.add_rgb("Skeleton color", initial_value=v._skeleton_color)
    mesh_color_picker = v.server.gui.add_rgb("Mesh color", initial_value=v._mesh_color)
    mesh_opacity_slider = v.server.gui.add_slider(
      "Mesh opacity",
      min=0.0,
      max=1.0,
      step=0.01,
      initial_value=v._mesh_opacity,
    )
    line_width_slider = v.server.gui.add_slider("Line width", min=1.0, max=8.0, step=0.1, initial_value=v._line_width)

    @show_skeleton_cb.on_update
    def _(_) -> None:
      self.actions.set_show_skeleton(bool(show_skeleton_cb.value))

    @show_mesh_cb.on_update
    def _(_) -> None:
      self.actions.set_show_mesh(bool(show_mesh_cb.value))

    @color_picker.on_update
    def _(_) -> None:
      self.actions.set_skeleton_color(tuple(int(c) for c in color_picker.value))

    @mesh_color_picker.on_update
    def _(_) -> None:
      self.actions.set_mesh_color(tuple(int(c) for c in mesh_color_picker.value))

    @mesh_opacity_slider.on_update
    def _(_) -> None:
      self.actions.set_mesh_opacity(float(mesh_opacity_slider.value))

    @line_width_slider.on_update
    def _(_) -> None:
      self.actions.set_line_width(float(line_width_slider.value))

    v._joint_inspector = JointInspector(
      v.server,
      show_frame_text=v._show_frame_text,
      show_frame_text_full_info=v._show_frame_text_full_info,
    )
    with v._joint_inspector.folder:
      frame_text_cb = v.server.gui.add_checkbox("Show frame text in scene", initial_value=v._show_frame_text)
      frame_text_full_info_cb = v.server.gui.add_checkbox(
        "Show full text info (xyz + quat)",
        initial_value=v._show_frame_text_full_info,
      )

    @frame_text_cb.on_update
    def _(_) -> None:
      self.actions.set_show_frame_text(bool(frame_text_cb.value))

    @frame_text_full_info_cb.on_update
    def _(_) -> None:
      self.actions.set_show_frame_text_full_info(bool(frame_text_full_info_cb.value))

  def attach(self, *_args: Any, **_kwargs: Any) -> None:
    self.build_tab()

  def tick(self, _dt: float) -> None:
    return None

  def dispose(self) -> None:
    return None
