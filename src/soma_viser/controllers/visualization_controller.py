"""Visualization tab controller for SomaViewer."""

from __future__ import annotations

from typing import Any

from ..viz.joint_inspector import JointInspector


class VisualizationController:
  """Owns Visualization tab UI and callbacks."""

  def __init__(self, viewer: Any) -> None:
    self.viewer = viewer

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
      v._show_skeleton = bool(show_skeleton_cb.value)
      v._line_handle.visible = v._show_skeleton

    @show_mesh_cb.on_update
    def _(_) -> None:
      v._show_mesh = bool(show_mesh_cb.value)
      v._needs_redraw = True

    @color_picker.on_update
    def _(_) -> None:
      v._skeleton_color = tuple(int(c) for c in color_picker.value)
      v._needs_redraw = True

    @mesh_color_picker.on_update
    def _(_) -> None:
      v._mesh_color = tuple(int(c) for c in mesh_color_picker.value)
      v._needs_redraw = True

    @mesh_opacity_slider.on_update
    def _(_) -> None:
      v._mesh_opacity = float(mesh_opacity_slider.value)
      if v._mesh_handle is not None:
        v._mesh_handle.opacity = v._mesh_opacity
      v._needs_redraw = True

    @line_width_slider.on_update
    def _(_) -> None:
      v._line_width = float(line_width_slider.value)
      v._line_handle.line_width = v._line_width

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
      v._show_frame_text = bool(frame_text_cb.value)
      if v._joint_inspector is not None:
        v._joint_inspector.show_frame_text = v._show_frame_text
        v._joint_inspector.update_text_visibility()

    @frame_text_full_info_cb.on_update
    def _(_) -> None:
      v._show_frame_text_full_info = bool(frame_text_full_info_cb.value)
      if v._joint_inspector is not None:
        v._joint_inspector.show_frame_text_full_info = v._show_frame_text_full_info
      v._needs_redraw = True

  def attach(self, *_args: Any, **_kwargs: Any) -> None:
    """Stable controller contract used by downstream integrations."""
    self.build_tab()

  def tick(self, _dt: float) -> None:
    """Visualization tab does not require a periodic update."""
    return None

  def dispose(self) -> None:
    """No explicit resources to dispose."""
    return None
