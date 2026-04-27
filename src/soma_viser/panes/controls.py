"""Controls tab pane for SomaViewer."""

from __future__ import annotations

from typing import Any

import numpy as np
import viser

from .common import PaneContext
from ..core.joint_overrides import is_main_joint
from ..core.math import euler_zyx_deg_to_wxyz, wxyz_to_euler_zyx_deg
from ..core.motion import export_current_pose_bvh


class ControlsPane:
  """Owns Controls tab UI and callbacks."""

  def __init__(self, context: PaneContext) -> None:
    self.context = context
    self.viewer = context.viewer
    self.actions = context.actions

  def build_tab(self) -> None:
    v = self.viewer
    v._show_joint_knobs = False
    show_gizmo_cb = v.server.gui.add_checkbox("Show root gizmo", initial_value=True)
    root_x = v.server.gui.add_number("Root offset X", initial_value=0.0, step=0.01)
    root_y = v.server.gui.add_number("Root offset Y", initial_value=0.0, step=0.01)
    root_z = v.server.gui.add_number("Root offset Z", initial_value=0.0, step=0.01)
    align_rx = v.server.gui.add_slider("Align rot X (deg)", -180.0, 180.0, 1.0, 90.0)
    align_ry = v.server.gui.add_slider("Align rot Y (deg)", -180.0, 180.0, 1.0, 0.0)
    align_rz = v.server.gui.add_slider("Align rot Z (deg)", -180.0, 180.0, 1.0, 90.0)
    scale_slider = v.server.gui.add_slider("Body scale", min=0.1, max=3.0, step=0.02, initial_value=1.0)
    export_name = v.server.gui.add_text("Export file", initial_value="pose_export.bvh")
    export_btn = v.server.gui.add_button("Export BVH (current config)")
    export_status = v.server.gui.add_text("Export status", initial_value="-", disabled=True)
    show_knobs_cb = v.server.gui.add_checkbox("Show joint gizmos", initial_value=v._show_joint_knobs)

    def _mark_redraw() -> None:
      self.actions.request_redraw()

    v._root_number_controls = (root_x, root_y, root_z)
    v._root_gizmo = v.server.scene.add_transform_controls(
      "/soma/root_gizmo",
      scale=0.14,
      line_width=1.4,
      position=tuple(float(val) for val in v._root_offset),
      wxyz=(1.0, 0.0, 0.0, 0.0),
      depth_test=False,
      disable_rotations=True,
      opacity=0.92,
      visible=True,
    )

    @show_gizmo_cb.on_update
    def _(_) -> None:
      if v._root_gizmo is not None:
        v._root_gizmo.visible = bool(show_gizmo_cb.value)

    @v._root_gizmo.on_update
    def _(event) -> None:
      p = np.asarray(event.target.position, dtype=np.float64)
      v._root_offset[:] = p
      if v._root_number_controls is not None:
        v._suppress_root_ui = True
        try:
          rx, ry, rz = v._root_number_controls
          rx.value = float(p[0])
          ry.value = float(p[1])
          rz.value = float(p[2])
        finally:
          v._suppress_root_ui = False
      _mark_redraw()

    @root_x.on_update
    def _(_) -> None:
      if v._suppress_root_ui:
        return
      v._root_offset[0] = float(root_x.value)
      if v._root_gizmo is not None:
        v._root_gizmo.position = tuple(float(val) for val in v._root_offset)
      _mark_redraw()

    @root_y.on_update
    def _(_) -> None:
      if v._suppress_root_ui:
        return
      v._root_offset[1] = float(root_y.value)
      if v._root_gizmo is not None:
        v._root_gizmo.position = tuple(float(val) for val in v._root_offset)
      _mark_redraw()

    @root_z.on_update
    def _(_) -> None:
      if v._suppress_root_ui:
        return
      v._root_offset[2] = float(root_z.value)
      if v._root_gizmo is not None:
        v._root_gizmo.position = tuple(float(val) for val in v._root_offset)
      _mark_redraw()

    @align_rx.on_update
    def _(_) -> None:
      v._align_euler[0] = float(align_rx.value)
      _mark_redraw()

    @align_ry.on_update
    def _(_) -> None:
      v._align_euler[1] = float(align_ry.value)
      _mark_redraw()

    @align_rz.on_update
    def _(_) -> None:
      v._align_euler[2] = float(align_rz.value)
      _mark_redraw()

    @scale_slider.on_update
    def _(_) -> None:
      v._body_scale = max(0.01, float(scale_slider.value))
      _mark_redraw()

    @export_btn.on_click
    def _(_) -> None:
      export_status.value = export_current_pose_bvh(v, str(export_name.value))

    with v.server.gui.add_folder("Joint controls (main, no fingers)"):
      v._joint_controls_folder = v.server.gui.add_folder("Joint list")

    @show_knobs_cb.on_update
    def _(_) -> None:
      self.actions.set_show_joint_knobs(bool(show_knobs_cb.value))

  def attach(self, *_args: Any, **_kwargs: Any) -> None:
    self.build_tab()

  def tick(self, _dt: float) -> None:
    return None

  def dispose(self) -> None:
    return None

  def rebuild_main_joint_controls(self) -> None:
    v = self.viewer
    folder = getattr(v, "_joint_controls_folder", None)
    if folder is None:
      return
    for c in v._main_joint_controls.values():
      c.remove()
    v._main_joint_controls.clear()
    for h in v._main_joint_knobs.values():
      h.remove()
    v._main_joint_knobs.clear()
    v._main_joint_overrides_deg.clear()
    v._main_joint_display_deg.clear()
    if v._clip is None:
      return

    with folder:
      for jname in v._joint_names:
        if not is_main_joint(jname):
          continue
        ctrl = v.server.gui.add_vector3(f"{jname} (Z,Y,X deg)", initial_value=(0.0, 0.0, 0.0), step=1.0)
        v._main_joint_controls[jname] = ctrl
        knob = v.server.scene.add_transform_controls(
          f"/soma/joint_knobs/{jname}",
          scale=0.12,
          line_width=1.8,
          disable_axes=True,
          disable_sliders=True,
          visible=v._show_joint_knobs,
          depth_test=False,
          opacity=1.0,
        )
        v._main_joint_knobs[jname] = knob
        v._main_joint_overrides_deg[jname] = np.zeros(3, dtype=np.float64)
        v._main_joint_display_deg[jname] = np.zeros(3, dtype=np.float64)

        def _bind(name: str, h: Any, k: viser.TransformControlsHandle) -> None:
          @h.on_update
          def _(_event) -> None:
            if v._suppress_joint_ui:
              return
            val = np.asarray(h.value, dtype=np.float64)
            prev = v._main_joint_display_deg.get(name, np.zeros(3, dtype=np.float64))
            delta = val - prev
            v._main_joint_overrides_deg[name] = v._main_joint_overrides_deg.get(name, np.zeros(3, dtype=np.float64)) + delta
            v._main_joint_display_deg[name] = val
            k.wxyz = euler_zyx_deg_to_wxyz(val)
            v._needs_redraw = True

        _bind(jname, ctrl, knob)

        def _bind_knob(name: str, h: Any, k: viser.TransformControlsHandle) -> None:
          @k.on_update
          def _(_event) -> None:
            if v._suppress_joint_ui:
              return
            val = wxyz_to_euler_zyx_deg(k.wxyz)
            prev = v._main_joint_display_deg.get(name, np.zeros(3, dtype=np.float64))
            delta = val - prev
            v._main_joint_overrides_deg[name] = v._main_joint_overrides_deg.get(name, np.zeros(3, dtype=np.float64)) + delta
            v._main_joint_display_deg[name] = val
            v._suppress_joint_ui = True
            try:
              h.value = (float(val[0]), float(val[1]), float(val[2]))
            finally:
              v._suppress_joint_ui = False
            v._needs_redraw = True

        _bind_knob(jname, ctrl, knob)
