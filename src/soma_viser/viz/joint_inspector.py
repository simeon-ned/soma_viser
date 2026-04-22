"""Joint frame inspector UI and scene helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FrameInspectorRow:
  """GUI + scene handles for one inspected joint frame."""

  joint_name: str
  checkbox: Any
  frame_handle: Any
  scene_label: Any


class JointInspector:
  """Build and update the Joint Frame Inspector panel."""

  def __init__(
    self,
    server: Any,
    *,
    show_frame_text: bool = True,
    show_frame_text_full_info: bool = True,
  ) -> None:
    self._server = server
    self.show_frame_text = show_frame_text
    self.show_frame_text_full_info = show_frame_text_full_info
    self.needs_redraw = False
    self.folder: Any = self._server.gui.add_folder("Joint frame inspector")
    self.rows: list[FrameInspectorRow] = []
    self._category_folders: list[Any] = []

  def rebuild(self, joint_names: list[str]) -> None:
    for folder in self._category_folders:
      folder.remove()
    self._category_folders.clear()

    for row in self.rows:
      row.checkbox.remove()
      row.frame_handle.remove()
      row.scene_label.remove()
    self.rows.clear()

    grouped: dict[str, list[str]] = {
      "Body": [],
      "Arms": [],
      "Hands": [],
      "Legs": [],
      "Head": [],
      "Other": [],
    }
    for joint_name in joint_names:
      grouped[self._joint_category(joint_name)].append(joint_name)

    with self.folder:
      for category in ("Body", "Head", "Arms", "Hands", "Legs", "Other"):
        names = grouped[category]
        if not names:
          continue
        category_folder = self._server.gui.add_folder(
          category, expand_by_default=(category == "Body")
        )
        self._category_folders.append(category_folder)
        with category_folder:
          for joint_name in names:
            checkbox = self._server.gui.add_checkbox(joint_name, initial_value=False)
            frame_handle = self._server.scene.add_frame(
              f"/soma/joint_frames/{joint_name}",
              axes_length=0.12,
              axes_radius=0.003,
              visible=False,
            )
            scene_label = self._server.scene.add_label(
              f"/soma/joint_frames/{joint_name}_text",
              text="",
              visible=False,
              anchor="top-left",
              depth_test=False,
              font_size_mode="screen",
              font_screen_scale=0.9,
            )
            row = FrameInspectorRow(joint_name, checkbox, frame_handle, scene_label)
            self.rows.append(row)

            @checkbox.on_update
            def _(_, _row=row) -> None:
              if not _row.checkbox.value:
                _row.frame_handle.visible = False
                _row.scene_label.visible = False
              self.needs_redraw = True

    self.needs_redraw = True

  def update_text_visibility(self) -> None:
    for row in self.rows:
      row.scene_label.visible = self.show_frame_text and row.checkbox.value

  @staticmethod
  def _joint_category(joint_name: str) -> str:
    name = joint_name.lower()
    if any(
      k in name
      for k in (
        "finger",
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
        "pinkie",
        "metacarp",
        "knuckle",
      )
    ):
      return "Hands"
    if any(k in name for k in ("eye", "jaw", "chin", "neck", "head", "skull")):
      return "Head"
    if any(k in name for k in ("arm", "shoulder", "elbow", "wrist", "hand", "clavicle")):
      return "Arms"
    if any(k in name for k in ("leg", "thigh", "knee", "ankle", "foot", "toe", "calf", "shin")):
      return "Legs"
    if any(k in name for k in ("hips", "pelvis", "spine", "chest", "root", "torso", "abdomen")):
      return "Body"
    return "Other"
