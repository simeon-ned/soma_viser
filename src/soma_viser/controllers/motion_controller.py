"""Motion tab controller for SomaViewer."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import viser

from ..motion import MotionEntry, index_bvh_files
from ..viz.playback import update_speed_index

_MOTION_MAX_OPTIONS = 300
_MOTION_FOLDER_SHOW_THRESHOLD = 120
_MOTION_SEARCH_DEBOUNCE_SEC = 0.2


class MotionController:
  """Owns Motion tab UI and filtering callbacks."""

  def __init__(self, viewer: Any, pose_keyframes: dict[str, Path]) -> None:
    self.viewer = viewer
    self.pose_keyframes = pose_keyframes

  def build_tab(self) -> None:
    v = self.viewer
    v._motion_search_text = v.server.gui.add_text("Search clips", initial_value="")
    v._motion_folder_dropdown = v.server.gui.add_dropdown(
      "Folder",
      options=["(all folders)"],
      initial_value="(all folders)",
    )
    v._motion_dropdown = v.server.gui.add_dropdown(
      "Motion clip",
      options=["(no .bvh files found)"],
      initial_value="(no .bvh files found)",
    )
    refresh_btn = v.server.gui.add_button("Refresh library")
    v._motion_page_text = v.server.gui.add_text(
      "Results",
      initial_value="No clips indexed.",
      disabled=True,
    )
    v._frame_slider = v.server.gui.add_slider("Frame", min=0, max=1, step=1, initial_value=0)
    v._status_html = v.server.gui.add_html("")
    play_btn = v.server.gui.add_button("Pause", icon=viser.Icon.PLAYER_PAUSE)
    speed_btn = v.server.gui.add_button_group("Speed", options=["Slower", "1x", "Faster"])
    loop_cb = v.server.gui.add_checkbox("Loop", initial_value=True)
    pose_dd = v.server.gui.add_dropdown(
      "Pose keyframe",
      options=list(self.pose_keyframes.keys()),
      initial_value="Calibration (frame0)",
    )

    @v._motion_search_text.on_update
    def _(_) -> None:
      v._motion_search_pending = True
      v._motion_search_deadline = time.perf_counter() + _MOTION_SEARCH_DEBOUNCE_SEC

    @v._motion_folder_dropdown.on_update
    def _(_) -> None:
      if v._suppress_motion_ui:
        return
      self.apply_motion_filters()

    @v._motion_dropdown.on_update
    def _(_) -> None:
      if v._suppress_motion_ui:
        return
      selected = str(v._motion_dropdown.value)
      if selected.startswith("("):
        return
      v._load_clip_by_relpath(selected)

    @refresh_btn.on_click
    def _(_) -> None:
      self.refresh_motion_library()

    @v._frame_slider.on_update
    def _(_) -> None:
      if v._suppress_gui_updates:
        return
      v._frame_idx = int(v._frame_slider.value)
      v._needs_redraw = True

    @play_btn.on_click
    def _(_) -> None:
      v._playing = not v._playing
      play_btn.label = "Pause" if v._playing else "Play"
      play_btn.icon = viser.Icon.PLAYER_PAUSE if v._playing else viser.Icon.PLAYER_PLAY
      v._update_status_text()

    @speed_btn.on_click
    def _(event) -> None:
      v._speed_idx = update_speed_index(v._speed_idx, event.target.value)
      v._update_status_text()

    @loop_cb.on_update
    def _(_) -> None:
      v._loop = bool(loop_cb.value)
      v._update_status_text()

    @pose_dd.on_update
    def _(_) -> None:
      pose_path = self.pose_keyframes.get(str(pose_dd.value))
      if pose_path is None or not pose_path.is_file():
        v._update_status_text("Pose keyframe file not found.")
        return
      v._load_clip_by_path(pose_path)

    self.refresh_motion_library()

  def attach(self, *_args: Any, **_kwargs: Any) -> None:
    """Stable controller contract used by downstream integrations."""
    self.build_tab()

  def tick(self, _dt: float) -> None:
    """Process pending debounced UI work."""
    v = self.viewer
    if v._motion_search_pending and time.perf_counter() >= v._motion_search_deadline:
      v._motion_search_pending = False
      self.apply_motion_filters()

  def dispose(self) -> None:
    """No explicit resources to dispose."""
    return None

  def refresh_motion_library(self) -> None:
    v = self.viewer
    v._motion_entries = index_bvh_files(v.motions_dir, recursive=True)
    v._motion_entry_by_relpath = {e.rel_path: e for e in v._motion_entries}

    folder_counts: dict[str, int] = {}
    folder_names = {"(all folders)"}
    for entry in v._motion_entries:
      parent = Path(entry.rel_path).parent.as_posix()
      parent = parent if parent not in ("", ".") else "root"
      folder_names.add(parent)
      folder_counts[parent] = folder_counts.get(parent, 0) + 1
    v._motion_folder_counts = folder_counts
    folder_options = sorted(folder_names, key=lambda value: (value != "(all folders)", value.lower()))
    if v._motion_folder_dropdown is not None:
      old_value = str(v._motion_folder_dropdown.value)
      v._suppress_motion_ui = True
      try:
        v._motion_folder_dropdown.options = folder_options
        v._motion_folder_dropdown.value = old_value if old_value in folder_options else "(all folders)"
      finally:
        v._suppress_motion_ui = False
    self.apply_motion_filters()

  def apply_motion_filters(self) -> None:
    v = self.viewer
    query = ""
    folder = "(all folders)"
    if v._motion_search_text is not None:
      query = str(v._motion_search_text.value).strip().lower()
    if v._motion_folder_dropdown is not None:
      folder = str(v._motion_folder_dropdown.value)

    terms = [t for t in query.split() if t]
    query_matches: list[MotionEntry] = []
    for entry in v._motion_entries:
      rel = entry.rel_path.lower()
      if terms and not all(term in rel for term in terms):
        continue
      query_matches.append(entry)

    has_subfolders = any(k != "root" for k in v._motion_folder_counts.keys())
    show_folder = len(query_matches) > _MOTION_FOLDER_SHOW_THRESHOLD and has_subfolders
    if v._motion_folder_dropdown is not None:
      v._motion_folder_dropdown.visible = show_folder
      if not show_folder and folder != "(all folders)":
        folder = "(all folders)"
        v._suppress_motion_ui = True
        try:
          v._motion_folder_dropdown.value = "(all folders)"
        finally:
          v._suppress_motion_ui = False

    filtered: list[MotionEntry] = []
    for entry in query_matches:
      parent = Path(entry.rel_path).parent.as_posix()
      parent = parent if parent not in ("", ".") else "root"
      if folder != "(all folders)" and parent != folder:
        continue
      filtered.append(entry)
    v._motion_filtered_entries = filtered
    visible_entries = filtered[:_MOTION_MAX_OPTIONS]
    options = [e.rel_path for e in visible_entries] or ["(no matches)"]
    selected = options[0]
    if v._motion_dropdown is not None:
      old_selected = str(v._motion_dropdown.value)
      if old_selected in options:
        selected = old_selected
      v._suppress_motion_ui = True
      try:
        v._motion_dropdown.options = options
        v._motion_dropdown.value = selected
      finally:
        v._suppress_motion_ui = False

    if v._motion_page_text is None:
      return
    if not filtered:
      v._motion_page_text.value = "No clips match current filters."
      return
    shown = len(visible_entries)
    if shown < len(filtered):
      v._motion_page_text.value = (
        f"Showing first {shown} of {len(filtered)} clips. "
        "Type more in Search clips to narrow results."
      )
    else:
      v._motion_page_text.value = f"Showing {shown} clip(s)."
