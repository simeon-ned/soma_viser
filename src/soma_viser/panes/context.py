"""Shared context objects for pane modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import PaneActions
from ..state.session import SessionState


@dataclass(frozen=True)
class PaneContext:
  """Dependencies shared across pane modules."""

  viewer: Any
  session: SessionState
  actions: PaneActions
