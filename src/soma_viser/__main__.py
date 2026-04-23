"""CLI entrypoint for soma_viser."""
from __future__ import annotations

import argparse
from pathlib import Path

from .app import SomaViewer

def _default_motions_dir() -> Path:
  return Path(__file__).resolve().parents[2] / "motions"

def main() -> None:
  parser = argparse.ArgumentParser(
    description="SOMA BVH viewer with Motion/Visualization/Controls tabs."
  )
  parser.add_argument(
    "--motions-dir",
    type=Path,
    default=_default_motions_dir(),
    help="Directory containing .bvh motion clips.",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8080,
    help="Viser server port (default: 8080).",
  )
  parser.add_argument(
    "--up-axis",
    choices=["+z", "+y"],
    default="+z",
    help="Scene up axis (default: +z).",
  )
  args = parser.parse_args()
  viewer = SomaViewer(
    motions_dir=args.motions_dir,
    port=args.port,
    up_axis=args.up_axis,
  )
  viewer.run()


if __name__ == "__main__":
  main()
