#!/usr/bin/env python3
"""Convert a directory of sequential image frames into an MP4 video."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import cv2


_NUMBER_RE = re.compile(r"(\d+)")


def _natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in _NUMBER_RE.split(path.name)
    )


def find_frames(frame_dir: Path, pattern: str = "frame_*.jpg") -> list[Path]:
    """Return matching frame paths in numeric filename order."""
    return sorted(frame_dir.glob(pattern), key=_natural_key)


def frames_to_mp4(
    frame_dir: str | Path,
    output_path: str | Path,
    *,
    fps: float = 10.0,
    pattern: str = "frame_*.jpg",
    codec: str = "mp4v",
) -> Path:
    """Encode sequential images from ``frame_dir`` as an MP4.

    All frames must have the same dimensions as the first frame. The function
    creates the output parent directory and returns the resolved output path.
    """
    frame_dir = Path(frame_dir).expanduser()
    output_path = Path(output_path).expanduser()
    if not frame_dir.is_dir():
        raise FileNotFoundError(f"Frame directory does not exist: {frame_dir}")
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if len(codec) != 4:
        raise ValueError(f"codec must contain exactly four characters, got {codec!r}")

    frames = find_frames(frame_dir, pattern)
    if not frames:
        raise ValueError(f"No frames matching {pattern!r} in {frame_dir}")

    first = cv2.imread(str(frames[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise ValueError(f"Unable to read frame: {frames[0]}")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open video writer for {output_path}")

    try:
        for index, frame_path in enumerate(frames):
            image = first if index == 0 else cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"Unable to read frame: {frame_path}")
            if image.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame size changed at {frame_path}: "
                    f"expected {width}x{height}, got {image.shape[1]}x{image.shape[0]}"
                )
            writer.write(image)
    except Exception:
        writer.release()
        output_path.unlink(missing_ok=True)
        raise
    writer.release()
    return output_path.resolve()


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frame_dir", type=Path, help="directory containing image frames")
    parser.add_argument("output", type=Path, help="output MP4 path")
    parser.add_argument("--fps", type=float, default=10.0, help="output frame rate (default: 10)")
    parser.add_argument("--pattern", default="frame_*.jpg", help="input glob (default: frame_*.jpg)")
    parser.add_argument("--codec", default="mp4v", help="four-character codec (default: mp4v)")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    output = frames_to_mp4(
        args.frame_dir,
        args.output,
        fps=args.fps,
        pattern=args.pattern,
        codec=args.codec,
    )
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
