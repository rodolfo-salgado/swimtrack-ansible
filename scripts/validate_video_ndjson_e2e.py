#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Validate the ordered NDJSON contract of the GPU video endpoint."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Raised when a video stream does not satisfy its public contract."""


def _number(value: Any, *, name: str, frame_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"frame {frame_index}: {name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"frame {frame_index}: {name} must be finite")
    return number


def _integer(value: Any, *, name: str, frame_index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"frame {frame_index}: {name} must be an integer")
    return value


def _validate_boxes(payload: dict[str, Any], *, frame_index: int, width: int, height: int) -> tuple[int, set[int]]:
    boxes = payload.get("boxes")
    if not isinstance(boxes, list):
        raise ValidationError(f"frame {frame_index}: boxes must be a list")
    frame_ids: set[int] = set()
    for box_index, box in enumerate(boxes):
        if not isinstance(box, dict):
            raise ValidationError(f"frame {frame_index}, box {box_index}: box must be an object")
        required = {"id", "x1", "y1", "x2", "y2", "conf"}
        missing = required.difference(box)
        if missing:
            raise ValidationError(f"frame {frame_index}, box {box_index}: missing fields {sorted(missing)}")
        track_id = _integer(box["id"], name="id", frame_index=frame_index)
        if track_id in frame_ids:
            raise ValidationError(f"frame {frame_index}: duplicate track id {track_id}")
        frame_ids.add(track_id)
        x1 = _number(box["x1"], name="x1", frame_index=frame_index)
        y1 = _number(box["y1"], name="y1", frame_index=frame_index)
        x2 = _number(box["x2"], name="x2", frame_index=frame_index)
        y2 = _number(box["y2"], name="y2", frame_index=frame_index)
        confidence = _number(box["conf"], name="conf", frame_index=frame_index)
        if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
            raise ValidationError(f"frame {frame_index}, box {box_index}: coordinates are outside the fixture")
        if not 0 <= confidence <= 1:
            raise ValidationError(f"frame {frame_index}, box {box_index}: confidence is outside [0, 1]")
    return len(boxes), frame_ids


def validate(args: argparse.Namespace) -> dict[str, Any]:
    raw_lines = Path(args.stream).read_text(encoding="utf-8").splitlines()
    lines = [line for line in raw_lines if line.strip()]
    if len(lines) != args.expected_events:
        raise ValidationError(f"expected {args.expected_events} NDJSON frames, received {len(lines)}")

    expected_interval_ms = 1_000.0 / args.fps
    previous_time_ms: float | None = None
    seen_ids: set[int] = set()
    frames_with_boxes = 0
    max_boxes = 0
    timestamps: list[float] = []
    for expected_frame_index, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"frame {expected_frame_index}: invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValidationError(f"frame {expected_frame_index}: data must be a JSON object")
        required = {"frame_index", "time_ms", "width", "height", "boxes"}
        missing = required.difference(payload)
        if missing:
            raise ValidationError(f"frame {expected_frame_index}: missing fields {sorted(missing)}")

        frame_index = _integer(payload["frame_index"], name="frame_index", frame_index=expected_frame_index)
        if frame_index != expected_frame_index:
            raise ValidationError(
                f"frame {expected_frame_index}: expected frame_index {expected_frame_index}, received {frame_index}"
            )
        width = _integer(payload["width"], name="width", frame_index=expected_frame_index)
        height = _integer(payload["height"], name="height", frame_index=expected_frame_index)
        if width != args.width or height != args.height:
            raise ValidationError(f"frame {expected_frame_index}: dimensions {width}x{height} do not match the fixture")
        time_ms = _number(payload["time_ms"], name="time_ms", frame_index=expected_frame_index)
        expected_time_ms = expected_frame_index * expected_interval_ms
        if abs(time_ms - expected_time_ms) > args.time_tolerance_ms:
            raise ValidationError(
                f"frame {expected_frame_index}: time_ms {time_ms} differs from expected {expected_time_ms} by more than {args.time_tolerance_ms}"
            )
        if previous_time_ms is not None and time_ms <= previous_time_ms:
            raise ValidationError(f"frame {expected_frame_index}: time_ms is not strictly increasing")
        previous_time_ms = time_ms
        timestamps.append(time_ms)

        boxes_count, frame_ids = _validate_boxes(
            payload,
            frame_index=expected_frame_index,
            width=width,
            height=height,
        )
        seen_ids.update(frame_ids)
        if boxes_count:
            frames_with_boxes += 1
        max_boxes = max(max_boxes, boxes_count)

    return {
        "event_count": len(lines),
        "frames_with_boxes": frames_with_boxes,
        "max_boxes_per_frame": max_boxes,
        "sample_track_ids": sorted(seen_ids)[:10],
        "unique_track_ids": len(seen_ids),
        "first_time_ms": timestamps[0],
        "last_time_ms": timestamps[-1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path)
    parser.add_argument("--expected-events", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--time-tolerance-ms", required=True, type=float)
    args = parser.parse_args()
    if args.expected_events < 1 or args.width < 1 or args.height < 1 or args.fps <= 0 or args.time_tolerance_ms < 0:
        parser.error("all dimensions, event count, and fps must be positive; time tolerance cannot be negative")
    return args


def main() -> int:
    try:
        print(json.dumps(validate(parse_args()), sort_keys=True))
    except (OSError, ValidationError) as exc:
        print(f"video NDJSON E2E validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
