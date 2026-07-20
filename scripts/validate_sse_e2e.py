#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Validate the public SwimTrack SSE contract without assuming model accuracy."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class ValidationError(ValueError):
    """Raised when the streamed response does not satisfy the public contract."""


def _number(value: Any, *, name: str, event_index: int) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValidationError(f"event {event_index}: {name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValidationError(f"event {event_index}: {name} must be finite")
    return number


def _integer(value: Any, *, name: str, event_index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"event {event_index}: {name} must be an integer")
    return value


def _parse_events(stream: str) -> list[tuple[str, str]]:
    events: list[tuple[str, str]] = []
    for block in stream.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip() or "message"
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line:
                raise ValidationError(f"invalid SSE field: {line!r}")
        if not data_lines:
            raise ValidationError(f"SSE event {event_name!r} has no data")
        events.append((event_name, "\n".join(data_lines)))
    return events


def _validate_event(
    payload: Any,
    *,
    event_index: int,
    width: int,
    height: int,
    expected_time: float,
    time_tolerance: float,
    seen_ids: set[int],
    previous_count: int,
    previous_confirmed_count: int | None,
    require_identity_summary: bool,
) -> tuple[int, int, set[int], int | None, int | None]:
    if not isinstance(payload, dict):
        raise ValidationError(f"event {event_index}: data must be a JSON object")
    required = {"time", "width", "height", "boxes", "count"}
    missing = required.difference(payload)
    if missing:
        raise ValidationError(f"event {event_index}: missing fields {sorted(missing)}")

    timestamp = _number(payload["time"], name="time", event_index=event_index)
    if abs(timestamp - expected_time) > time_tolerance:
        raise ValidationError(
            f"event {event_index}: time {timestamp} differs from expected {expected_time} by more than {time_tolerance}"
        )
    if _integer(payload["width"], name="width", event_index=event_index) != width:
        raise ValidationError(f"event {event_index}: width does not match the fixture")
    if _integer(payload["height"], name="height", event_index=event_index) != height:
        raise ValidationError(f"event {event_index}: height does not match the fixture")

    boxes = payload["boxes"]
    if not isinstance(boxes, list):
        raise ValidationError(f"event {event_index}: boxes must be a list")
    frame_ids: set[int] = set()
    frame_identity_ids: set[int] = set()
    for box_index, box in enumerate(boxes):
        if not isinstance(box, dict):
            raise ValidationError(f"event {event_index}, box {box_index}: box must be an object")
        required_box = {"id", "x1", "y1", "x2", "y2", "conf"}
        missing_box = required_box.difference(box)
        if missing_box:
            raise ValidationError(f"event {event_index}, box {box_index}: missing fields {sorted(missing_box)}")
        legacy_id = _integer(box["id"], name="id", event_index=event_index)
        if legacy_id in frame_ids:
            raise ValidationError(f"event {event_index}: duplicate legacy box id {legacy_id}")
        frame_ids.add(legacy_id)
        identity_id = box.get("identity_id")
        if identity_id is None:
            if require_identity_summary:
                raise ValidationError(f"event {event_index}, box {box_index}: missing identity_id")
        else:
            identity_id = _integer(identity_id, name="identity_id", event_index=event_index)
            if identity_id < 1:
                raise ValidationError(f"event {event_index}, box {box_index}: identity_id must be positive")
            if identity_id in frame_identity_ids:
                raise ValidationError(f"event {event_index}: duplicate canonical identity_id {identity_id}")
            frame_identity_ids.add(identity_id)
        track_id = box.get("track_id")
        if track_id is not None:
            track_id = _integer(track_id, name="track_id", event_index=event_index)
            if track_id < 1:
                raise ValidationError(f"event {event_index}, box {box_index}: track_id must be positive")
        x1 = _number(box["x1"], name="x1", event_index=event_index)
        y1 = _number(box["y1"], name="y1", event_index=event_index)
        x2 = _number(box["x2"], name="x2", event_index=event_index)
        y2 = _number(box["y2"], name="y2", event_index=event_index)
        confidence = _number(box["conf"], name="conf", event_index=event_index)
        if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
            raise ValidationError(f"event {event_index}, box {box_index}: coordinates are outside the fixture")
        if not 0 <= confidence <= 1:
            raise ValidationError(f"event {event_index}, box {box_index}: confidence is outside [0, 1]")

    seen_ids.update(frame_ids)
    count = _integer(payload["count"], name="count", event_index=event_index)
    if count < previous_count:
        raise ValidationError(f"event {event_index}: count decreased")
    if count != len(seen_ids):
        raise ValidationError(f"event {event_index}: count does not equal the accumulated unique track ids")

    identity_summary = payload.get("identity_summary")
    if identity_summary is None:
        if require_identity_summary:
            raise ValidationError(f"event {event_index}: missing identity_summary")
        return len(boxes), count, frame_ids, None, None
    if not isinstance(identity_summary, dict):
        raise ValidationError(f"event {event_index}: identity_summary must be an object")
    if {"confirmed_count", "active_count"}.difference(identity_summary):
        raise ValidationError(f"event {event_index}: identity_summary is incomplete")
    confirmed_count = _integer(
        identity_summary["confirmed_count"],
        name="identity_summary.confirmed_count",
        event_index=event_index,
    )
    active_count = _integer(
        identity_summary["active_count"],
        name="identity_summary.active_count",
        event_index=event_index,
    )
    if confirmed_count < 0 or active_count < 0 or active_count > confirmed_count:
        raise ValidationError(f"event {event_index}: identity_summary contains invalid counts")
    if previous_confirmed_count is not None and confirmed_count < previous_confirmed_count:
        raise ValidationError(f"event {event_index}: confirmed identity count decreased")
    return len(boxes), count, frame_ids, confirmed_count, active_count


def validate(args: argparse.Namespace) -> dict[str, Any]:
    events = _parse_events(Path(args.stream).read_text(encoding="utf-8"))
    errors = [data for event_name, data in events if event_name == "error"]
    if errors:
        raise ValidationError(f"stream emitted event: error: {errors[0]}")
    messages = [(event_name, data) for event_name, data in events if event_name == "message"]
    if len(messages) != len(events):
        unexpected = sorted({event_name for event_name, _ in events if event_name != "message"})
        raise ValidationError(f"stream emitted unsupported event types: {unexpected}")
    if len(messages) != args.expected_events:
        raise ValidationError(f"expected {args.expected_events} SSE events, received {len(messages)}")

    seen_ids: set[int] = set()
    previous_count = 0
    expected_confirmed_identities = getattr(args, "expected_confirmed_identities", None)
    require_identity_summary = expected_confirmed_identities is not None
    previous_confirmed_count: int | None = None
    final_confirmed_count: int | None = None
    max_confirmed_count = 0
    max_active_count = 0
    frames_with_boxes = 0
    max_boxes = 0
    timestamps: list[float] = []
    for event_index, (_, data) in enumerate(messages):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            raise ValidationError(f"event {event_index}: invalid JSON") from exc
        (
            boxes_count,
            previous_count,
            _,
            confirmed_count,
            active_count,
        ) = _validate_event(
            payload,
            event_index=event_index,
            width=args.width,
            height=args.height,
            expected_time=event_index / args.fps,
            time_tolerance=args.time_tolerance,
            seen_ids=seen_ids,
            previous_count=previous_count,
            previous_confirmed_count=previous_confirmed_count,
            require_identity_summary=require_identity_summary,
        )
        if confirmed_count is not None:
            previous_confirmed_count = confirmed_count
            final_confirmed_count = confirmed_count
            max_confirmed_count = max(max_confirmed_count, confirmed_count)
        if active_count is not None:
            max_active_count = max(max_active_count, active_count)
        timestamps.append(float(payload["time"]))
        if boxes_count:
            frames_with_boxes += 1
        max_boxes = max(max_boxes, boxes_count)

    if require_identity_summary and (
        final_confirmed_count != expected_confirmed_identities
        or max_confirmed_count != expected_confirmed_identities
    ):
        raise ValidationError(
            "expected final and maximum confirmed identity count "
            f"{expected_confirmed_identities}, received final={final_confirmed_count} "
            f"max={max_confirmed_count}"
        )

    return {
        "event_count": len(messages),
        "frames_with_boxes": frames_with_boxes,
        "max_boxes_per_frame": max_boxes,
        "max_cumulative_track_count": previous_count,
        "sample_track_ids": sorted(seen_ids)[:10],
        "unique_track_ids": len(seen_ids),
        "first_time_seconds": timestamps[0],
        "last_time_seconds": timestamps[-1],
        "final_confirmed_identity_count": final_confirmed_count,
        "max_confirmed_identity_count": max_confirmed_count,
        "max_active_identity_count": max_active_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path)
    parser.add_argument("--expected-events", required=True, type=int)
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--fps", required=True, type=float)
    parser.add_argument("--time-tolerance", required=True, type=float)
    parser.add_argument("--expected-confirmed-identities", type=int)
    args = parser.parse_args()
    if args.expected_events < 1 or args.width < 1 or args.height < 1 or args.fps <= 0 or args.time_tolerance < 0:
        parser.error("all dimensions, event count, and fps must be positive; time tolerance cannot be negative")
    if args.expected_confirmed_identities is not None and args.expected_confirmed_identities < 1:
        parser.error("expected-confirmed-identities must be positive")
    return args


def main() -> int:
    try:
        print(json.dumps(validate(parse_args()), sort_keys=True))
    except (OSError, ValidationError) as exc:
        print(f"SSE E2E validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
