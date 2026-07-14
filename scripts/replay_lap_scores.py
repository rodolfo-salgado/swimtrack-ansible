#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy>=1.26.4,<2.0.0",
#   "opencv-python-headless>=4.11.0.86,<5.0.0",
#   "pydantic>=2.10,<3.0",
# ]
# ///
"""Replay historical tracked boxes through the current local SwimTrack lap analyzer."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from pathlib import Path
from statistics import median
from typing import Any

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AI_SOURCE = PROJECT_DIR.parent / "swimtrack-ai"


class ReplayError(ValueError):
    """Raised when an SSE stream cannot be replayed safely."""


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ReplayError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ReplayError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise ReplayError(f"{name} must be at least {minimum:g}")
    return result


def _dimension(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReplayError(f"{name} must be a positive integer")
    return value


def parse_messages(stream: str) -> list[dict[str, Any]]:
    """Parse the message-only SSE dialect emitted by swimtrack-front."""

    messages: list[dict[str, Any]] = []
    normalized = stream.replace("\r\n", "\n").replace("\r", "\n")
    for block_index, block in enumerate(normalized.split("\n\n")):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if not line or line.startswith(":"):
                continue
            field, separator, raw_value = line.partition(":")
            value = raw_value.lstrip(" ") if separator else ""
            if field == "event":
                event_name = value or "message"
            elif field == "data":
                data_lines.append(value)
            elif field not in {"id", "retry"}:
                raise ReplayError(f"SSE block {block_index} contains an invalid field: {line!r}")
        if event_name == "error":
            raise ReplayError(f"stream contains an error event: {' '.join(data_lines)}")
        if event_name != "message":
            raise ReplayError(f"stream contains unsupported event type {event_name!r}")
        if not data_lines:
            raise ReplayError(f"SSE block {block_index} has no data")
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError as exc:
            raise ReplayError(f"SSE block {block_index} contains invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ReplayError(f"SSE block {block_index} data must be an object")
        messages.append(payload)
    if not messages:
        raise ReplayError("stream does not contain message events")
    return messages


def infer_fps(messages: list[dict[str, Any]]) -> float:
    times = [
        _number(message.get("time"), f"frame {index}.time", minimum=0)
        for index, message in enumerate(messages)
    ]
    if any(current < previous for previous, current in zip(times, times[1:])):
        raise ReplayError("frame timestamps must be non-decreasing")
    deltas = [
        current - previous
        for previous, current in zip(times, times[1:])
        if current > previous
    ]
    if not deltas:
        raise ReplayError("cannot infer fps without at least two distinct timestamps; pass --fps")
    return 1.0 / median(deltas)


def _load_analyzer(ai_source: Path) -> tuple[type[Any], type[Any]]:
    source_dir = ai_source.resolve() / "src"
    package_dir = source_dir / "swimtrack_ai"
    if not (package_dir / "lap_analysis.py").is_file() or not (
        package_dir / "schemas.py"
    ).is_file():
        raise ReplayError(f"{ai_source} is not a swimtrack-ai source tree")

    sys.path.insert(0, str(source_dir))
    try:
        lap_module = importlib.import_module("swimtrack_ai.lap_analysis")
        schemas_module = importlib.import_module("swimtrack_ai.schemas")
    except (ImportError, RuntimeError) as exc:
        raise ReplayError(f"could not import swimtrack-ai from {ai_source}: {exc}") from exc
    finally:
        sys.path.pop(0)

    loaded_path = Path(lap_module.__file__).resolve()
    if not loaded_path.is_relative_to(package_dir.resolve()):
        raise ReplayError(
            f"swimtrack_ai was already imported from a different source tree: {loaded_path}"
        )
    return lap_module.LapAnalyzer, schemas_module.BoundingBox


def _analysis_boxes(
    message: dict[str, Any],
    raw_boxes: list[Any],
    BoundingBox: type[Any],
    frame_index: int,
) -> list[Any]:
    """Reconstruct the boxes seen by trajectory-v4 without changing the payload."""

    try:
        tracked_boxes = [BoundingBox(**box) for box in raw_boxes]
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"frame {frame_index} contains an invalid box: {exc}") from exc
    if tracked_boxes:
        return tracked_boxes

    diagnostics = message.get("tracking_diagnostics")
    if diagnostics is None:
        return tracked_boxes
    if not isinstance(diagnostics, dict):
        raise ReplayError(f"frame {frame_index}.tracking_diagnostics must be an object")
    lanes = diagnostics.get("lanes")
    if not isinstance(lanes, list):
        raise ReplayError(f"frame {frame_index}.tracking_diagnostics.lanes must be a list")

    fallback_boxes: list[Any] = []
    for lane_index, lane in enumerate(lanes):
        name = f"frame {frame_index}.tracking_diagnostics.lanes[{lane_index}]"
        if not isinstance(lane, dict):
            raise ReplayError(f"{name} must be an object")
        lane_id = lane.get("lane_id")
        if not isinstance(lane_id, str) or not lane_id:
            raise ReplayError(f"{name}.lane_id must be a non-empty string")
        after_roi = lane.get("after_roi")
        if not isinstance(after_roi, dict):
            raise ReplayError(f"{name}.after_roi must be an object")
        diagnostic_boxes = after_roi.get("boxes")
        if diagnostic_boxes is None:
            continue
        if not isinstance(diagnostic_boxes, list):
            raise ReplayError(f"{name}.after_roi.boxes must be a list")
        for box_index, box in enumerate(diagnostic_boxes):
            if not isinstance(box, dict):
                raise ReplayError(f"{name}.after_roi.boxes[{box_index}] must be an object")
            try:
                fallback_boxes.append(
                    BoundingBox(
                        id=-1,
                        lane_id=None if lane_id == "global" else lane_id,
                        **box,
                    )
                )
            except (TypeError, ValueError) as exc:
                raise ReplayError(
                    f"{name}.after_roi.boxes[{box_index}] is invalid: {exc}"
                ) from exc
    return fallback_boxes


def replay_messages(
    messages: list[dict[str, Any]],
    *,
    ai_source: Path,
    fps: float,
    calibration_id: str,
) -> list[dict[str, Any]]:
    """Return messages with only lap_scores recalculated from their existing boxes."""

    fps = _number(fps, "fps", minimum=1e-12)
    LapAnalyzer, BoundingBox = _load_analyzer(ai_source)
    try:
        analyzer = LapAnalyzer(fps, calibration_id)
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"could not initialize LapAnalyzer: {exc}") from exc
    replayed: list[dict[str, Any]] = []
    previous_time_ms: float | None = None

    for frame_index, message in enumerate(messages):
        time_seconds = _number(message.get("time"), f"frame {frame_index}.time", minimum=0)
        time_ms = time_seconds * 1000.0
        if previous_time_ms is not None and time_ms < previous_time_ms:
            raise ReplayError("frame timestamps must be non-decreasing")
        previous_time_ms = time_ms
        width = _dimension(message.get("width"), f"frame {frame_index}.width")
        height = _dimension(message.get("height"), f"frame {frame_index}.height")
        raw_boxes = message.get("boxes")
        if not isinstance(raw_boxes, list):
            raise ReplayError(f"frame {frame_index}.boxes must be a list")
        boxes = _analysis_boxes(message, raw_boxes, BoundingBox, frame_index)

        try:
            scores = analyzer.observe(
                time_ms=time_ms,
                width=width,
                height=height,
                boxes=boxes,
            )
        except (TypeError, ValueError) as exc:
            raise ReplayError(f"could not score frame {frame_index}: {exc}") from exc
        payload = dict(message)
        payload["lap_scores"] = [score.model_dump(mode="json", exclude_none=True) for score in scores]
        replayed.append(payload)
    return replayed


def serialize_messages(messages: list[dict[str, Any]]) -> str:
    """Serialize messages in the SSE form consumed by evaluate_lap_events.py."""

    try:
        return "".join(
            f"data: {json.dumps(message, ensure_ascii=False, allow_nan=False)}\n\n"
            for message in messages
        )
    except (TypeError, ValueError) as exc:
        raise ReplayError(f"could not serialize replayed stream: {exc}") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stream",
        required=True,
        type=Path,
        help="Historical SSE stream containing tracked boxes.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination for the replayed SSE stream.",
    )
    parser.add_argument(
        "--ai-source",
        type=Path,
        default=DEFAULT_AI_SOURCE,
        help=f"Local swimtrack-ai repository whose lap analyzer should be used (default: {DEFAULT_AI_SOURCE}).",
    )
    parser.add_argument("--fps", type=float, help="Frame rate; inferred from timestamps when omitted.")
    parser.add_argument(
        "--calibration-id",
        default="fixed-camera-v1",
        help="Lap calibration passed to LapAnalyzer.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        stream_path = args.stream.resolve()
        output_path = args.output.resolve()
        if stream_path == output_path:
            raise ReplayError("--output must not overwrite --stream")
        messages = parse_messages(stream_path.read_text(encoding="utf-8"))
        fps = (
            infer_fps(messages)
            if args.fps is None
            else _number(args.fps, "fps", minimum=1e-12)
        )
        replayed = replay_messages(
            messages,
            ai_source=args.ai_source,
            fps=fps,
            calibration_id=args.calibration_id,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(serialize_messages(replayed), encoding="utf-8")
        score_versions = sorted(
            {str(score["score_version"]) for message in replayed for score in message["lap_scores"]}
        )
        print(
            json.dumps(
                {
                    "ai_source": str(args.ai_source.resolve()),
                    "fps": fps,
                    "frames": len(replayed),
                    "output": str(output_path),
                    "score_versions": score_versions,
                },
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ReplayError) as exc:
        print(f"Lap score replay failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
