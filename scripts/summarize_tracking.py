#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Summarize tracker continuity and diagnostics from a SwimTrack SSE stream."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


class SummaryError(ValueError):
    """Raised when an SSE stream cannot be summarized safely."""


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SummaryError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SummaryError(f"{name} must be finite")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SummaryError(f"{name} must be an integer")
    return value


def parse_messages(stream: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for block_index, block in enumerate(stream.replace("\r\n", "\n").split("\n\n")):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line[6:].strip() or "message"
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif line:
                raise SummaryError(f"SSE block {block_index} contains an invalid field: {line!r}")
        if event_name == "error":
            raise SummaryError(f"stream contains an error event: {' '.join(data_lines)}")
        if event_name != "message":
            raise SummaryError(f"stream contains unsupported event type {event_name!r}")
        if not data_lines:
            raise SummaryError(f"SSE block {block_index} has no data")
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError as exc:
            raise SummaryError(f"SSE block {block_index} contains invalid JSON") from exc
        if not isinstance(payload, dict):
            raise SummaryError(f"SSE block {block_index} data must be an object")
        messages.append(payload)
    if not messages:
        raise SummaryError("stream does not contain message events")
    return messages


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def _infer_fps(times: list[float]) -> float | None:
    deltas = [current - previous for previous, current in zip(times, times[1:]) if current > previous]
    if not deltas:
        return None
    frame_delta = median(deltas)
    return 1.0 / frame_delta if frame_delta > 0 else None


def _stage_summary(counts: list[int], frame_count: int) -> dict[str, int | float]:
    frames_nonempty = sum(count > 0 for count in counts)
    return {
        "observations": sum(counts),
        "frames_nonempty": frames_nonempty,
        "frame_coverage": frames_nonempty / frame_count,
        "mean_per_frame": mean(counts) if counts else 0.0,
        "max_per_frame": max(counts, default=0),
    }


def _empty_runs(presence: list[bool], *, internal_only: bool) -> list[int]:
    if not presence:
        return []
    start = 0
    stop = len(presence)
    if internal_only:
        try:
            start = presence.index(True)
            stop = len(presence) - list(reversed(presence)).index(True)
        except ValueError:
            return []
    runs: list[int] = []
    current = 0
    for present in presence[start:stop]:
        if present:
            if current:
                runs.append(current)
                current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    return runs


def _gap_summary(presence: list[bool], fps: float | None, *, internal_only: bool) -> dict[str, Any]:
    gaps = _empty_runs(presence, internal_only=internal_only)
    seconds = [gap / fps for gap in gaps] if fps else []
    return {
        "count": len(gaps),
        "frames": {
            "p50": percentile([float(gap) for gap in gaps], 0.50),
            "p95": percentile([float(gap) for gap in gaps], 0.95),
            "max": max(gaps, default=0),
        },
        "seconds": {
            "p50": percentile(seconds, 0.50),
            "p95": percentile(seconds, 0.95),
            "max": max(seconds, default=0.0),
        },
    }


def _diagnostic_stage(stage: Any, name: str) -> int:
    if not isinstance(stage, dict):
        raise SummaryError(f"{name} must be an object")
    count = _integer(stage.get("count"), f"{name}.count")
    if count < 0:
        raise SummaryError(f"{name}.count cannot be negative")
    boxes = stage.get("boxes")
    if boxes is not None:
        if not isinstance(boxes, list):
            raise SummaryError(f"{name}.boxes must be a list when present")
        if len(boxes) != count:
            raise SummaryError(f"{name}.boxes does not match {name}.count")
    return count


def _frame_boxes(frame: dict[str, Any], frame_index: int) -> list[dict[str, Any]]:
    boxes = frame.get("boxes")
    if not isinstance(boxes, list):
        raise SummaryError(f"frame {frame_index}: boxes must be a list")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[int] = set()
    for box_index, box in enumerate(boxes):
        if not isinstance(box, dict):
            raise SummaryError(f"frame {frame_index}, box {box_index}: box must be an object")
        track_id = _integer(box.get("id"), f"frame {frame_index}, box {box_index}.id")
        if track_id in seen_ids:
            raise SummaryError(f"frame {frame_index}: duplicate active track id {track_id}")
        seen_ids.add(track_id)
        normalized.append(box)
    return normalized


def _tracking_runs(
    track_frames: dict[int, list[int]],
    times: list[float],
    fps: float | None,
) -> tuple[list[dict[str, Any]], dict[str, Any], int, int]:
    tracks: list[dict[str, Any]] = []
    longest: dict[str, Any] = {
        "track_id": None,
        "frames": 0,
        "seconds": 0.0,
        "start_time_seconds": None,
        "end_time_seconds": None,
    }
    reacquisitions = 0
    longest_reacquisition_gap = 0
    for track_id, indices in sorted(track_frames.items()):
        previous = indices[0]
        longest_for_track = 1
        longest_start = indices[0]
        current_start = indices[0]
        current_length = 1
        track_reacquisitions = 0
        track_longest_gap = 0
        for index in indices[1:]:
            if index == previous + 1:
                current_length += 1
            else:
                gap = index - previous - 1
                track_reacquisitions += 1
                track_longest_gap = max(track_longest_gap, gap)
                if current_length > longest_for_track:
                    longest_for_track = current_length
                    longest_start = current_start
                current_start = index
                current_length = 1
            previous = index
        if current_length > longest_for_track:
            longest_for_track = current_length
            longest_start = current_start
        reacquisitions += track_reacquisitions
        longest_reacquisition_gap = max(longest_reacquisition_gap, track_longest_gap)
        longest_end = longest_start + longest_for_track - 1
        track_summary = {
            "track_id": track_id,
            "observed_frames": len(indices),
            "first_time_seconds": times[indices[0]],
            "last_time_seconds": times[indices[-1]],
            "span_seconds": times[indices[-1]] - times[indices[0]],
            "longest_consecutive_frames": longest_for_track,
            "longest_consecutive_seconds": longest_for_track / fps if fps else None,
            "reacquisitions": track_reacquisitions,
            "longest_reacquisition_gap_frames": track_longest_gap,
        }
        tracks.append(track_summary)
        if longest_for_track > longest["frames"]:
            longest = {
                "track_id": track_id,
                "frames": longest_for_track,
                "seconds": longest_for_track / fps if fps else None,
                "start_time_seconds": times[longest_start],
                "end_time_seconds": times[longest_end],
            }
    return tracks, longest, reacquisitions, longest_reacquisition_gap


def _lap_summary(messages: list[dict[str, Any]], expected_turns_ms: list[float]) -> dict[str, Any]:
    entries: list[tuple[float, dict[str, Any]]] = []
    frames_with_scores = 0
    for frame_index, frame in enumerate(messages):
        scores = frame.get("lap_scores")
        if scores is None:
            continue
        if not isinstance(scores, list):
            raise SummaryError(f"frame {frame_index}: lap_scores must be a list")
        frames_with_scores += 1
        frame_time = _number(frame.get("time"), f"frame {frame_index}.time")
        for score_index, score in enumerate(scores):
            if not isinstance(score, dict):
                raise SummaryError(f"frame {frame_index}, lap score {score_index}: score must be an object")
            _number(score.get("lap_score"), f"frame {frame_index}, lap score {score_index}.lap_score")
            entries.append((frame_time, score))
    if not entries:
        return {"frames_with_scores": frames_with_scores, "maximum": None}
    frame_time, maximum = max(entries, key=lambda item: float(item[1]["lap_score"]))
    candidate_time = maximum.get("candidate_time_ms")
    if candidate_time is not None:
        candidate_time = _number(candidate_time, "maximum lap candidate_time_ms")
    nearest_expected = None
    candidate_error = None
    if candidate_time is not None and expected_turns_ms:
        nearest_expected = min(expected_turns_ms, key=lambda expected: abs(expected - candidate_time))
        candidate_error = abs(candidate_time - nearest_expected)
    return {
        "frames_with_scores": frames_with_scores,
        "maximum": {
            "lap_score": float(maximum["lap_score"]),
            "frame_time_seconds": frame_time,
            "candidate_time_ms": candidate_time,
            "endpoint": maximum.get("endpoint"),
            "lane_id": maximum.get("lane_id"),
            "observation_quality": maximum.get("observation_quality"),
            "evaluable": maximum.get("evaluable"),
            "evidence": maximum.get("evidence"),
            "nearest_expected_turn_ms": nearest_expected,
            "candidate_error_ms": candidate_error,
        },
    }


def summarize(messages: list[dict[str, Any]], expected_turns_ms: list[float]) -> dict[str, Any]:
    times = [_number(frame.get("time"), f"frame {index}.time") for index, frame in enumerate(messages)]
    if any(current < previous for previous, current in zip(times, times[1:])):
        raise SummaryError("frame times must not move backwards")
    fps = _infer_fps(times)
    frame_count = len(messages)

    active_counts: list[int] = []
    active_presence: list[bool] = []
    track_frames: dict[int, list[int]] = defaultdict(list)
    diagnostics_frames = 0
    diagnostic_floors: set[float] = set()
    candidate_counts: list[int] = []
    accepted_counts: list[int] = []
    weak_candidate_counts: list[int] = []
    roi_counts: list[int] = []
    weak_roi_counts: list[int] = []
    diagnostic_active_counts: list[int] = []
    weak_reactivated_counts: list[int] = []
    lost_counts: list[int] = []
    accepted_no_track_frames = 0
    detector_accepted_no_track_frames = 0
    lane_track_ids: dict[str, set[int]] = defaultdict(set)
    weak_reactivated_track_ids: dict[str, set[int]] = defaultdict(set)

    for frame_index, frame in enumerate(messages):
        boxes = _frame_boxes(frame, frame_index)
        active_ids = {int(box["id"]) for box in boxes}
        active_counts.append(len(boxes))
        active_presence.append(bool(boxes))
        for track_id in active_ids:
            track_frames[track_id].append(frame_index)

        diagnostics = frame.get("tracking_diagnostics")
        if diagnostics is None:
            continue
        if not isinstance(diagnostics, dict):
            raise SummaryError(f"frame {frame_index}: tracking_diagnostics must be an object")
        diagnostics_frames += 1
        diagnostic_floors.add(_number(diagnostics.get("diagnostic_floor"), f"frame {frame_index}.diagnostic_floor"))
        candidate_count = _diagnostic_stage(diagnostics.get("person_candidates"), f"frame {frame_index}.person_candidates")
        accepted_count = _diagnostic_stage(diagnostics.get("detector_accepted"), f"frame {frame_index}.detector_accepted")
        weak_candidate_count = _diagnostic_stage(diagnostics.get("weak_candidates"), f"frame {frame_index}.weak_candidates")
        lanes = diagnostics.get("lanes")
        if not isinstance(lanes, list) or not lanes:
            raise SummaryError(f"frame {frame_index}: diagnostics.lanes must be a non-empty list")

        roi_count = 0
        weak_roi_count = 0
        retained_lost = 0
        diagnostic_active_ids: set[int] = set()
        frame_weak_reactivated_ids: set[tuple[str, int]] = set()
        seen_lanes: set[str] = set()
        for lane_index, lane in enumerate(lanes):
            if not isinstance(lane, dict):
                raise SummaryError(f"frame {frame_index}, lane {lane_index}: lane diagnostics must be an object")
            lane_id = lane.get("lane_id")
            if not isinstance(lane_id, str) or not lane_id or lane_id in seen_lanes:
                raise SummaryError(f"frame {frame_index}, lane {lane_index}: lane_id must be unique and non-empty")
            seen_lanes.add(lane_id)
            roi_count += _diagnostic_stage(lane.get("after_roi"), f"frame {frame_index}, lane {lane_id}.after_roi")
            weak_roi_count += _diagnostic_stage(
                lane.get("weak_candidates_after_roi"),
                f"frame {frame_index}, lane {lane_id}.weak_candidates_after_roi",
            )
            lane_active_ids = lane.get("active_track_ids")
            if not isinstance(lane_active_ids, list):
                raise SummaryError(f"frame {frame_index}, lane {lane_id}: active_track_ids must be a list")
            for value in lane_active_ids:
                track_id = _integer(value, f"frame {frame_index}, lane {lane_id}.active_track_id")
                if track_id in diagnostic_active_ids:
                    raise SummaryError(f"frame {frame_index}: duplicate diagnostic active track id {track_id}")
                diagnostic_active_ids.add(track_id)
                lane_track_ids[lane_id].add(track_id)
            lane_weak_reactivated_ids = lane.get("weak_reactivated_track_ids")
            if not isinstance(lane_weak_reactivated_ids, list):
                raise SummaryError(f"frame {frame_index}, lane {lane_id}: weak_reactivated_track_ids must be a list")
            for value in lane_weak_reactivated_ids:
                track_id = _integer(value, f"frame {frame_index}, lane {lane_id}.weak_reactivated_track_id")
                key = (lane_id, track_id)
                if key in frame_weak_reactivated_ids:
                    raise SummaryError(f"frame {frame_index}: duplicate weak reactivation for track {key}")
                frame_weak_reactivated_ids.add(key)
                weak_reactivated_track_ids[lane_id].add(track_id)
            lane_lost = _integer(
                lane.get("retained_lost_track_count"),
                f"frame {frame_index}, lane {lane_id}.retained_lost_track_count",
            )
            if lane_lost < 0:
                raise SummaryError(f"frame {frame_index}, lane {lane_id}: lost count cannot be negative")
            retained_lost += lane_lost

        if diagnostic_active_ids != active_ids:
            raise SummaryError(
                f"frame {frame_index}: diagnostic active ids {sorted(diagnostic_active_ids)} "
                f"do not match SSE boxes {sorted(active_ids)}"
            )
        if roi_count > accepted_count:
            raise SummaryError(f"frame {frame_index}: after_roi exceeds detector_accepted")
        if weak_candidate_count > candidate_count:
            raise SummaryError(f"frame {frame_index}: weak_candidates exceeds person_candidates")
        if weak_roi_count > weak_candidate_count:
            raise SummaryError(f"frame {frame_index}: weak_candidates_after_roi exceeds weak_candidates")

        candidate_counts.append(candidate_count)
        accepted_counts.append(accepted_count)
        weak_candidate_counts.append(weak_candidate_count)
        roi_counts.append(roi_count)
        weak_roi_counts.append(weak_roi_count)
        diagnostic_active_counts.append(len(diagnostic_active_ids))
        weak_reactivated_counts.append(len(frame_weak_reactivated_ids))
        lost_counts.append(retained_lost)
        if roi_count > 0 and not diagnostic_active_ids:
            accepted_no_track_frames += 1
        if accepted_count > 0 and not diagnostic_active_ids:
            detector_accepted_no_track_frames += 1

    tracks, longest_run, reacquisitions, longest_reacquisition_gap = _tracking_runs(track_frames, times, fps)
    if lane_track_ids:
        fragmentations = sum(max(0, len(track_ids) - 1) for track_ids in lane_track_ids.values())
    else:
        fragmentations = max(0, len(track_frames) - 1)

    diagnostics_available = diagnostics_frames > 0
    diagnostics_summary: dict[str, Any] = {
        "available": diagnostics_available,
        "frames": diagnostics_frames,
        "frame_coverage": diagnostics_frames / frame_count,
        "diagnostic_floors": sorted(diagnostic_floors),
        "stages": None,
        "funnel": None,
        "roi_rejected_observations": None,
        "weak_roi_rejected_observations": None,
        "accepted_no_track_frames": None,
        "detector_accepted_no_track_frames": None,
        "retained_lost": None,
    }
    if diagnostics_available:
        diagnostics_summary.update(
            {
                "stages": {
                    "person_candidates": _stage_summary(candidate_counts, frame_count),
                    "detector_accepted": _stage_summary(accepted_counts, frame_count),
                    "weak_candidates": _stage_summary(weak_candidate_counts, frame_count),
                    "after_roi": _stage_summary(roi_counts, frame_count),
                    "weak_candidates_after_roi": _stage_summary(weak_roi_counts, frame_count),
                    "active_tracks": _stage_summary(diagnostic_active_counts, frame_count),
                },
                "funnel": {
                    "candidate_to_accepted": sum(accepted_counts) / sum(candidate_counts) if sum(candidate_counts) else None,
                    "accepted_to_roi": sum(roi_counts) / sum(accepted_counts) if sum(accepted_counts) else None,
                    "candidate_to_weak": sum(weak_candidate_counts) / sum(candidate_counts) if sum(candidate_counts) else None,
                    "weak_to_roi": sum(weak_roi_counts) / sum(weak_candidate_counts) if sum(weak_candidate_counts) else None,
                },
                "roi_rejected_observations": sum(accepted_counts) - sum(roi_counts),
                "weak_roi_rejected_observations": sum(weak_candidate_counts) - sum(weak_roi_counts),
                "accepted_no_track_frames": accepted_no_track_frames,
                "detector_accepted_no_track_frames": detector_accepted_no_track_frames,
                "retained_lost": {
                    "frames_nonempty": sum(count > 0 for count in lost_counts),
                    "mean_per_diagnostic_frame": mean(lost_counts),
                    "peak": max(lost_counts, default=0),
                },
            }
        )

    return {
        "schema_version": 1,
        "event_count": frame_count,
        "first_time_seconds": times[0],
        "last_time_seconds": times[-1],
        "timeline_span_seconds": times[-1] - times[0],
        "inferred_fps": fps,
        "diagnostics": diagnostics_summary,
        "tracking": {
            "active_stage": _stage_summary(active_counts, frame_count),
            "unique_track_ids": len(track_frames),
            "fragmentations": fragmentations,
            "track_ids_by_lane": {lane_id: sorted(ids) for lane_id, ids in sorted(lane_track_ids.items())},
            "longest_consecutive_run": longest_run,
            "internal_active_gaps": _gap_summary(active_presence, fps, internal_only=True),
            "all_active_gaps": _gap_summary(active_presence, fps, internal_only=False),
            "same_id_reacquisitions": reacquisitions,
            "longest_same_id_reacquisition_gap_frames": longest_reacquisition_gap,
            "weak_reactivations": {
                "events": sum(weak_reactivated_counts),
                "frames_nonempty": sum(count > 0 for count in weak_reactivated_counts),
                "unique_track_ids": sum(len(track_ids) for track_ids in weak_reactivated_track_ids.values()),
                "track_ids_by_lane": {
                    lane_id: sorted(track_ids) for lane_id, track_ids in sorted(weak_reactivated_track_ids.items())
                },
            },
            "tracks": tracks,
        },
        "lap": _lap_summary(messages, expected_turns_ms),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path, help="SSE stream produced by the published Front endpoint.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    parser.add_argument(
        "--expected-turn-ms",
        action="append",
        type=float,
        default=[],
        help="Optional expected turn timestamp; may be passed more than once.",
    )
    parser.add_argument(
        "--require-diagnostics",
        action="store_true",
        help="Fail unless every SSE frame contains tracking_diagnostics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        messages = parse_messages(args.stream.read_text(encoding="utf-8"))
        result = summarize(messages, args.expected_turn_ms)
        if args.require_diagnostics and result["diagnostics"]["frames"] != result["event_count"]:
            raise SummaryError(
                "tracking diagnostics are required on every frame; deploy the Front with "
                "VISION_TRACKING_DIAGNOSTICS=counts or boxes"
            )
        serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
    except (OSError, SummaryError) as exc:
        print(f"Tracking summary failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
