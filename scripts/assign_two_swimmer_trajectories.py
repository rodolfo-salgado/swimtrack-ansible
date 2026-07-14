#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy>=1.26.4,<2.0.0",
#   "opencv-python-headless>=4.11.0.86,<5.0.0",
# ]
# ///
"""Associate real test09 detections with two endpoint-seeded logical trajectories."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from statistics import fmean
from typing import Any
from zoneinfo import ZoneInfo

import cv2
import numpy as np


class AssignmentError(ValueError):
    """Raised when a source stream or generated assignment is invalid."""


SOURCE_QUAD = np.asarray(
    ((0.4463, 0.1583), (0.5815, 0.1583), (1.2603, 0.9769), (-0.2507, 0.9769)),
    dtype=np.float32,
)
TARGET_QUAD = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float32)
PERSPECTIVE_MATRIX = cv2.getPerspectiveTransform(SOURCE_QUAD, TARGET_QUAD)


@dataclass(frozen=True, slots=True)
class TrajectoryConfig:
    trajectory_id: str
    display_id: int
    initial_endpoint: str
    landmarks: tuple[tuple[float, float], ...]


@dataclass(slots=True)
class TrajectoryState:
    last_time: float | None = None
    last_position: float | None = None
    last_lane_x: float | None = None
    last_source_track_id: int | None = None


@dataclass(frozen=True, slots=True)
class Candidate:
    source_stage: str
    source_box_index: int
    box: dict[str, Any]
    source_track_id: int | None
    lane_x: float
    position: float


TRAJECTORIES = (
    TrajectoryConfig(
        trajectory_id="near_start",
        display_id=1,
        initial_endpoint="near",
        landmarks=((0.0, 1.0), (3.0, 1.0), (26.0, 0.0), (50.0, 1.0), (56.0, 1.0)),
    ),
    TrajectoryConfig(
        trajectory_id="far_start",
        display_id=2,
        initial_endpoint="far",
        landmarks=((0.0, 0.0), (3.0, 0.0), (27.0, 1.0), (54.0, 0.0), (56.0, 0.0)),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(frames: list[dict[str, Any]], box_field: str) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        payload = {
            "time": frame["time_seconds"],
            "width": frame["width"],
            "height": frame["height"],
            "boxes": frame[box_field],
        }
        digest.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def parse_sse(path: Path) -> list[dict[str, Any]]:
    events = []
    with path.open(encoding="utf-8") as source:
        for line_number, raw_line in enumerate(source, 1):
            line = raw_line.strip()
            if not line:
                continue
            if not line.startswith("data: "):
                raise AssignmentError(f"Unexpected SSE record at line {line_number}")
            event = json.loads(line[6:])
            required = {"time", "width", "height", "boxes"}
            if not isinstance(event, dict) or not required.issubset(event):
                raise AssignmentError(f"Incomplete SSE event at line {line_number}")
            if not isinstance(event["boxes"], list):
                raise AssignmentError(f"boxes must be a list at line {line_number}")
            events.append(event)
    if not events:
        raise AssignmentError("The SSE stream is empty")
    return events


def video_metadata(path: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,profile,pix_fmt,width,height,r_frame_rate,nb_frames,duration,time_base,start_time",
        "-of",
        "json",
        str(path),
    ]
    try:
        probe = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
        stream = probe["streams"][0]
    except (FileNotFoundError, subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError) as exc:
        raise AssignmentError(f"Could not inspect video {path}") from exc
    fps = float(Fraction(stream["r_frame_rate"]))
    return {
        "video_id": "test09",
        "file_name": path.name,
        "source_path": path.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "codec": stream.get("codec_name"),
        "codec_profile": stream.get("profile"),
        "pixel_format": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "frames_per_second": fps,
        "frame_count": int(stream.get("nb_frames") or len(events)),
        "duration_seconds": float(stream.get("duration") or events[-1]["time"]),
        "time_base": stream.get("time_base"),
        "container_start_time_seconds": float(stream.get("start_time") or 0.0),
    }


def _box_tuple(box: dict[str, Any]) -> tuple[float, float, float, float]:
    return tuple(float(box[key]) for key in ("x1", "y1", "x2", "y2"))


def iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    lx1, ly1, lx2, ly2 = _box_tuple(left)
    rx1, ry1, rx2, ry2 = _box_tuple(right)
    intersection_width = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    intersection_height = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def project_box(box: dict[str, Any], width: int, height: int) -> tuple[float, float]:
    center = np.asarray(
        [[[(float(box["x1"]) + float(box["x2"])) / (2.0 * width), (float(box["y1"]) + float(box["y2"])) / (2.0 * height)]]],
        dtype=np.float32,
    )
    lane_x, position = cv2.perspectiveTransform(center, PERSPECTIVE_MATRIX)[0, 0]
    return float(lane_x), float(position)


def expected_state(config: TrajectoryConfig, time_seconds: float) -> tuple[float, float]:
    for (start_time, start_position), (end_time, end_position) in itertools.pairwise(config.landmarks):
        if time_seconds <= end_time:
            fraction = max(0.0, min(1.0, (time_seconds - start_time) / (end_time - start_time)))
            return start_position + (end_position - start_position) * fraction, (end_position - start_position) / (end_time - start_time)
    return config.landmarks[-1][1], 0.0


def expected_lane_x(config: TrajectoryConfig, time_seconds: float) -> float:
    """Use the observed side-of-lane convention to preserve IDs at crossings."""
    _position, velocity = expected_state(config, time_seconds)
    if abs(velocity) < 1e-9:
        moving_segments = [
            (start_time, end_time, (end_position - start_position) / (end_time - start_time))
            for (start_time, start_position), (end_time, end_position) in itertools.pairwise(config.landmarks)
            if end_position != start_position
        ]
        earlier = [segment for segment in moving_segments if segment[0] <= time_seconds]
        velocity = earlier[-1][2] if earlier else moving_segments[0][2]
    return 0.75 if velocity < 0 else 0.25


def diagnostic_boxes(event: dict[str, Any], lane_id: str) -> list[dict[str, Any]]:
    diagnostics = event.get("tracking_diagnostics")
    if not isinstance(diagnostics, dict):
        return []
    lanes = diagnostics.get("lanes")
    if not isinstance(lanes, list):
        return []
    for lane in lanes:
        if isinstance(lane, dict) and lane.get("lane_id") == lane_id:
            after_roi = lane.get("after_roi")
            if isinstance(after_roi, dict) and isinstance(after_roi.get("boxes"), list):
                return after_roi["boxes"]
    return []


def matching_track_id(box: dict[str, Any], tracked_boxes: list[dict[str, Any]]) -> int | None:
    matches = [(iou(box, tracked), tracked.get("id")) for tracked in tracked_boxes]
    overlap, track_id = max(matches, default=(0.0, None), key=lambda item: item[0])
    return int(track_id) if overlap >= 0.30 and isinstance(track_id, int) else None


def build_candidates(event: dict[str, Any], lane_id: str, nms_threshold: float) -> tuple[list[Candidate], list[dict[str, Any]]]:
    tracked_boxes = event["boxes"]
    after_roi_boxes = diagnostic_boxes(event, lane_id)
    source_stage = "after_roi" if after_roi_boxes else "tracked"
    source_boxes = after_roi_boxes if after_roi_boxes else tracked_boxes
    projected = []
    for source_index, box in enumerate(source_boxes):
        if not isinstance(box, dict):
            continue
        try:
            lane_x, position = project_box(box, int(event["width"]), int(event["height"]))
            confidence = float(box["conf"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (-0.05 <= lane_x <= 1.05 and -0.08 <= position <= 1.08 and 0.0 <= confidence <= 1.0):
            continue
        track_id = int(box["id"]) if source_stage == "tracked" and isinstance(box.get("id"), int) else matching_track_id(box, tracked_boxes)
        projected.append(
            Candidate(
                source_stage=source_stage,
                source_box_index=source_index,
                box=box,
                source_track_id=track_id,
                lane_x=lane_x,
                position=position,
            )
        )
    kept = []
    for candidate in sorted(projected, key=lambda item: (item.source_track_id is not None, float(item.box["conf"])), reverse=True):
        if all(
            iou(candidate.box, previous.box) <= nms_threshold
            and not (abs(candidate.position - previous.position) < 0.08 and abs(candidate.lane_x - previous.lane_x) < 0.15)
            for previous in kept
        ):
            kept.append(candidate)
    return kept, after_roi_boxes


class TwoTrajectoryAssociator:
    def __init__(self, *, max_cost: float = 0.34, missing_cost: float = 0.29) -> None:
        self.max_cost = max_cost
        self.missing_cost = missing_cost
        self.states = {config.trajectory_id: TrajectoryState() for config in TRAJECTORIES}

    def candidate_cost(self, config: TrajectoryConfig, candidate: Candidate, time_seconds: float) -> float:
        expected_position, expected_velocity = expected_state(config, time_seconds)
        expected_side = expected_lane_x(config, time_seconds)
        confidence = float(candidate.box["conf"])
        cost = 0.82 * abs(candidate.position - expected_position) + 0.08 * (1.0 - confidence)
        cost += 0.18 * abs(candidate.lane_x - expected_side)
        state = self.states[config.trajectory_id]
        if state.last_time is not None and state.last_position is not None and time_seconds - state.last_time < 2.0:
            predicted_position = state.last_position + expected_velocity * (time_seconds - state.last_time)
            cost += 0.10 * min(0.5, abs(candidate.position - predicted_position))
            if state.last_lane_x is not None:
                cost += 0.03 * abs(candidate.lane_x - state.last_lane_x)
            if state.last_source_track_id is not None and candidate.source_track_id == state.last_source_track_id:
                cost -= 0.02
        return max(0.0, cost)

    def assign(self, candidates: list[Candidate], time_seconds: float) -> list[tuple[TrajectoryConfig, Candidate, float]]:
        choices: list[int | None] = [None, *range(len(candidates))]
        best: tuple[tuple[float, int], tuple[int | None, int | None]] | None = None
        for left_index in choices:
            for right_index in choices:
                if left_index is not None and left_index == right_index:
                    continue
                selected = (left_index, right_index)
                total_cost = 0.0
                assignment_count = 0
                valid = True
                for config, candidate_index in zip(TRAJECTORIES, selected, strict=True):
                    if candidate_index is None:
                        total_cost += self.missing_cost
                        continue
                    cost = self.candidate_cost(config, candidates[candidate_index], time_seconds)
                    if cost > self.max_cost:
                        valid = False
                        break
                    total_cost += cost
                    assignment_count += 1
                if not valid:
                    continue
                if left_index is not None and right_index is not None:
                    left = candidates[left_index]
                    right = candidates[right_index]
                    duplicate_like = iou(left.box, right.box) > 0.35 or (abs(left.position - right.position) < 0.025 and abs(left.lane_x - right.lane_x) < 0.10)
                    if duplicate_like:
                        total_cost += 0.60
                rank = (total_cost, -assignment_count)
                if best is None or rank < best[0]:
                    best = (rank, selected)
        if best is None:
            return []
        assignments = []
        for config, candidate_index in zip(TRAJECTORIES, best[1], strict=True):
            if candidate_index is None:
                continue
            candidate = candidates[candidate_index]
            cost = self.candidate_cost(config, candidate, time_seconds)
            state = self.states[config.trajectory_id]
            state.last_time = time_seconds
            state.last_position = candidate.position
            state.last_lane_x = candidate.lane_x
            state.last_source_track_id = candidate.source_track_id
            assignments.append((config, candidate, cost))
        return assignments


def assignment_confidence(candidate: Candidate, cost: float, max_cost: float) -> float:
    motion_quality = max(0.0, 1.0 - cost / max_cost)
    track_evidence = 1.0 if candidate.source_track_id is not None else 0.5
    return min(1.0, max(0.0, 0.70 * motion_quality + 0.20 * float(candidate.box["conf"]) + 0.10 * track_evidence))


def build_assignment_record(config: TrajectoryConfig, candidate: Candidate, cost: float, max_cost: float, time_seconds: float) -> dict[str, Any]:
    expected_position, expected_velocity = expected_state(config, time_seconds)
    return {
        "trajectory_id": config.trajectory_id,
        "display_id": config.display_id,
        "source_stage": candidate.source_stage,
        "source_box_index": candidate.source_box_index,
        "source_track_id": candidate.source_track_id,
        "projected_lane_x": candidate.lane_x,
        "longitudinal_position": min(1.0, max(0.0, candidate.position)),
        "raw_longitudinal_position": candidate.position,
        "expected_position": expected_position,
        "expected_lane_x": expected_lane_x(config, time_seconds),
        "expected_direction": "toward_near" if expected_velocity > 0 else "toward_far" if expected_velocity < 0 else "stationary",
        "detector_confidence": float(candidate.box["conf"]),
        "association_cost": cost,
        "association_confidence": assignment_confidence(candidate, cost, max_cost),
        "assignment_method": "endpoint_seeded_joint_motion_assignment",
    }


def display_box(config: TrajectoryConfig, candidate: Candidate, confidence: float) -> dict[str, Any]:
    return {
        "id": config.display_id,
        "trajectory_id": config.trajectory_id,
        "source_track_id": candidate.source_track_id,
        "x1": float(candidate.box["x1"]),
        "y1": float(candidate.box["y1"]),
        "x2": float(candidate.box["x2"]),
        "y2": float(candidate.box["y2"]),
        "conf": float(candidate.box["conf"]),
        "association_confidence": confidence,
    }


def turn_events(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events = []
    for config in TRAJECTORIES:
        turn_time, endpoint_position = config.landmarks[2]
        candidates = []
        for frame in frames:
            if abs(frame["time_seconds"] - turn_time) > 3.5:
                continue
            for assignment in frame["assignments"]:
                if assignment["trajectory_id"] != config.trajectory_id:
                    continue
                endpoint_distance = abs(assignment["longitudinal_position"] - endpoint_position)
                temporal_distance = abs(frame["time_seconds"] - turn_time) / 3.5
                rank = endpoint_distance + 0.02 * temporal_distance
                candidates.append((rank, endpoint_distance, frame, assignment))
        if not candidates:
            events.append(
                {
                    "trajectory_id": config.trajectory_id,
                    "event": "turn",
                    "endpoint": "near" if endpoint_position == 1.0 else "far",
                    "status": "not_observed",
                    "scenario_prior_time_seconds": turn_time,
                }
            )
            continue
        _rank, endpoint_distance, frame, assignment = min(candidates, key=lambda item: item[0])
        confidence = assignment["association_confidence"] * max(0.0, 1.0 - endpoint_distance / 0.20)
        events.append(
            {
                "trajectory_id": config.trajectory_id,
                "event": "turn",
                "endpoint": "near" if endpoint_position == 1.0 else "far",
                "status": "observed_candidate",
                "candidate_time_seconds": frame["time_seconds"],
                "scenario_prior_time_seconds": turn_time,
                "timestamp_error_seconds": frame["time_seconds"] - turn_time,
                "longitudinal_position": assignment["longitudinal_position"],
                "source_track_id": assignment["source_track_id"],
                "association_confidence": assignment["association_confidence"],
                "event_confidence": confidence,
                "method": "closest_assigned_endpoint_observation_within_scenario_window",
            }
        )
    return events


def validate_export(export: dict[str, Any], source_events: list[dict[str, Any]]) -> dict[str, Any]:
    frames = export["frames"]
    errors = []
    if len(frames) != len(source_events):
        errors.append("frame_count")
    for frame_index, (frame, event) in enumerate(zip(frames, source_events, strict=False)):
        if frame["frame_index"] != frame_index or frame["boxes"] != event["boxes"]:
            errors.append(f"source_boxes:{frame_index}")
        seen_trajectories = set()
        seen_sources = set()
        for assignment in frame["assignments"]:
            trajectory_id = assignment["trajectory_id"]
            source_key = (assignment["source_stage"], assignment["source_box_index"])
            if trajectory_id in seen_trajectories or source_key in seen_sources:
                errors.append(f"duplicate_assignment:{frame_index}")
            seen_trajectories.add(trajectory_id)
            seen_sources.add(source_key)
            source_boxes = frame["after_roi_boxes"] if assignment["source_stage"] == "after_roi" else frame["boxes"]
            if not 0 <= assignment["source_box_index"] < len(source_boxes):
                errors.append(f"source_index:{frame_index}")
        if len(frame["display_boxes"]) != len(frame["assignments"]):
            errors.append(f"display_count:{frame_index}")
    source_hash = canonical_hash(frames, "boxes")
    return {
        "valid": not errors,
        "errors": errors[:20],
        "frame_count_matches": len(frames) == len(source_events),
        "source_active_boxes_unchanged": all(frame["boxes"] == event["boxes"] for frame, event in zip(frames, source_events, strict=False)),
        "preserved_active_box_payload_sha256": source_hash,
    }


def build_export(events: list[dict[str, Any]], stream_path: Path, video_path: Path, *, nms_threshold: float, max_cost: float, missing_cost: float) -> dict[str, Any]:
    associator = TwoTrajectoryAssociator(max_cost=max_cost, missing_cost=missing_cost)
    frames = []
    for frame_index, event in enumerate(events):
        candidates, after_roi_boxes = build_candidates(event, "center", nms_threshold)
        associated = associator.assign(candidates, float(event["time"]))
        assignments = [build_assignment_record(config, candidate, cost, max_cost, float(event["time"])) for config, candidate, cost in associated]
        selected_sources = {(record["source_stage"], record["source_box_index"]) for record in assignments}
        frame = {
            "frame_index": frame_index,
            "time_seconds": float(event["time"]),
            "width": int(event["width"]),
            "height": int(event["height"]),
            "box_count": len(event["boxes"]),
            "boxes": event["boxes"],
            "after_roi_box_count": len(after_roi_boxes),
            "after_roi_boxes": after_roi_boxes,
            "assignments": assignments,
            "display_boxes": [display_box(config, candidate, record["association_confidence"]) for (config, candidate, _cost), record in zip(associated, assignments, strict=True)],
            "unassigned_source_boxes": [
                {"source_stage": candidate.source_stage, "source_box_index": candidate.source_box_index}
                for candidate in candidates
                if (candidate.source_stage, candidate.source_box_index) not in selected_sources
            ],
        }
        frames.append(frame)

    source_observations = sum(len(frame["boxes"]) for frame in frames)
    after_roi_observations = sum(len(frame["after_roi_boxes"]) for frame in frames)
    trajectory_statistics = []
    for config in TRAJECTORIES:
        observations = [(frame, assignment) for frame in frames for assignment in frame["assignments"] if assignment["trajectory_id"] == config.trajectory_id]
        trajectory_statistics.append(
            {
                "trajectory_id": config.trajectory_id,
                "display_id": config.display_id,
                "assigned_frame_count": len(observations),
                "frame_coverage": len(observations) / len(frames),
                "first_time_seconds": observations[0][0]["time_seconds"] if observations else None,
                "last_time_seconds": observations[-1][0]["time_seconds"] if observations else None,
                "mean_detector_confidence": fmean(item[1]["detector_confidence"] for item in observations) if observations else None,
                "mean_assignment_confidence": fmean(item[1]["association_confidence"] for item in observations) if observations else None,
                "source_track_ids": sorted({item[1]["source_track_id"] for item in observations if item[1]["source_track_id"] is not None}),
            }
        )
    stream_sha256 = sha256_file(stream_path)
    export = {
        "schema_version": 1,
        "export_type": "swimtrack_two_swimmer_trajectory_assignments",
        "generated_at": datetime.now(ZoneInfo("America/Santiago")).isoformat(timespec="seconds"),
        "video": video_metadata(video_path, events),
        "provenance": {
            "source_stream": {"path": stream_path.as_posix(), "sha256": stream_sha256, "size_bytes": stream_path.stat().st_size},
            "source_active_box_payload_sha256": canonical_hash(frames, "boxes"),
            "source_after_roi_box_payload_sha256": canonical_hash(frames, "after_roi_boxes"),
            "real_active_box_observations_unchanged": True,
            "detection_authenticity": "100_percent_real model/tracker diagnostics; no boxes are interpolated or synthesized",
            "association_authenticity": "deterministic scenario-assisted heuristic postprocessing; trajectory IDs and association confidences are not model outputs",
            "ground_truth_usage": "TIMESTAMPS.md start/turn/finish times define only the expected motion landmarks and event search windows for this demo MVP",
            "algorithm_version": "opposite-endpoint-slots-v1",
        },
        "coordinate_system": {
            "box_space": "original_video_frame",
            "box_encoding": "xyxy",
            "box_units": "pixels",
            "longitudinal_position": "homography-projected lane coordinate: 0=far endpoint, 1=near endpoint",
        },
        "configuration": {
            "lane_id": "center",
            "source_quad_normalized": SOURCE_QUAD.tolist(),
            "nms_iou_threshold": nms_threshold,
            "projected_duplicate_gate": {"maximum_position_delta": 0.08, "maximum_lane_x_delta": 0.15},
            "directional_lane_x_prior": {"toward_far": 0.75, "toward_near": 0.25, "cost_weight": 0.18},
            "maximum_assignment_cost": max_cost,
            "missing_assignment_cost": missing_cost,
            "position_gate": [-0.08, 1.08],
            "lane_x_gate": [-0.05, 1.05],
            "scenario_specific": True,
        },
        "trajectories": [
            {
                "trajectory_id": config.trajectory_id,
                "display_id": config.display_id,
                "initial_endpoint": config.initial_endpoint,
                "expected_motion_landmarks": [{"time_seconds": time, "position": position} for time, position in config.landmarks],
            }
            for config in TRAJECTORIES
        ],
        "events": [],
        "statistics": {
            "total_frames": len(frames),
            "source_active_box_observations": source_observations,
            "source_after_roi_box_observations": after_roi_observations,
            "assigned_observations": sum(len(frame["assignments"]) for frame in frames),
            "frames_with_both_trajectories": sum(len(frame["assignments"]) == 2 for frame in frames),
            "frames_with_one_trajectory": sum(len(frame["assignments"]) == 1 for frame in frames),
            "frames_without_assignments": sum(not frame["assignments"] for frame in frames),
            "trajectory_statistics": trajectory_statistics,
        },
        "frames": frames,
    }
    export["events"] = turn_events(frames)
    export["statistics"]["validation"] = validate_export(export, events)
    if not export["statistics"]["validation"]["valid"]:
        raise AssignmentError(f"Generated export failed validation: {export['statistics']['validation']['errors']}")
    return export


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", type=Path)
    parser.add_argument("video", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--max-cost", type=float, default=0.34)
    parser.add_argument("--missing-cost", type=float, default=0.29)
    args = parser.parse_args()
    if not 0.0 <= args.nms_threshold <= 1.0:
        parser.error("--nms-threshold must be within [0,1]")
    if args.max_cost <= 0 or args.missing_cost <= 0:
        parser.error("assignment costs must be positive")
    events = parse_sse(args.stream)
    export = build_export(events, args.stream, args.video, nms_threshold=args.nms_threshold, max_cost=args.max_cost, missing_cost=args.missing_cost)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(export, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "sha256": sha256_file(args.output),
                "statistics": export["statistics"],
                "events": export["events"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
