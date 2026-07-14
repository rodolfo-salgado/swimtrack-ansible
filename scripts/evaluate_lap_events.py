#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6.0.2,<7"]
# ///
"""Evaluate thresholded lap events against timestamp ground truth."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from summarize_tracking import SummaryError, parse_messages

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "e2e" / "lap-ground-truth.yml"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class EvaluationError(ValueError):
    """Raised when ground truth or prediction data cannot be evaluated safely."""


@dataclass(frozen=True)
class MatchState:
    matches: tuple[tuple[int, int], ...] = ()
    error_ms: float = 0.0


def _number(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise EvaluationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise EvaluationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise EvaluationError(f"{name} must be at least {minimum:g}")
    return result


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationError(f"{name} must be a non-empty string")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise EvaluationError(f"could not read manifest {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise EvaluationError("manifest schema_version must be 1")

    source = raw.get("source")
    if not isinstance(source, dict):
        raise EvaluationError("manifest source must be an object")
    _string(source.get("path"), "source.path")
    source_sha = _string(source.get("sha256"), "source.sha256")
    if SHA256_PATTERN.fullmatch(source_sha) is None:
        raise EvaluationError("source.sha256 must be a lowercase SHA256")

    policy = raw.get("temporal_evaluation")
    if not isinstance(policy, dict):
        raise EvaluationError("temporal_evaluation must be an object")
    timestamp_uncertainty = _number(
        policy.get("timestamp_uncertainty_ms"),
        "temporal_evaluation.timestamp_uncertainty_ms",
        minimum=0,
    )
    turn_duration = _number(
        policy.get("turn_duration_ms"),
        "temporal_evaluation.turn_duration_ms",
        minimum=0,
    )
    match_tolerance = _number(
        policy.get("match_tolerance_ms"),
        "temporal_evaluation.match_tolerance_ms",
        minimum=0,
    )
    deduplication = _number(
        policy.get("prediction_deduplication_ms"),
        "temporal_evaluation.prediction_deduplication_ms",
        minimum=0,
    )
    derived_tolerance = timestamp_uncertainty + turn_duration / 2
    if not math.isclose(match_tolerance, derived_tolerance, abs_tol=1e-6):
        raise EvaluationError(
            "temporal_evaluation.match_tolerance_ms must equal timestamp uncertainty plus half the turn duration"
        )
    if policy.get("matching_policy") != "one_to_one_minimum_absolute_error":
        raise EvaluationError("unsupported temporal_evaluation.matching_policy")
    primary_ids = policy.get("primary_video_ids")
    secondary_ids = policy.get("secondary_video_ids")
    if (
        not isinstance(primary_ids, list)
        or not primary_ids
        or not all(isinstance(value, str) for value in primary_ids)
    ):
        raise EvaluationError(
            "temporal_evaluation.primary_video_ids must be a non-empty string list"
        )
    if not isinstance(secondary_ids, list) or not all(
        isinstance(value, str) for value in secondary_ids
    ):
        raise EvaluationError(
            "temporal_evaluation.secondary_video_ids must be a string list"
        )
    if len(set(primary_ids + secondary_ids)) != len(primary_ids) + len(secondary_ids):
        raise EvaluationError(
            "primary and secondary video ids must be unique and disjoint"
        )

    videos_raw = raw.get("videos")
    if not isinstance(videos_raw, list) or not videos_raw:
        raise EvaluationError("videos must be a non-empty list")
    videos: dict[str, dict[str, Any]] = {}
    for index, video in enumerate(videos_raw):
        name = f"videos[{index}]"
        if not isinstance(video, dict):
            raise EvaluationError(f"{name} must be an object")
        video_id = _string(video.get("id"), f"{name}.id")
        if video_id in videos:
            raise EvaluationError(f"duplicate video id {video_id!r}")
        tier = video.get("evaluation_tier")
        if tier not in {"primary", "secondary"}:
            raise EvaluationError(
                f"{name}.evaluation_tier must be primary or secondary"
            )
        expected_tier = (
            "primary"
            if video_id in primary_ids
            else "secondary"
            if video_id in secondary_ids
            else None
        )
        if tier != expected_tier:
            raise EvaluationError(
                f"{video_id} tier does not match temporal_evaluation video id lists"
            )
        _string(video.get("file"), f"{name}.file")
        checksum = _string(video.get("source_sha256"), f"{name}.source_sha256")
        if SHA256_PATTERN.fullmatch(checksum) is None:
            raise EvaluationError(f"{name}.source_sha256 must be a lowercase SHA256")
        media = video.get("media")
        if not isinstance(media, dict):
            raise EvaluationError(f"{name}.media must be an object")
        duration_ms = _number(
            media.get("duration_ms"), f"{name}.media.duration_ms", minimum=0
        )
        for field in ("width", "height", "fps", "frame_count"):
            _number(media.get(field), f"{name}.media.{field}", minimum=1)

        subjects_raw = video.get("subjects")
        if not isinstance(subjects_raw, list) or not subjects_raw:
            raise EvaluationError(f"{name}.subjects must be a non-empty list")
        subjects: dict[str, dict[str, Any]] = {}
        for subject_index, subject in enumerate(subjects_raw):
            subject_name = f"{name}.subjects[{subject_index}]"
            if not isinstance(subject, dict):
                raise EvaluationError(f"{subject_name} must be an object")
            subject_id = _string(subject.get("id"), f"{subject_name}.id")
            if subject_id in subjects:
                raise EvaluationError(f"{name} has duplicate subject id {subject_id!r}")
            lane_id = _string(subject.get("lane_id"), f"{subject_name}.lane_id")
            start_ms = _number(
                subject.get("start_ms"), f"{subject_name}.start_ms", minimum=0
            )
            end_ms = _number(subject.get("end_ms"), f"{subject_name}.end_ms", minimum=0)
            if end_ms <= start_ms or end_ms > duration_ms:
                raise EvaluationError(
                    f"{subject_name} must have 0 <= start_ms < end_ms <= media duration"
                )
            subjects[subject_id] = {
                **subject,
                "lane_id": lane_id,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }

        events_raw = video.get("turn_events")
        if not isinstance(events_raw, list):
            raise EvaluationError(f"{name}.turn_events must be a list")
        events: list[dict[str, Any]] = []
        event_ids: set[str] = set()
        for event_index, event in enumerate(events_raw):
            event_name = f"{name}.turn_events[{event_index}]"
            if not isinstance(event, dict):
                raise EvaluationError(f"{event_name} must be an object")
            event_id = _string(event.get("id"), f"{event_name}.id")
            if event_id in event_ids:
                raise EvaluationError(
                    f"{name} has duplicate turn event id {event_id!r}"
                )
            event_ids.add(event_id)
            subject_id = _string(event.get("subject_id"), f"{event_name}.subject_id")
            if subject_id not in subjects:
                raise EvaluationError(
                    f"{event_name} references unknown subject {subject_id!r}"
                )
            lane_id = _string(event.get("lane_id"), f"{event_name}.lane_id")
            if lane_id != subjects[subject_id]["lane_id"]:
                raise EvaluationError(
                    f"{event_name}.lane_id does not match its subject"
                )
            timestamp_ms = _number(
                event.get("timestamp_ms"), f"{event_name}.timestamp_ms", minimum=0
            )
            subject = subjects[subject_id]
            if not subject["start_ms"] <= timestamp_ms <= subject["end_ms"]:
                raise EvaluationError(
                    f"{event_name}.timestamp_ms is outside its subject interval"
                )
            events.append({**event, "timestamp_ms": timestamp_ms})
        events.sort(
            key=lambda event: (event["lane_id"], event["timestamp_ms"], event["id"])
        )
        videos[video_id] = {
            **video,
            "subjects": list(subjects.values()),
            "turn_events": events,
        }

    expected_video_ids = set(primary_ids + secondary_ids)
    if set(videos) != expected_video_ids:
        missing = sorted(expected_video_ids - set(videos))
        extra = sorted(set(videos) - expected_video_ids)
        raise EvaluationError(
            f"video id lists and videos differ; missing={missing}, extra={extra}"
        )
    return {
        "schema_version": 1,
        "source": source,
        "temporal_evaluation": {
            **policy,
            "timestamp_uncertainty_ms": timestamp_uncertainty,
            "turn_duration_ms": turn_duration,
            "match_tolerance_ms": match_tolerance,
            "prediction_deduplication_ms": deduplication,
        },
        "videos": videos,
    }


def _candidate_in_scope(
    candidate: dict[str, Any], subjects: list[dict[str, Any]]
) -> bool:
    return any(
        subject["lane_id"] == candidate["lane_id"]
        and subject["start_ms"] <= candidate["timestamp_ms"] <= subject["end_ms"]
        for subject in subjects
    )


def extract_predictions(
    messages: list[dict[str, Any]],
    video: dict[str, Any],
    threshold: float,
    deduplication_ms: float,
) -> tuple[list[dict[str, Any]], int]:
    candidates: list[dict[str, Any]] = []
    ignored_outside_scope = 0
    for frame_index, frame in enumerate(messages):
        frame_time_ms = (
            _number(frame.get("time"), f"frame {frame_index}.time", minimum=0) * 1000
        )
        scores = frame.get("lap_scores")
        if scores is None:
            continue
        if not isinstance(scores, list):
            raise EvaluationError(f"frame {frame_index}.lap_scores must be a list")
        for score_index, score in enumerate(scores):
            name = f"frame {frame_index}.lap_scores[{score_index}]"
            if not isinstance(score, dict):
                raise EvaluationError(f"{name} must be an object")
            lap_score = _number(score.get("lap_score"), f"{name}.lap_score")
            if not 0 <= lap_score <= 1:
                raise EvaluationError(f"{name}.lap_score must be in [0, 1]")
            if (
                score.get("evaluable") is not True
                or lap_score < threshold
                or score.get("candidate_time_ms") is None
            ):
                continue
            candidate = {
                "lane_id": _string(score.get("lane_id"), f"{name}.lane_id"),
                "timestamp_ms": _number(
                    score.get("candidate_time_ms"),
                    f"{name}.candidate_time_ms",
                    minimum=0,
                ),
                "lap_score": lap_score,
                "frame_time_ms": frame_time_ms,
                "endpoint": score.get("endpoint"),
                "track_id": score.get("track_id"),
                "candidate_episode_id": score.get("candidate_episode_id"),
            }
            episode_id = candidate["candidate_episode_id"]
            if episode_id is not None and (
                isinstance(episode_id, bool)
                or not isinstance(episode_id, int)
                or episode_id < 1
            ):
                raise EvaluationError(f"{name}.candidate_episode_id must be a positive integer")
            if not _candidate_in_scope(candidate, video["subjects"]):
                ignored_outside_scope += 1
                continue
            candidates.append(candidate)

    episode_candidates: dict[tuple[str, int], dict[str, Any]] = {}
    legacy_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        episode_id = candidate["candidate_episode_id"]
        if episode_id is None:
            legacy_candidates.append(candidate)
            continue
        key = (candidate["lane_id"], episode_id)
        existing = episode_candidates.get(key)
        candidate_rank = (
            candidate["lap_score"],
            -candidate["frame_time_ms"],
            -candidate["timestamp_ms"],
        )
        existing_rank = (
            existing["lap_score"],
            -existing["frame_time_ms"],
            -existing["timestamp_ms"],
        ) if existing is not None else None
        if existing_rank is None or candidate_rank > existing_rank:
            episode_candidates[key] = candidate

    selected: list[dict[str, Any]] = list(episode_candidates.values())
    for candidate in sorted(
        legacy_candidates,
        key=lambda item: (
            -item["lap_score"],
            item["timestamp_ms"],
            item["frame_time_ms"],
            item["lane_id"],
        ),
    ):
        if any(
            existing["lane_id"] == candidate["lane_id"]
            and abs(existing["timestamp_ms"] - candidate["timestamp_ms"])
            <= deduplication_ms
            for existing in selected
        ):
            continue
        selected.append(candidate)
    selected.sort(
        key=lambda item: (item["lane_id"], item["timestamp_ms"], -item["lap_score"])
    )
    return selected, ignored_outside_scope


def _better(left: MatchState, right: MatchState) -> MatchState:
    left_key = (-len(left.matches), left.error_ms, left.matches)
    right_key = (-len(right.matches), right.error_ms, right.matches)
    return left if left_key <= right_key else right


def _match_lane(
    expected: list[dict[str, Any]], predicted: list[dict[str, Any]], tolerance_ms: float
) -> MatchState:
    rows = len(expected) + 1
    columns = len(predicted) + 1
    states = [[MatchState() for _ in range(columns)] for _ in range(rows)]
    for expected_index in range(1, rows):
        for predicted_index in range(1, columns):
            best = _better(
                states[expected_index - 1][predicted_index],
                states[expected_index][predicted_index - 1],
            )
            error = abs(
                expected[expected_index - 1]["timestamp_ms"]
                - predicted[predicted_index - 1]["timestamp_ms"]
            )
            if error <= tolerance_ms:
                previous = states[expected_index - 1][predicted_index - 1]
                paired = MatchState(
                    matches=previous.matches
                    + ((expected_index - 1, predicted_index - 1),),
                    error_ms=previous.error_ms + error,
                )
                best = _better(best, paired)
            states[expected_index][predicted_index] = best
    return states[-1][-1]


def match_events(
    expected: list[dict[str, Any]], predicted: list[dict[str, Any]], tolerance_ms: float
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    lane_ids = sorted(
        {event["lane_id"] for event in expected}
        | {event["lane_id"] for event in predicted}
    )
    for lane_id in lane_ids:
        lane_expected = sorted(
            (event for event in expected if event["lane_id"] == lane_id),
            key=lambda event: event["timestamp_ms"],
        )
        lane_predicted = sorted(
            (event for event in predicted if event["lane_id"] == lane_id),
            key=lambda event: event["timestamp_ms"],
        )
        state = _match_lane(lane_expected, lane_predicted, tolerance_ms)
        for expected_index, predicted_index in state.matches:
            truth = lane_expected[expected_index]
            prediction = lane_predicted[predicted_index]
            matches.append(
                {
                    "event_id": truth["id"],
                    "subject_id": truth["subject_id"],
                    "lane_id": lane_id,
                    "expected_time_ms": truth["timestamp_ms"],
                    "predicted_time_ms": prediction["timestamp_ms"],
                    "absolute_error_ms": abs(
                        truth["timestamp_ms"] - prediction["timestamp_ms"]
                    ),
                    "lap_score": prediction["lap_score"],
                }
            )
    matches.sort(
        key=lambda match: (
            match["lane_id"],
            match["expected_time_ms"],
            match["event_id"],
        )
    )
    return matches


def _rates(
    true_positives: int, false_positives: int, false_negatives: int
) -> dict[str, float | None]:
    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator if precision_denominator else None
    )
    recall = true_positives / recall_denominator if recall_denominator else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_video(
    video_id: str,
    messages: list[dict[str, Any]],
    manifest: dict[str, Any],
    threshold: float,
) -> dict[str, Any]:
    if video_id not in manifest["videos"]:
        raise EvaluationError(f"unknown video id {video_id!r}")
    video = manifest["videos"][video_id]
    policy = manifest["temporal_evaluation"]
    predictions, ignored_outside_scope = extract_predictions(
        messages, video, threshold, policy["prediction_deduplication_ms"]
    )
    expected = video["turn_events"]
    matches = match_events(expected, predictions, policy["match_tolerance_ms"])
    matched_event_ids = {match["event_id"] for match in matches}
    matched_prediction_keys = {
        (match["lane_id"], match["predicted_time_ms"]) for match in matches
    }
    false_negatives = [
        event for event in expected if event["id"] not in matched_event_ids
    ]
    false_positives = [
        prediction
        for prediction in predictions
        if (prediction["lane_id"], prediction["timestamp_ms"])
        not in matched_prediction_keys
    ]
    counts = {
        "true_positives": len(matches),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
    }
    half_duration = policy["turn_duration_ms"] / 2
    expected_with_windows = [
        {
            **event,
            "turn_interval_ms": [
                event["timestamp_ms"] - half_duration,
                event["timestamp_ms"] + half_duration,
            ],
            "acceptable_prediction_window_ms": [
                event["timestamp_ms"] - policy["match_tolerance_ms"],
                event["timestamp_ms"] + policy["match_tolerance_ms"],
            ],
        }
        for event in expected
    ]
    return {
        "video_id": video_id,
        "video_file": video["file"],
        "evaluation_tier": video["evaluation_tier"],
        "included_in_primary_metrics": video["evaluation_tier"] == "primary",
        "primary_exclusion_reason": video.get("primary_exclusion_reason"),
        "confidence_threshold": threshold,
        "policy": {
            "timestamp_uncertainty_ms": policy["timestamp_uncertainty_ms"],
            "turn_duration_ms": policy["turn_duration_ms"],
            "match_tolerance_ms": policy["match_tolerance_ms"],
            "prediction_deduplication_ms": policy["prediction_deduplication_ms"],
            "matching_policy": policy["matching_policy"],
        },
        "expected_events": expected_with_windows,
        "predicted_events": predictions,
        "ignored_candidates_outside_subject_intervals": ignored_outside_scope,
        "matches": matches,
        "false_positive_events": false_positives,
        "false_negative_events": false_negatives,
        "counts": counts,
        "metrics": {**counts, **_rates(**counts)},
    }


def aggregate(
    evaluations: list[dict[str, Any]], *, primary_only: bool
) -> dict[str, Any]:
    selected = [
        evaluation
        for evaluation in evaluations
        if not primary_only or evaluation["included_in_primary_metrics"]
    ]
    counts = {
        key: sum(evaluation["counts"][key] for evaluation in selected)
        for key in ("true_positives", "false_positives", "false_negatives")
    }
    return {
        "video_ids": [evaluation["video_id"] for evaluation in selected],
        "counts": counts,
        "metrics": {**counts, **_rates(**counts)},
    }


def _stream_argument(value: str) -> tuple[str, Path]:
    video_id, separator, raw_path = value.partition("=")
    if not separator or not video_id or not raw_path:
        raise argparse.ArgumentTypeError("stream must use VIDEO_ID=PATH syntax")
    return video_id, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--stream",
        action="append",
        type=_stream_argument,
        required=True,
        metavar="VIDEO_ID=PATH",
        help="SSE stream to evaluate; may be repeated.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Inclusive lap_score event threshold in [0, 1].",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
            raise EvaluationError("threshold must be finite and in [0, 1]")
        stream_ids = [video_id for video_id, _ in args.stream]
        if len(set(stream_ids)) != len(stream_ids):
            raise EvaluationError("each video id may be supplied only once")
        manifest = load_manifest(args.manifest.resolve())
        evaluations: list[dict[str, Any]] = []
        for video_id, stream_path in args.stream:
            try:
                messages = parse_messages(stream_path.read_text(encoding="utf-8"))
            except SummaryError as exc:
                raise EvaluationError(f"invalid stream for {video_id}: {exc}") from exc
            evaluations.append(
                evaluate_video(video_id, messages, manifest, args.threshold)
            )
        primary_aggregate = aggregate(evaluations, primary_only=True)
        expected_primary_ids = manifest["temporal_evaluation"]["primary_video_ids"]
        evaluated_primary_ids = set(primary_aggregate["video_ids"])
        primary_aggregate.update(
            {
                "expected_video_ids": expected_primary_ids,
                "missing_video_ids": [
                    video_id
                    for video_id in expected_primary_ids
                    if video_id not in evaluated_primary_ids
                ],
                "complete": evaluated_primary_ids == set(expected_primary_ids),
            }
        )
        result = {
            "schema_version": 1,
            "ground_truth_manifest": str(args.manifest.resolve()),
            "ground_truth_source": manifest["source"],
            "confidence_threshold": args.threshold,
            "videos": evaluations,
            "aggregate_primary": primary_aggregate,
            "aggregate_all": aggregate(evaluations, primary_only=False),
        }
        serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        print(json.dumps(result, sort_keys=True))
        return 0
    except (EvaluationError, OSError, yaml.YAMLError) as exc:
        print(f"Lap event evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
