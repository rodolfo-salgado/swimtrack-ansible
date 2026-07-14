#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6.0.2,<7"]
# ///
"""Build lap-confidence windows and evaluate strict and tolerant predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import yaml

from evaluate_lap_events import EvaluationError, load_manifest, match_events
from summarize_tracking import SummaryError, parse_messages

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_DIR / "e2e" / "lap-ground-truth.yml"


class WindowEvaluationError(ValueError):
    """Raised when a stream cannot produce trustworthy window metrics."""


def _number(
    value: Any,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WindowEvaluationError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise WindowEvaluationError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise WindowEvaluationError(f"{name} must be at least {minimum:g}")
    if maximum is not None and result > maximum:
        raise WindowEvaluationError(f"{name} must be at most {maximum:g}")
    return result


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WindowEvaluationError(f"{name} must be a non-empty string")
    return value


def validate_configuration(
    *,
    window_size_ms: float,
    stride_ms: float,
    anchor_ms: float,
    coverage_threshold: float,
    confidence_threshold: float,
) -> dict[str, float]:
    """Validate public numeric parameters and return normalized floats."""
    normalized = {
        "window_size_ms": _number(
            window_size_ms, "window_size_ms", minimum=math.nextafter(0.0, 1.0)
        ),
        "stride_ms": _number(stride_ms, "stride_ms", minimum=math.nextafter(0.0, 1.0)),
        "anchor_ms": _number(anchor_ms, "anchor_ms"),
        "coverage_threshold": _number(
            coverage_threshold, "coverage_threshold", minimum=0, maximum=1
        ),
        "confidence_threshold": _number(
            confidence_threshold, "confidence_threshold", minimum=0, maximum=1
        ),
    }
    if normalized["stride_ms"] > normalized["window_size_ms"]:
        raise WindowEvaluationError(
            "stride_ms must not exceed window_size_ms because uncovered gaps make window labels undefined"
        )
    return normalized


def _merge_intervals(
    intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start_ms, end_ms in sorted(intervals):
        if not merged or start_ms > merged[-1][1]:
            merged.append([start_ms, end_ms])
        else:
            merged[-1][1] = max(merged[-1][1], end_ms)
    return [(start_ms, end_ms) for start_ms, end_ms in merged]


def _lane_intervals(video: dict[str, Any]) -> dict[str, list[tuple[float, float]]]:
    raw: dict[str, list[tuple[float, float]]] = {}
    for subject in video["subjects"]:
        raw.setdefault(subject["lane_id"], []).append(
            (subject["start_ms"], subject["end_ms"])
        )
    return {lane_id: _merge_intervals(intervals) for lane_id, intervals in raw.items()}


def _interval_overlap_ms(
    start_ms: float,
    end_ms: float,
    intervals: list[tuple[float, float]],
) -> float:
    return sum(
        max(0.0, min(end_ms, active_end) - max(start_ms, active_start))
        for active_start, active_end in intervals
    )


def _in_intervals(timestamp_ms: float, intervals: list[tuple[float, float]]) -> bool:
    return any(start_ms <= timestamp_ms < end_ms for start_ms, end_ms in intervals)


def _window_indices(
    intervals: list[tuple[float, float]],
    *,
    window_size_ms: float,
    stride_ms: float,
    anchor_ms: float,
) -> list[int]:
    indices: set[int] = set()
    for start_ms, end_ms in intervals:
        first = math.floor((start_ms - window_size_ms - anchor_ms) / stride_ms) + 1
        last = math.ceil((end_ms - anchor_ms) / stride_ms) - 1
        indices.update(range(first, last + 1))
    return sorted(indices)


def _normalize_frames(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    frames: list[dict[str, Any]] = []
    score_versions: set[str] = set()
    previous_time_ms: float | None = None
    for frame_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise WindowEvaluationError(f"frame {frame_index} must be an object")
        time_ms = (
            _number(message.get("time"), f"frame {frame_index}.time", minimum=0) * 1000
        )
        if previous_time_ms is not None and time_ms <= previous_time_ms:
            raise WindowEvaluationError("frame times must be strictly increasing")
        previous_time_ms = time_ms
        raw_scores = message.get("lap_scores")
        if raw_scores is None:
            raw_scores = []
        if not isinstance(raw_scores, list):
            raise WindowEvaluationError(
                f"frame {frame_index}.lap_scores must be a list"
            )
        lane_scores: dict[str, dict[str, Any]] = {}
        for score_index, raw_score in enumerate(raw_scores):
            name = f"frame {frame_index}.lap_scores[{score_index}]"
            if not isinstance(raw_score, dict):
                raise WindowEvaluationError(f"{name} must be an object")
            lane_id = _string(raw_score.get("lane_id"), f"{name}.lane_id")
            if lane_id in lane_scores:
                raise WindowEvaluationError(
                    f"frame {frame_index} has duplicate lap scores for lane {lane_id!r}"
                )
            lap_score = _number(
                raw_score.get("lap_score"),
                f"{name}.lap_score",
                minimum=0,
                maximum=1,
            )
            evaluable = raw_score.get("evaluable")
            if not isinstance(evaluable, bool):
                raise WindowEvaluationError(f"{name}.evaluable must be a boolean")
            score_version = _string(
                raw_score.get("score_version"), f"{name}.score_version"
            )
            score_versions.add(score_version)
            quality = raw_score.get("observation_quality")
            if quality is not None:
                quality = _number(
                    quality,
                    f"{name}.observation_quality",
                    minimum=0,
                    maximum=1,
                )
            candidate_time_ms = raw_score.get("candidate_time_ms")
            if candidate_time_ms is not None:
                candidate_time_ms = _number(
                    candidate_time_ms, f"{name}.candidate_time_ms", minimum=0
                )
            episode_id = raw_score.get("candidate_episode_id")
            if episode_id is not None and (
                isinstance(episode_id, bool)
                or not isinstance(episode_id, int)
                or episode_id < 1
            ):
                raise WindowEvaluationError(
                    f"{name}.candidate_episode_id must be a positive integer"
                )
            if (candidate_time_ms is None) != (episode_id is None):
                raise WindowEvaluationError(
                    f"{name}.candidate_time_ms and candidate_episode_id must appear together"
                )
            lane_scores[lane_id] = {
                "lane_id": lane_id,
                "lap_score": lap_score,
                "evaluable": evaluable,
                "score_version": score_version,
                "observation_quality": quality,
                "candidate_time_ms": candidate_time_ms,
                "candidate_episode_id": episode_id,
                "endpoint": raw_score.get("endpoint"),
                "track_id": raw_score.get("track_id"),
            }
        frames.append(
            {"frame_index": frame_index, "time_ms": time_ms, "scores": lane_scores}
        )
    if not score_versions:
        raise WindowEvaluationError("stream contains no lap score version")
    if len(score_versions) > 1:
        raise WindowEvaluationError(
            f"stream mixes score versions: {sorted(score_versions)!r}"
        )
    return frames, next(iter(score_versions))


def _reduce_episodes(
    frames: list[dict[str, Any]],
    lane_intervals: dict[str, list[tuple[float, float]]],
) -> tuple[list[dict[str, Any]], int]:
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    ignored_outside_scope = 0
    for frame in frames:
        for score in frame["scores"].values():
            candidate_time_ms = score["candidate_time_ms"]
            episode_id = score["candidate_episode_id"]
            if (
                candidate_time_ms is None
                or episode_id is None
                or not score["evaluable"]
            ):
                continue
            intervals = lane_intervals.get(score["lane_id"], [])
            if not _in_intervals(candidate_time_ms, intervals):
                ignored_outside_scope += 1
                continue
            candidate = {
                "lane_id": score["lane_id"],
                "candidate_episode_id": episode_id,
                "candidate_time_ms": candidate_time_ms,
                "lap_score": score["lap_score"],
                "score_version": score["score_version"],
                "observation_quality": score["observation_quality"],
                "endpoint": score["endpoint"],
                "track_id": score["track_id"],
                "frame_time_ms": frame["time_ms"],
            }
            key = (candidate["lane_id"], episode_id)
            existing = selected.get(key)
            rank = (
                candidate["lap_score"],
                -candidate["frame_time_ms"],
                -candidate["candidate_time_ms"],
            )
            existing_rank = (
                (
                    existing["lap_score"],
                    -existing["frame_time_ms"],
                    -existing["candidate_time_ms"],
                )
                if existing is not None
                else None
            )
            if existing_rank is None or rank > existing_rank:
                selected[key] = candidate
    episodes = sorted(
        selected.values(),
        key=lambda item: (
            item["lane_id"],
            item["candidate_time_ms"],
            item["candidate_episode_id"],
        ),
    )
    return episodes, ignored_outside_scope


def build_video_dataset(
    video_id: str,
    messages: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    window_size_ms: float,
    stride_ms: float,
    anchor_ms: float,
    coverage_threshold: float,
    expected_score_version: str | None = None,
) -> dict[str, Any]:
    """Reduce an SSE stream into threshold-independent window rows."""
    if video_id not in manifest["videos"]:
        raise WindowEvaluationError(f"unknown video id {video_id!r}")
    config = validate_configuration(
        window_size_ms=window_size_ms,
        stride_ms=stride_ms,
        anchor_ms=anchor_ms,
        coverage_threshold=coverage_threshold,
        confidence_threshold=0,
    )
    video = manifest["videos"][video_id]
    lane_intervals = _lane_intervals(video)
    for event in video["turn_events"]:
        if not _in_intervals(event["timestamp_ms"], lane_intervals[event["lane_id"]]):
            raise WindowEvaluationError(
                f"{video_id} event {event['id']!r} is outside its half-open active interval"
            )
    frames, score_version = _normalize_frames(messages)
    if expected_score_version is not None and score_version != expected_score_version:
        raise WindowEvaluationError(
            f"{video_id} score version is {score_version!r}, expected {expected_score_version!r}"
        )
    episodes, ignored = _reduce_episodes(frames, lane_intervals)
    tolerance_ms = manifest["temporal_evaluation"]["match_tolerance_ms"]
    rows: list[dict[str, Any]] = []
    for lane_id, intervals in sorted(lane_intervals.items()):
        for window_index in _window_indices(
            intervals,
            window_size_ms=config["window_size_ms"],
            stride_ms=config["stride_ms"],
            anchor_ms=config["anchor_ms"],
        ):
            window_start_ms = config["anchor_ms"] + window_index * config["stride_ms"]
            window_end_ms = window_start_ms + config["window_size_ms"]
            active_overlap_ms = _interval_overlap_ms(
                window_start_ms, window_end_ms, intervals
            )
            if active_overlap_ms <= 0:
                continue
            window_frames = [
                frame
                for frame in frames
                if window_start_ms <= frame["time_ms"] < window_end_ms
                and _in_intervals(frame["time_ms"], intervals)
            ]
            lane_scores = [frame["scores"].get(lane_id) for frame in window_frames]
            evaluable_frames = sum(
                score is not None and score["evaluable"] for score in lane_scores
            )
            evaluable_fraction = (
                evaluable_frames / len(window_frames) if window_frames else 0.0
            )
            qualities = [
                score["observation_quality"]
                for score in lane_scores
                if score is not None and score["observation_quality"] is not None
            ]
            window_episodes = [
                episode
                for episode in episodes
                if episode["lane_id"] == lane_id
                and window_start_ms <= episode["candidate_time_ms"] < window_end_ms
            ]
            selected_episode = max(
                window_episodes,
                key=lambda item: (
                    item["lap_score"],
                    -item["candidate_time_ms"],
                    -item["candidate_episode_id"],
                ),
                default=None,
            )
            ground_truth_events = [
                event
                for event in video["turn_events"]
                if event["lane_id"] == lane_id
                and window_start_ms <= event["timestamp_ms"] < window_end_ms
            ]
            ambiguous = any(
                event["lane_id"] == lane_id
                and max(window_start_ms, event["timestamp_ms"] - tolerance_ms)
                < min(window_end_ms, event["timestamp_ms"] + tolerance_ms)
                for event in video["turn_events"]
            )
            is_evaluable = evaluable_fraction >= config["coverage_threshold"]
            rows.append(
                {
                    "schema_version": 1,
                    "video_id": video_id,
                    "video_file": video["file"],
                    "evaluation_tier": video["evaluation_tier"],
                    "lane_id": lane_id,
                    "window_index": window_index,
                    "window_start_ms": window_start_ms,
                    "window_end_ms": window_end_ms,
                    "active_interval_overlap_ms": active_overlap_ms,
                    "ground_truth_label": ("lap" if ground_truth_events else "no_lap"),
                    "ambiguous": ambiguous,
                    "ground_truth_event_ids": [
                        event["id"] for event in ground_truth_events
                    ],
                    "raw_lap_score": (
                        selected_episode["lap_score"]
                        if selected_episode is not None
                        else 0.0
                    ),
                    "score_version": score_version,
                    "candidate_episode_ids": sorted(
                        episode["candidate_episode_id"] for episode in window_episodes
                    ),
                    "selected_candidate_episode_id": (
                        selected_episode["candidate_episode_id"]
                        if selected_episode is not None
                        else None
                    ),
                    "candidate_time_ms": (
                        selected_episode["candidate_time_ms"]
                        if selected_episode is not None
                        else None
                    ),
                    "frame_count": len(window_frames),
                    "evaluable_frame_count": evaluable_frames,
                    "evaluable_fraction": evaluable_fraction,
                    "observation_quality_mean": mean(qualities) if qualities else None,
                    "observation_quality_min": min(qualities, default=None),
                    "observation_quality_max": max(qualities, default=None),
                    "coverage_threshold": config["coverage_threshold"],
                    "evaluable": is_evaluable,
                }
            )
    rows.sort(key=lambda row: (row["lane_id"], row["window_index"]))
    return {
        "video_id": video_id,
        "video_file": video["file"],
        "evaluation_tier": video["evaluation_tier"],
        "included_in_primary_metrics": video["evaluation_tier"] == "primary",
        "primary_exclusion_reason": video.get("primary_exclusion_reason"),
        "score_version": score_version,
        "episodes": episodes,
        "ignored_candidates_outside_active_intervals": ignored,
        "rows": rows,
    }


def _rates(counts: dict[str, int]) -> dict[str, float | None]:
    tp = counts["true_positives"]
    tn = counts["true_negatives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    negative_predictive_value = tn / (tn + fn) if tn + fn else None
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn else None
    f1_denominator = 2 * tp + fp + fn
    f1 = 2 * tp / f1_denominator if f1_denominator else None
    balanced_accuracy = (
        (recall + specificity) / 2
        if recall is not None and specificity is not None
        else None
    )
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn - fp * fn) / denominator) if denominator else None
    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "negative_predictive_value": negative_predictive_value,
        "accuracy": accuracy,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "matthews_correlation_coefficient": mcc,
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    evaluable = sum(row["evaluable"] for row in rows)
    abstentions = total - evaluable
    positive = sum(row["ground_truth_label"] == "lap" for row in rows)
    evaluable_positive = sum(
        row["evaluable"] and row["ground_truth_label"] == "lap" for row in rows
    )
    return {
        "total_windows": total,
        "evaluable_windows": evaluable,
        "abstentions": abstentions,
        "coverage": evaluable / total if total else None,
        "abstention_rate": abstentions / total if total else None,
        "ground_truth_lap_windows": positive,
        "ground_truth_no_lap_windows": total - positive,
        "evaluable_ground_truth_lap_windows": evaluable_positive,
        "evaluable_ground_truth_no_lap_windows": evaluable - evaluable_positive,
        "abstained_ground_truth_lap_windows": positive - evaluable_positive,
        "abstained_ground_truth_no_lap_windows": (
            total - positive - evaluable + evaluable_positive
        ),
    }


def _strict_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "true_positives": 0,
        "true_negatives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }
    for row in rows:
        if row["predicted_label"] == "abstain":
            continue
        truth = row["ground_truth_label"] == "lap"
        predicted = row["predicted_label"] == "lap"
        key = (
            "true_positives"
            if truth and predicted
            else "false_negatives"
            if truth
            else "false_positives"
            if predicted
            else "true_negatives"
        )
        counts[key] += 1
    return {"counts": counts, "metrics": {**counts, **_rates(counts)}}


def _canonical_row(
    rows: list[dict[str, Any]], lane_id: str, timestamp_ms: float
) -> dict[str, Any] | None:
    candidates = [
        row
        for row in rows
        if row["lane_id"] == lane_id
        and row["window_start_ms"] <= timestamp_ms < row["window_end_ms"]
    ]
    return max(
        candidates,
        key=lambda row: (row["window_start_ms"], row["window_index"]),
        default=None,
    )


def _prediction_key(prediction: dict[str, Any]) -> tuple[str, float, float]:
    return (
        prediction["lane_id"],
        prediction["timestamp_ms"],
        prediction["lap_score"],
    )


def _tolerant_metrics(
    dataset: dict[str, Any],
    rows: list[dict[str, Any]],
    video: dict[str, Any],
    tolerance_ms: float,
    threshold: float,
) -> dict[str, Any]:
    expected: list[dict[str, Any]] = []
    abstained_expected: list[dict[str, Any]] = []
    for event in video["turn_events"]:
        row = _canonical_row(rows, event["lane_id"], event["timestamp_ms"])
        if row is None or not row["evaluable"]:
            abstained_expected.append(event)
        else:
            expected.append(event)
    predicted: list[dict[str, Any]] = []
    abstained_predictions: list[dict[str, Any]] = []
    prediction_rows: dict[tuple[str, int], dict[str, Any]] = {}
    for episode in dataset["episodes"]:
        if episode["lap_score"] < threshold:
            continue
        row = _canonical_row(rows, episode["lane_id"], episode["candidate_time_ms"])
        prediction = {
            **episode,
            "timestamp_ms": episode["candidate_time_ms"],
        }
        if row is None or not row["evaluable"]:
            abstained_predictions.append(prediction)
            continue
        predicted.append(prediction)
        prediction_rows[(episode["lane_id"], episode["candidate_episode_id"])] = row
    matches = match_events(expected, predicted, tolerance_ms)
    matched_event_ids = {match["event_id"] for match in matches}
    matched_prediction_counter = Counter(
        (
            match["lane_id"],
            match["predicted_time_ms"],
            match["lap_score"],
        )
        for match in matches
    )
    false_positives: list[dict[str, Any]] = []
    for prediction in predicted:
        key = _prediction_key(prediction)
        if matched_prediction_counter[key]:
            matched_prediction_counter[key] -= 1
        else:
            false_positives.append(prediction)
    false_negatives = [
        event for event in expected if event["id"] not in matched_event_ids
    ]
    negative_rows = {
        (row["lane_id"], row["window_index"])
        for row in rows
        if row["evaluable"] and row["ground_truth_label"] == "no_lap"
    }
    false_positive_rows = {
        (
            prediction_rows[
                (prediction["lane_id"], prediction["candidate_episode_id"])
            ]["lane_id"],
            prediction_rows[
                (prediction["lane_id"], prediction["candidate_episode_id"])
            ]["window_index"],
        )
        for prediction in false_positives
        if (
            prediction_rows[
                (prediction["lane_id"], prediction["candidate_episode_id"])
            ]["lane_id"],
            prediction_rows[
                (prediction["lane_id"], prediction["candidate_episode_id"])
            ]["window_index"],
        )
        in negative_rows
    }
    counts = {
        "true_positives": len(matches),
        "true_negatives": len(negative_rows - false_positive_rows),
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
    }
    return {
        "counts": counts,
        "metrics": {**counts, **_rates(counts)},
        "matches": matches,
        "false_positive_episodes": false_positives,
        "false_negative_events": false_negatives,
        "abstained_ground_truth_events": abstained_expected,
        "abstained_prediction_episodes": abstained_predictions,
    }


def evaluate_video_dataset(
    dataset: dict[str, Any],
    manifest: dict[str, Any],
    *,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Apply a confidence threshold without reparsing or re-windowing the stream."""
    threshold = _number(
        confidence_threshold,
        "confidence_threshold",
        minimum=0,
        maximum=1,
    )
    rows: list[dict[str, Any]] = []
    for base_row in dataset["rows"]:
        evaluable = base_row["evaluable"]
        raw_lap_score = base_row["raw_lap_score"]
        rows.append(
            {
                key: value
                for key, value in base_row.items()
                if key not in {"raw_lap_score", "evaluable"}
            }
            | {
                "lap_score": raw_lap_score if evaluable else None,
                "confidence_threshold": threshold,
                "predicted_label": (
                    "abstain"
                    if not evaluable
                    else "lap"
                    if base_row["selected_candidate_episode_id"] is not None
                    and raw_lap_score >= threshold
                    else "no_lap"
                ),
            }
        )
    video = manifest["videos"][dataset["video_id"]]
    return {
        "video_id": dataset["video_id"],
        "video_file": dataset["video_file"],
        "evaluation_tier": dataset["evaluation_tier"],
        "included_in_primary_metrics": dataset["included_in_primary_metrics"],
        "primary_exclusion_reason": dataset["primary_exclusion_reason"],
        "score_version": dataset["score_version"],
        "confidence_threshold": threshold,
        "ignored_candidates_outside_active_intervals": dataset[
            "ignored_candidates_outside_active_intervals"
        ],
        "episodes": dataset["episodes"],
        "windows": rows,
        "coverage": _coverage(dataset["rows"]),
        "strict_window": _strict_metrics(rows),
        "temporal_tolerant": _tolerant_metrics(
            dataset,
            dataset["rows"],
            video,
            manifest["temporal_evaluation"]["match_tolerance_ms"],
            threshold,
        ),
    }


def _aggregate_metric(
    evaluations: list[dict[str, Any]], metric_name: str
) -> dict[str, Any]:
    keys = ("true_positives", "true_negatives", "false_positives", "false_negatives")
    counts = {
        key: sum(evaluation[metric_name]["counts"][key] for evaluation in evaluations)
        for key in keys
    }
    return {"counts": counts, "metrics": {**counts, **_rates(counts)}}


def aggregate(
    evaluations: list[dict[str, Any]],
    *,
    tier: str | None,
    expected_video_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Aggregate window and event metrics for one evaluation tier or all tiers."""
    selected = [
        evaluation
        for evaluation in evaluations
        if tier is None or evaluation["evaluation_tier"] == tier
    ]
    coverage_keys = (
        "total_windows",
        "evaluable_windows",
        "abstentions",
        "ground_truth_lap_windows",
        "ground_truth_no_lap_windows",
        "evaluable_ground_truth_lap_windows",
        "evaluable_ground_truth_no_lap_windows",
        "abstained_ground_truth_lap_windows",
        "abstained_ground_truth_no_lap_windows",
    )
    coverage = {
        key: sum(evaluation["coverage"][key] for evaluation in selected)
        for key in coverage_keys
    }
    total = coverage["total_windows"]
    coverage.update(
        {
            "coverage": coverage["evaluable_windows"] / total if total else None,
            "abstention_rate": coverage["abstentions"] / total if total else None,
        }
    )
    result = {
        "video_ids": [evaluation["video_id"] for evaluation in selected],
        "coverage": coverage,
        "strict_window": _aggregate_metric(selected, "strict_window"),
        "temporal_tolerant": _aggregate_metric(selected, "temporal_tolerant"),
    }
    if expected_video_ids is not None:
        supplied = set(result["video_ids"])
        result.update(
            {
                "expected_video_ids": expected_video_ids,
                "missing_video_ids": [
                    video_id
                    for video_id in expected_video_ids
                    if video_id not in supplied
                ],
                "complete": supplied == set(expected_video_ids),
            }
        )
    return result


def evaluate_datasets(
    datasets: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    confidence_threshold: float,
) -> dict[str, Any]:
    evaluations = [
        evaluate_video_dataset(
            dataset, manifest, confidence_threshold=confidence_threshold
        )
        for dataset in datasets
    ]
    policy = manifest["temporal_evaluation"]
    return {
        "confidence_threshold": confidence_threshold,
        "videos": evaluations,
        "aggregate_primary": aggregate(
            evaluations,
            tier="primary",
            expected_video_ids=policy["primary_video_ids"],
        ),
        "aggregate_secondary": aggregate(
            evaluations,
            tier="secondary",
            expected_video_ids=policy["secondary_video_ids"],
        ),
        "aggregate_all": aggregate(evaluations, tier=None),
    }


def _stream_argument(value: str) -> tuple[str, Path]:
    video_id, separator, raw_path = value.partition("=")
    if not separator or not video_id or not raw_path:
        raise argparse.ArgumentTypeError("stream must use VIDEO_ID=PATH syntax")
    return video_id, Path(raw_path)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_DIR.parent).as_posix()
    except ValueError:
        return resolved.as_posix()


def _file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    size_bytes = 0
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
    return {
        "path": _display_path(path),
        "size_bytes": size_bytes,
        "sha256": digest.hexdigest(),
    }


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
    parser.add_argument("--window-size-ms", type=float, default=2000)
    parser.add_argument("--stride-ms", type=float, default=2000)
    parser.add_argument("--anchor-ms", type=float, default=0)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
        help="Inclusive lap confidence threshold in [0, 1].",
    )
    parser.add_argument(
        "--sweep-threshold",
        action="append",
        type=float,
        default=[],
        help="Additional threshold to evaluate without reparsing streams.",
    )
    parser.add_argument(
        "--expected-score-version",
        help="Reject a stream whose lap score version does not equal this value.",
    )
    parser.add_argument("--output", type=Path, help="Optional structured JSON output.")
    parser.add_argument(
        "--rows-output", type=Path, help="Optional selected-threshold window JSONL."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = validate_configuration(
            window_size_ms=args.window_size_ms,
            stride_ms=args.stride_ms,
            anchor_ms=args.anchor_ms,
            coverage_threshold=args.coverage_threshold,
            confidence_threshold=args.threshold,
        )
        sweep_thresholds = [
            _number(value, "sweep_threshold", minimum=0, maximum=1)
            for value in args.sweep_threshold
        ]
        stream_ids = [video_id for video_id, _ in args.stream]
        if len(set(stream_ids)) != len(stream_ids):
            raise WindowEvaluationError("each video id may be supplied only once")
        manifest = load_manifest(args.manifest.resolve())
        datasets: list[dict[str, Any]] = []
        input_streams: list[dict[str, Any]] = []
        for video_id, stream_path in args.stream:
            try:
                messages = parse_messages(stream_path.read_text(encoding="utf-8"))
            except SummaryError as exc:
                raise WindowEvaluationError(
                    f"invalid stream for {video_id}: {exc}"
                ) from exc
            datasets.append(
                build_video_dataset(
                    video_id,
                    messages,
                    manifest,
                    window_size_ms=config["window_size_ms"],
                    stride_ms=config["stride_ms"],
                    anchor_ms=config["anchor_ms"],
                    coverage_threshold=config["coverage_threshold"],
                    expected_score_version=args.expected_score_version,
                )
            )
            input_streams.append({"video_id": video_id, **_file_metadata(stream_path)})
        versions = {dataset["score_version"] for dataset in datasets}
        if len(versions) > 1:
            raise WindowEvaluationError(
                f"supplied streams mix score versions: {sorted(versions, key=str)!r}"
            )
        selected = evaluate_datasets(
            datasets, manifest, confidence_threshold=config["confidence_threshold"]
        )
        sweep: list[dict[str, Any]] = []
        for threshold in dict.fromkeys(sweep_thresholds):
            evaluation = evaluate_datasets(
                datasets, manifest, confidence_threshold=threshold
            )
            sweep.append(
                {
                    "confidence_threshold": threshold,
                    "aggregate_primary": evaluation["aggregate_primary"],
                    "aggregate_secondary": evaluation["aggregate_secondary"],
                    "aggregate_all": evaluation["aggregate_all"],
                }
            )
        result = {
            "schema_version": 1,
            "ground_truth_manifest": _file_metadata(args.manifest),
            "ground_truth_source": manifest["source"],
            "input_streams": input_streams,
            "development_only": True,
            "confidence_interpretation": "heuristic_score_not_calibrated_probability",
            "configuration": {
                **config,
                "match_tolerance_ms": manifest["temporal_evaluation"][
                    "match_tolerance_ms"
                ],
                "expected_score_version": args.expected_score_version,
            },
            **selected,
            "threshold_sweep": sweep,
        }
        serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        if args.rows_output is not None:
            args.rows_output.parent.mkdir(parents=True, exist_ok=True)
            with args.rows_output.open("w", encoding="utf-8") as rows_file:
                for evaluation in selected["videos"]:
                    for row in evaluation["windows"]:
                        rows_file.write(json.dumps(row, sort_keys=True) + "\n")
        print(json.dumps(result, sort_keys=True))
        return 0
    except (
        EvaluationError,
        OSError,
        SummaryError,
        WindowEvaluationError,
        yaml.YAMLError,
    ) as exc:
        print(f"Lap window evaluation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
