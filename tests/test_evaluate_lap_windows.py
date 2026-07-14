#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6.0.2,<7"]
# ///
"""Unit tests for lap confidence windows and their temporal metrics."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from evaluate_lap_windows import (  # noqa: E402
    WindowEvaluationError,
    aggregate,
    build_video_dataset,
    evaluate_video_dataset,
    validate_configuration,
)


def manifest(
    *,
    start_ms: float = 0,
    end_ms: float = 6000,
    events: list[float] | None = None,
    secondary: bool = False,
) -> dict:
    video_id = "secondary" if secondary else "primary"
    tier = "secondary" if secondary else "primary"
    return {
        "temporal_evaluation": {
            "match_tolerance_ms": 2000.0,
            "primary_video_ids": [] if secondary else [video_id],
            "secondary_video_ids": [video_id] if secondary else [],
        },
        "videos": {
            video_id: {
                "file": f"{video_id}.mp4",
                "evaluation_tier": tier,
                "subjects": [
                    {
                        "id": "swimmer_1",
                        "lane_id": "center",
                        "start_ms": float(start_ms),
                        "end_ms": float(end_ms),
                    }
                ],
                "turn_events": [
                    {
                        "id": f"turn_{index}",
                        "subject_id": "swimmer_1",
                        "lane_id": "center",
                        "timestamp_ms": float(timestamp_ms),
                    }
                    for index, timestamp_ms in enumerate(events or [], start=1)
                ],
                **(
                    {"primary_exclusion_reason": "synthetic secondary case"}
                    if secondary
                    else {}
                ),
            }
        },
    }


def frame(
    time_seconds: float,
    *,
    evaluable: bool = True,
    lap_score: float = 0.0,
    candidate_ms: float | None = None,
    episode_id: int | None = None,
    quality: float = 0.5,
    score_version: str = "trajectory-test",
    include_lane: bool = True,
) -> dict:
    scores = []
    if include_lane:
        scores.append(
            {
                "lane_id": "center",
                "lap_score": lap_score,
                "evaluable": evaluable,
                "observation_quality": quality,
                "score_version": score_version,
                **(
                    {"candidate_time_ms": candidate_ms}
                    if candidate_ms is not None
                    else {}
                ),
                **(
                    {"candidate_episode_id": episode_id}
                    if episode_id is not None
                    else {}
                ),
            }
        )
    return {"time": time_seconds, "boxes": [], "lap_scores": scores}


def build(
    messages: list[dict],
    ground_truth: dict,
    *,
    coverage_threshold: float = 0.5,
) -> dict:
    video_id = next(iter(ground_truth["videos"]))
    return build_video_dataset(
        video_id,
        messages,
        ground_truth,
        window_size_ms=2000,
        stride_ms=2000,
        anchor_ms=0,
        coverage_threshold=coverage_threshold,
        expected_score_version="trajectory-test",
    )


class LapWindowEvaluationTests(unittest.TestCase):
    def test_episode_reduction_keeps_maximum_and_associated_candidate_time(
        self,
    ) -> None:
        ground_truth = manifest()
        dataset = build(
            [
                frame(0),
                frame(1, lap_score=0.2, candidate_ms=1000, episode_id=1),
                frame(2, lap_score=0.8, candidate_ms=2500, episode_id=1),
                frame(3, lap_score=0.7, candidate_ms=2600, episode_id=1),
                frame(3.5, lap_score=0.75, candidate_ms=2700, episode_id=2),
                frame(4),
                frame(5),
            ],
            ground_truth,
        )

        self.assertEqual(len(dataset["episodes"]), 2)
        first = dataset["episodes"][0]
        self.assertEqual(first["candidate_episode_id"], 1)
        self.assertEqual(first["lap_score"], 0.8)
        self.assertEqual(first["candidate_time_ms"], 2500)
        middle = dataset["rows"][1]
        self.assertEqual(middle["candidate_episode_ids"], [1, 2])
        self.assertEqual(middle["selected_candidate_episode_id"], 1)
        self.assertEqual(middle["raw_lap_score"], 0.8)

        evaluation = evaluate_video_dataset(
            dataset, ground_truth, confidence_threshold=0.5
        )
        positive_windows = [
            row for row in evaluation["windows"] if row["predicted_label"] == "lap"
        ]
        self.assertEqual(len(positive_windows), 1)
        self.assertEqual(evaluation["strict_window"]["counts"]["false_positives"], 1)
        self.assertEqual(
            evaluation["temporal_tolerant"]["counts"]["false_positives"], 2
        )

    def test_half_open_boundaries_anchor_and_partial_overlap(self) -> None:
        ground_truth = manifest(start_ms=500, end_ms=4500, events=[2000])
        dataset = build(
            [
                frame(0),
                frame(0.5),
                frame(1),
                frame(1.5),
                frame(2, lap_score=0.7, candidate_ms=2000, episode_id=1),
                frame(2.5),
                frame(3),
                frame(3.5),
                frame(4),
                frame(4.5),
            ],
            ground_truth,
        )

        self.assertEqual([row["window_index"] for row in dataset["rows"]], [0, 1, 2])
        self.assertEqual(
            [row["active_interval_overlap_ms"] for row in dataset["rows"]],
            [1500, 2000, 500],
        )
        self.assertEqual(
            [row["ground_truth_label"] for row in dataset["rows"]],
            ["no_lap", "lap", "no_lap"],
        )
        self.assertEqual(dataset["rows"][0]["candidate_episode_ids"], [])
        self.assertEqual(dataset["rows"][1]["candidate_episode_ids"], [1])
        self.assertEqual(dataset["rows"][2]["frame_count"], 1)

    def test_low_coverage_abstains_and_does_not_become_true_negative(self) -> None:
        ground_truth = manifest(end_ms=4000)
        dataset = build(
            [
                frame(0, include_lane=False),
                frame(0.5, include_lane=False),
                frame(1, include_lane=False),
                frame(1.5),
                frame(2),
                frame(2.5),
                frame(3),
                frame(3.5),
            ],
            ground_truth,
        )
        evaluation = evaluate_video_dataset(
            dataset, ground_truth, confidence_threshold=0.5
        )

        self.assertEqual(dataset["rows"][0]["evaluable_fraction"], 0.25)
        self.assertEqual(evaluation["windows"][0]["predicted_label"], "abstain")
        self.assertIsNone(evaluation["windows"][0]["lap_score"])
        self.assertEqual(
            evaluation["coverage"],
            {
                "total_windows": 2,
                "evaluable_windows": 1,
                "abstentions": 1,
                "coverage": 0.5,
                "abstention_rate": 0.5,
                "ground_truth_lap_windows": 0,
                "ground_truth_no_lap_windows": 2,
                "evaluable_ground_truth_lap_windows": 0,
                "evaluable_ground_truth_no_lap_windows": 1,
                "abstained_ground_truth_lap_windows": 0,
                "abstained_ground_truth_no_lap_windows": 1,
            },
        )
        self.assertEqual(
            evaluation["strict_window"]["counts"],
            {
                "true_positives": 0,
                "true_negatives": 1,
                "false_positives": 0,
                "false_negatives": 0,
            },
        )

    def test_tolerant_matching_resolves_strict_boundary_artifact(self) -> None:
        ground_truth = manifest(events=[3000])
        dataset = build(
            [
                frame(0),
                frame(1),
                frame(2),
                frame(3),
                frame(4),
                frame(5, lap_score=0.8, candidate_ms=4500, episode_id=1),
            ],
            ground_truth,
        )
        evaluation = evaluate_video_dataset(
            dataset, ground_truth, confidence_threshold=0.5
        )

        self.assertEqual(
            evaluation["strict_window"]["counts"],
            {
                "true_positives": 0,
                "true_negatives": 1,
                "false_positives": 1,
                "false_negatives": 1,
            },
        )
        self.assertEqual(
            evaluation["temporal_tolerant"]["counts"],
            {
                "true_positives": 1,
                "true_negatives": 2,
                "false_positives": 0,
                "false_negatives": 0,
            },
        )
        self.assertEqual(
            evaluation["temporal_tolerant"]["matches"][0]["absolute_error_ms"],
            1500,
        )

    def test_tolerant_matching_is_one_to_one(self) -> None:
        ground_truth = manifest(events=[3000, 4000])
        dataset = build(
            [
                frame(0),
                frame(1),
                frame(2),
                frame(3.5, lap_score=0.8, candidate_ms=3500, episode_id=1),
                frame(4),
                frame(5),
            ],
            ground_truth,
        )
        evaluation = evaluate_video_dataset(
            dataset, ground_truth, confidence_threshold=0.5
        )
        tolerant = evaluation["temporal_tolerant"]

        self.assertEqual(tolerant["counts"]["true_positives"], 1)
        self.assertEqual(tolerant["counts"]["false_negatives"], 1)
        self.assertEqual(len(tolerant["matches"]), 1)
        self.assertEqual(len(tolerant["false_negative_events"]), 1)

    def test_threshold_can_be_changed_without_rebuilding_rows(self) -> None:
        ground_truth = manifest()
        dataset = build(
            [
                frame(0),
                frame(1, lap_score=0.6, candidate_ms=1000, episode_id=1),
                frame(2),
                frame(3),
                frame(4),
                frame(5),
            ],
            ground_truth,
        )
        low = evaluate_video_dataset(dataset, ground_truth, confidence_threshold=0.5)
        high = evaluate_video_dataset(dataset, ground_truth, confidence_threshold=0.7)

        self.assertEqual(low["windows"][0]["predicted_label"], "lap")
        self.assertEqual(high["windows"][0]["predicted_label"], "no_lap")
        self.assertEqual(dataset["rows"][0]["raw_lap_score"], 0.6)
        self.assertNotIn("predicted_label", dataset["rows"][0])

    def test_zero_threshold_does_not_classify_empty_windows_as_lap(self) -> None:
        ground_truth = manifest()
        dataset = build([frame(index) for index in range(6)], ground_truth)

        evaluation = evaluate_video_dataset(
            dataset, ground_truth, confidence_threshold=0
        )

        self.assertTrue(
            all(row["predicted_label"] == "no_lap" for row in evaluation["windows"])
        )
        self.assertEqual(evaluation["strict_window"]["counts"]["false_positives"], 0)

    def test_f1_is_zero_when_predictions_and_events_exist_without_matches(self) -> None:
        ground_truth = manifest(events=[3000])
        dataset = build(
            [
                frame(0, lap_score=0.8, candidate_ms=0, episode_id=1),
                frame(1),
                frame(2),
                frame(3),
                frame(4),
                frame(5),
            ],
            ground_truth,
        )

        evaluation = evaluate_video_dataset(
            dataset, ground_truth, confidence_threshold=0.5
        )

        self.assertEqual(evaluation["strict_window"]["metrics"]["f1"], 0)
        self.assertEqual(evaluation["temporal_tolerant"]["metrics"]["f1"], 0)

    def test_event_at_exclusive_active_interval_end_is_rejected(self) -> None:
        ground_truth = manifest(end_ms=6000, events=[6000])

        with self.assertRaisesRegex(
            WindowEvaluationError, "outside its half-open active interval"
        ):
            build([frame(index) for index in range(6)], ground_truth)

    def test_secondary_evaluation_is_excluded_from_primary_aggregate(self) -> None:
        primary_truth = manifest(events=[3000])
        secondary_truth = manifest(events=[3000], secondary=True)
        messages = [frame(index) for index in range(6)]
        primary_dataset = build(messages, primary_truth)
        secondary_dataset = build(messages, secondary_truth)
        primary_eval = evaluate_video_dataset(
            primary_dataset, primary_truth, confidence_threshold=0.5
        )
        secondary_eval = evaluate_video_dataset(
            secondary_dataset, secondary_truth, confidence_threshold=0.5
        )

        primary_aggregate = aggregate([primary_eval, secondary_eval], tier="primary")
        secondary_aggregate = aggregate(
            [primary_eval, secondary_eval], tier="secondary"
        )
        self.assertEqual(primary_aggregate["video_ids"], ["primary"])
        self.assertEqual(secondary_aggregate["video_ids"], ["secondary"])
        self.assertEqual(
            primary_aggregate["temporal_tolerant"]["counts"]["false_negatives"],
            1,
        )

    def test_invalid_parameters_scores_versions_and_episode_ids_are_rejected(
        self,
    ) -> None:
        with self.assertRaises(WindowEvaluationError):
            validate_configuration(
                window_size_ms=math.nan,
                stride_ms=2000,
                anchor_ms=0,
                coverage_threshold=0.5,
                confidence_threshold=0.5,
            )
        with self.assertRaises(WindowEvaluationError):
            validate_configuration(
                window_size_ms=2000,
                stride_ms=0,
                anchor_ms=0,
                coverage_threshold=0.5,
                confidence_threshold=0.5,
            )
        with self.assertRaises(WindowEvaluationError):
            validate_configuration(
                window_size_ms=2000,
                stride_ms=3000,
                anchor_ms=0,
                coverage_threshold=0.5,
                confidence_threshold=0.5,
            )
        with self.assertRaises(WindowEvaluationError):
            build([frame(0, lap_score=1.1)], manifest())
        with self.assertRaises(WindowEvaluationError):
            build(
                [frame(0, lap_score=0.5, candidate_ms=0, episode_id=0)],
                manifest(),
            )
        with self.assertRaises(WindowEvaluationError):
            build([frame(0, lap_score=0.5, episode_id=1)], manifest())
        with self.assertRaisesRegex(WindowEvaluationError, "no lap score version"):
            build_video_dataset(
                "primary",
                [frame(0, include_lane=False)],
                manifest(),
                window_size_ms=2000,
                stride_ms=2000,
                anchor_ms=0,
                coverage_threshold=0.5,
            )
        with self.assertRaises(WindowEvaluationError):
            build_video_dataset(
                "primary",
                [frame(0, score_version="wrong")],
                manifest(),
                window_size_ms=2000,
                stride_ms=2000,
                anchor_ms=0,
                coverage_threshold=0.5,
                expected_score_version="trajectory-test",
            )


if __name__ == "__main__":
    unittest.main()
