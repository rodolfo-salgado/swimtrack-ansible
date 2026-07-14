#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6.0.2,<7"]
# ///
"""Unit tests for temporal lap-event evaluation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from evaluate_lap_events import aggregate, evaluate_video, load_manifest  # noqa: E402


def frame(
    time_seconds: float,
    candidate_ms: float,
    score: float,
    *,
    lane_id: str = "center",
    episode_id: int | None = None,
) -> dict:
    return {
        "time": time_seconds,
        "boxes": [],
        "lap_scores": [
            {
                "lane_id": lane_id,
                "lap_score": score,
                "evaluable": True,
                "candidate_time_ms": candidate_ms,
                "endpoint": "far",
                "track_id": 1,
                **(
                    {"candidate_episode_id": episode_id}
                    if episode_id is not None
                    else {}
                ),
            }
        ],
    }


class LapEventEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(PROJECT_DIR / "e2e" / "lap-ground-truth.yml")

    def test_manifest_preserves_primary_and_secondary_scope(self) -> None:
        policy = self.manifest["temporal_evaluation"]
        self.assertEqual(
            policy["primary_video_ids"], [f"test{index:02d}" for index in range(1, 9)]
        )
        self.assertEqual(policy["secondary_video_ids"], ["test09"])
        self.assertEqual(
            sum(
                len(video["turn_events"]) for video in self.manifest["videos"].values()
            ),
            8,
        )
        self.assertEqual(
            self.manifest["videos"]["test06"]["turn_events"][0]["timestamp_ms"], 19000
        )

    def test_threshold_deduplication_and_one_to_one_temporal_matching(self) -> None:
        messages = [
            frame(24.0, 22100, 0.80),
            frame(24.1, 23000, 0.70),
            frame(47.0, 49100, 0.90),
            frame(50.0, 52000, 0.49),
        ]
        result = evaluate_video("test08", messages, self.manifest, threshold=0.50)
        self.assertEqual(
            result["counts"],
            {"true_positives": 1, "false_positives": 1, "false_negatives": 1},
        )
        self.assertEqual(len(result["predicted_events"]), 2)
        self.assertEqual(result["matches"][0]["absolute_error_ms"], 1900)
        self.assertEqual(result["false_negative_events"][0]["timestamp_ms"], 47000)

    def test_candidates_outside_subject_interval_are_ignored(self) -> None:
        result = evaluate_video(
            "test01", [frame(1.0, 1000, 0.95)], self.manifest, threshold=0.50
        )
        self.assertEqual(
            result["counts"],
            {"true_positives": 0, "false_positives": 0, "false_negatives": 0},
        )
        self.assertEqual(result["ignored_candidates_outside_subject_intervals"], 1)

    def test_candidate_episode_is_counted_once_beyond_legacy_time_window(self) -> None:
        result = evaluate_video(
            "test04",
            [
                frame(30.0, 29000, 0.70, episode_id=1),
                frame(32.0, 31500, 0.90, episode_id=1),
            ],
            self.manifest,
            threshold=0.50,
        )
        self.assertEqual(len(result["predicted_events"]), 1)
        self.assertEqual(result["predicted_events"][0]["candidate_episode_id"], 1)
        self.assertEqual(
            result["counts"],
            {"true_positives": 1, "false_positives": 0, "false_negatives": 0},
        )

    def test_distinct_candidate_episodes_remain_distinct_predictions(self) -> None:
        result = evaluate_video(
            "test04",
            [
                frame(30.0, 29500, 0.80, episode_id=1),
                frame(30.1, 30500, 0.70, episode_id=2),
            ],
            self.manifest,
            threshold=0.50,
        )
        self.assertEqual(len(result["predicted_events"]), 2)
        self.assertEqual(
            result["counts"],
            {"true_positives": 1, "false_positives": 1, "false_negatives": 0},
        )

    def test_secondary_video_does_not_contribute_to_primary_aggregate(self) -> None:
        primary = evaluate_video("test04", [], self.manifest, threshold=0.50)
        secondary = evaluate_video("test09", [], self.manifest, threshold=0.50)
        primary_result = aggregate([primary, secondary], primary_only=True)
        all_result = aggregate([primary, secondary], primary_only=False)
        self.assertEqual(primary_result["video_ids"], ["test04"])
        self.assertEqual(primary_result["counts"]["false_negatives"], 1)
        self.assertEqual(all_result["counts"]["false_negatives"], 3)


if __name__ == "__main__":
    unittest.main()
