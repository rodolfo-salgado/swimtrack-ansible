#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["opencv-python-headless>=4.11.0.86,<5"]
# ///
"""Unit tests for tracker diagnostic aggregation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from summarize_tracking import SummaryError, summarize  # noqa: E402


def _frame(
    time_seconds: float,
    *,
    boxes: list[dict] | None = None,
    candidates: int = 2,
    accepted: int = 1,
    weak_candidates: int = 0,
    after_roi: int = 1,
    weak_after_roi: int = 0,
    active_ids: list[int] | None = None,
    weak_reactivated_ids: list[int] | None = None,
    lost: int = 0,
) -> dict:
    return {
        "time": time_seconds,
        "boxes": boxes or [],
        "tracking_diagnostics": {
            "diagnostic_floor": 0.05,
            "person_candidates": {"count": candidates},
            "detector_accepted": {"count": accepted},
            "weak_candidates": {"count": weak_candidates},
            "lanes": [
                {
                    "lane_id": "center",
                    "after_roi": {"count": after_roi},
                    "weak_candidates_after_roi": {"count": weak_after_roi},
                    "active_track_ids": active_ids or [],
                    "retained_lost_track_count": lost,
                    "weak_reactivated_track_ids": weak_reactivated_ids or [],
                }
            ],
        },
    }


class SummarizeTrackingTests(unittest.TestCase):
    def test_weak_diagnostic_funnel_and_reactivations_are_aggregated(self) -> None:
        box = {"id": 8, "x1": 1.0, "y1": 1.0, "x2": 3.0, "y2": 4.0, "conf": 0.9}
        summary = summarize(
            [
                _frame(0.0, boxes=[box], active_ids=[8]),
                _frame(1 / 60, boxes=[], accepted=0, weak_candidates=1, after_roi=0, weak_after_roi=1, lost=1),
                _frame(
                    2 / 60,
                    boxes=[box],
                    accepted=0,
                    weak_candidates=1,
                    after_roi=0,
                    weak_after_roi=1,
                    active_ids=[8],
                    weak_reactivated_ids=[8],
                ),
            ],
            [],
        )

        stages = summary["diagnostics"]["stages"]
        self.assertEqual(stages["person_candidates"]["observations"], 6)
        self.assertEqual(stages["weak_candidates"]["observations"], 2)
        self.assertEqual(stages["weak_candidates_after_roi"]["observations"], 2)
        self.assertEqual(
            summary["diagnostics"]["funnel"],
            {
                "candidate_to_accepted": 1 / 6,
                "accepted_to_roi": 1.0,
                "candidate_to_weak": 2 / 6,
                "weak_to_roi": 1.0,
            },
        )
        self.assertEqual(summary["diagnostics"]["weak_roi_rejected_observations"], 0)
        self.assertEqual(
            summary["tracking"]["weak_reactivations"],
            {
                "events": 1,
                "frames_nonempty": 1,
                "unique_track_ids": 1,
                "track_ids_by_lane": {"center": [8]},
            },
        )

    def test_rejects_weak_candidates_outside_the_diagnostic_funnel(self) -> None:
        with self.assertRaisesRegex(SummaryError, "weak_candidates_after_roi exceeds weak_candidates"):
            summarize(
                [
                    _frame(
                        0.0,
                        boxes=[],
                        accepted=0,
                        weak_candidates=0,
                        after_roi=0,
                        weak_after_roi=1,
                    )
                ],
                [],
            )


if __name__ == "__main__":
    unittest.main()
