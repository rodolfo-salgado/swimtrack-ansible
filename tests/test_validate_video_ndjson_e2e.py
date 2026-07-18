#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Unit tests for the GPU video NDJSON E2E validator."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from validate_video_ndjson_e2e import ValidationError, validate  # noqa: E402


def _frame(frame_index: int, time_ms: float) -> dict:
    return {
        "frame_index": frame_index,
        "time_ms": time_ms,
        "width": 8,
        "height": 6,
        "boxes": [
            {
                "id": 3,
                "x1": 1.0,
                "y1": 1.0,
                "x2": 4.0,
                "y2": 5.0,
                "conf": 0.9,
            }
        ],
    }


class ValidateVideoNdjsonE2ETests(unittest.TestCase):
    def _validate(self, frames: list[dict]) -> dict:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stream = Path(temporary_directory) / "stream.ndjson"
            stream.write_text(
                "\n".join(json.dumps(frame) for frame in frames) + "\n",
                encoding="utf-8",
            )
            return validate(
                argparse.Namespace(
                    stream=stream,
                    expected_events=2,
                    width=8,
                    height=6,
                    fps=10.0,
                    time_tolerance_ms=0.1,
                )
            )

    def test_accepts_ordered_frames_with_expected_timestamps(self) -> None:
        summary = self._validate([_frame(0, 0.0), _frame(1, 100.0)])
        self.assertEqual(summary["event_count"], 2)
        self.assertEqual(summary["unique_track_ids"], 1)
        self.assertEqual(summary["last_time_ms"], 100.0)

    def test_rejects_out_of_order_frame_indexes(self) -> None:
        with self.assertRaisesRegex(ValidationError, "expected frame_index 1"):
            self._validate([_frame(0, 0.0), _frame(2, 100.0)])

    def test_rejects_non_monotonic_timestamps(self) -> None:
        with self.assertRaisesRegex(ValidationError, "differs from expected"):
            self._validate([_frame(0, 0.0), _frame(1, 0.0)])


if __name__ == "__main__":
    unittest.main()
