#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Unit tests for CPU-only lap score replay."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from replay_lap_scores import (  # noqa: E402
    infer_fps,
    parse_messages,
    replay_messages,
    serialize_messages,
)
from summarize_tracking import parse_messages as parse_evaluation_messages  # noqa: E402


def create_fake_ai_source(root: Path) -> Path:
    ai_source = root / "swimtrack-ai"
    package = ai_source / "src" / "swimtrack_ai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "schemas.py").write_text(
        "class BoundingBox:\n"
        "    def __init__(self, **values):\n"
        "        self.values = values\n",
        encoding="utf-8",
    )
    (package / "lap_analysis.py").write_text(
        "class Score:\n"
        "    def __init__(self, value):\n"
        "        self.value = value\n"
        "    def model_dump(self, *, mode, exclude_none):\n"
        "        return {'lane_id': 'center', 'lap_score': self.value, 'score_version': 'test-current'}\n"
        "class LapAnalyzer:\n"
        "    def __init__(self, fps, calibration_id):\n"
        "        self.fps = fps\n"
        "    def observe(self, *, time_ms, width, height, boxes):\n"
        "        return [Score((time_ms / 1000.0) / self.fps + len(boxes) / 10.0)]\n",
        encoding="utf-8",
    )
    return ai_source


class LapScoreReplayTests(unittest.TestCase):
    def test_parse_replay_preserves_frame_data_and_replaces_scores(self) -> None:
        original = [
            {
                "time": 0.0,
                "width": 100,
                "height": 80,
                "boxes": [
                    {
                        "id": 7,
                        "x1": 1.0,
                        "y1": 2.0,
                        "x2": 5.0,
                        "y2": 8.0,
                        "conf": 0.9,
                    }
                ],
                "lap_scores": [{"score_version": "old"}],
                "count": 1,
            },
            {
                "time": 0.5,
                "width": 100,
                "height": 80,
                "boxes": [],
                "lap_scores": [],
                "count": 1,
                "tracking_diagnostics": {
                    "lanes": [
                        {
                            "lane_id": "center",
                            "after_roi": {
                                "count": 1,
                                "boxes": [
                                    {
                                        "x1": 2.0,
                                        "y1": 3.0,
                                        "x2": 6.0,
                                        "y2": 9.0,
                                        "conf": 0.4,
                                    }
                                ],
                            },
                        }
                    ]
                },
            },
        ]
        stream = "".join(f": frame\r\ndata: {json.dumps(frame)}\r\n\r\n" for frame in original)

        with tempfile.TemporaryDirectory() as temporary:
            ai_source = create_fake_ai_source(Path(temporary))
            parsed = parse_messages(stream)
            fps = infer_fps(parsed)
            replayed = replay_messages(
                parsed,
                ai_source=ai_source,
                fps=fps,
                calibration_id="fixed-camera-v1",
            )

        self.assertEqual(fps, 2.0)
        for source, replay in zip(original, replayed):
            self.assertEqual(replay["time"], source["time"])
            self.assertEqual(replay["width"], source["width"])
            self.assertEqual(replay["height"], source["height"])
            self.assertEqual(replay["boxes"], source["boxes"])
            self.assertEqual(replay["count"], source["count"])
        self.assertEqual(replayed[0]["lap_scores"][0]["score_version"], "test-current")
        self.assertNotEqual(replayed[0]["lap_scores"], original[0]["lap_scores"])
        self.assertAlmostEqual(replayed[1]["lap_scores"][0]["lap_score"], 0.35)

        evaluation_messages = parse_evaluation_messages(serialize_messages(replayed))
        self.assertEqual(evaluation_messages, replayed)

    def test_parser_rejects_error_events(self) -> None:
        with self.assertRaisesRegex(ValueError, "error event"):
            parse_messages('event: error\ndata: {"detail":"failed"}\n\n')

    def test_fps_requires_distinct_non_decreasing_timestamps(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot infer fps"):
            infer_fps([{"time": 1.0}, {"time": 1.0}])
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            infer_fps([{"time": 1.0}, {"time": 0.5}])


if __name__ == "__main__":
    unittest.main()
