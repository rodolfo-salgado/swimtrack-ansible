#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Unit tests for canonical-identity SSE contract validation."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from validate_sse_e2e import ValidationError, validate  # noqa: E402


def _frame(
    frame_index: int,
    time: float,
    *,
    confirmed_count: int | None = 1,
    active_count: int | None = 1,
) -> dict:
    frame = {
        "time": time,
        "width": 8,
        "height": 6,
        "count": 1,
        "boxes": [
            {
                "id": 3,
                "track_id": 3,
                "identity_id": 1,
                "x1": 1.0,
                "y1": 1.0,
                "x2": 4.0,
                "y2": 5.0,
                "conf": 0.9,
            }
        ],
    }
    if confirmed_count is not None and active_count is not None:
        frame["identity_summary"] = {
            "confirmed_count": confirmed_count,
            "active_count": active_count,
        }
    return frame


class ValidateSseE2ETests(unittest.TestCase):
    def _validate(self, frames: list[dict], *, expected_confirmed_identities: int | None = 1) -> dict:
        with tempfile.TemporaryDirectory() as temporary_directory:
            stream = Path(temporary_directory) / "stream.sse"
            stream.write_text(
                "".join(f"data: {json.dumps(frame)}\n\n" for frame in frames),
                encoding="utf-8",
            )
            return validate(
                argparse.Namespace(
                    stream=stream,
                    expected_events=2,
                    width=8,
                    height=6,
                    fps=10.0,
                    time_tolerance=0.001,
                    expected_confirmed_identities=expected_confirmed_identities,
                )
            )

    def test_accepts_the_canonical_identity_contract(self) -> None:
        summary = self._validate([_frame(0, 0.0), _frame(1, 0.1)])

        self.assertEqual(summary["final_confirmed_identity_count"], 1)
        self.assertEqual(summary["max_confirmed_identity_count"], 1)
        self.assertEqual(summary["max_active_identity_count"], 1)

    def test_requires_identity_summary_when_an_expected_person_count_is_given(self) -> None:
        with self.assertRaisesRegex(ValidationError, "missing identity_summary"):
            self._validate([_frame(0, 0.0), _frame(1, 0.1, confirmed_count=None, active_count=None)])

    def test_rejects_a_different_final_identity_count(self) -> None:
        with self.assertRaisesRegex(ValidationError, "expected final and maximum confirmed identity count"):
            self._validate([_frame(0, 0.0), _frame(1, 0.1, confirmed_count=2, active_count=2)])


if __name__ == "__main__":
    unittest.main()
