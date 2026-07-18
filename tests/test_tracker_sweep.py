#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6.0.2,<7"]
# ///
"""Unit tests for the reproducible tracker sweep contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from run_tracker_sweep import CONFIG_KEYS, _comparison_row, _extra_vars, load_manifest  # noqa: E402


class TrackerSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = load_manifest(PROJECT_DIR / "e2e" / "tracker-sweep.yml")

    def test_weak_reactivation_pair_is_isolated_and_manifest_is_explicit(self) -> None:
        variants = {variant["id"]: variant for variant in self.manifest["variants"]}
        disabled = variants["weak_reactivation_off"]
        enabled = variants["weak_reactivation_default"]

        self.assertTrue(self.manifest["restore_defaults"]["weak_reactivation_enabled"])
        for variant in self.manifest["variants"]:
            self.assertTrue(set(CONFIG_KEYS).issubset(variant))
        for key in CONFIG_KEYS:
            if key == "weak_reactivation_enabled":
                continue
            self.assertEqual(disabled[key], enabled[key])
        self.assertFalse(disabled["weak_reactivation_enabled"])
        self.assertTrue(enabled["weak_reactivation_enabled"])

    def test_extra_vars_propagate_every_weak_reactivation_setting(self) -> None:
        configuration = next(
            variant
            for variant in self.manifest["variants"]
            if variant["id"] == "weak_reactivation_default"
        )
        extra_vars = _extra_vars("weak_reactivation_default", configuration)

        self.assertEqual(
            {
                key: extra_vars[f"tracker_{key}"]
                for key in (
                    "weak_reactivation_enabled",
                    "weak_reactivation_score_threshold",
                    "weak_reactivation_min_box_area",
                    "weak_reactivation_max_gap_seconds",
                    "weak_reactivation_max_center_distance",
                )
            },
            {
                key: configuration[key]
                for key in (
                    "weak_reactivation_enabled",
                    "weak_reactivation_score_threshold",
                    "weak_reactivation_min_box_area",
                    "weak_reactivation_max_gap_seconds",
                    "weak_reactivation_max_center_distance",
                )
            },
        )

    def test_configure_playbook_requires_weak_reactivation_inputs_without_defaults(self) -> None:
        playbook = yaml.safe_load(
            (PROJECT_DIR / "playbooks" / "configure-tracker-variant.yml").read_text(encoding="utf-8")
        )
        self.assertIsInstance(playbook, list)
        variables = playbook[0]["vars"]
        required = next(
            task for task in playbook[0]["tasks"] if task["name"] == "Require every tracker experiment variable"
        )["ansible.builtin.assert"]["that"]

        for key in (
            "weak_reactivation_enabled",
            "weak_reactivation_score_threshold",
            "weak_reactivation_min_box_area",
            "weak_reactivation_max_gap_seconds",
            "weak_reactivation_max_center_distance",
        ):
            self.assertNotIn(f"tracker_{key}", variables)
            self.assertIn(f"tracker_{key} is defined", required)

    def test_comparison_row_contains_weak_reactivation_observations(self) -> None:
        variant = next(variant for variant in self.manifest["variants"] if variant["id"] == "weak_reactivation_default")
        video = self.manifest["videos"][0]
        summary = {
            "event_count": 3,
            "diagnostics": {
                "frame_coverage": 1.0,
                "stages": {
                    "person_candidates": {"frame_coverage": 1.0},
                    "detector_accepted": {"frame_coverage": 1.0},
                    "after_roi": {"frame_coverage": 1.0},
                    "weak_candidates": {"frame_coverage": 2 / 3},
                    "weak_candidates_after_roi": {"frame_coverage": 1 / 3},
                },
                "accepted_no_track_frames": 1,
                "retained_lost": {"peak": 2},
            },
            "tracking": {
                "active_stage": {"frame_coverage": 2 / 3},
                "unique_track_ids": 1,
                "fragmentations": 0,
                "longest_consecutive_run": {"frames": 1, "seconds": 1 / 60},
                "internal_active_gaps": {"frames": {"p95": 1.0, "max": 1}},
                "weak_reactivations": {"events": 1, "frames_nonempty": 1, "unique_track_ids": 1},
            },
            "lap": {"maximum": None},
        }

        row = _comparison_row("run", variant, video, summary)

        self.assertEqual(row["weak_reactivation_events"], 1)
        self.assertEqual(row["weak_reactivation_frames"], 1)
        self.assertEqual(row["weak_reactivation_unique_track_ids"], 1)
        self.assertEqual(row["weak_candidates_coverage"], 2 / 3)
        self.assertEqual(row["weak_candidates_after_roi_coverage"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
