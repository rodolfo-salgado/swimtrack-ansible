#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6.0.2,<7"]
# ///
"""Run a reproducible tracker-configuration sweep through the published system."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
DEFAULT_MANIFEST = PROJECT_DIR / "e2e" / "tracker-sweep.yml"
DEFAULT_RESULTS_ROOT = WORKSPACE_DIR / "results" / "tracker-sweeps"
CONFIG_KEYS = (
    "lane_roi_enabled",
    "score_threshold",
    "min_box_area",
    "track_threshold",
    "track_buffer",
    "match_threshold",
    "mot20",
)
ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class SweepError(ValueError):
    """Raised when a sweep definition or result is invalid."""


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _configuration(mapping: Any, name: str) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        raise SweepError(f"{name} must be an object")
    missing = [key for key in CONFIG_KEYS if key not in mapping]
    if missing:
        raise SweepError(f"{name} is missing {missing}")
    result = {key: mapping[key] for key in CONFIG_KEYS}
    if not isinstance(result["lane_roi_enabled"], bool) or not isinstance(result["mot20"], bool):
        raise SweepError(f"{name} lane_roi_enabled and mot20 must be booleans")
    for key in ("score_threshold", "min_box_area", "track_threshold", "match_threshold"):
        if not _is_number(result[key]):
            raise SweepError(f"{name}.{key} must be a number")
    if not isinstance(result["track_buffer"], int) or isinstance(result["track_buffer"], bool):
        raise SweepError(f"{name}.track_buffer must be an integer")
    if not 0.05 <= float(result["score_threshold"]) <= 1.0:
        raise SweepError(f"{name}.score_threshold must be between 0.05 and 1")
    if float(result["min_box_area"]) < 0:
        raise SweepError(f"{name}.min_box_area cannot be negative")
    for key in ("track_threshold", "match_threshold"):
        if not 0.0 <= float(result[key]) <= 1.0:
            raise SweepError(f"{name}.{key} must be between zero and one")
    if result["track_buffer"] <= 0:
        raise SweepError(f"{name}.track_buffer must be positive")
    return result


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or ID_PATTERN.fullmatch(value) is None:
        raise SweepError(f"{name} must match {ID_PATTERN.pattern}")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SweepError(f"could not read manifest {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise SweepError("manifest schema_version must be 1")

    restore_defaults = _configuration(raw.get("restore_defaults"), "restore_defaults")
    variants_raw = raw.get("variants")
    videos_raw = raw.get("videos")
    if not isinstance(variants_raw, list) or not variants_raw:
        raise SweepError("manifest variants must be a non-empty list")
    if not isinstance(videos_raw, list) or not videos_raw:
        raise SweepError("manifest videos must be a non-empty list")

    variants: list[dict[str, Any]] = []
    variant_ids: set[str] = set()
    for index, value in enumerate(variants_raw):
        if not isinstance(value, dict):
            raise SweepError(f"variants[{index}] must be an object")
        variant_id = _identifier(value.get("id"), f"variants[{index}].id")
        if variant_id in variant_ids:
            raise SweepError(f"duplicate variant id {variant_id!r}")
        variant_ids.add(variant_id)
        description = value.get("description", "")
        if not isinstance(description, str):
            raise SweepError(f"variant {variant_id}.description must be a string")
        variants.append(
            {
                "id": variant_id,
                "description": description,
                **_configuration(value, f"variant {variant_id}"),
            }
        )

    videos: list[dict[str, Any]] = []
    video_ids: set[str] = set()
    for index, value in enumerate(videos_raw):
        if not isinstance(value, dict):
            raise SweepError(f"videos[{index}] must be an object")
        video_id = _identifier(value.get("id"), f"videos[{index}].id")
        if video_id in video_ids:
            raise SweepError(f"duplicate video id {video_id!r}")
        video_ids.add(video_id)
        stem = value.get("video_stem")
        checksum = value.get("source_sha256")
        if not isinstance(stem, str) or not stem:
            raise SweepError(f"video {video_id}.video_stem must be a non-empty string")
        if not isinstance(checksum, str) or SHA256_PATTERN.fullmatch(checksum) is None:
            raise SweepError(f"video {video_id}.source_sha256 must be a lowercase SHA256")
        expected_events = value.get("expected_events")
        width = value.get("width")
        height = value.get("height")
        fps = value.get("fps")
        if any(not isinstance(item, int) or isinstance(item, bool) or item <= 0 for item in (expected_events, width, height)):
            raise SweepError(f"video {video_id} event count and dimensions must be positive integers")
        if not _is_number(fps) or float(fps) <= 0:
            raise SweepError(f"video {video_id}.fps must be positive")
        truth = value.get("ground_truth")
        if not isinstance(truth, dict) or truth.get("label") not in {"lap", "no_lap"}:
            raise SweepError(f"video {video_id}.ground_truth.label must be lap or no_lap")
        turn_times = truth.get("turn_times_ms")
        if not isinstance(turn_times, list) or any(not _is_number(item) or float(item) < 0 for item in turn_times):
            raise SweepError(f"video {video_id}.ground_truth.turn_times_ms must contain non-negative numbers")
        if truth["label"] == "lap" and not turn_times:
            raise SweepError(f"video {video_id} is labeled lap but has no turn timestamp")
        videos.append(
            {
                "id": video_id,
                "video_stem": stem,
                "source_sha256": checksum,
                "expected_events": expected_events,
                "width": width,
                "height": height,
                "fps": float(fps),
                "ground_truth": truth,
            }
        )
    return {
        "schema_version": 1,
        "restore_defaults": restore_defaults,
        "variants": variants,
        "videos": videos,
    }


def _select(items: list[dict[str, Any]], selected_ids: list[str] | None, kind: str) -> list[dict[str, Any]]:
    if not selected_ids:
        return items
    duplicates = sorted({item for item in selected_ids if selected_ids.count(item) > 1})
    if duplicates:
        raise SweepError(f"duplicate --{kind} selection(s): {duplicates}")
    by_id = {item["id"]: item for item in items}
    missing = [item_id for item_id in selected_ids if item_id not in by_id]
    if missing:
        raise SweepError(f"unknown --{kind} selection(s): {missing}")
    return [by_id[item_id] for item_id in selected_ids]


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _run(command: list[str]) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=PROJECT_DIR, check=True)


def _extra_vars(variant_id: str, configuration: dict[str, Any]) -> dict[str, Any]:
    return {
        "tracker_variant_id": variant_id,
        "tracker_lane_roi_enabled": configuration["lane_roi_enabled"],
        "tracker_score_threshold": configuration["score_threshold"],
        "tracker_min_box_area": configuration["min_box_area"],
        "tracker_track_threshold": configuration["track_threshold"],
        "tracker_track_buffer": configuration["track_buffer"],
        "tracker_match_threshold": configuration["match_threshold"],
        "tracker_mot20": configuration["mot20"],
    }


def configure_variant(ansible_run: Path, variant_id: str, configuration: dict[str, Any]) -> None:
    _run(
        [
            str(ansible_run),
            "ansible-playbook",
            "playbooks/configure-tracker-variant.yml",
            "--extra-vars",
            json.dumps(_extra_vars(variant_id, configuration), separators=(",", ":")),
        ]
    )


def process_video(ansible_run: Path, video: dict[str, Any], result_dir: Path) -> None:
    extra_vars = {
        "video_stem": video["video_stem"],
        "source_sha256": video["source_sha256"],
        "expected_events": video["expected_events"],
        "result_dir": str(result_dir),
    }
    _run(
        [
            str(ansible_run),
            "ansible-playbook",
            "playbooks/process-first-video.yml",
            "--extra-vars",
            json.dumps(extra_vars, separators=(",", ":")),
        ]
    )


def summarize_video(uv_path: str, video: dict[str, Any], result_dir: Path) -> None:
    command = [
        uv_path,
        "run",
        "--script",
        "scripts/summarize_tracking.py",
        "--stream",
        str(result_dir / "stream.sse"),
        "--output",
        str(result_dir / "tracking-summary.json"),
        "--require-diagnostics",
    ]
    for turn_time in video["ground_truth"]["turn_times_ms"]:
        command.extend(("--expected-turn-ms", str(turn_time)))
    _run(command)


def _nested(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _comparison_row(
    run_id: str,
    variant: dict[str, Any],
    video: dict[str, Any],
    summary: dict[str, Any],
) -> dict[str, Any]:
    lap_maximum = _nested(summary, "lap", "maximum") or {}
    row = {
        "run_id": run_id,
        "variant_id": variant["id"],
        "video_id": video["id"],
        "video_stem": video["video_stem"],
        "ground_truth_label": video["ground_truth"]["label"],
        **{key: variant[key] for key in CONFIG_KEYS},
        "event_count": summary.get("event_count"),
        "diagnostics_coverage": _nested(summary, "diagnostics", "frame_coverage"),
        "person_candidates_coverage": _nested(
            summary, "diagnostics", "stages", "person_candidates", "frame_coverage"
        ),
        "detector_accepted_coverage": _nested(
            summary, "diagnostics", "stages", "detector_accepted", "frame_coverage"
        ),
        "after_roi_coverage": _nested(summary, "diagnostics", "stages", "after_roi", "frame_coverage"),
        "active_track_coverage": _nested(summary, "tracking", "active_stage", "frame_coverage"),
        "accepted_no_track_frames": _nested(summary, "diagnostics", "accepted_no_track_frames"),
        "unique_track_ids": _nested(summary, "tracking", "unique_track_ids"),
        "fragmentations": _nested(summary, "tracking", "fragmentations"),
        "longest_run_frames": _nested(summary, "tracking", "longest_consecutive_run", "frames"),
        "longest_run_seconds": _nested(summary, "tracking", "longest_consecutive_run", "seconds"),
        "internal_gap_p95_frames": _nested(summary, "tracking", "internal_active_gaps", "frames", "p95"),
        "internal_gap_max_frames": _nested(summary, "tracking", "internal_active_gaps", "frames", "max"),
        "lost_peak": _nested(summary, "diagnostics", "retained_lost", "peak"),
        "lap_score_max": lap_maximum.get("lap_score"),
        "lap_candidate_time_ms": lap_maximum.get("candidate_time_ms"),
        "lap_candidate_error_ms": lap_maximum.get("candidate_error_ms"),
        "lap_endpoint": lap_maximum.get("endpoint"),
        "lap_observation_quality": lap_maximum.get("observation_quality"),
    }
    return row


def write_comparison(
    run_dir: Path,
    run_id: str,
    variants: list[dict[str, Any]],
    videos: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for video in videos:
            summary_path = run_dir / variant["id"] / video["video_stem"] / "tracking-summary.json"
            if not summary_path.is_file():
                continue
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise SweepError(f"could not read {summary_path}: {exc}") from exc
            if not isinstance(summary, dict):
                raise SweepError(f"{summary_path} must contain a JSON object")
            rows.append(_comparison_row(run_id, variant, video, summary))

    pairwise: list[dict[str, Any]] = []
    rows_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        rows_by_variant.setdefault(row["variant_id"], {})[row["ground_truth_label"]] = row
    for variant in variants:
        by_label = rows_by_variant.get(variant["id"], {})
        positive = by_label.get("lap")
        negative = by_label.get("no_lap")
        if positive is None or negative is None:
            continue
        positive_score = positive.get("lap_score_max")
        negative_score = negative.get("lap_score_max")
        pairwise.append(
            {
                "variant_id": variant["id"],
                "lap_score": positive_score,
                "no_lap_score": negative_score,
                "lap_score_margin": (
                    float(positive_score) - float(negative_score)
                    if positive_score is not None and negative_score is not None
                    else None
                ),
                "lap_candidate_error_ms": positive.get("lap_candidate_error_ms"),
            }
        )

    comparison = {
        "schema_version": 1,
        "run_id": run_id,
        "completed_results": len(rows),
        "expected_results": len(variants) * len(videos),
        "rows": rows,
        "pairwise_lap_separation": pairwise,
    }
    _atomic_json(run_dir / "comparison.json", comparison)
    if rows:
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        _atomic_text(run_dir / "comparison.csv", buffer.getvalue())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--run-id", help="Stable run identifier; defaults to a UTC timestamp.")
    parser.add_argument("--variant", action="append", help="Run only this variant; may be repeated.")
    parser.add_argument("--video", action="append", help="Run only this video id; may be repeated.")
    parser.add_argument("--ansible-run", type=Path, default=PROJECT_DIR / "ansible-run")
    parser.add_argument("--resume", action="store_true", help="Resume an existing run and skip completed results.")
    parser.add_argument("--list", action="store_true", help="Validate and print the selected matrix without running it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest_path = args.manifest.resolve()
        manifest = load_manifest(manifest_path)
        variants = _select(manifest["variants"], args.variant, "variant")
        videos = _select(manifest["videos"], args.video, "video")
        run_id = args.run_id or datetime.now(UTC).strftime("%Y%m%dt%H%M%Sz")
        _identifier(run_id, "run_id")
        selection = {
            "run_id": run_id,
            "manifest": str(manifest_path),
            "restore_defaults": manifest["restore_defaults"],
            "variants": variants,
            "videos": videos,
        }
        if args.list:
            print(json.dumps(selection, indent=2, sort_keys=True))
            return 0

        ansible_run = args.ansible_run.resolve()
        if not ansible_run.is_file():
            raise SweepError(f"ansible runner does not exist: {ansible_run}")
        uv_path = shutil.which("uv")
        if uv_path is None:
            raise SweepError("uv is required to execute the tracking summarizer")

        run_dir = args.results_root.resolve() / run_id
        run_state_path = run_dir / "run.json"
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        run_state = {
            "schema_version": 1,
            "run_id": run_id,
            "status": "running",
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "selected_variant_ids": [variant["id"] for variant in variants],
            "selected_video_ids": [video["id"] for video in videos],
            "started_at": datetime.now(UTC).isoformat(),
            "finished_at": None,
        }
        if run_dir.exists() and not args.resume:
            raise SweepError(f"result directory already exists: {run_dir}; use --resume or another --run-id")
        if args.resume:
            if not run_state_path.is_file():
                raise SweepError(f"cannot resume without {run_state_path}")
            previous = json.loads(run_state_path.read_text(encoding="utf-8"))
            for key in ("manifest_sha256", "selected_variant_ids", "selected_video_ids"):
                if previous.get(key) != run_state[key]:
                    raise SweepError(f"cannot resume because {key} changed")
            run_state["started_at"] = previous.get("started_at", run_state["started_at"])
        run_dir.mkdir(parents=True, exist_ok=True)
        _atomic_json(run_state_path, run_state)
        _atomic_json(run_dir / "selection.json", selection)

        try:
            try:
                for variant in variants:
                    configure_variant(ansible_run, variant["id"], variant)
                    for video in videos:
                        result_dir = run_dir / variant["id"] / video["video_stem"]
                        tracking_summary = result_dir / "tracking-summary.json"
                        if args.resume and tracking_summary.is_file():
                            print(f"Skipping completed result {variant['id']}/{video['id']}", flush=True)
                            continue
                        if result_dir.exists() and any(result_dir.iterdir()):
                            raise SweepError(f"partial result exists: {result_dir}; remove it before resuming")
                        result_dir.mkdir(parents=True, exist_ok=True)
                        _atomic_json(
                            result_dir / "experiment.json",
                            {
                                "run_id": run_id,
                                "variant": variant,
                                "video": video,
                            },
                        )
                        process_video(ansible_run, video, result_dir)
                        summarize_video(uv_path, video, result_dir)
                        write_comparison(run_dir, run_id, variants, videos)
            finally:
                print("Restoring tracker defaults", flush=True)
                configure_variant(ansible_run, "restore_defaults", manifest["restore_defaults"])
        except (Exception, KeyboardInterrupt):
            run_state["status"] = "failed"
            run_state["finished_at"] = datetime.now(UTC).isoformat()
            _atomic_json(run_state_path, run_state)
            write_comparison(run_dir, run_id, variants, videos)
            raise

        write_comparison(run_dir, run_id, variants, videos)
        run_state["status"] = "completed"
        run_state["finished_at"] = datetime.now(UTC).isoformat()
        _atomic_json(run_state_path, run_state)
        print(json.dumps({"run_id": run_id, "result_dir": str(run_dir), "status": "completed"}, sort_keys=True))
        return 0
    except (OSError, SweepError, subprocess.CalledProcessError, yaml.YAMLError) as exc:
        print(f"Tracker sweep failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
