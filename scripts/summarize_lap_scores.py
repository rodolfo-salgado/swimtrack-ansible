#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Summarize heuristic lane scores carried by a SwimTrack SSE stream."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def parse_messages(stream: str) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for block in stream.replace("\r\n", "\n").split("\n\n"):
        if not block.strip():
            continue
        event_name = "message"
        data_lines: list[str] = []
        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[6:].strip() or "message"
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if event_name == "error":
            raise ValueError(f"stream contains an error event: {' '.join(data_lines)}")
        if event_name != "message" or not data_lines:
            continue
        payload = json.loads("\n".join(data_lines))
        if not isinstance(payload, dict):
            raise ValueError("SSE data must be a JSON object")
        messages.append(payload)
    return messages


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def summarize(messages: list[dict[str, Any]]) -> dict[str, Any]:
    lane_scores: dict[str, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    candidate_scores: dict[tuple[str, str, float], tuple[float, float, dict[str, Any]]] = {}
    frames_with_scores = 0
    for frame in messages:
        scores = frame.get("lap_scores")
        if scores is None:
            continue
        if not isinstance(scores, list):
            raise ValueError("lap_scores must be a list")
        frames_with_scores += 1
        frame_time = float(frame["time"])
        for score in scores:
            if not isinstance(score, dict):
                raise ValueError("each lap score must be an object")
            lane_id = str(score["lane_id"])
            lap_score = float(score["lap_score"])
            lane_scores[lane_id].append((frame_time, score))
            candidate_time = score.get("candidate_time_ms")
            endpoint = score.get("endpoint")
            if candidate_time is None or endpoint is None:
                continue
            key = (lane_id, str(endpoint), round(float(candidate_time), 3))
            previous = candidate_scores.get(key)
            if previous is None or lap_score > previous[0]:
                candidate_scores[key] = (lap_score, frame_time, score)

    if not lane_scores:
        raise ValueError("stream does not contain lap_scores")

    lanes: dict[str, Any] = {}
    for lane_id, entries in lane_scores.items():
        scores = [float(score["lap_score"]) for _, score in entries]
        qualities = [float(score["observation_quality"]) for _, score in entries]
        evaluable = [(time_seconds, score) for time_seconds, score in entries if score.get("evaluable") is True]
        maximum_time, maximum = max(entries, key=lambda item: float(item[1]["lap_score"]))
        lanes[lane_id] = {
            "score_version": sorted({str(score["score_version"]) for _, score in entries}),
            "frames": len(entries),
            "evaluable_frames": len(evaluable),
            "first_evaluable_time_seconds": evaluable[0][0] if evaluable else None,
            "lap_score": {
                "min": min(scores),
                "mean": mean(scores),
                "p50": percentile(scores, 0.50),
                "p95": percentile(scores, 0.95),
                "max": max(scores),
                "max_frame_time_seconds": maximum_time,
                "max_candidate_time_ms": maximum.get("candidate_time_ms"),
                "max_endpoint": maximum.get("endpoint"),
                "max_evidence": maximum.get("evidence"),
            },
            "observation_quality": {
                "min": min(qualities),
                "mean": mean(qualities),
                "max": max(qualities),
            },
        }

    top_candidates = sorted(candidate_scores.values(), key=lambda item: item[0], reverse=True)[:10]
    return {
        "event_count": len(messages),
        "frames_with_lap_scores": frames_with_scores,
        "lanes": lanes,
        "top_candidates": [
            {
                "lap_score": score_value,
                "frame_time_seconds": frame_time,
                "lane_id": score["lane_id"],
                "endpoint": score.get("endpoint"),
                "candidate_time_ms": score.get("candidate_time_ms"),
                "observation_quality": score["observation_quality"],
                "evidence": score["evidence"],
            }
            for score_value, frame_time, score in top_candidates
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = summarize(parse_messages(args.stream.read_text(encoding="utf-8")))
    serialized = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
