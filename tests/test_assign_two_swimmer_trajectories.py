from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from assign_two_swimmer_trajectories import (  # noqa: E402
    Candidate,
    TRAJECTORIES,
    TwoTrajectoryAssociator,
    expected_lane_x,
    expected_state,
)


def _candidate(index: int, *, position: float, lane_x: float, track_id: int, x: float) -> Candidate:
    return Candidate(
        source_stage="after_roi",
        source_box_index=index,
        box={"x1": x, "y1": 200.0, "x2": x + 45.0, "y2": 240.0, "conf": 0.8},
        source_track_id=track_id,
        lane_x=lane_x,
        position=position,
    )


def test_expected_motion_uses_opposite_endpoints_and_reverses() -> None:
    near, far = TRAJECTORIES
    assert expected_state(near, 3.1) == pytest.approx((1.0 - 0.1 / 23.0, -1.0 / 23.0))
    assert expected_state(far, 3.1) == pytest.approx((0.1 / 24.0, 1.0 / 24.0))
    assert expected_state(near, 30.0)[1] > 0
    assert expected_state(far, 30.0)[1] < 0
    assert expected_lane_x(near, 14.5) == 0.75
    assert expected_lane_x(far, 14.5) == 0.25
    assert expected_lane_x(near, 40.0) == 0.25
    assert expected_lane_x(far, 40.0) == 0.75


@pytest.mark.parametrize(
    ("time_seconds", "near_lane_x", "far_lane_x"),
    ((14.75, 0.75, 0.25), (39.5, 0.25, 0.75)),
)
def test_joint_assignment_preserves_identity_at_crossings(time_seconds: float, near_lane_x: float, far_lane_x: float) -> None:
    associator = TwoTrajectoryAssociator()
    near_position = expected_state(TRAJECTORIES[0], time_seconds)[0]
    far_position = expected_state(TRAJECTORIES[1], time_seconds)[0]
    candidates = [
        _candidate(0, position=near_position, lane_x=near_lane_x, track_id=101, x=600.0),
        _candidate(1, position=far_position, lane_x=far_lane_x, track_id=202, x=400.0),
    ]
    assignments = associator.assign(candidates, time_seconds)
    assigned_tracks = {config.trajectory_id: candidate.source_track_id for config, candidate, _cost in assignments}
    assert assigned_tracks == {"near_start": 101, "far_start": 202}


def test_gap_reacquires_near_slot_after_bytetrack_id_change() -> None:
    associator = TwoTrajectoryAssociator()
    first = _candidate(0, position=0.70, lane_x=0.75, track_id=10, x=600.0)
    initial = associator.assign([first], 10.0)
    assert [(config.trajectory_id, candidate.source_track_id) for config, candidate, _cost in initial] == [("near_start", 10)]

    reacquired = _candidate(0, position=0.64, lane_x=0.73, track_id=99, x=590.0)
    after_gap = associator.assign([reacquired], 11.5)
    assert [(config.trajectory_id, candidate.source_track_id) for config, candidate, _cost in after_gap] == [("near_start", 99)]


def test_duplicate_detection_is_never_assigned_to_both_slots() -> None:
    associator = TwoTrajectoryAssociator()
    time_seconds = 14.75
    position = sum(expected_state(config, time_seconds)[0] for config in TRAJECTORIES) / 2.0
    candidates = [
        _candidate(0, position=position, lane_x=0.50, track_id=10, x=500.0),
        _candidate(1, position=position + 0.005, lane_x=0.51, track_id=11, x=502.0),
    ]
    assignments = associator.assign(candidates, time_seconds)
    assert len(assignments) <= 1
    assert len({candidate.source_box_index for _config, candidate, _cost in assignments}) == len(assignments)
