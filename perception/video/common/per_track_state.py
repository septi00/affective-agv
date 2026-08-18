"""
per_track_state.py
One instance of some stateful object per tracked person, created on first
sight and dropped when the track dies.

This is the fix for the shared-state bug. `pose_live.py` builds ONE
PoseFeatures before the pipeline and calls `pf.update(pts)` for every
detection in the frame, so `self.prev` alternates between people and the
"velocity" it computes is the gap between two different bodies. Same class of
bug applies to EmaSmoother on the face side: one EMA vector fed two people
returns the average of their emotional states as if it were fact.

Deliberately generic -- it takes a zero-argument factory and knows nothing
about PoseFeatures, EmaSmoother, or emotion. That keeps it importable from
both video/pose_est/ and video/face_rec/src/ without either directory having
to import the other, and it stays a pure data structure with no I/O.

Usage:
    pose_state = PerTrackState(lambda: PoseFeatures(intrinsics=cam_intrinsics))
    ema_state  = PerTrackState(EmaSmoother)

    for tid, box in tracker.update(boxes):
        op, mean, burst = pose_state.get(tid).update(pts_for[tid])

    pose_state.evict(tracker.dead_ids)      # <- do not skip this
    ema_state.evict(tracker.dead_ids)
"""
from __future__ import annotations

from typing import Callable, Iterable


class PerTrackState:
    """factory: called with no arguments to build the state object for a newly
    seen track ID. Bind any constructor arguments with a lambda or partial."""

    def __init__(self, factory: Callable[[], object]):
        self._factory = factory
        self._state: dict[int, object] = {}

    def get(self, track_id: int):
        """State for this track, created on first request."""
        if track_id not in self._state:
            self._state[track_id] = self._factory()
        return self._state[track_id]

    def has(self, track_id: int) -> bool:
        return track_id in self._state

    def evict(self, track_ids: Iterable[int]) -> None:
        """Drop state for dead tracks. Call every frame with tracker.dead_ids.
        Without this the dict grows for the whole run -- one PoseFeatures with
        a 15-frame deque per person who ever walked past the robot."""
        for tid in track_ids:
            self._state.pop(tid, None)

    def clear(self) -> None:
        self._state.clear()

    def __len__(self) -> int:
        return len(self._state)

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._state
