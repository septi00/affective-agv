"""
target_selector.py
Which tracked person the robot is currently attending to.

Multi-person perception and multi-person OUTPUT are different things. Every
track gets its own features and its own emotion estimate -- that part is now
genuinely multi-person. But the fusion layer emits one 6-vector, and the robot
behaves toward one interaction partner, so something has to choose. This does.

Frame it as ATTENTION, not as a limitation: a robot attending to one partner
at a time is standard HRI practice and is what the person in front of it
expects. "Attention target selection" reads as design. "Only handles one
person" reads as unfinished.

POLICY: largest box area. It is a monotone proxy for proximity (a closer
person subtends a bigger box), it needs no depth, and it aligns selection with
reliability -- the biggest face is also the highest-quality face input, so the
attended person is the one the perception stack reads best.

HYSTERESIS: a challenger must beat the incumbent by `margin` for `patience`
consecutive frames before attention switches. Without it, two people at similar
distance trade the largest-box title every few frames and the reported emotion
jitters between two different humans.

Because per-track state lives in PerTrackState, switching attention costs
nothing -- the previous person's PoseFeatures window and EMA are still intact
if attention comes back. There is no reset-on-switch to remember, which is
exactly why the tracker was worth building rather than selecting first and
keeping one shared state object.
"""
from __future__ import annotations


def box_area(box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class TargetSelector:
    """margin: how much bigger a challenger must be to be considered at all
        (1.25 = 25% more box area). Below this, the incumbent keeps attention.
    patience: consecutive frames the challenger must hold that advantage."""

    def __init__(self, margin: float = 1.25, patience: int = 5):
        self.margin = margin
        self.patience = patience
        self.current: int | None = None
        self._challenger: int | None = None
        self._count = 0

    def update(self, tracks) -> int | None:
        """tracks: [(track_id, box), ...] from PersonTracker.update().
        Returns the attended track_id, or None if nobody is visible."""
        if not tracks:
            self.current = None
            self._reset_challenge()
            return None

        areas = {tid: box_area(box) for tid, box in tracks}
        best = max(areas, key=areas.__getitem__)

        # incumbent gone (left frame, or track died) -> take the best available
        if self.current not in areas:
            self.current = best
            self._reset_challenge()
            return self.current

        # incumbent is still the biggest -> nothing to decide
        if best == self.current:
            self._reset_challenge()
            return self.current

        # a challenger leads, but not by enough to matter
        if areas[best] < areas[self.current] * self.margin:
            self._reset_challenge()
            return self.current

        # same challenger as last frame -> build confidence, else start over
        if best == self._challenger:
            self._count += 1
        else:
            self._challenger = best
            self._count = 1

        if self._count >= self.patience:
            self.current = best
            self._reset_challenge()

        return self.current

    def _reset_challenge(self):
        self._challenger = None
        self._count = 0


if __name__ == "__main__":
    # B is much bigger from frame 2 on, but attention must not switch instantly.
    sel = TargetSelector(margin=1.25, patience=3)
    frames = [
        [(0, (0, 0, 100, 200)), (1, (200, 0, 260, 120))],   # A bigger
        [(0, (0, 0, 100, 200)), (1, (200, 0, 400, 300))],   # B takes the lead
        [(0, (0, 0, 100, 200)), (1, (200, 0, 400, 300))],
        [(0, (0, 0, 100, 200)), (1, (200, 0, 400, 300))],   # switch lands here
        [(0, (0, 0, 100, 200)), (1, (200, 0, 400, 300))],
    ]
    for i, f in enumerate(frames):
        print(f"frame {i}: attending track", sel.update(f))
