"""
person_tracker.py
Greedy IoU tracker: per-frame person boxes -> stable integer track IDs.

One job: answer "is this the same person as last frame?" Nothing else.
No keypoints, no features, no drawing, no hardware.

WHY THIS EXISTS: PoseFeatures and EmaSmoother are both STATEFUL over time
(prev wrists, speed window, EMA). Temporal state is only meaningful if every
update belongs to the same person. Without stable IDs, a two-person frame
makes PoseFeatures subtract person A's wrist position from person B's, which
is not a velocity -- it's the distance between two people, and it registers as
huge activity even when nobody moves.

Video analog of multi_tracker.py on the audio side: that one associates SRP-PHAT
peaks across time into DOA tracks; this one associates person boxes across time
into person tracks. Same problem, different measurement space (pixels vs azimuth).

Deliberately NOT a Kalman tracker. Person boxes at 30fps move a few pixels per
frame and overlap heavily between consecutive frames, so IoU association is
sufficient and has no tuning surface beyond one threshold. A motion model would
buy prediction through occlusion, which matters for long dropouts -- documented
as a limitation rather than built.
"""
from __future__ import annotations

from dataclasses import dataclass


def iou(a, b) -> float:
    """Intersection-over-union of two (x1, y1, x2, y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    id: int
    box: tuple          # (x1, y1, x2, y2) as of the last frame this track was seen
    age: int = 0        # consecutive frames since last matched (0 = seen this frame)
    hits: int = 1       # total frames this track has been matched


class PersonTracker:
    """Stateful across frames. One instance per video stream.

    iou_thresh: minimum overlap to call two boxes the same person. 0.3 is loose
        enough for fast walking at 30fps, tight enough that two people standing
        side by side don't swap.
    max_age: frames a track survives unmatched before it is declared dead and
        its ID retired. Covers brief detector dropout and partial occlusion.
    min_hits: frames a new track must be matched before it is reported. Filters
        single-frame false positives, which would otherwise each spawn a track,
        allocate per-track state, and immediately die.
    """

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 15, min_hits: int = 3):
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self.min_hits = min_hits
        self._tracks: list[Track] = []
        self._next_id = 0
        self.dead_ids: list[int] = []   # IDs retired by the LAST update() call

    def update(self, boxes) -> list[tuple[int, tuple]]:
        """boxes: list of (x1, y1, x2, y2) person boxes detected this frame.

        Returns [(track_id, box), ...] for CONFIRMED tracks seen THIS frame --
        i.e. only real, currently-visible people. Coasting tracks (matched
        recently but not this frame) are deliberately withheld: feeding a stale
        box into PoseFeatures would fabricate zero motion for someone who may
        well be moving.

        After calling this, read `dead_ids` and evict the corresponding
        per-track state, or it leaks for the lifetime of the process.
        """
        boxes = [tuple(float(v) for v in b) for b in boxes]
        tracks = self._tracks

        # --- greedy association: best overlaps claimed first ---
        pairs = []
        for ti, t in enumerate(tracks):
            for di, b in enumerate(boxes):
                v = iou(t.box, b)
                if v >= self.iou_thresh:
                    pairs.append((v, ti, di))
        pairs.sort(key=lambda p: -p[0])

        matched_t: set[int] = set()
        matched_d: set[int] = set()
        for _v, ti, di in pairs:
            if ti in matched_t or di in matched_d:
                continue
            matched_t.add(ti)
            matched_d.add(di)
            t = tracks[ti]
            t.box = boxes[di]
            t.age = 0
            t.hits += 1

        # --- age unmatched tracks, retire the ones past max_age ---
        self.dead_ids = []
        kept: list[Track] = []
        for ti, t in enumerate(tracks):
            if ti not in matched_t:
                t.age += 1
                if t.age > self.max_age:
                    self.dead_ids.append(t.id)
                    continue
            kept.append(t)

        # --- unmatched detections become new tracks ---
        for di, b in enumerate(boxes):
            if di not in matched_d:
                kept.append(Track(id=self._next_id, box=b))
                self._next_id += 1

        self._tracks = kept
        return [(t.id, t.box) for t in self._tracks
                if t.age == 0 and t.hits >= self.min_hits]

    def reset(self):
        """Drop all tracks. IDs continue counting up, so a retired ID is never
        reused for a different person -- per-track state keyed by ID stays safe."""
        self._tracks = []
        self.dead_ids = []


if __name__ == "__main__":
    # Two people, one walking right, one standing still. IDs must stay stable.
    tr = PersonTracker(min_hits=1)
    for f in range(5):
        a = (100 + f * 12, 50, 180 + f * 12, 250)   # moving
        b = (300, 60, 380, 260)                      # still
        print(f"frame {f}:", tr.update([a, b]))

    print("-- person A leaves --")
    for f in range(5, 9):
        print(f"frame {f}:", tr.update([(300, 60, 380, 260)]), "dead:", tr.dead_ids)
